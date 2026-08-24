r"""使用 LeRobot 原生 eval_split/eval_steps 完成训练和 validation。

这个脚本只使用最新的 training/dataN，不读取 validation 文件夹。
LeRobot 会在同一个 dataset 内按 episode 做划分：默认 80% train、20% eval，
并在每隔 2500 个 training steps 时输出 eval_loss。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
VENV_TRAIN = WORKSPACE / "venv" / "Scripts" / "lerobot-train.exe"
TOTAL_STEPS = 20_000
EVAL_STEPS = 2_500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with native LeRobot train/eval split")
    parser.add_argument("--dataset-number", type=int, default=None)
    parser.add_argument("--eval-split", type=float, default=0.2, help="fraction of episodes held out for eval")
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "torchcodec"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_dataset(dataset_number: int | None) -> tuple[int, Path]:
    training_root = WORKSPACE / "training"
    if dataset_number is not None:
        if dataset_number < 1:
            raise ValueError("dataset-number must be positive")
        candidates = [training_root / f"data{dataset_number}"]
    else:
        candidates = sorted(
            [
                path for path in training_root.glob("data*")
                if path.is_dir() and path.name.removeprefix("data").isdigit()
            ],
            key=lambda path: int(path.name.removeprefix("data")),
        )
    if not candidates or not candidates[-1].is_dir():
        raise FileNotFoundError(f"No training dataset found: {training_root}")
    dataset = candidates[-1]
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset: {dataset}")
    return int(dataset.name.removeprefix("data")), dataset


def checkpoints(run_dir: Path) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        [
            path for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
            and (path / "pretrained_model" / "train_config.json").is_file()
        ],
        key=lambda path: int(path.name),
    )


def previous_policy(dataset_number: int) -> Path | None:
    for number in range(dataset_number - 1, 0, -1):
        candidates = checkpoints(WORKSPACE / "outputs" / "train" / f"act_data{number}")
        if candidates:
            return candidates[-1] / "pretrained_model"
    return None


def make_command(
    dataset_number: int,
    dataset_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    resume_config: Path | None,
    initial_policy: Path | None,
) -> list[str]:
    command = [
        str(VENV_TRAIN),
        f"--dataset.repo_id=data{dataset_number}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={args.video_backend}",
        f"--dataset.eval_split={args.eval_split}",
        "--max_eval_samples=0",
        f"--policy.device={'cpu' if args.cpu else 'cuda'}",
        f"--policy.use_amp={'true' if args.amp else 'false'}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name=act_data{dataset_number}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.total_steps}",
        f"--eval_steps={args.eval_steps}",
        "--env_eval_freq=0",
        f"--save_freq={args.eval_steps}",
        f"--num_workers={args.num_workers}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
        "--save_checkpoint_to_hub=false",
    ]
    if resume_config is not None:
        command.extend(["--resume=true", f"--config_path={resume_config}"])
    elif initial_policy is not None:
        command.append(f"--policy.path={initial_policy}")
    else:
        command.append("--policy.type=act")
    return command


def parse_metrics(text: str) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train: list[tuple[int, float]] = []
    evaluation: list[tuple[int, float]] = []
    train_pattern = re.compile(r"step:(\d+(?:\.\d+)?)([KMG]?).*?\bloss:([0-9.eE+-]+)")
    eval_pattern = re.compile(r"step\s+(\d+):\s+eval_loss=([0-9.eE+-]+)")
    multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}
    for match in train_pattern.finditer(text):
        train.append((int(float(match.group(1)) * multipliers[match.group(2)]), float(match.group(3))))
    for match in eval_pattern.finditer(text):
        evaluation.append((int(match.group(1)), float(match.group(2))))
    return train, evaluation


def save_plot(output_dir: Path, train: list[tuple[int, float]], evaluation: list[tuple[int, float]]) -> None:
    if not train and not evaluation:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "native_train_validation_curve.csv"
    train_by_step = dict(train)
    eval_by_step = dict(evaluation)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "train_loss", "validation_loss"])
        for step in sorted(set(train_by_step) | set(eval_by_step)):
            writer.writerow([step, train_by_step.get(step, ""), eval_by_step.get(step, "")])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    if train_by_step:
        points = sorted(train_by_step.items())
        plt.plot([x for x, _ in points], [y for _, y in points], label="train loss")
    if eval_by_step:
        points = sorted(eval_by_step.items())
        plt.plot([x for x, _ in points], [y for _, y in points], marker="o", label="validation loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"{output_dir.name}: native train / validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "native_train_validation_curve.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Loss curve saved: {plot_path}")
    print(f"Loss data saved: {csv_path}")


def run(command: list[str], args: argparse.Namespace, log_path: Path) -> str:
    print("\n$", " ".join(f'"{part}"' if " " in part else part for part in command))
    if args.dry_run:
        return ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": str(WORKSPACE / "lerobot" / "src"),
        "HF_HOME": str(WORKSPACE / "training" / "hf_home"),
        "HF_DATASETS_CACHE": str(WORKSPACE / "training" / "hf_cache"),
    }
    process = subprocess.Popen(
        command,
        cwd=WORKSPACE,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            output.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)
    return "".join(output)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.eval_split < 1.0:
        raise ValueError("eval-split must be between 0 and 1")
    if args.total_steps < 1 or args.eval_steps < 1:
        raise ValueError("total-steps and eval-steps must be positive")
    if not VENV_TRAIN.is_file():
        raise FileNotFoundError(f"Training executable not found: {VENV_TRAIN}")

    dataset_number, dataset_root = find_dataset(args.dataset_number)
    output_dir = WORKSPACE / "outputs" / "train" / f"act_data{dataset_number}"
    current_checkpoints = checkpoints(output_dir)
    if output_dir.exists() and not current_checkpoints:
        if args.dry_run:
            print(f"Residual directory will be removed during a real run: {output_dir}")
        else:
            print(f"Removing residual directory without checkpoints: {output_dir}")
            shutil.rmtree(output_dir)
    current_checkpoints = checkpoints(output_dir)
    current_step = int(current_checkpoints[-1].name) if current_checkpoints else 0
    if current_step >= args.total_steps:
        print(f"{output_dir} already has checkpoint {current_step}; nothing to train.")
        return 0

    if current_step:
        resume_config = current_checkpoints[-1] / "pretrained_model" / "train_config.json"
        initial_policy = None
    else:
        resume_config = None
        initial_policy = previous_policy(dataset_number)

    print(f"dataset: {dataset_root}")
    print(f"output: {output_dir}")
    print(f"native split: {1.0 - args.eval_split:.0%} train / {args.eval_split:.0%} eval")
    print(f"steps: {current_step} -> {args.total_steps}; eval every {args.eval_steps} steps")
    if initial_policy:
        print(f"initializing from previous checkpoint: {initial_policy}")
    elif current_step:
        print(f"resuming from checkpoint: {current_checkpoints[-1]}")

    log_path = output_dir / "native_train.log" if output_dir.exists() else WORKSPACE / "outputs" / "train_logs" / f"act_data{dataset_number}_native.log"
    output = run(
        make_command(dataset_number, dataset_root, output_dir, args, resume_config, initial_policy),
        args,
        log_path,
    )
    if not args.dry_run:
        train, evaluation = parse_metrics(output)
        save_plot(output_dir, train, evaluation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
