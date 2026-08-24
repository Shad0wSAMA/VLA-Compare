r"""一键训练并定期验证 LeRobot ACT。

默认行为：
  1. 自动选择 training/dataN 中编号最大的 dataset；
  2. 每 2500 步保存 checkpoint 并在 ./validation 上验证；
  3. 总训练步数固定为 20000，不做 early stopping；
  4. 当前 act_dataN 有 checkpoint 时恢复训练；如果只有残留目录则删除后重新开始，并尝试加载 act_dataN-1 的最后模型。

示例：
    .\venv\Scripts\python.exe scripts\train_with_validation.py
    .\venv\Scripts\python.exe scripts\train_with_validation.py --dataset-number 7 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
VENV_TRAIN = WORKSPACE / "venv" / "Scripts" / "lerobot-train.exe"
VENV_PYTHON = WORKSPACE / "venv" / "Scripts" / "python.exe"
VALIDATION_SCRIPT = WORKSPACE / "scripts" / "eval_validation.py"
CHUNK_STEPS = 2_500
TOTAL_STEPS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT and validate every 2500 steps")
    parser.add_argument("--dataset-number", type=int, default=None, help="数据集编号 N；默认自动选择最新 dataN")
    parser.add_argument("--output-dir", type=Path, default=None, help="自定义训练输出目录")
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--validation-every", type=int, default=CHUNK_STEPS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=24)
    parser.add_argument("--validation-num-workers", type=int, default=0)
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "torchcodec"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行训练和验证")
    return parser.parse_args()


def find_dataset(dataset_number: int | None) -> tuple[int, Path]:
    training_root = WORKSPACE / "training"
    if dataset_number is not None:
        if dataset_number < 1:
            raise ValueError("dataset-number 必须是正整数")
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
        raise FileNotFoundError(f"找不到训练数据集: {training_root}")
    dataset = candidates[-1]
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"不是有效的 LeRobot dataset: {dataset}")
    return int(dataset.name.removeprefix("data")), dataset


def split_episode_indices(dataset_root: Path) -> tuple[list[int], list[int]]:
    with (dataset_root / "meta" / "info.json").open(encoding="utf-8") as file:
        total_episodes = int(json.load(file)["total_episodes"])
    if total_episodes < 2:
        raise ValueError("至少需要 2 个 episode 才能按 80/20 划分")
    train_count = max(1, min(total_episodes - 1, int(total_episodes * 0.8)))
    indices = list(range(total_episodes))
    return indices[:train_count], indices[train_count:]


def episode_argument(indices: list[int]) -> str:
    return "[" + ",".join(str(index) for index in indices) + "]"


def valid_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    return sorted(
        [
            path for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
            and (path / "pretrained_model" / "train_config.json").is_file()
        ],
        key=lambda path: int(path.name),
    )


def latest_previous_policy(dataset_number: int) -> Path | None:
    for number in range(dataset_number - 1, 0, -1):
        run_dir = WORKSPACE / "outputs" / "train" / f"act_data{number}"
        checkpoints = valid_checkpoints(run_dir)
        if checkpoints:
            return checkpoints[-1] / "pretrained_model"
    return None


def train_command(
    dataset_number: int,
    dataset_root: Path,
    output_dir: Path,
    target_steps: int,
    args: argparse.Namespace,
    resume_config: Path | None,
    initial_policy: Path | None,
    train_episodes: list[int],
) -> list[str]:
    command = [
        str(VENV_TRAIN),
        f"--dataset.repo_id=data{dataset_number}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={args.video_backend}",
        f"--dataset.episodes={episode_argument(train_episodes)}",
        f"--policy.device={'cpu' if args.cpu else 'cuda'}",
        f"--policy.use_amp={'true' if args.amp else 'false'}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name=act_data{dataset_number}",
        f"--batch_size={args.batch_size}",
        f"--steps={target_steps}",
        f"--save_freq={args.validation_every}",
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


def validation_command(
    output_dir: Path,
    dataset_number: int,
    dataset_root: Path,
    validation_episodes: list[int],
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(VENV_PYTHON),
        str(VALIDATION_SCRIPT),
        f"--run-dir={output_dir}",
        f"--model-number={dataset_number}",
        f"--dataset={dataset_root}",
        f"--episodes={episode_argument(validation_episodes)}",
        f"--batch-size={args.validation_batch_size}",
        f"--num-workers={args.validation_num_workers}",
        f"--video-backend={args.video_backend}",
        f"--device={'cpu' if args.cpu else 'cuda'}",
    ]


def parse_train_losses(text: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    suffixes = {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}
    pattern = re.compile(r"step:(\d+(?:\.\d+)?)([KMG]?).*?\bloss:([0-9.eE+-]+)")
    for match in pattern.finditer(text):
        step = int(float(match.group(1)) * suffixes[match.group(2)])
        points.append((step, float(match.group(3))))
    return points


def parse_eval_loss(text: str) -> float | None:
    matches = re.findall(r"eval_loss:\s*([0-9.eE+-]+)", text)
    return float(matches[-1]) if matches else None


def load_existing_history(output_dir: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train_points: list[tuple[int, float]] = []
    validation_points: list[tuple[int, float]] = []
    for log_path in sorted(output_dir.glob("train_until_*.log")):
        train_points.extend(parse_train_losses(log_path.read_text(encoding="utf-8", errors="replace")))
    history_path = output_dir / "train_validation_curve.csv"
    if history_path.is_file():
        with history_path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("validation_loss"):
                    validation_points.append((int(row["step"]), float(row["validation_loss"])))
    return train_points, validation_points


def save_loss_plot(
    output_dir: Path,
    train_points: list[tuple[int, float]],
    validation_points: list[tuple[int, float]],
) -> None:
    if not train_points and not validation_points:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "train_validation_curve.csv"
    validation_by_step = dict(validation_points)
    steps = sorted(set(step for step, _ in train_points) | set(validation_by_step))
    train_by_step = {}
    for step, loss in train_points:
        train_by_step[step] = loss
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "train_loss", "validation_loss"])
        for step in steps:
            writer.writerow([step, train_by_step.get(step, ""), validation_by_step.get(step, "")])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    if train_by_step:
        train_sorted = sorted(train_by_step.items())
        plt.plot([x for x, _ in train_sorted], [y for _, y in train_sorted], label="train loss", alpha=0.8)
    if validation_by_step:
        validation_sorted = sorted(validation_by_step.items())
        plt.plot(
            [x for x, _ in validation_sorted],
            [y for _, y in validation_sorted],
            marker="o",
            linewidth=2,
            label="validation loss",
        )
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"{output_dir.name}: train / validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "train_validation_curve.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Loss curve saved: {plot_path}")


def run_command(command: list[str], dry_run: bool, log_path: Path | None = None) -> str:
    print("\n$", " ".join(f'"{item}"' if " " in item else item for item in command))
    if dry_run:
        return ""
    env = {
        **os.environ,
        "PYTHONPATH": str(WORKSPACE / "lerobot" / "src"),
        "HF_HOME": str(WORKSPACE / "training" / "hf_home"),
        "HF_DATASETS_CACHE": str(WORKSPACE / "training" / "hf_cache"),
    }
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8") if log_path else None
    process = subprocess.Popen(
        command,
        cwd=WORKSPACE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output.append(line)
        if log_file:
            log_file.write(line)
    return_code = process.wait()
    if log_file:
        log_file.close()
    if return_code != 0:
        raise SystemExit(return_code)
    return "".join(output)


def main() -> int:
    args = parse_args()
    if args.total_steps < 1 or args.validation_every < 1:
        raise ValueError("total-steps 和 validation-every 必须是正整数")
    if args.validation_every > args.total_steps:
        raise ValueError("validation-every 不能大于 total-steps")
    if args.batch_size < 1 or args.num_workers < 0 or args.validation_batch_size < 1:
        raise ValueError("batch-size 和 worker 参数不合法")
    if not VENV_TRAIN.is_file() or not VENV_PYTHON.is_file():
        raise FileNotFoundError("找不到 workspace venv 中的训练器或 Python")

    dataset_number, dataset_root = find_dataset(args.dataset_number)
    train_episodes, validation_episodes = split_episode_indices(dataset_root)
    base_output_dir = WORKSPACE / "outputs" / "train" / f"act_data{dataset_number}"
    output_dir = args.output_dir.resolve() if args.output_dir is not None else base_output_dir
    if output_dir.exists() and not valid_checkpoints(output_dir):
        if args.dry_run:
            print(f"检测到无 checkpoint 的残留目录；实际运行时将删除: {output_dir}")
        else:
            print(f"检测到无 checkpoint 的残留目录，删除后重新训练: {output_dir}")
            shutil.rmtree(output_dir)
    current_checkpoints = valid_checkpoints(output_dir)
    current_step = int(current_checkpoints[-1].name) if current_checkpoints else 0
    train_points, validation_points = load_existing_history(output_dir)

    if current_step >= args.total_steps:
        print(f"act_data{dataset_number} 已经有 {current_step} 步 checkpoint，无需继续训练。")
        return 0

    print(f"dataset: {dataset_root}")
    print(f"output: {output_dir}")
    print(f"episode split: {len(train_episodes)} train / {len(validation_episodes)} validation (80/20)")
    print(f"progress: {current_step} -> {args.total_steps}, validation every {args.validation_every} steps")

    initial_policy = None
    if current_step == 0:
        initial_policy = latest_previous_policy(dataset_number)
        if initial_policy:
            print(f"从上一个 dataset 的 checkpoint 初始化: {initial_policy}")
        else:
            print("没有找到上一个 checkpoint，将从头初始化 ACT。")

    while current_step < args.total_steps:
        target_step = min(current_step + args.validation_every, args.total_steps)
        if current_step > 0:
            checkpoint_dir = output_dir / "checkpoints" / f"{current_step:06d}"
            resume_config = checkpoint_dir / "pretrained_model" / "train_config.json"
            if not args.dry_run and not resume_config.is_file():
                raise FileNotFoundError(f"checkpoint 缺少 train_config.json: {resume_config}")
        else:
            resume_config = None

        if output_dir.exists():
            train_log_path = output_dir / f"train_until_{target_step:06d}.log"
        else:
            train_log_path = WORKSPACE / "outputs" / "train_logs" / f"act_data{dataset_number}_until_{target_step:06d}.log"
        train_output = run_command(
            train_command(
                dataset_number, dataset_root, output_dir, target_step,
                args, resume_config, initial_policy, train_episodes,
            ),
            args.dry_run,
            train_log_path,
        )
        if not args.dry_run:
            train_points.extend(parse_train_losses(train_output))

        validation_log_path = output_dir / f"validation_until_{target_step:06d}.log"
        validation_output = run_command(
        validation_command(output_dir, dataset_number, dataset_root, validation_episodes, args),
            args.dry_run,
            validation_log_path,
        )
        validation_loss = parse_eval_loss(validation_output)
        if validation_loss is not None:
            validation_points.append((target_step, validation_loss))
            save_loss_plot(output_dir, train_points, validation_points)
        current_step = target_step
        initial_policy = None

    print(f"训练和 validation 完成：act_data{dataset_number}，共 {args.total_steps} 步。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
