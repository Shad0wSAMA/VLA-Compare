"""One-command SmolVLA fine-tuning launcher for the local LeRobot dataset.

Typical AutoDL usage from the workspace root::

    python scripts/train_smolvla.py

The defaults below are intentionally kept together so the common settings can
also be edited directly in this file. Every important setting can be overridden
from the command line; run ``python scripts/train_smolvla.py --help`` for details.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]

# ----------------------------- Default settings -----------------------------
DEFAULT_DATASET_NAME = "manual_data2"
DEFAULT_DATASET_ROOT = WORKSPACE / "training" / DEFAULT_DATASET_NAME
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs" / "train" / f"smolvla_{DEFAULT_DATASET_NAME}"
DEFAULT_POLICY = "lerobot/smolvla_base"
DEFAULT_RENAME_MAP = {
    "observation.images.fixed": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}

DEFAULT_STEPS = 10000
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 12
DEFAULT_SAVE_CHECKPOINT = True
DEFAULT_SAVE_FREQ = 4000
DEFAULT_RESUME = True
DEFAULT_CHECKPOINT: Path | None = None

DEFAULT_DEVICE = "cuda"
DEFAULT_USE_AMP = True
DEFAULT_MIXED_PRECISION = "bf16"
DEFAULT_VIDEO_BACKEND = "torchcodec"
DEFAULT_LOG_FREQ = 50
DEFAULT_WANDB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on a local LeRobot dataset")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="Base model for a fresh run")
    parser.add_argument(
        "--rename-map",
        type=json.loads,
        default=DEFAULT_RENAME_MAP,
        help="JSON mapping from dataset camera keys to the pretrained policy camera keys",
    )

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Total target training steps")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_CHECKPOINT,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME,
        help="Automatically resume the latest checkpoint in output-dir",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Explicit checkpoint directory, pretrained_model directory, or train_config.json",
    )

    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_AMP)
    parser.add_argument(
        "--mixed-precision",
        default=DEFAULT_MIXED_PRECISION,
        choices=("bf16", "fp16"),
        help="Accelerate mixed precision mode used when AMP is enabled",
    )
    parser.add_argument("--video-backend", default=DEFAULT_VIDEO_BACKEND, choices=("pyav", "torchcodec"))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=DEFAULT_WANDB)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting training")
    return parser.parse_args()


def valid_checkpoints(output_dir: Path) -> list[Path]:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and (path / "pretrained_model" / "train_config.json").is_file()
        ),
        key=lambda path: int(path.name),
    )


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = (
        path,
        path / "train_config.json",
        path / "pretrained_model" / "train_config.json",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "train_config.json":
            return candidate
    raise FileNotFoundError(f"Could not find train_config.json under checkpoint: {path}")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset, missing: {dataset_root / 'meta' / 'info.json'}")
    if args.steps < 1 or args.batch_size < 1 or args.save_freq < 1 or args.log_freq < 1:
        raise ValueError("steps, batch-size, save-freq, and log-freq must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if not isinstance(args.rename_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in args.rename_map.items()
    ):
        raise ValueError("rename-map must be a JSON object whose keys and values are strings")
    if args.checkpoint is not None and not args.resume:
        raise ValueError("--checkpoint requires --resume")

    return dataset_root, output_dir


def build_command(args: argparse.Namespace, dataset_root: Path, output_dir: Path) -> tuple[list[str], int]:
    checkpoint_config: Path | None = None
    checkpoint_step = 0

    if args.checkpoint is not None:
        checkpoint_config = resolve_checkpoint(args.checkpoint)
        checkpoint_dir = checkpoint_config.parents[1]
        if checkpoint_dir.name.isdigit():
            checkpoint_step = int(checkpoint_dir.name)
    elif args.resume:
        checkpoints = valid_checkpoints(output_dir)
        if checkpoints:
            checkpoint_dir = checkpoints[-1]
            checkpoint_config = checkpoint_dir / "pretrained_model" / "train_config.json"
            checkpoint_step = int(checkpoint_dir.name)

    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.dataset_name}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={args.video_backend}",
        f"--dataset.return_uint8=true",
        f"--rename_map={json.dumps(args.rename_map, separators=(',', ':'))}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.amp).lower()}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name=smolvla_{args.dataset_name}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--num_workers={args.num_workers}",
        f"--save_checkpoint={str(args.save_checkpoint).lower()}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--wandb.enable={str(args.wandb).lower()}",
        "--save_checkpoint_to_hub=false",
    ]

    if checkpoint_config is not None:
        command.extend(("--resume=true", f"--config_path={checkpoint_config}"))
    else:
        if output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists and has no resumable checkpoint: {output_dir}\n"
                "Use a different --output-dir, provide --checkpoint, or remove the incomplete output manually."
            )
        command.append(f"--policy.path={args.policy}")

    return command, checkpoint_step


def stream_process(command: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=WORKSPACE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return process.wait()


def main() -> int:
    args = parse_args()
    dataset_root, output_dir = validate_args(args)
    command, checkpoint_step = build_command(args, dataset_root, output_dir)

    if checkpoint_step >= args.steps:
        print(f"Training is already complete: checkpoint {checkpoint_step} >= target {args.steps} steps")
        return 0

    env = os.environ.copy()
    source_dir = str(WORKSPACE / "lerobot" / "src")
    env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", str(WORKSPACE / ".cache" / "huggingface"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["ACCELERATE_MIXED_PRECISION"] = args.mixed_precision if args.amp else "no"

    log_path = WORKSPACE / "outputs" / "train_logs" / f"smolvla_{args.dataset_name}.log"
    print(f"Dataset:    {dataset_root}")
    print(f"Output:     {output_dir}")
    print(f"Steps:      {checkpoint_step} -> {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Precision:  {args.mixed_precision if args.amp else 'float32'}")
    print(f"Log:        {log_path}")
    print("Command:", shlex.join(command))

    if args.dry_run:
        return 0

    if args.device.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed in the current Python environment") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; run with the CUDA-enabled AutoDL Python environment")
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return stream_process(command, env, log_path)


if __name__ == "__main__":
    raise SystemExit(main())
