r"""Evaluate LeRobot checkpoints on an independent validation dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "lerobot" / "src"))
os.environ.setdefault("HF_HOME", str(WORKSPACE / "training" / "hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", str(WORKSPACE / "training" / "hf_cache_eval"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LeRobot policy on validation data")
    parser.add_argument(
        "--model-number", type=int, default=None,
        help="N in outputs/train/act_dataN; omitted means the newest model",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="直接指定训练输出目录")
    parser.add_argument(
        "--all-checkpoints", action=argparse.BooleanOptionalAction, default=False,
        help="validate every numbered checkpoint and save a loss curve",
    )
    parser.add_argument("--dataset", type=Path, default=WORKSPACE / "validation")
    parser.add_argument("--episodes", default=None, help="JSON episode list, for example [120,121,122]")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "torchcodec"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    return parser.parse_args()


def resolve_run_dir(model_number: int | None, explicit_run_dir: Path | None = None) -> Path:
    train_root = WORKSPACE / "outputs" / "train"
    if explicit_run_dir is not None:
        run_dir = explicit_run_dir.resolve()
    elif model_number is not None:
        if model_number < 1:
            raise ValueError("model-number must be a positive integer")
        run_dir = train_root / f"act_data{model_number}"
    else:
        run_dirs = sorted(
            [
                path for path in train_root.glob("act_data*")
                if path.is_dir() and path.name.removeprefix("act_data").isdigit()
            ],
            key=lambda path: int(path.name.removeprefix("act_data")),
        )
        if not run_dirs:
            raise FileNotFoundError(f"No act_dataN output directories found: {train_root}")
        run_dir = run_dirs[-1]
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training output directory not found: {run_dir}")
    return run_dir


def numbered_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints = [
        path for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "pretrained_model" / "config.json").is_file()
    ] if checkpoints_dir.is_dir() else []
    return sorted(checkpoints, key=lambda path: int(path.name))


def latest_policy_path(run_dir: Path) -> Path:
    last = run_dir / "checkpoints" / "last" / "pretrained_model"
    if (last / "config.json").is_file():
        return last
    checkpoints = numbered_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No valid checkpoints found: {run_dir / 'checkpoints'}")
    return checkpoints[-1] / "pretrained_model"


def evaluate_checkpoint(
    policy_path: Path,
    dataset: LeRobotDataset,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, int]:
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = str(device)
    # make_policy only loads model.safetensors when pretrained_path is set.
    policy_cfg.pretrained_path = policy_path
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    policy.eval()

    total_loss = 0.0
    total_samples = 0
    total_batches = 0
    progress_total = len(loader) if not max_batches else min(len(loader), max_batches)
    with torch.inference_mode():
        progress = tqdm(loader, total=progress_total, desc=f"Checkpoint {policy_path.parent.name}", unit="batch")
        for batch in progress:
            if max_batches and total_batches >= max_batches:
                break
            for cam_key in dataset.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            batch_size = next(iter(batch.values())).shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            total_batches += 1
            progress.set_postfix(eval_loss=f"{total_loss / total_samples:.4f}")
        progress.close()

    del preprocessor, policy
    if total_samples == 0:
        raise RuntimeError("Validation dataset returned no samples")
    return total_loss / total_samples, total_samples


def save_curve(run_dir: Path, results: list[tuple[int, float]]) -> None:
    csv_path = run_dir / "validation_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["checkpoint", "eval_loss"])
        writer.writerows(results)

    import matplotlib.pyplot as plt

    steps, losses = zip(*results)
    plt.figure(figsize=(9, 5))
    plt.plot(steps, losses, marker="o")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Validation eval_loss")
    plt.title(f"{run_dir.name} validation curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    image_path = run_dir / "validation_curve.png"
    plt.savefig(image_path, dpi=160)
    plt.close()
    print(f"Saved curve: {image_path}")
    print(f"Saved results: {csv_path}")


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.model_number, args.run_dir)
    dataset_root = args.dataset.resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset: {dataset_root}")
    if args.batch_size < 1 or args.num_workers < 0 or args.max_batches < 0:
        raise ValueError("batch-size, num-workers, and max-batches must be non-negative/positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    with (dataset_root / "meta" / "info.json").open(encoding="utf-8") as file:
        dataset_info = json.load(file)
    episode_indices = json.loads(args.episodes) if args.episodes else None
    if episode_indices is not None and (
        not isinstance(episode_indices, list) or any(not isinstance(index, int) for index in episode_indices)
    ):
        raise ValueError("episodes must be a JSON list of integers")

    all_checkpoints = numbered_checkpoints(run_dir)
    if args.all_checkpoints:
        if not all_checkpoints:
            raise FileNotFoundError(f"No numbered checkpoints found: {run_dir / 'checkpoints'}")
        policy_paths = [path / "pretrained_model" for path in all_checkpoints]
    else:
        policy_paths = [latest_policy_path(run_dir)]

    first_cfg = PreTrainedConfig.from_pretrained(policy_paths[0])
    first_cfg.device = str(device)
    dataset_meta = LeRobotDatasetMetadata(repo_id="validation", root=dataset_root)
    dataset = LeRobotDataset(
        repo_id="validation",
        root=dataset_root,
        episodes=episode_indices,
        delta_timestamps=resolve_delta_timestamps(first_cfg, dataset_meta),
        video_backend=args.video_backend,
        return_uint8=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    print(f"model: {run_dir.name}")
    print(f"validation: {dataset_root} ({dataset.num_episodes} episodes, {dataset.num_frames} frames)")
    results: list[tuple[int, float]] = []
    for policy_path in policy_paths:
        checkpoint = policy_path.parent.name
        print(f"\nEvaluating checkpoint {checkpoint}: {policy_path}")
        loss, samples = evaluate_checkpoint(policy_path, dataset, loader, device, args.max_batches)
        if args.all_checkpoints:
            results.append((int(checkpoint), loss))
        print(f"eval_loss: {loss:.6f}; samples: {samples}")

    if args.all_checkpoints:
        save_curve(run_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
