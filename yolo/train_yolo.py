"""Train an Ultralytics YOLO object detector on the converted dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
DEFAULT_DATA = SCRIPT_DIR / "dataset" / "dataset.yaml"
DEFAULT_PROJECT = WORKSPACE_DIR / "outputs" / "yolo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained weights or model YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Device: 0, 0,1, cpu, etc. Default: auto")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="cube_detector")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    parser.add_argument(
        "--degrees",
        type=float,
        default=20.0,
        help="Random training rotation range in degrees (default: +/-15)",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a last.pt checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(
            f"Dataset YAML does not exist: {data_path}\n"
            "Run labelme_to_yolo.py before training."
        )
    for name in ("epochs", "imgsz", "batch"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name} must be positive")
    if not 0.0 <= args.degrees <= 180.0:
        raise SystemExit("--degrees must be between 0 and 180")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install it with: pip install ultralytics") from exc

    if args.resume is not None:
        checkpoint = args.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"Resume checkpoint does not exist: {checkpoint}")
        model = YOLO(str(checkpoint))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        train_options = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": str(args.project.expanduser().resolve()),
            "name": args.name,
            "patience": args.patience,
            "degrees": args.degrees,
        }
        if args.device not in (None, ""):
            train_options["device"] = args.device
        results = model.train(**train_options)

    save_dir = Path(results.save_dir).resolve()
    print(f"Training finished. Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {save_dir / 'weights' / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
