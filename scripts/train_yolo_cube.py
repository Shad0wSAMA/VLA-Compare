"""Train an Ultralytics YOLO detector for the tabletop cube."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("training/yolo_cube/dataset.yaml"))
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained .pt or model .yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Examples: cpu, 0, 0,1. Default: auto")
    parser.add_argument("--project", type=Path, default=Path("outputs/yolo"))
    parser.add_argument("--name", default="cube")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a last.pt checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data_path}")
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
        train_args = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": str(args.project.expanduser().resolve()),
            "name": args.name,
        }
        if args.device not in (None, ""):
            train_args["device"] = args.device
        results = model.train(**train_args)

    save_dir = Path(results.save_dir)
    print(f"Training complete. Best weights: {save_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
