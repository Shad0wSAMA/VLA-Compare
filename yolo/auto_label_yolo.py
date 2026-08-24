"""Create LabelMe annotations for images that have no JSON annotation yet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = SCRIPT_DIR / "dataset_raw"
DEFAULT_WEIGHTS = WORKSPACE_DIR / "outputs" / "yolo" / "cube_detector" / "weights" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Folder containing images")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="YOLO best.pt; default: cube_detector/best.pt or the newest best.pt",
    )
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default=None, help="Device: 0, cpu, etc. Default: auto")
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="Also write empty JSON files when no object is detected",
    )
    return parser.parse_args()


def find_weights(requested: Path | None) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Weights do not exist: {path}")
        return path

    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS.resolve()

    candidates = list((WORKSPACE_DIR / "outputs" / "yolo").glob("**/weights/best.pt"))
    if not candidates:
        raise SystemExit(
            "No trained best.pt was found. Train the model first or pass --weights PATH.\n"
            f"Expected default location: {DEFAULT_WEIGHTS}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names[class_id])
    return str(names[class_id])


def build_labelme_document(result: Any, image_name: str) -> dict[str, Any]:
    height, width = (int(value) for value in result.orig_shape)
    shapes: list[dict[str, Any]] = []

    if result.boxes is not None:
        coordinates = result.boxes.xyxy.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        for xyxy, raw_class_id, confidence in zip(coordinates, class_ids, confidences):
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            class_id = int(raw_class_id)
            shapes.append(
                {
                    "label": class_name(result.names, class_id),
                    "points": [
                        [max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1))],
                        [max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))],
                    ],
                    "group_id": None,
                    "description": f"YOLO auto-label confidence: {confidence:.4f}",
                    "shape_type": "rectangle",
                    "flags": {"auto_labeled": True},
                    "mask": None,
                }
            )

    return {
        "version": "7.0.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source folder does not exist: {source}")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--iou must be between 0 and 1")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")

    weights = find_weights(args.weights)
    images = sorted(
        path
        for path in source.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.with_suffix(".json").exists()
    )
    if not images:
        print(f"No unannotated images found in {source}")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install it with: pip install ultralytics") from exc

    print(f"Weights: {weights}")
    print(f"Unannotated images: {len(images)}")
    print(f"Confidence threshold: {args.confidence}")
    model = YOLO(str(weights))
    predict_options: dict[str, Any] = {
        "source": [str(path) for path in images],
        "conf": args.confidence,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "stream": True,
        "verbose": False,
    }
    if args.device not in (None, ""):
        predict_options["device"] = args.device

    saved = 0
    empty = 0
    skipped_empty = 0
    for result in model.predict(**predict_options):
        image_path = Path(result.path).resolve()
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            print(f"Skip existing annotation: {json_path.name}")
            continue

        document = build_labelme_document(result, image_path.name)
        if not document["shapes"]:
            empty += 1
            if not args.save_empty:
                skipped_empty += 1
                print(f"No detection, JSON not written: {image_path.name}")
                continue

        temporary_path = json_path.with_suffix(json_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(json_path)
        saved += 1
        print(f"Saved: {json_path.name} ({len(document['shapes'])} box(es))")

    print(f"Finished. JSON files saved: {saved}")
    print(f"Images with no detection: {empty} (not saved: {skipped_empty})")
    print("Please review auto-labeled boxes in LabelMe before training again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
