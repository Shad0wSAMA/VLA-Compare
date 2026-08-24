"""Run real-time YOLO detection and capture LabelMe image/JSON pairs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset_raw"
WINDOW_NAME = "YOLO Real-time LabelMe Capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS (default: 30)")
    parser.add_argument("--weights", type=Path, default=None, help="YOLO best.pt path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Image and JSON output folder")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default=None, help="Device: 0, cpu, etc. Default: auto")
    return parser.parse_args()


def find_weights(requested: Path | None) -> Path:
    if requested is not None:
        weights = requested.expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"weights do not exist: {weights}")
        return weights

    candidates = list((WORKSPACE_DIR / "outputs" / "yolo").glob("**/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "no best.pt found; train the model first or pass --weights PATH"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def open_camera(index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height and fps must be positive")

    if hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open camera {index}")

    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names[class_id])
    return str(names[class_id])


def labelme_document(result: Any, image_name: str) -> dict[str, Any]:
    height, width = (int(value) for value in result.orig_shape)
    shapes: list[dict[str, Any]] = []
    if result.boxes is not None:
        coordinates = result.boxes.xyxy.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        for xyxy, raw_class_id, confidence in zip(coordinates, class_ids, confidences):
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            shapes.append(
                {
                    "label": class_name(result.names, int(raw_class_id)),
                    "points": [
                        [max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1))],
                        [max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))],
                    ],
                    "group_id": None,
                    "description": f"YOLO capture confidence: {confidence:.4f}",
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


def next_capture_paths(output: Path) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = f"realtime_{timestamp}"
    image_path = output / f"{stem}.jpg"
    suffix = 1
    while image_path.exists() or image_path.with_suffix(".json").exists():
        image_path = output / f"{stem}_{suffix:02d}.jpg"
        suffix += 1
    return image_path, image_path.with_suffix(".json")


def save_capture(output: Path, frame: Any, result: Any) -> tuple[Path, int]:
    image_path, json_path = next_capture_paths(output)
    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f"failed to save image: {image_path}")

    document = labelme_document(result, image_path.name)
    temporary_json = json_path.with_suffix(".json.tmp")
    try:
        temporary_json.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_json.replace(json_path)
    except Exception:
        temporary_json.unlink(missing_ok=True)
        # The image was created by this failed capture and has no usable JSON pair.
        image_path.unlink(missing_ok=True)
        raise
    return image_path, len(document["shapes"])


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--iou must be between 0 and 1")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        weights = find_weights(args.weights)
        capture = open_camera(args.camera, args.width, args.height, args.fps)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        capture.release()
        raise SystemExit("Missing ultralytics. Install it with: pip install ultralytics")

    model = YOLO(str(weights))
    print(f"Weights: {weights}")
    print(f"Output: {output}")
    print(
        f"Camera {args.camera}: {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{capture.get(cv2.CAP_PROP_FPS):.1f} FPS"
    )
    print("Press Space to save the image and LabelMe JSON; press Q or Esc to quit.")

    predict_options: dict[str, Any] = {
        "conf": args.confidence,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device not in (None, ""):
        predict_options["device"] = args.device

    display_fps = 0.0
    previous_time = time.perf_counter()
    saved_count = 0
    status_text = ""
    status_until = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Error: failed to read a frame from the camera")
                return 1
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            result = model.predict(source=frame, **predict_options)[0]
            annotated = result.plot()
            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-9)
            previous_time = now
            display_fps = instantaneous_fps if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instantaneous_fps
            detection_count = 0 if result.boxes is None else len(result.boxes)

            cv2.putText(
                annotated,
                f"FPS: {display_fps:.1f}  Objects: {detection_count}  Saved: {saved_count}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "SPACE: save image + LabelMe JSON   Q/ESC: quit",
                (10, args.height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if status_text and now < status_until:
                cv2.putText(
                    annotated,
                    status_text,
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                try:
                    image_path, box_count = save_capture(output, frame, result)
                    saved_count += 1
                    status_text = f"Saved {image_path.name} ({box_count} boxes)"
                    status_until = time.perf_counter() + 2.0
                    print(status_text)
                except OSError as exc:
                    status_text = f"Save failed: {exc}"
                    status_until = time.perf_counter() + 3.0
                    print(f"Error: {exc}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Finished. Saved {saved_count} image/JSON pair(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
