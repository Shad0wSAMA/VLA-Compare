"""Run real-time YOLO detection with a USB camera."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2


WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = WORKSPACE_DIR / "outputs" / "yolo" / "cube_detector" / "weights" / "best.pt"
WINDOW_NAME = "YOLO Real-time Detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS (default: 30)")
    parser.add_argument("--weights", type=Path, default=None, help="YOLO best.pt path")
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

    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS.resolve()

    candidates = list((WORKSPACE_DIR / "outputs" / "yolo").glob("**/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "no best.pt found; train the model first or pass --weights PATH"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def open_camera(index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height and fps must be positive")

    # DirectShow generally applies USB camera format settings reliably on Windows.
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

    # MJPG commonly enables 640x480 at 30 FPS on USB cameras.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--iou must be between 0 and 1")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")

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
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_camera_fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"Weights: {weights}")
    print(f"Camera {args.camera}: {actual_width}x{actual_height} @ {actual_camera_fps:.1f} FPS")
    print("Press Q or Esc to quit.")

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
                f"FPS: {display_fps:.1f}  Objects: {detection_count}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
