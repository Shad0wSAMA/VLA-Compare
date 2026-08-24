"""Collect raw images from a camera for a YOLO dataset.

Controls:
    Space / S: save the current frame
    Q / Esc:   quit
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset_raw"
WINDOW_NAME = "YOLO Dataset Collector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture raw images for YOLO training.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Image output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def next_image_path(output_dir: Path) -> Path:
    """Return a collision-resistant, chronologically sortable image path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = output_dir / f"image_{timestamp}.jpg"
    suffix = 1
    while path.exists():
        path = output_dir / f"image_{timestamp}_{suffix:02d}.jpg"
        suffix += 1
    return path


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    # CAP_DSHOW usually starts USB cameras faster on Windows. Fall back to the
    # default backend for platforms/cameras where DirectShow is unavailable.
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

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main() -> int:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        capture = open_camera(args.camera, args.width, args.height)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    saved_count = 0
    print(f"Camera: {args.camera}, requested resolution: {args.width}x{args.height}")
    print(f"Save directory: {output_dir}")
    print("Press Space/S to capture; press Q/Esc to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Error: failed to read a frame from the camera.")
                return 1

            # Some camera drivers ignore the requested capture resolution.
            # Resize before saving so every dataset image is exactly 640x480
            # (or the dimensions supplied on the command line).
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            preview = frame.copy()
            cv2.putText(
                preview,
                f"SPACE/S: capture  Q/ESC: quit  Saved: {saved_count}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S"), 32):
                image_path = next_image_path(output_dir)
                if cv2.imwrite(str(image_path), frame):
                    saved_count += 1
                    print(f"Saved [{saved_count}]: {image_path}")
                else:
                    print(f"Warning: failed to save {image_path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Finished. Saved {saved_count} image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
