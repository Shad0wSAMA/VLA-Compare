"""Load YOLO weights and optionally run cube selection on one image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import DEFAULT_CONFIG
from aruco_pick_place import load_config, resolve_local_path, select_cube


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--image", type=Path, help="RGB inference test image; omitted means model-load test only")
    args = parser.parse_args()
    config, config_dir = load_config(args.config)
    weights = args.weights.resolve() if args.weights else resolve_local_path(config["perception"].get("weights"), config_dir)
    if weights is None or not weights.is_file():
        print("SKIP: no YOLO weights yet. Pass --weights path/to/best.pt after training.")
        return 0
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if args.image is None:
        print(f"PASS: YOLO model loaded: {weights}")
        return 0
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {args.image}")
    height, width = image_bgr.shape[:2]
    bounds = np.asarray(config["perception"]["table_xy_bounds_m"], dtype=np.float32)
    src = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    dst = np.asarray(
        [[bounds[0, 0], bounds[1, 0]], [bounds[0, 1], bounds[1, 0]],
         [bounds[0, 1], bounds[1, 1]], [bounds[0, 0], bounds[1, 1]]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    detection = select_cube(model, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), homography, config["perception"])
    print(
        f"PASS: confidence={detection.confidence:.3f}, pixel={detection.pixel_xy.round(1).tolist()}, "
        f"normalized-table={detection.table_xy_m.round(5).tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
