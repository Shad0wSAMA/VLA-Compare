"""Test dual-camera configuration; add --open to access camera hardware."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import DEFAULT_CONFIG
from aruco_pick_place import load_config
from aruco_pick_place.hardware import make_camera_configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--open", action="store_true", help="Explicitly open both physical cameras")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/camera_test"))
    args = parser.parse_args()
    config, _ = load_config(args.config)
    camera_configs = make_camera_configs(config["robot"])
    assert set(camera_configs) == {"wrist", "fixed"}
    for name, camera_config in camera_configs.items():
        print(
            f"{name}: source={camera_config.index_or_path}, "
            f"{camera_config.width}x{camera_config.height}@{camera_config.fps}, fourcc={camera_config.fourcc}"
        )
    if not args.open:
        print("PASS: both LeRobot camera configurations constructed. Hardware was not opened (use --open).")
        return 0

    from lerobot.cameras.opencv import OpenCVCamera

    cameras = {name: OpenCVCamera(camera_config) for name, camera_config in camera_configs.items()}
    try:
        for camera in cameras.values():
            camera.connect()
        frames = {}
        for _ in range(max(1, args.frames)):
            frames = {name: camera.read() for name, camera in cameras.items()}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame_rgb in frames.items():
            output = args.output_dir / f"{name}.jpg"
            if not cv2.imwrite(str(output), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not write {output}")
            print(f"saved {name}: shape={frame_rgb.shape} -> {output.resolve()}")
    finally:
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
    print("PASS: both physical cameras opened and returned frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
