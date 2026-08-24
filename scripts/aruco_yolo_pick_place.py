"""Run the modular ArUco + YOLO + IKPy SO-101 pick-and-place pipeline.

Use ``--check-config`` first.  It performs no camera or robot I/O.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Re-export the old public helpers so existing imports remain compatible.
from aruco_pick_place import (  # noqa: F401
    Detection,
    detect_marker_boundary_points,
    detect_marker_centers,
    estimate_table_homography,
    load_config,
    pixel_to_table_xy,
    resolve_local_path,
    select_cube,
    table_target_to_robot_pose,
    undistort_if_configured,
    validate_config,
)
from aruco_pick_place.pipeline import PickPlaceRunner


LOGGER = logging.getLogger("aruco_yolo_pick_place")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/aruco_yolo_pick_place.example.json"),
        help="JSON configuration file.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Only validate configuration and local files; do not open cameras or the robot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        config, config_dir = load_config(args.config)
        errors = validate_config(config, config_dir)
        if errors:
            print("Configuration is not ready:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 2
        if args.check_config:
            print("Configuration and required local files look ready.")
            return 0
        PickPlaceRunner(config, config_dir).run()
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user; disconnect cleanup will run.")
        return 130
    except Exception:
        LOGGER.exception("Pick-and-place failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
