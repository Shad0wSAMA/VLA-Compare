"""Offline test for JSON structure, paths, camera mapping, and known placeholders."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_CONFIG
from aruco_pick_place.config import load_config, resolve_local_path, validate_config


EXPECTED_SETUP_BLOCKERS = (
    "perception.weights",
    "table_to_robot_confirmed",
    "motion.poses_confirmed",
    "recording.repo_id",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config, config_dir = load_config(Path(args.config))
    errors = validate_config(config, config_dir)
    cameras = config["robot"]["cameras"]
    assert cameras["wrist"]["index_or_path"] == 2
    assert cameras["fixed"]["index_or_path"] == 1
    assert config["robot"]["perception_camera"] == "fixed"
    urdf = resolve_local_path(config["kinematics"]["urdf_path"], config_dir)
    assert urdf is not None and urdf.is_file(), f"URDF missing: {urdf}"
    unexpected = [error for error in errors if not any(token in error for token in EXPECTED_SETUP_BLOCKERS)]
    if unexpected:
        print("FAIL: unexpected configuration errors:")
        for error in unexpected:
            print(f"  - {error}")
        return 1
    print("PASS: config structure, dual-camera mapping, and URDF path are valid.")
    if errors:
        print("Expected setup blockers (must be filled before running the robot):")
        for error in errors:
            print(f"  - {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
