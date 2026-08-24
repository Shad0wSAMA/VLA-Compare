"""Calibrate the configured LeRobot robot or leader arm.

Examples::

    python scripts/calibrate.py robot
    python scripts/calibrate.py teleop
"""

from __future__ import annotations

import argparse

import lerobot_config as defaults
from _lerobot_cli import run_lerobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("robot", "teleop"), help="Which device to calibrate")
    parser.add_argument("--type", dest="device_type", default=None, help="Override the configured device type")
    parser.add_argument("--port", default=None, help="Override the configured serial port")
    parser.add_argument("--id", dest="device_id", default=None, help="Override the configured calibration ID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "robot":
        prefix = "robot"
        device_type = args.device_type or defaults.ROBOT_TYPE
        port = args.port or defaults.ROBOT_PORT
        device_id = args.device_id or defaults.ROBOT_ID
    else:
        prefix = "teleop"
        device_type = args.device_type or defaults.TELEOP_TYPE
        port = args.port or defaults.TELEOP_PORT
        device_id = args.device_id or defaults.TELEOP_ID

    return run_lerobot(
        "lerobot-calibrate",
        [f"--{prefix}.type={device_type}", f"--{prefix}.port={port}", f"--{prefix}.id={device_id}"],
    )


if __name__ == "__main__":
    raise SystemExit(main())

