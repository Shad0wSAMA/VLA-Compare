"""Set up motor IDs and baudrate using the configured device."""

from __future__ import annotations

import argparse

import lerobot_config as defaults
from _lerobot_cli import run_lerobot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("robot", "teleop"))
    parser.add_argument("--type", dest="device_type", default=None)
    parser.add_argument("--port", default=None)
    args = parser.parse_args()

    if args.device == "robot":
        prefix, device_type, port = "robot", defaults.ROBOT_TYPE, defaults.ROBOT_PORT
    else:
        prefix, device_type, port = "teleop", defaults.TELEOP_TYPE, defaults.TELEOP_PORT

    return run_lerobot(
        "lerobot-setup-motors",
        [f"--{prefix}.type={args.device_type or device_type}", f"--{prefix}.port={args.port or port}"],
    )


if __name__ == "__main__":
    raise SystemExit(main())

