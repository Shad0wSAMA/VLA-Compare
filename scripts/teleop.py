"""Start LeRobot teleoperation with the local defaults.

Typical use from the workspace root::

    python scripts/teleop.py

Temporary overrides are supported, for example::

    python scripts/teleop.py --robot-port COM5 --display-data
"""

from __future__ import annotations

import argparse

import lerobot_config as defaults
from _lerobot_cli import run_lerobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-type", default=defaults.ROBOT_TYPE)
    parser.add_argument("--robot-port", default=defaults.ROBOT_PORT)
    parser.add_argument("--robot-id", default=defaults.ROBOT_ID)
    parser.add_argument("--teleop-type", default=defaults.TELEOP_TYPE)
    parser.add_argument("--teleop-port", default=defaults.TELEOP_PORT)
    parser.add_argument("--teleop-id", default=defaults.TELEOP_ID)
    parser.add_argument("--camera", default=defaults.CAMERA, help="LeRobot camera config string")
    parser.add_argument("--fps", type=int, default=defaults.FPS)
    parser.add_argument("--duration", type=float, default=None, help="Stop after N seconds")
    parser.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=defaults.DISPLAY_DATA)
    parser.add_argument("--display-mode", choices=("rerun", "foxglove"), default="rerun")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        f"--robot.type={args.robot_type}",
        f"--robot.port={args.robot_port}",
        f"--robot.id={args.robot_id}",
        f"--teleop.type={args.teleop_type}",
        f"--teleop.port={args.teleop_port}",
        f"--teleop.id={args.teleop_id}",
        f"--fps={args.fps}",
        f"--display_data={str(args.display_data).lower()}",
        f"--display_mode={args.display_mode}",
    ]
    if args.camera:
        command.append(f"--robot.cameras={args.camera}")
    if args.duration is not None:
        command.append(f"--teleop_time_s={args.duration}")
    return run_lerobot("lerobot-teleoperate", command)


if __name__ == "__main__":
    raise SystemExit(main())

