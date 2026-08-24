"""Record a LeRobot dataset using the local defaults.

Configure the dataset values in ``lerobot_config.py`` once, then run::

    python scripts/record.py

Use ``--dry-run`` to inspect the generated LeRobot command without opening
the serial ports or camera.
"""

from __future__ import annotations

import argparse
import shlex

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
    parser.add_argument("--repo-id", default=defaults.DATASET_REPO_ID)
    parser.add_argument("--task", default=defaults.DATASET_SINGLE_TASK)
    parser.add_argument("--fps", type=int, default=defaults.DATASET_FPS)
    parser.add_argument("--num-episodes", type=int, default=defaults.DATASET_NUM_EPISODES)
    parser.add_argument("--episode-time", type=float, default=defaults.DATASET_EPISODE_TIME_S)
    parser.add_argument("--reset-time", type=float, default=defaults.DATASET_RESET_TIME_S)
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=defaults.DATASET_PUSH_TO_HUB,
    )
    parser.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=defaults.DISPLAY_DATA)
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=defaults.PLAY_SOUNDS)
    parser.add_argument("--resume", action="store_true", help="Resume an existing dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        f"--robot.type={args.robot_type}",
        f"--robot.port={args.robot_port}",
        f"--robot.id={args.robot_id}",
        f"--teleop.type={args.teleop_type}",
        f"--teleop.port={args.teleop_port}",
        f"--teleop.id={args.teleop_id}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.fps={args.fps}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.reset_time_s={args.reset_time}",
        f"--dataset.push_to_hub={str(args.push_to_hub).lower()}",
        f"--display_data={str(args.display_data).lower()}",
        f"--play_sounds={str(args.play_sounds).lower()}",
        f"--resume={str(args.resume).lower()}",
    ]
    if args.camera:
        command.append(f"--robot.cameras={args.camera}")
    return command


def main() -> int:
    args = parse_args()
    if args.repo_id.startswith("your_name/"):
        raise SystemExit("请先在 scripts/lerobot_config.py 中把 DATASET_REPO_ID 改成你的 Hugging Face 用户名/数据集名。")
    command = build_command(args)
    if args.dry_run:
        print("lerobot-record " + shlex.join(command))
        return 0
    return run_lerobot("lerobot-record", command)


if __name__ == "__main__":
    raise SystemExit(main())
