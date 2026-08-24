"""One-time local configuration for the LeRobot convenience scripts.

Edit this file once for your hardware.  The command-line scripts in this
directory use these values by default, so the usual commands stay short.
Environment variables with the same names can also override the values.
"""

from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# SO-101 is the default.  Change to so100_* if you have an SO-100.
ROBOT_TYPE = _env("LEROBOT_ROBOT_TYPE", "so101_follower")
ROBOT_PORT = _env("LEROBOT_ROBOT_PORT", "COM3")
ROBOT_ID = _env("LEROBOT_ROBOT_ID", "my_follower")

TELEOP_TYPE = _env("LEROBOT_TELEOP_TYPE", "so101_leader")
TELEOP_PORT = _env("LEROBOT_TELEOP_PORT", "COM4")
TELEOP_ID = _env("LEROBOT_TELEOP_ID", "my_leader")

# Set the camera names here.  For a fixed camera plus a wrist camera, use
# unique names such as ``top`` and ``wrist``.  Replace 0/1 with the IDs shown
# by ``lerobot-find-cameras opencv`` on this computer.
# CAMERA = "{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }"
CAMERA = os.getenv("LEROBOT_CAMERA") or None

FPS = int(_env("LEROBOT_FPS", "60"))
DISPLAY_DATA = _env("LEROBOT_DISPLAY_DATA", "false").lower() in {"1", "true", "yes", "on"}

# Recording defaults.  The record script appends a timestamp to REPO_ID when
# it creates a new dataset, so repeated recording sessions do not overwrite
# one another.
DATASET_REPO_ID = _env("LEROBOT_DATASET_REPO_ID", "your_name/soarm101_task")
DATASET_SINGLE_TASK = _env("LEROBOT_DATASET_SINGLE_TASK", "Pick up the cube and place it in the box")
DATASET_FPS = int(_env("LEROBOT_DATASET_FPS", "30"))
DATASET_NUM_EPISODES = int(_env("LEROBOT_DATASET_NUM_EPISODES", "50"))
DATASET_EPISODE_TIME_S = float(_env("LEROBOT_DATASET_EPISODE_TIME_S", "30"))
DATASET_RESET_TIME_S = float(_env("LEROBOT_DATASET_RESET_TIME_S", "10"))
DATASET_PUSH_TO_HUB = _env("LEROBOT_DATASET_PUSH_TO_HUB", "true").lower() in {"1", "true", "yes", "on"}
PLAY_SOUNDS = _env("LEROBOT_PLAY_SOUNDS", "true").lower() in {"1", "true", "yes", "on"}
