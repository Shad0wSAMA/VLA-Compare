"""LeRobotDataset episode recording helpers."""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import resolve_local_path


LOGGER = logging.getLogger(__name__)


class EpisodeRecorder:
    """Small recording adapter; a no-op when recording is disabled."""

    def __init__(self, robot: Any, config: dict[str, Any], config_dir: Path):
        self.robot = robot
        self.config = config
        self.config_dir = config_dir
        self.dataset: Any | None = None

    def start(self) -> Any | None:
        recording = self.config["recording"]
        if not recording.get("enabled", True):
            return None
        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = resolve_local_path(recording.get("root"), self.config_dir)
        if root is not None:
            root = root / timestamp
        use_videos = bool(recording.get("video", True))
        features = combine_feature_dicts(
            hw_to_dataset_features(self.robot.action_features, ACTION, use_video=use_videos),
            hw_to_dataset_features(self.robot.observation_features, OBS_STR, use_video=use_videos),
        )
        self.dataset = LeRobotDataset.create(
            repo_id=f"{recording['repo_id']}_{timestamp}",
            fps=int(recording.get("fps", 30)),
            root=root,
            robot_type=self.robot.name,
            features=features,
            use_videos=use_videos,
            image_writer_processes=0,
            image_writer_threads=int(recording.get("image_writer_threads", 4)),
            streaming_encoding=bool(recording.get("streaming_encoding", False)),
            encoder_threads=recording.get("encoder_threads"),
        )
        LOGGER.info("LeRobot recording started: %s", self.dataset.root)
        return self.dataset

    def video_manager(self) -> Any:
        if self.dataset is None:
            return contextlib.nullcontext()
        from lerobot.datasets import VideoEncodingManager

        return VideoEncodingManager(self.dataset)

    def add_step(self, observation: dict[str, Any], action: dict[str, float]) -> None:
        if self.dataset is None:
            return
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        observation_frame = build_dataset_frame(self.dataset.features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.dataset.features, action, prefix=ACTION)
        self.dataset.add_frame(
            {**observation_frame, **action_frame, "task": self.config["recording"]["single_task"]}
        )

    def save_episode(self) -> None:
        if self.dataset is not None:
            self.dataset.save_episode()
            LOGGER.info("Recording stopped; episode saved.")

    def push_if_enabled(self) -> None:
        recording = self.config["recording"]
        if self.dataset is not None and recording.get("push_to_hub", False):
            self.dataset.push_to_hub(private=recording.get("private"))
