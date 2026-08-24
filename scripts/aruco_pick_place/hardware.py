"""Factories for LeRobot cameras, SO-101 follower, and IK backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import resolve_local_path


MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def make_camera_configs(robot_config: dict[str, Any]) -> dict[str, Any]:
    from lerobot.cameras.opencv import OpenCVCameraConfig

    camera_configs: dict[str, OpenCVCameraConfig] = {}
    for camera_name, camera in robot_config["cameras"].items():
        raw_index = camera["index_or_path"]
        index_or_path: int | Path
        if isinstance(raw_index, int) or (isinstance(raw_index, str) and raw_index.isdigit()):
            index_or_path = int(raw_index)
        else:
            index_or_path = Path(str(raw_index)).expanduser()
        camera_configs[str(camera_name)] = OpenCVCameraConfig(
            index_or_path=index_or_path,
            fps=int(camera["fps"]),
            width=int(camera["width"]),
            height=int(camera["height"]),
            fourcc=camera.get("fourcc"),
            color_mode="rgb",
        )
    return camera_configs


def create_robot(config: dict[str, Any], config_dir: Path) -> Any:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot_config = config["robot"]
    return SO101Follower(
        SO101FollowerConfig(
            port=str(robot_config["port"]),
            id=robot_config.get("id"),
            calibration_dir=resolve_local_path(robot_config.get("calibration_dir"), config_dir),
            cameras=make_camera_configs(robot_config),
            use_degrees=True,
            max_relative_target=robot_config.get("max_relative_target_deg", 10.0),
        )
    )


def create_kinematics(config: dict[str, Any], config_dir: Path) -> Any:
    cfg = config["kinematics"]
    urdf_path = resolve_local_path(cfg["urdf_path"], config_dir)
    if urdf_path is None:
        raise ValueError("kinematics.urdf_path is empty")
    backend = str(cfg.get("backend", "placo")).lower()
    common_args = {
        "urdf_path": str(urdf_path),
        "target_frame_name": str(cfg["target_frame_name"]),
        "joint_names": [str(name) for name in cfg["joint_names"]],
    }
    if backend == "ikpy":
        from lerobot.model.kinematics_ikpy import RobotKinematics

        return RobotKinematics(
            **common_args,
            orientation_mode=cfg.get("orientation_mode", "Z"),
            max_iterations=int(cfg.get("max_iterations", 200)),
            regularization_parameter=cfg.get("regularization_parameter", 1e-6),
            symbolic=bool(cfg.get("symbolic", False)),
        )
    if backend == "placo":
        from lerobot.model.kinematics import RobotKinematics

        return RobotKinematics(**common_args)
    raise ValueError(f"Unsupported kinematics backend: {backend}")
