"""End-to-end orchestration built from independently testable components."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import resolve_local_path
from .geometry import build_pick_place_poses
from .hardware import create_kinematics, create_robot
from .motion import MotionController, current_joint_vector
from .perception import (
    Detection,
    detect_marker_boundary_points,
    detect_marker_centers,
    estimate_table_homography,
    select_cube,
    undistort_if_configured,
)
from .recording import EpisodeRecorder


LOGGER = logging.getLogger(__name__)


def collect_table_calibration(
    robot: Any, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray | None]:
    """Collect stable border-corner samples from the configured fixed camera."""
    camera_name = str(config["robot"]["perception_camera"])
    camera_config = config["robot"]["cameras"][camera_name]
    samples = int(config["perception"].get("aruco_samples", 12))
    marker_samples: list[dict[int, np.ndarray]] = []
    last_image: np.ndarray | None = None
    last_corners: list[np.ndarray] = []
    last_ids: np.ndarray | None = None
    LOGGER.info("Collecting %d ArUco samples from '%s'...", samples, camera_name)
    for _ in range(samples):
        observation = robot.get_observation()
        image = undistort_if_configured(observation[camera_name], config["perception"])
        _centers, corners, ids = detect_marker_centers(image, config["perception"])
        marker_samples.append(detect_marker_boundary_points(corners, ids, config["perception"]))
        last_image, last_corners, last_ids = image, corners, ids
        time.sleep(1.0 / max(1, int(camera_config["fps"])))
    if last_image is None:
        raise RuntimeError("The perception camera returned no image.")
    homography, _averaged_points = estimate_table_homography(marker_samples, config["perception"])
    return homography, last_image, last_corners, last_ids


def save_detection_preview(
    config: dict[str, Any],
    config_dir: Path,
    image_rgb: np.ndarray,
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    detection: Detection,
) -> None:
    output = resolve_local_path(config["perception"].get("preview_path"), config_dir)
    if output is None:
        return
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image_bgr, corners, ids)
    x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"cube table=({detection.table_xy_m[0]:.3f}, {detection.table_xy_m[1]:.3f})m"
    cv2.putText(image_bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image_bgr):
        raise RuntimeError(f"Failed to write detection preview: {output}")
    LOGGER.info("Detection preview: %s", output)


def execute_pick_place_sequence(
    robot: Any,
    controller: MotionController,
    recorder: EpisodeRecorder,
    poses: dict[str, np.ndarray],
    motion: dict[str, Any],
) -> None:
    """Execute the already-planned sequence; separated for fake-hardware testing."""
    with recorder.video_manager():
        controller.last_observation = robot.get_observation()
        open_value = float(motion["gripper_open"])
        closed_value = float(motion["gripper_closed"])
        controller.send_interpolated_action(
            current_joint_vector(controller.last_observation),
            open_value,
            float(motion["gripper_duration_s"]),
            "open gripper",
        )
        controller.move_to_pose(poses["grasp approach"], open_value, "grasp approach")
        controller.move_to_pose(poses["grasp"], open_value, "descend to cube")
        assert controller.last_observation is not None
        controller.send_interpolated_action(
            current_joint_vector(controller.last_observation),
            closed_value,
            float(motion["gripper_duration_s"]),
            "close gripper",
        )
        controller.move_to_pose(poses["grasp lift"], closed_value, "lift cube")
        controller.move_to_pose(poses["place approach"], closed_value, "place approach")
        controller.move_to_pose(poses["place"], closed_value, "descend to place")
        assert controller.last_observation is not None
        controller.send_interpolated_action(
            current_joint_vector(controller.last_observation),
            open_value,
            float(motion["gripper_duration_s"]),
            "release cube",
        )
        controller.move_to_pose(poses["place approach"], open_value, "retreat")
        recorder.save_episode()


class PickPlaceRunner:
    def __init__(self, config: dict[str, Any], config_dir: Path):
        self.config = config
        self.config_dir = config_dir
        self.robot: Any | None = None

    def run(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Missing ultralytics. Install it with: pip install ultralytics") from exc

        weights = resolve_local_path(self.config["perception"]["weights"], self.config_dir)
        if weights is None:
            raise ValueError("perception.weights is empty")
        model = YOLO(str(weights))
        self.robot = create_robot(self.config, self.config_dir)
        solver = create_kinematics(self.config, self.config_dir)

        try:
            self.robot.connect()
            homography, image, corners, ids = collect_table_calibration(self.robot, self.config)
            detection = select_cube(model, image, homography, self.config["perception"])
            save_detection_preview(self.config, self.config_dir, image, corners, ids, detection)
            LOGGER.info(
                "Selected cube: confidence=%.3f, pixel=%s, table_xy_m=%s",
                detection.confidence,
                detection.pixel_xy.round(1).tolist(),
                detection.table_xy_m.round(4).tolist(),
            )
            poses = build_pick_place_poses(detection.table_xy_m, self.config)

            # Exactly as in the previous integrated script: recording starts before
            # the first gripper/motion command, after perception has selected a cube.
            recorder = EpisodeRecorder(self.robot, self.config, self.config_dir)
            recorder.start()
            controller = MotionController(self.robot, solver, self.config, recorder.add_step)
            execute_pick_place_sequence(self.robot, controller, recorder, poses, self.config["motion"])
            recorder.push_if_enabled()
        finally:
            if self.robot is not None and self.robot.is_connected:
                self.robot.disconnect()
