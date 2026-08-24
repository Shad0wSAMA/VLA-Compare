"""Run a trained Cartesian-action policy on a physical SO-101 in real time.

Expected policy action: [x, y, z, wrist_yaw, wrist_roll, gripper].
The first five values are converted to arm joints with the same FK/IK model as
the GUI and dataset converter. Gripper stays in the original LeRobot 0..100
dataset units and is sent directly to the follower driver.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
LEROBOT_SRC = WORKSPACE / "lerobot" / "src"
for import_path in (WORKSPACE, LEROBOT_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from robot.so101_kinematics import ACTION_NAMES, ARM_JOINTS, IKError, SO101Kinematics  # noqa: E402


MOTOR_NAMES = (*ARM_JOINTS, "gripper")
CAMERA_RENAME = {
    "fixed": "camera1",
    "wrist": "camera2",
}


def resolve_policy_path(path: Path) -> Path:
    """Accept a pretrained_model directory, run directory, or numbered checkpoint."""
    path = path.expanduser().resolve()
    candidates = (
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    numbered = path / "checkpoints"
    if numbered.is_dir():
        valid = sorted(
            (
                item / "pretrained_model"
                for item in numbered.iterdir()
                if item.is_dir() and item.name.isdigit() and (item / "pretrained_model" / "config.json").is_file()
            ),
            key=lambda item: int(item.parent.name),
        )
        if valid:
            return valid[-1]
    raise FileNotFoundError(f"找不到模型 config.json：{path}")


def load_action_bounds(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"]["action"]["names"]
    if names != list(ACTION_NAMES):
        raise ValueError(f"数据集 action 名称不匹配：期望 {list(ACTION_NAMES)}，实际 {names}")
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text(encoding="utf-8"))["action"]
    lower = np.asarray(stats["min"], dtype=float)
    upper = np.asarray(stats["max"], dtype=float)
    if lower.shape != (6,) or upper.shape != (6,):
        raise ValueError("数据集 action 统计必须是六维")
    return lower, upper


def policy_action_array(action: Any) -> np.ndarray:
    """Convert postprocessor output to one finite six-dimensional NumPy action."""
    if isinstance(action, dict):
        action = action.get("action", action.get("actions"))
    if action is None:
        raise ValueError("模型后处理器没有返回 action")
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    values = np.asarray(action, dtype=float).reshape(-1)
    if values.size != 6 or not np.all(np.isfinite(values)):
        raise ValueError(f"模型 action 必须是六个有限数值，实际 shape={values.shape}")
    return values


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=WORKSPACE / "outputs" / "train" / "smolvla_manual_data2_fk",
        help="训练运行目录、checkpoint 或 pretrained_model 目录。",
    )
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE / "training" / "manual_data2_fk")
    parser.add_argument("--urdf", type=Path, default=WORKSPACE / "robot" / "so101.urdf")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--fixed-camera", type=int, default=0)
    parser.add_argument("--wrist-camera", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--task", default="Pick up one cube and place it in the target area.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--fps", type=float, default=10.0, help="控制循环目标频率。")
    parser.add_argument("--duration", type=float, default=0.0, help="秒；0 表示运行到 Ctrl+C。")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--max-relative-target", type=float, default=5.0, help="每步每个电机最大变化。")
    parser.add_argument("--max-ik-failures", type=int, default=3)
    parser.add_argument(
        "--clamp-training-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="把模型 FK 输出裁剪到训练数据 action 的逐维 min/max。",
    )
    parser.add_argument("--check-config", action="store_true", help="只检查数据与参数，不连接硬件或加载模型。")
    parser.add_argument("--log", type=Path, default=None, help="CSV 日志路径；默认写入 outputs/inference_logs。")
    return parser


def validate_args(args: argparse.Namespace) -> tuple[Path, np.ndarray, np.ndarray]:
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"无效数据集：{dataset_root}")
    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF 不存在：{args.urdf}")
    if args.fps <= 0 or args.duration < 0 or args.countdown < 0:
        raise ValueError("fps 必须大于 0；duration/countdown 不能为负")
    if args.max_relative_target <= 0 or args.max_ik_failures < 1:
        raise ValueError("max-relative-target 必须大于 0；max-ik-failures 至少为 1")
    lower, upper = load_action_bounds(dataset_root)
    return dataset_root, lower, upper


def run(args: argparse.Namespace) -> None:
    dataset_root, action_lower, action_upper = validate_args(args)
    kinematics = SO101Kinematics(args.urdf)
    print(f"Dataset: {dataset_root}")
    print(f"Action:  {list(ACTION_NAMES)}")
    print(f"Range:   min={np.round(action_lower, 4)}, max={np.round(action_upper, 4)}")
    print(f"Robot:   {args.robot_id} on {args.port}")
    print(f"Cameras: fixed={args.fixed_camera}->camera1, wrist={args.wrist_camera}->camera2")
    if args.check_config:
        print("配置检查通过（未加载模型、未连接机械臂）。")
        return

    import torch
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.feature_utils import hw_to_dataset_features

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda，但 PyTorch 未检测到 CUDA；可改用 --device=cpu")
    device = torch.device(args.device)
    policy_path = resolve_policy_path(args.policy)
    print(f"Policy:  {policy_path}")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = str(device)
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(policy_path, config=policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    policy.eval()
    policy.reset()

    cameras = {
        CAMERA_RENAME["fixed"]: OpenCVCameraConfig(
            index_or_path=args.fixed_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc="MJPG",
        ),
        CAMERA_RENAME["wrist"]: OpenCVCameraConfig(
            index_or_path=args.wrist_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc="MJPG",
        ),
    }
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            cameras=cameras,
            use_degrees=True,
            max_relative_target=args.max_relative_target,
        )
    )
    if not robot.calibration:
        raise RuntimeError(f"找不到 robot-id={args.robot_id} 的校准文件，请先完成 LeRobot 校准")

    log_path = args.log
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = WORKSPACE / "outputs" / "inference_logs" / f"fk_policy_{timestamp}.csv"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "time_s", "loop_s", *[f"fk.{name}" for name in ACTION_NAMES],
        *[f"ik.{name}" for name in MOTOR_NAMES], "ik_position_error_m", "ik_yaw_error_deg",
    ]

    try:
        robot.connect(calibrate=False)
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        print("机械臂已连接。按 Ctrl+C 可随时停止并断开扭矩。")
        for remaining in range(args.countdown, 0, -1):
            print(f"{remaining}…")
            time.sleep(1.0)

        started = time.monotonic()
        consecutive_ik_failures = 0
        frame_index = 0
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(header)
            while args.duration == 0 or time.monotonic() - started < args.duration:
                loop_started = time.monotonic()
                observation = robot.get_observation()
                current_joints = np.asarray(
                    [observation[f"{name}.pos"] for name in MOTOR_NAMES], dtype=float
                )
                inference_frame = build_inference_frame(
                    observation=observation,
                    ds_features=obs_features,
                    device=device,
                    task=args.task,
                    robot_type="so_follower",
                )
                processed = preprocessor(inference_frame)
                with torch.inference_mode():
                    predicted = policy.select_action(processed)
                    predicted = postprocessor(predicted)
                fk_action = policy_action_array(predicted)
                if args.clamp_training_range:
                    fk_action = np.clip(fk_action, action_lower, action_upper)
                # The recorded gripper dimension remains LeRobot RANGE_0_100.
                fk_action[5] = float(np.clip(fk_action[5], 0.0, 100.0))

                try:
                    solved = kinematics.inverse(fk_action, current_joints, stop_at_first_valid=True)
                except IKError as error:
                    consecutive_ik_failures += 1
                    print(
                        f"[frame {frame_index}] IK失败 "
                        f"({consecutive_ik_failures}/{args.max_ik_failures}): {error}"
                    )
                    if consecutive_ik_failures >= args.max_ik_failures:
                        raise RuntimeError("连续 IK 失败达到上限，已停止推理") from error
                    continue
                consecutive_ik_failures = 0
                command = {
                    f"{name}.pos": float(value)
                    for name, value in zip(MOTOR_NAMES, solved.joints, strict=True)
                }
                sent = robot.send_action(command)
                loop_elapsed = time.monotonic() - loop_started
                writer.writerow([
                    time.monotonic() - started,
                    loop_elapsed,
                    *fk_action.tolist(),
                    *[sent[f"{name}.pos"] for name in MOTOR_NAMES],
                    solved.position_error_m,
                    solved.yaw_error_deg,
                ])
                log_file.flush()
                if frame_index % max(1, round(args.fps)) == 0:
                    print(
                        f"frame={frame_index} loop={loop_elapsed * 1000:.0f}ms "
                        f"xyz={np.round(fk_action[:3], 3)} yaw={fk_action[3]:.1f} "
                        f"roll={fk_action[4]:.1f} grip={fk_action[5]:.1f}"
                    )
                frame_index += 1
                sleep_s = 1.0 / args.fps - (time.monotonic() - loop_started)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在停止。")
    finally:
        if robot.is_connected:
            robot.disconnect()
        print(f"机械臂已断开；日志：{log_path}")


def main() -> None:
    args = make_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
