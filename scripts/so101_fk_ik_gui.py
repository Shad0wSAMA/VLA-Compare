"""Tkinter FK/IK test panel for the SO-101 (native Windows compatible)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from robot.so101_kinematics import ACTION_NAMES, ARM_JOINTS, IKError, SO101Kinematics  # noqa: E402


JOINT_LABELS = (*ARM_JOINTS, "gripper")
GRIPPER_CLOSE_DEG = -20.0
# GUI gripper zero is the old LeRobot normalized-0 endpoint. Negative values
# intentionally continue beyond that calibration endpoint toward jaw closure.
GRIPPER_CLOSE_LIMIT_DEG = -40.0
GRIPPER_DEFAULT_OPEN_LIMIT_DEG = 130.0
GRIPPER_SPEED_DEG_S = 30.0
PICK_GRIPPER_OPEN_DEG = 15.0
PICK_X_OFFSET_M = -0.01
PICK_Y_OFFSET_M = 0.01
PICK_ABOVE_Z_M = 0.05
PICK_GRASP_Z_M = -0.01
PLACE_RELEASE_Z_M = 0.0
PICK_WRIST_YAW_DEG = 80.0
PICK_WRIST_ROLL_DEG = -90.0
INITIAL_POSE_PATH = WORKSPACE / "robot" / "so101_gui_initial_pose.json"


def synchronized_arm_speeds(delta_deg: np.ndarray, maximum_speed_deg_s: float) -> np.ndarray:
    """Scale five arm-joint speeds so all nonzero moves finish together."""
    distances = np.abs(np.asarray(delta_deg, dtype=float))
    if distances.shape != (5,):
        raise ValueError("delta_deg must contain the five arm joints")
    farthest = float(np.max(distances))
    if farthest <= 0.0:
        return np.zeros(5, dtype=float)
    return maximum_speed_deg_s * distances / farthest


class SO101Panel(tk.Tk):
    def __init__(self, model: SO101Kinematics, port: str, robot_id: str) -> None:
        super().__init__()
        self.model = model
        self.robot = None
        self.motion_thread: threading.Thread | None = None
        self.stop_motion = threading.Event()
        self.title("SO-101 FK / IK 工具")
        self.geometry("1180x1000")
        self.minsize(1050, 900)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status = tk.StringVar(value="离线模式：可进行 FK / IK 与虚拟姿态测试")
        self.port = tk.StringVar(value=port)
        self.robot_id = tk.StringVar(value=robot_id)
        self.joint_speed = tk.StringVar(value="30.0")
        self.joint_step_limit = tk.StringVar(value="10.0")
        self.cube_x = tk.StringVar(value="0.25")
        self.cube_y = tk.StringVar(value="0.00")
        self.place_x = tk.StringVar(value="0.28")
        self.place_y = tk.StringVar(value="0.00")
        self.action_vars = [tk.StringVar() for _ in ACTION_NAMES]
        self.joint_vars: list[tk.DoubleVar] = []
        self.scales: list[ttk.Scale] = []

        self._build_ui()
        initial = self._load_initial_pose()
        if initial is None:
            initial = np.array([0.0, 20.0, -40.0, 20.0, 0.0, 0.0])
        self._set_joint_values(initial)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        view = ttk.Frame(outer)
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure((0, 1), weight=1)
        view.rowconfigure(0, weight=1)

        joints_box = ttk.LabelFrame(controls, text="关节空间（度）", padding=8)
        joints_box.pack(fill="x")
        limits = self.model.joint_limits_deg.tolist() + [
            [GRIPPER_CLOSE_LIMIT_DEG, GRIPPER_DEFAULT_OPEN_LIMIT_DEG]
        ]
        for row, (name, (lower, upper)) in enumerate(zip(JOINT_LABELS, limits, strict=True)):
            variable = tk.DoubleVar()
            self.joint_vars.append(variable)
            ttk.Label(joints_box, text=name, width=15).grid(row=row, column=0, sticky="w")
            scale = ttk.Scale(
                joints_box, from_=lower, to=upper, variable=variable, length=215,
                command=lambda _value: self._on_joint_slider(),
            )
            scale.grid(row=row, column=1, padx=5)
            self.scales.append(scale)
            label = ttk.Label(joints_box, width=8, anchor="e")
            label.grid(row=row, column=2)
            variable.trace_add("write", lambda *_args, v=variable, out=label: out.configure(text=f"{v.get():7.2f}"))

        button_row = ttk.Frame(joints_box)
        button_row.grid(row=len(JOINT_LABELS), column=0, columnspan=3, pady=(8, 0), sticky="ew")
        ttk.Button(button_row, text="关节 → FK", command=self._run_fk).pack(side="left", expand=True, fill="x")
        ttk.Button(button_row, text="读取机械臂", command=self._read_robot).pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(button_row, text="发送关节目标", command=self._send_robot).pack(side="left", expand=True, fill="x")
        ttk.Button(
            joints_box,
            text=f"夹爪并拢预设（{GRIPPER_CLOSE_DEG:g}°）",
            command=self._set_gripper_closed,
        ).grid(row=len(JOINT_LABELS) + 1, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        home_row = ttk.Frame(joints_box)
        home_row.grid(row=len(JOINT_LABELS) + 2, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        ttk.Button(home_row, text="设当前位置为初始位姿", command=self._save_current_as_initial).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(home_row, text="载入初始位姿", command=self._apply_initial_pose).pack(
            side="left", expand=True, fill="x", padx=(5, 0)
        )

        action_box = ttk.LabelFrame(controls, text="6D 动作空间", padding=8)
        action_box.pack(fill="x", pady=10)
        units = ("m", "m", "m", "deg", "deg", "deg")
        for row, (name, unit, variable) in enumerate(zip(ACTION_NAMES, units, self.action_vars, strict=True)):
            ttk.Label(action_box, text=name, width=15).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(action_box, textvariable=variable, width=17).grid(row=row, column=1, padx=5)
            ttk.Label(action_box, text=unit, width=7).grid(row=row, column=2, sticky="w")
        ttk.Button(action_box, text="动作 → IK", command=self._run_ik).grid(
            row=len(ACTION_NAMES), column=0, columnspan=3, sticky="ew", pady=(7, 0)
        )

        pick_box = ttk.LabelFrame(controls, text="方块抓取", padding=8)
        pick_box.pack(fill="x", pady=(0, 10))
        ttk.Label(pick_box, text="方块 x", width=15).grid(row=0, column=0, sticky="w")
        ttk.Entry(pick_box, textvariable=self.cube_x, width=17).grid(row=0, column=1, padx=5)
        ttk.Label(pick_box, text="m").grid(row=0, column=2, sticky="w")
        ttk.Label(pick_box, text="方块 y", width=15).grid(row=1, column=0, sticky="w")
        ttk.Entry(pick_box, textvariable=self.cube_y, width=17).grid(row=1, column=1, padx=5)
        ttk.Label(pick_box, text="m").grid(row=1, column=2, sticky="w")
        ttk.Button(pick_box, text="抓取方块", command=self._pick_cube).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )
        ttk.Label(
            pick_box,
            text="路径使用 (x-0.01, y+0.01)，上方 z=0.05 m，抓取 z=-0.01 m。",
            foreground="#555", wraplength=370,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))

        place_box = ttk.LabelFrame(controls, text="方块放置", padding=8)
        place_box.pack(fill="x", pady=(0, 10))
        ttk.Label(place_box, text="放置 x", width=15).grid(row=0, column=0, sticky="w")
        ttk.Entry(place_box, textvariable=self.place_x, width=17).grid(row=0, column=1, padx=5)
        ttk.Label(place_box, text="m").grid(row=0, column=2, sticky="w")
        ttk.Label(place_box, text="放置 y", width=15).grid(row=1, column=0, sticky="w")
        ttk.Entry(place_box, textvariable=self.place_y, width=17).grid(row=1, column=1, padx=5)
        ttk.Label(place_box, text="m").grid(row=1, column=2, sticky="w")
        ttk.Button(place_box, text="放置方块", command=self._place_cube).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )
        ttk.Label(
            place_box,
            text="路径使用 (x-0.01, y+0.01)，默认 x=0.28 m、y=0 m。",
            foreground="#555", wraplength=370,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))

        hardware = ttk.LabelFrame(controls, text="可选：LeRobot 实机连接", padding=8)
        hardware.pack(fill="x")
        ttk.Label(hardware, text="串口").grid(row=0, column=0, sticky="w")
        ttk.Entry(hardware, textvariable=self.port, width=14).grid(row=0, column=1, padx=5)
        ttk.Label(hardware, text="ID").grid(row=1, column=0, sticky="w")
        ttk.Entry(hardware, textvariable=self.robot_id, width=14).grid(row=1, column=1, padx=5)
        ttk.Button(hardware, text="连接", command=self._connect_robot).grid(row=0, column=2, rowspan=2, sticky="ns")
        ttk.Label(hardware, text="机械臂最大速度").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(hardware, textvariable=self.joint_speed, width=14).grid(row=2, column=1, padx=5, pady=(6, 0))
        ttk.Label(hardware, text="°/s").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Label(hardware, text="单次关节限幅").grid(row=3, column=0, sticky="w")
        ttk.Entry(hardware, textvariable=self.joint_step_limit, width=14).grid(row=3, column=1, padx=5)
        ttk.Label(hardware, text="°/指令").grid(row=3, column=2, sticky="w")
        self.stop_button = ttk.Button(hardware, text="停止运动", command=self._stop_robot, state="disabled")
        self.stop_button.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(
            hardware,
            text="点击发送后按设定速度自动走到目标；可随时停止。\n"
            "五轴同比例同步到达；夹爪固定 30°/s。\n"
            "夹爪 0°=旧归一化 0 端点，负角度继续并拢。",
            foreground="#555", wraplength=370,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.top_canvas = tk.Canvas(view, background="#fafafa", highlightthickness=1, highlightbackground="#bbb")
        self.side_canvas = tk.Canvas(view, background="#fafafa", highlightthickness=1, highlightbackground="#bbb")
        self.top_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.side_canvas.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.top_canvas.bind("<Configure>", lambda _event: self._draw())
        self.side_canvas.bind("<Configure>", lambda _event: self._draw())

        status_bar = ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=5)
        status_bar.pack(fill="x", side="bottom")

    def _joint_values(self) -> np.ndarray:
        return np.asarray([variable.get() for variable in self.joint_vars], dtype=float)

    def _set_joint_values(self, values: np.ndarray) -> None:
        for variable, value in zip(self.joint_vars, values, strict=True):
            variable.set(float(value))
        self._run_fk()

    def _on_joint_slider(self) -> None:
        if len(self.joint_vars) == 6:
            self._run_fk(quiet=True)

    def _set_gripper_closed(self) -> None:
        self.joint_vars[5].set(GRIPPER_CLOSE_DEG)
        self._run_fk()
        self.status.set(f"已设置夹爪并拢目标 {GRIPPER_CLOSE_DEG:g}°；点击“发送关节目标”执行")

    def _load_initial_pose(self) -> np.ndarray | None:
        if not INITIAL_POSE_PATH.is_file():
            return None
        try:
            values = np.asarray(json.loads(INITIAL_POSE_PATH.read_text(encoding="utf-8"))["joints_deg"], dtype=float)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError("初始位姿必须包含六个有限数值")
            return values
        except Exception as exc:
            self.after(0, lambda error=exc: messagebox.showwarning("初始位姿无效", str(error), parent=self))
            return None

    def _write_initial_pose(self, values: np.ndarray) -> None:
        payload = {
            "joint_order": list(JOINT_LABELS),
            "joints_deg": [float(value) for value in values],
        }
        INITIAL_POSE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _save_current_as_initial(self) -> None:
        if self.motion_thread is not None and self.motion_thread.is_alive():
            messagebox.showwarning("正在运动", "请先停止机械臂再设置初始位姿。", parent=self)
            return

        if self.robot is None or not self.robot.is_connected:
            values = self._joint_values().copy()
            self._write_initial_pose(values)
            self.status.set(f"已把当前界面姿态保存为软件初始位姿：{INITIAL_POSE_PATH.name}")
            return

        def read_and_save() -> str:
            assert self.robot is not None
            observation = self.robot.get_observation()
            robot_values = np.asarray([observation[f"{name}.pos"] for name in JOINT_LABELS], dtype=float)
            gui_values = self._robot_to_gui_values(robot_values)
            self._write_initial_pose(gui_values)
            self.after(0, lambda: self._set_joint_values(gui_values))
            return "已读取实机并保存为软件初始位姿"

        self.status.set("正在读取当前实机位置…")
        self._run_worker(read_and_save, "初始位姿设置完成")

    def _apply_initial_pose(self) -> None:
        values = self._load_initial_pose()
        if values is None:
            messagebox.showwarning("没有初始位姿", "请先设置当前位置为初始位姿。", parent=self)
            return
        self._set_joint_values(values)
        self.status.set("已载入软件初始位姿；点击“发送关节目标”后机械臂才会运动")

    def _run_fk(self, quiet: bool = False) -> None:
        try:
            action = self.model.forward(self._joint_values())
            for index, (variable, value) in enumerate(zip(self.action_vars, action, strict=True)):
                variable.set(f"{value:.6f}" if index < 3 else f"{value:.3f}")
            self._draw()
            if not quiet:
                self.status.set("FK 完成：XYZ 单位为米，角度单位为度")
        except Exception as exc:
            self.status.set(f"FK 错误：{exc}")

    def _run_ik(self) -> None:
        try:
            target = np.asarray([float(variable.get()) for variable in self.action_vars])
            result = self.model.inverse(target, self._joint_values())
            self._set_joint_values(result.joints)
            self.status.set(
                f"IK 完成：位置误差 {result.position_error_m * 1000:.4f} mm，"
                f"yaw 误差 {result.yaw_error_deg:.5f}°，迭代 {result.iterations} 次"
            )
        except (ValueError, IKError) as exc:
            self.status.set(f"IK 失败：{exc}")
            messagebox.showerror("IK 失败", str(exc), parent=self)

    def _draw(self) -> None:
        if len(self.joint_vars) != 6:
            return
        points = self.model.link_positions(self._joint_values()[:5])
        ordered_names = [
            "base_link", "shoulder_link", "upper_arm_link", "lower_arm_link",
            "wrist_link", "gripper_link", "gripper_frame_link",
        ]
        xyz = np.asarray([points[name] for name in ordered_names])
        self._draw_projection(self.top_canvas, xyz[:, [0, 1]], "俯视图  X / Y", "X", "Y")
        radial = np.sign(xyz[:, 0] + 1e-12) * np.hypot(xyz[:, 0], xyz[:, 1])
        self._draw_projection(self.side_canvas, np.column_stack((radial, xyz[:, 2])), "侧视图  R / Z", "R", "Z")

    @staticmethod
    def _draw_projection(canvas: tk.Canvas, points: np.ndarray, title: str, x_name: str, y_name: str) -> None:
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        scale = min(width, height) / 1.05  # about 0.5 m from centre to edge
        cx, cy = width / 2, height * 0.58
        canvas.create_line(20, cy, width - 20, cy, fill="#aaa")
        canvas.create_line(cx, 20, cx, height - 20, fill="#aaa")
        canvas.create_text(12, 12, text=title, anchor="nw", font=("Segoe UI", 12, "bold"))
        canvas.create_text(width - 15, cy - 12, text=x_name, anchor="e", fill="#666")
        canvas.create_text(cx + 12, 25, text=y_name, anchor="w", fill="#666")
        screen = np.column_stack((cx + points[:, 0] * scale, cy - points[:, 1] * scale))
        canvas.create_line(*screen.reshape(-1).tolist(), fill="#1769aa", width=5, smooth=True)
        for index, (x, y) in enumerate(screen):
            color = "#d32f2f" if index == len(screen) - 1 else "#ffb300"
            canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, outline="#333")

    def _run_worker(self, operation, success_text: str) -> None:
        def worker() -> None:
            try:
                result = operation()
                self.after(0, lambda: self.status.set(success_text if result is None else f"{success_text}: {result}"))
            except Exception as exc:
                self.after(0, lambda error=exc: messagebox.showerror("机械臂错误", str(error), parent=self))
                self.after(0, lambda error=exc: self.status.set(f"机械臂错误：{error}"))

        threading.Thread(target=worker, daemon=True).start()

    def _connect_robot(self) -> None:
        if self.robot is not None and self.robot.is_connected:
            self.status.set("机械臂已经连接")
            return

        port = self.port.get().strip()
        robot_id = self.robot_id.get().strip()
        try:
            step_limit = self._positive_setting(self.joint_step_limit, "单次关节限幅")
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return

        def connect() -> None:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            from lerobot.motors import MotorNormMode

            self.robot = SO101Follower(
                SO101FollowerConfig(
                    port=port, id=robot_id,
                    use_degrees=True, max_relative_target=step_limit,
                )
            )
            # Never launch interactive calibration from a GUI. Existing calibration
            # is loaded automatically; first-time calibration must use LeRobot CLI.
            if not self.robot.calibration:
                raise RuntimeError(
                    f"未找到 ID={robot_id} 的校准文件。请先用 LeRobot 完成机械臂校准。"
                )
            # SOFollower normally exposes the gripper as RANGE_0_100. This GUI
            # deliberately exposes every actuator in calibrated degrees so a
            # physical close target such as -30 degrees is not clipped to zero.
            self.robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
            self.robot.connect(calibrate=False)
            # configure() restores the servo's hardware limit to calibration.range_min.
            # Extend only the closing side by the GUI's explicitly bounded -40 deg,
            # otherwise every negative GUI command would still stop at the old 0 point.
            calibration = self.robot.calibration["gripper"]
            close_extension_steps = round(GRIPPER_CLOSE_LIMIT_DEG * 4095.0 / 360.0)
            extended_min = max(0, calibration.range_min + close_extension_steps)
            self.robot.bus.write("Min_Position_Limit", "gripper", extended_min)

        self.status.set("正在连接机械臂…")
        self._run_worker(connect, "机械臂已连接")

    def _read_robot(self) -> None:
        if self.robot is None or not self.robot.is_connected:
            messagebox.showwarning("未连接", "请先连接机械臂。", parent=self)
            return

        def read() -> str:
            observation = self.robot.get_observation()
            robot_values = np.asarray([observation[f"{name}.pos"] for name in JOINT_LABELS])
            values = self._robot_to_gui_values(robot_values)
            self.after(0, lambda: self._set_joint_values(values))
            return "已读取当前位置"

        self._run_worker(read, "读取完成")

    def _send_robot(self) -> None:
        if self.robot is None or not self.robot.is_connected:
            messagebox.showwarning("未连接", "请先连接机械臂。", parent=self)
            return
        if self.motion_thread is not None and self.motion_thread.is_alive():
            messagebox.showwarning("正在运动", "机械臂正在执行目标；请先停止或等待完成。", parent=self)
            return
        try:
            speed = self._positive_setting(self.joint_speed, "机械臂最大速度")
            step_limit = self._positive_setting(self.joint_step_limit, "单次关节限幅")
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return
        if not messagebox.askokcancel(
            "确认运动",
            f"确认以机械臂最大 {speed:g}°/s 同步移动到目标？\n"
            f"夹爪固定 {GRIPPER_SPEED_DEG_S:g}°/s；单条指令限幅：{step_limit:g}°",
            parent=self,
        ):
            return
        target = self._joint_values().copy()
        gripper_open_limit = self._calibrated_gripper_span_deg()
        if not GRIPPER_CLOSE_LIMIT_DEG <= target[5] <= gripper_open_limit:
            messagebox.showerror(
                "夹爪目标超限",
                f"夹爪目标必须在 {GRIPPER_CLOSE_LIMIT_DEG:.2f}° .. {gripper_open_limit:.2f}° 内。",
                parent=self,
            )
            return
        self.stop_motion.clear()
        self.stop_button.configure(state="normal")
        self.status.set(f"正在规划同步运动：最大 {speed:g}°/s，夹爪 {GRIPPER_SPEED_DEG_S:g}°/s…")

        def move() -> None:
            try:
                current = self._execute_joint_target(target, speed, step_limit, "关节目标")
                stopped = self.stop_motion.is_set()
                self.after(0, lambda values=current.copy(): self._set_joint_values(values))
                self.after(
                    0,
                    lambda: self.status.set("运动已停止" if stopped else "已到达关节目标"),
                )
            except Exception as exc:
                self.after(0, lambda error=exc: messagebox.showerror("机械臂错误", str(error), parent=self))
                self.after(0, lambda error=exc: self.status.set(f"机械臂错误：{error}"))
            finally:
                self.after(0, lambda: self.stop_button.configure(state="disabled"))

        self.motion_thread = threading.Thread(target=move, daemon=True)
        self.motion_thread.start()

    def _execute_joint_target(
        self, target: np.ndarray, speed: float, step_limit: float, stage_name: str
    ) -> np.ndarray:
        """Blocking worker-thread motion primitive used by manual and pick moves."""
        assert self.robot is not None
        self.robot.config.max_relative_target = step_limit
        observation = self.robot.get_observation()
        robot_current = np.asarray(
            [observation[f"{name}.pos"] for name in JOINT_LABELS], dtype=float
        )
        current = self._robot_to_gui_values(robot_current)
        period_s = 0.05
        effective_max_speed = min(speed, step_limit / period_s)
        initial_arm_delta = target[:5] - current[:5]
        arm_speeds = synchronized_arm_speeds(initial_arm_delta, effective_max_speed)
        arm_step_limits = arm_speeds * period_s
        gripper_step_limit = min(GRIPPER_SPEED_DEG_S * period_s, step_limit)
        farthest_distance = float(np.max(np.abs(initial_arm_delta)))
        planned_duration = farthest_distance / effective_max_speed if farthest_distance else 0.0
        self.after(
            0,
            lambda: self.status.set(
                f"{stage_name}：同步规划 {planned_duration:.2f}s，"
                f"机械臂最大 {effective_max_speed:g}°/s，夹爪 {GRIPPER_SPEED_DEG_S:g}°/s"
            ),
        )
        started = time.monotonic()
        while not self.stop_motion.is_set():
            delta = target - current
            if float(np.max(np.abs(delta))) <= 0.05:
                break
            next_position = current.copy()
            next_position[:5] += np.clip(delta[:5], -arm_step_limits, arm_step_limits)
            next_position[5] += float(np.clip(delta[5], -gripper_step_limit, gripper_step_limit))
            robot_next_position = self._gui_to_robot_values(next_position)
            command = {
                f"{name}.pos": float(value)
                for name, value in zip(JOINT_LABELS, robot_next_position, strict=True)
            }
            sent = self.robot.send_action(command)
            robot_sent = np.asarray([sent[f"{name}.pos"] for name in JOINT_LABELS], dtype=float)
            current = self._robot_to_gui_values(robot_sent)
            elapsed = time.monotonic() - started
            self.after(
                0,
                lambda values=current.copy(), seconds=elapsed: self.status.set(
                    f"{stage_name}：已执行 {seconds:.1f}s，最大剩余误差 "
                    f"{np.max(np.abs(target - values)):.2f}°"
                ),
            )
            time.sleep(period_s)
        return current

    def _pick_cube(self) -> None:
        if self.robot is None or not self.robot.is_connected:
            messagebox.showwarning("未连接", "请先连接机械臂。", parent=self)
            return
        if self.motion_thread is not None and self.motion_thread.is_alive():
            messagebox.showwarning("正在运动", "机械臂正在执行目标；请先停止或等待完成。", parent=self)
            return
        try:
            cube_x = float(self.cube_x.get())
            cube_y = float(self.cube_y.get())
            speed = self._positive_setting(self.joint_speed, "机械臂最大速度")
            step_limit = self._positive_setting(self.joint_step_limit, "单次关节限幅")
            if not np.all(np.isfinite([cube_x, cube_y])):
                raise ValueError("方块坐标必须是有限数值")
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return

        target_x = cube_x + PICK_X_OFFSET_M
        target_y = cube_y + PICK_Y_OFFSET_M
        if not messagebox.askokcancel(
            "确认抓取",
            f"确认执行方块抓取？\n目标 XY=({target_x:.3f}, {target_y:.3f}) m\n"
            f"上方 Z={PICK_ABOVE_Z_M:.3f} m，抓取 Z={PICK_GRASP_Z_M:.3f} m\n"
            f"wrist_yaw={PICK_WRIST_YAW_DEG:g}°，wrist_roll={PICK_WRIST_ROLL_DEG:g}°\n"
            f"夹爪打开 {PICK_GRIPPER_OPEN_DEG:g}°，闭合 {GRIPPER_CLOSE_DEG:g}°",
            parent=self,
        ):
            return

        cartesian_stages = (
            ("1/4 移到方块上方并打开夹爪", np.array([
                target_x, target_y, PICK_ABOVE_Z_M,
                PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, PICK_GRIPPER_OPEN_DEG
            ])),
            ("2/4 下降到方块", np.array([
                target_x, target_y, PICK_GRASP_Z_M,
                PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, PICK_GRIPPER_OPEN_DEG
            ])),
            ("3/4 合上夹爪", np.array([
                target_x, target_y, PICK_GRASP_Z_M,
                PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, GRIPPER_CLOSE_DEG
            ])),
            ("4/4 抓取后上升", np.array([
                target_x, target_y, PICK_ABOVE_Z_M,
                PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, GRIPPER_CLOSE_DEG
            ])),
        )
        self._start_cartesian_sequence(cartesian_stages, speed, step_limit, "抓取")

    def _place_cube(self) -> None:
        if self.robot is None or not self.robot.is_connected:
            messagebox.showwarning("未连接", "请先连接机械臂。", parent=self)
            return
        if self.motion_thread is not None and self.motion_thread.is_alive():
            messagebox.showwarning("正在运动", "机械臂正在执行目标；请先停止或等待完成。", parent=self)
            return
        try:
            place_x = float(self.place_x.get())
            place_y = float(self.place_y.get())
            speed = self._positive_setting(self.joint_speed, "机械臂最大速度")
            step_limit = self._positive_setting(self.joint_step_limit, "单次关节限幅")
            if not np.all(np.isfinite([place_x, place_y])):
                raise ValueError("放置坐标必须是有限数值")
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return

        target_x = place_x + PICK_X_OFFSET_M
        target_y = place_y + PICK_Y_OFFSET_M
        if not messagebox.askokcancel(
            "确认放置",
            f"确认执行方块放置？\n目标 XY=({target_x:.3f}, {target_y:.3f}) m\n"
            f"上方 Z={PICK_ABOVE_Z_M:.3f} m，放置 Z={PLACE_RELEASE_Z_M:.3f} m\n"
            f"wrist_yaw={PICK_WRIST_YAW_DEG:g}°，wrist_roll={PICK_WRIST_ROLL_DEG:g}°\n"
            f"下降时夹爪保持 {GRIPPER_CLOSE_DEG:g}°，释放到 {PICK_GRIPPER_OPEN_DEG:g}°",
            parent=self,
        ):
            return

        cartesian_stages = (
            ("1/4 移到放置点上方", np.array([
                target_x, target_y, PICK_ABOVE_Z_M,
                PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, GRIPPER_CLOSE_DEG
            ])),
            ("2/4 下降到放置点", np.array([
                target_x, target_y, PLACE_RELEASE_Z_M,
                PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, GRIPPER_CLOSE_DEG
            ])),
            ("3/4 打开夹爪释放方块", np.array([
                target_x, target_y, PLACE_RELEASE_Z_M,
                PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, PICK_GRIPPER_OPEN_DEG
            ])),
            ("4/4 释放后上升", np.array([
                target_x, target_y, PICK_ABOVE_Z_M,
                PICK_WRIST_YAW_DEG, PICK_WRIST_ROLL_DEG, PICK_GRIPPER_OPEN_DEG
            ])),
        )
        self._start_cartesian_sequence(cartesian_stages, speed, step_limit, "放置")

    def _start_cartesian_sequence(
        self,
        cartesian_stages: tuple[tuple[str, np.ndarray], ...],
        speed: float,
        step_limit: float,
        operation_name: str,
    ) -> None:
        self.stop_motion.clear()
        self.stop_button.configure(state="normal")

        def run_sequence() -> None:
            try:
                assert self.robot is not None
                observation = self.robot.get_observation()
                robot_values = np.asarray(
                    [observation[f"{name}.pos"] for name in JOINT_LABELS], dtype=float
                )
                current = self._robot_to_gui_values(robot_values)
                planned_stages: list[tuple[str, np.ndarray]] = []
                seed = current.copy()
                self.after(0, lambda: self.status.set(f"正在检查全部{operation_name}路径的 IK…"))
                for stage_name, action in cartesian_stages:
                    result = self.model.inverse(action, seed)
                    seed = result.joints
                    planned_stages.append((stage_name, result.joints))

                for stage_name, joint_target in planned_stages:
                    if self.stop_motion.is_set():
                        break
                    current = self._execute_joint_target(joint_target, speed, step_limit, stage_name)
                    self.after(0, lambda values=current.copy(): self._set_joint_values(values))
                stopped = self.stop_motion.is_set()
                self.after(
                    0,
                    lambda: self.status.set(
                        f"{operation_name}流程已停止" if stopped else f"{operation_name}流程完成"
                    ),
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda error=exc: messagebox.showerror(f"{operation_name}失败", str(error), parent=self),
                )
                self.after(0, lambda error=exc: self.status.set(f"{operation_name}失败：{error}"))
            finally:
                self.after(0, lambda: self.stop_button.configure(state="disabled"))

        self.motion_thread = threading.Thread(target=run_sequence, daemon=True)
        self.motion_thread.start()

    @staticmethod
    def _positive_setting(variable: tk.StringVar, name: str) -> float:
        try:
            value = float(variable.get())
        except ValueError as exc:
            raise ValueError(f"{name}必须是数字") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name}必须大于 0")
        return value

    def _stop_robot(self) -> None:
        self.stop_motion.set()
        self.status.set("正在停止运动…")

    def _gripper_driver_zero_deg(self) -> float:
        """Driver-degree value corresponding to the old normalized gripper zero."""
        if self.robot is None or "gripper" not in self.robot.calibration:
            return -GRIPPER_DEFAULT_OPEN_LIMIT_DEG / 2.0
        calibration = self.robot.calibration["gripper"]
        midpoint = (calibration.range_min + calibration.range_max) / 2.0
        return (calibration.range_min - midpoint) * 360.0 / 4095.0

    def _calibrated_gripper_span_deg(self) -> float:
        if self.robot is None or "gripper" not in self.robot.calibration:
            return GRIPPER_DEFAULT_OPEN_LIMIT_DEG
        calibration = self.robot.calibration["gripper"]
        return abs(calibration.range_max - calibration.range_min) * 360.0 / 4095.0

    def _gui_to_robot_values(self, values: np.ndarray) -> np.ndarray:
        converted = np.asarray(values, dtype=float).copy()
        converted[5] += self._gripper_driver_zero_deg()
        return converted

    def _robot_to_gui_values(self, values: np.ndarray) -> np.ndarray:
        converted = np.asarray(values, dtype=float).copy()
        converted[5] -= self._gripper_driver_zero_deg()
        return converted

    def _on_close(self) -> None:
        self.stop_motion.set()
        if self.motion_thread is not None and self.motion_thread.is_alive():
            self.motion_thread.join(timeout=0.5)
        robot = self.robot
        self.robot = None
        if robot is not None and robot.is_connected:
            try:
                robot.disconnect()
            except Exception:
                pass
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=WORKSPACE / "robot" / "so101.urdf")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--robot-id", default="my_follower")
    args = parser.parse_args()
    SO101Panel(SO101Kinematics(args.urdf), args.port, args.robot_id).mainloop()


if __name__ == "__main__":
    main()
