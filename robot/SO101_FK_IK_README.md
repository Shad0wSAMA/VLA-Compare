# SO-101 FK / IK 工具（Windows）

这套工具直接读取 `so101.urdf`，仅依赖 Python、NumPy 和标准库 Tkinter，不使用 Placo、IKPy 或 LeRobot 的运动学实现。

## 数据约定

关节空间和动作空间均为固定顺序：

```text
joints = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
action = [x, y, z, wrist_yaw, wrist_roll, gripper]
```

- `x/y/z`：URDF `base_link` 坐标系下 `gripper_frame_link` 的位置，单位 m。
- 所有关节角、`wrist_yaw` 和 `wrist_roll`：单位 deg。
- `wrist_yaw = shoulder_lift + elbow_flex + wrist_flex`。
- `wrist_roll` 和 `gripper` 在 FK/IK 间原样传递。
- 核心 FK/IK 对 `gripper` 原样透传；GUI 将它解释为校准后的角度（deg）。
- FK 完整使用 URDF 中的原点、关节轴和固定末端偏置。虽然 roll 数值原样传递，但 URDF 末端点相对 roll 轴有横向偏置，所以 roll 仍会轻微影响 XYZ，IK 已对此建模。

## Python API

```python
from robot.so101_kinematics import SO101Kinematics

kin = SO101Kinematics("robot/so101.urdf")

action = kin.forward([0, 20, -40, 20, 0, 50])
result = kin.inverse(action, current_joints_deg=[0, 20, -40, 20, 0, 50])
print(result.joints)
print(result.position_error_m, result.yaw_error_deg)
```

`current_joints_deg` 可省略；提供当前关节角时，IK 会优先选取运动距离更短的有效分支。目标不可达或违反腕旋转限位时会抛出 `IKError`，不会静默裁剪成错误姿态。

## 启动 GUI

在工作区根目录执行：

```powershell
python scripts/so101_fk_ik_gui.py
```

也可以双击 `scripts/run_so101_fk_ik_gui.bat`。GUI 支持：

- 拖动六个关节滑块并实时做 FK；
- 输入 6D 动作并做 IK；
- 俯视与侧视姿态显示；
- 可选连接 LeRobot SO-101 实机、读取当前位置、确认后发送目标。
- 在界面调节最大关节速度（deg/s）和 LeRobot 单条指令关节限幅（deg）；
- 点击一次后自动插值到最终目标，并可用“停止运动”立即停止后续指令。
- 夹爪 GUI 零点是原 LeRobot 归一化 0 的位置；负角度从该位置继续向并拢方向运动，并提供 `-20°` 并拢预设按钮。
- 可把当前实机位置保存为持久化的软件初始位姿，并可安全地载入为下一次目标。

实机默认串口为 `COM3`、机器人 ID 为 `my_follower`，可在界面修改，也可这样启动：

```powershell
python scripts/so101_fk_ik_gui.py --port COM5 --robot-id my_follower
```

GUI 不会启动交互式校准。首次连接前请先用 LeRobot CLI 完成同一 robot ID 的校准。连接后拖动滑块不会自动驱动实机；只有点击“发送关节目标”并再次确认才会发送。

“机械臂最大速度”默认是 `30 deg/s`。每次运动前，GUI 会计算五个机械臂关节的位移：位移最大的关节使用该最大速度，其余关节按照 `本关节位移 / 最大位移` 同比例缩放，因此五个机械臂关节同时到达目标。夹爪不参与比例规划，始终使用 `30 deg/s`。

单条指令关节限幅默认是 `10 deg`。GUI 以 20 Hz 插值发送；如果最大速度产生的单步角度超过该限幅，实际最大速度会被限幅降低。关节限幅还会写入 LeRobot 的 `max_relative_target`，由底层对每一条指令再次检查。

LeRobot 的 SOFollower 默认把夹爪暴露为 0..100 归一化量。本 GUI 将原归一化 0 端点定义成界面夹爪 `0°`，所以 `-20°` 表示从旧 0 位置再向并拢方向转 20°。连接时 GUI 会把夹爪切换到度模式，并只把硬件的并拢侧限位扩展到 `-40°`；张开侧仍使用校准文件的原始范围。关闭 GUI、断开机械臂后，下次 LeRobot 连接会重新写入原校准限位。

“设当前位置为初始位姿”在已连接时会读取实机位置，未连接时保存界面位置。结果保存在 `robot/so101_gui_initial_pose.json`。“载入初始位姿”只把保存值载入为 GUI 目标，不会自动运动；仍需点击“发送关节目标”并确认。

## 方块抓取

输入方块的 `x/y`（单位 m）并点击“抓取方块”后，GUI 固定使用 `wrist_yaw=80°`、`wrist_roll=-90°`，依次执行：

1. 移到 `(x-0.01, y+0.01, 0.05)`，同时把夹爪打开到 `+15°`；
2. 下降到 `(x-0.01, y+0.01, -0.01)`，保持夹爪打开；
3. 在抓取点把夹爪闭合到 `-20°`；
4. 保持闭合并上升回 `z=0.05 m`。

每个阶段都使用 IK、同步关节速度与单条指令限幅。“停止运动”会中断当前阶段并取消余下流程。

“方块放置”有独立的 `x/y` 输入框，默认 `x=0.28 m、y=0 m`，同样应用 `x-0.01、y+0.01` 偏移。放置流程为：

1. 保持夹爪 `-20°`，移到放置点上方 `z=0.05 m`；
2. 保持夹爪闭合并下降到 `z=0 m`；
3. 打开夹爪到 `+15°`，释放方块；
4. 保持夹爪打开并上升回 `z=0.05 m`。

## 将关节动作数据集转换为 FK 动作

`scripts/convert_joint_actions_to_so101_fk.py` 直接导入本目录的 `so101_kinematics.py`，把 LeRobot 数据集的六维关节 `action` 转换为：

```text
[x, y, z, wrist_yaw, wrist_roll, gripper]
```

默认命令：

```powershell
python scripts/convert_joint_actions_to_so101_fk.py
```

默认读取 `training/manual_data2_joints`，生成 `training/manual_data2_fk`。源数据不会被覆盖，`observation.state` 保持关节空间，视频在同一磁盘上优先使用硬链接。脚本同时更新动作字段名称与 `meta/stats.json`。如果输出目录已存在，脚本会停止并要求显式选择其他 `--output`，避免误覆盖。

自定义路径示例：

```powershell
python scripts/convert_joint_actions_to_so101_fk.py `
  --input training/manual_data2_joints `
  --output training/manual_data2_fk_v2 `
  --urdf robot/so101.urdf
```

## FK 动作模型实时推理

训练完成后，`scripts/infer_so101_fk_policy.py` 负责以下实时链路：

```text
双相机 + 关节观测 → 策略模型 → FK 动作 → SO101Kinematics.inverse() → 电机关节目标
```

先做不连接硬件的配置检查：

```powershell
python scripts/infer_so101_fk_policy.py --check-config
```

使用默认 SmolVLA 训练目录运行：

```powershell
python scripts/infer_so101_fk_policy.py
```

也可双击 `commands/inference_fk.bat`，或指定模型：

```powershell
python scripts/infer_so101_fk_policy.py `
  --policy outputs/train/smolvla_manual_data2_fk/checkpoints/last/pretrained_model `
  --port COM3 `
  --robot-id my_follower `
  --fixed-camera 0 `
  --wrist-camera 1 `
  --fps 10
```

脚本默认把模型输出裁剪到训练数据逐维范围，每次电机目标最多变化 `5`，连续三次 IK 失败会停机。启动时有 3 秒倒计时，运行中按 `Ctrl+C` 会断开机械臂并释放扭矩。CSV 日志写到 `outputs/inference_logs/`。

训练数据中的夹爪仍是 LeRobot 原始 0..100 归一化值，因此实时推理脚本不使用 GUI 的自定义负角度夹爪坐标；模型夹爪输出会限制在 0..100 后直接交给 SOFollower。

## 离线验证

```powershell
python scripts/test_so101_kinematics.py
```

测试包括固定姿态 FK、动作字段约定、随机 FK→IK→FK 往返以及关节限位拒绝。
