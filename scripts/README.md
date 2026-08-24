# LeRobot 快捷脚本

先编辑一次 [`lerobot_config.py`](./lerobot_config.py)，把 `COM7`、`COM8` 和设备 ID 改成自己的实际值。之后在工作区根目录执行：

```powershell
# 校准 follower 和 leader（第一次使用或重新装配后执行）
python scripts/calibrate.py robot
python scripts/calibrate.py teleop

# 可选：设置电机 ID / baudrate
python scripts/setup_motors.py robot
python scripts/setup_motors.py teleop

# 开始 teleop，Ctrl+C 停止
python scripts/teleop.py

# 按配置文件中的参数一键录制数据集
python scripts/record.py
```

录制前还要在 `lerobot_config.py` 中设置：

```python
DATASET_REPO_ID = "你的用户名/soarm101_pick_cube"
DATASET_SINGLE_TASK = "Pick up the cube and place it in the box"
DATASET_NUM_EPISODES = 50
DATASET_EPISODE_TIME_S = 30
DATASET_RESET_TIME_S = 10
DATASET_PUSH_TO_HUB = True
```

每次新建数据集时，LeRobot 会自动在 repo id 后加时间戳，避免覆盖之前的录制。运行 `python scripts/record.py --dry-run` 可以先查看命令，不会连接机械臂。

相机可以在配置文件中设置 `CAMERA`。固定相机和手腕相机建议使用 `top`、`wrist` 这两个名字，例如：

```python
CAMERA = "{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }"
```

先运行 `lerobot-find-cameras opencv` 确认两个摄像头的实际编号；Windows 重启或重新插拔后编号可能变化。

临时参数可以覆盖默认值，例如：

```powershell
python scripts/teleop.py --robot-port COM5 --teleop-port COM6 --display-data
python scripts/calibrate.py teleop --port COM6
```

如果系统安装了 `uv`，脚本会自动通过 `uv run` 调用 `lerobot`；否则使用工作区的 `venv`。
