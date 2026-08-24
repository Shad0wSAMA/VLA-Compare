# SO101：ArUco + YOLO 自动抓取

主程序是 `scripts/aruco_yolo_pick_place.py`。它完成以下流程：

1. 从固定相机连续读取多帧，检测四个 ArUco Marker。
2. 用四个 Marker 中心点及其桌面坐标计算“像素 → 桌面 XY”的单应变换。
3. 用 YOLO 检测方块，将检测框中心转换为桌面坐标。
4. 用配置的 `table_to_robot` 把桌面目标转换到 URDF 基座坐标。
5. 默认调用 `lerobot.model.kinematics_ikpy.RobotKinematics.inverse_kinematics()` 计算 SO101 关节角。
6. 按“张开、接近、下降、夹紧、抬起、移至固定点、下降、释放、撤离”执行平滑关节轨迹。
7. 第一条运动指令前创建 LeRobotDataset；每一帧记录相机、关节观测及实际发送动作；成功后保存一个 episode 并结束录制。

固定相机 `fixed` 用于 ArUco/YOLO，腕部相机 `wrist` 仅作为观测。两者都会进入 LeRobotDataset。当前数据集时间轴为 30 FPS：wrist 相机可以按 60 FPS 采集，但控制循环每次保存其最新帧，因此最终 wrist 视频仍约为每秒 30 个数据帧。

## 安装

在工作区根目录运行：

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements_pick_place.txt
```

配置默认使用纯 Python 的 IKPy 后端，可直接在当前原生 Windows 环境运行，不再要求 Placo。若以后需要切回原实现，可将 `kinematics.backend` 改成 `placo` 并安装对应 LeRobot extra。

### 单独使用 IKPy 工具

新工具保持了原 `RobotKinematics` 的主要接口，外部关节角仍使用度，末端位姿仍使用米制 4×4 齐次矩阵：

```python
import numpy as np

from lerobot.model.kinematics_ikpy import RobotKinematics

kinematics = RobotKinematics(
    urdf_path="models/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=[
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ],
    orientation_mode="Z",
)

current_deg = np.array([0.0, -20.0, 40.0, 20.0, 0.0, 90.0])
current_pose = kinematics.forward_kinematics(current_deg)
target_pose = current_pose.copy()
target_pose[2, 3] += 0.02
target_deg = kinematics.inverse_kinematics(current_deg, target_pose)
```

返回值会保留输入向量中五个机械臂关节之后的值，因此上例中的第六项夹爪状态不会被 IK 改写。`orientation_weight <= 0` 时只求目标位置；正值会启用配置的 `orientation_mode`。适配器通过 SciPy least-squares 对 IKPy 链进行求解，因此 `position_weight` 和 `orientation_weight` 会实际缩放对应误差，`max_iterations` 也会限制函数评估次数。

## YOLO 数据和训练

把图像放入：

- `training/yolo_cube/images/train`
- `training/yolo_cube/images/val`

把同名 YOLO 标签文本放入：

- `training/yolo_cube/labels/train`
- `training/yolo_cube/labels/val`

每行标签格式为 `class_id x_center y_center width height`，后四项均为相对图像宽高的 0–1 数值；本模板只有类别 `0: cube`。

训练：

```powershell
.\venv\Scripts\python.exe scripts\train_yolo_cube.py --epochs 100 --device 0
```

CPU 训练时使用 `--device cpu`。训练结束后，把输出的 `best.pt` 路径写入配置的 `perception.weights`。

## 必须补齐和实测的内容

不要直接把示例里的三个 `*_confirmed` 改成 `true`。应先逐项测量、验证：

1. **YOLO 数据和权重**：当前没有标注图片，也没有 `best.pt`。
2. **SO101 URDF**：工作区目前没有 `models/SO101/so101_new_calib.urdf`。建议从 [TheRobotStudio/SO-ARM100 的 SO101 仿真目录](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)复制整个 `SO101` 文件夹，并确认末端 frame 名称和五个关节名称。IKPy 只解析运动链，不要求 mesh 文件，但保留完整目录便于后续可视化和其他工具使用。
3. **四个 Marker 外框边界**：坐标变换使用 Marker 外框朝外的角点，不再使用中心。OpenCV 角点顺序是左上 `0`、右上 `1`、右下 `2`、左下 `3`；当前配置对应 Marker 0/1/2/3 的角点 0/1/2/3。边界 0→1 为 19.2 mm，0→3 为 27.1 mm，单位已换算为米写入 `marker_boundary.*.xy_m`。该角点配置假定四个 Marker 的印刷方向一致；若某个 Marker 旋转安装，需要相应修改它的 `corner_index`。
4. **桌面到机器人基座的外参**：`table_to_robot` 是把桌面齐次坐标左乘变换到 URDF 基座坐标的 4×4 矩阵。单位必须为米，不能使用示例单位阵。
5. **相机内参**：强烈建议标定并填写 `camera_matrix` 与 `dist_coeffs`；两者保持 `null` 时仍能用平面单应变换，但镜头畸变会降低边缘区域精度。
6. **抓取高度和姿态**：需要根据方块尺寸、夹爪 TCP、URDF 末端 frame 调整 `grasp_z_m`、放置 Z、高度、`tool_rotation_rpy_deg_in_table` 和开合值。
7. **安全范围**：实测后填写五组关节限位 `joint_limits_deg`，收紧 `workspace_bounds_robot_m`；主程序会拒绝缺少关节限位的配置。首次运行应低速、空载、随时准备断电。
8. **端口、相机编号、机械臂标定**：确认 `COM3`、fixed 相机 `1`、wrist 相机 `2`、robot id 及 LeRobot 电机标定文件。
9. **数据集身份**：把 `recording.repo_id` 的 `your_name` 改掉。仅本地保存时无需 Hugging Face 登录；上传时还需登录并启用 `push_to_hub`。

## 检查和运行

只做配置检查，不连接硬件：

```powershell
.\venv\Scripts\python.exe scripts\aruco_yolo_pick_place.py `
  --config configs\aruco_yolo_pick_place.example.json `
  --check-config
```

所有检查通过后，移除 `--check-config` 运行。程序会将最终识别预览写到 `outputs/aruco_yolo_detection.jpg`。任何 Marker 不全、识别失败、目标超工作区、IK 误差过大或关节越界都会在发送该段动作前停止。

## 视觉可视化测试

该脚本只打开 `robot.perception_camera`（默认 `fixed`），不会连接机械臂：

```powershell
# 尚无 YOLO 权重时，只测试四个 Marker 和桌面坐标变换
.\venv\Scripts\python.exe scripts\visualize_aruco_yolo.py --aruco-only

# 配置中的 perception.weights 已填写后，同时测试方块识别
.\venv\Scripts\python.exe scripts\visualize_aruco_yolo.py

# 临时指定权重，不修改 JSON
.\venv\Scripts\python.exe scripts\visualize_aruco_yolo.py --weights outputs\yolo\cube\weights\best.pt
```

画面会显示 Marker ID、被用于变换的紫色外框角点、桌面有效范围、坐标网格、XY 轴、固定放置点、边界角点坐标残差、YOLO 方框中心及其桌面坐标。当前边界只有毫米量级，建议用 `--grid-step-m 0.005` 显示每 5 mm 网格。按 `r` 清空样本并重新估计单应矩阵，按 `s` 保存截图，按 `q`/Esc 退出；鼠标左键点击任意位置可查看对应桌面坐标。

无窗口录制带标注的测试视频：

```powershell
.\venv\Scripts\python.exe scripts\visualize_aruco_yolo.py `
  --aruco-only --no-display --max-frames 300 `
  --output-video outputs\vision_debug\aruco_test.mp4
```
