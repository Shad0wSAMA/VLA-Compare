# ArUco + YOLO + SO-101 分模块测试

这些脚本可以直接运行，不依赖 `pytest`。除非命令中明确带有硬件或写入开关，否则不会打开相机、串口或创建数据集。

统一运行所有安全/离线测试：

```powershell
venv\Scripts\python.exe scripts\run_pick_place_component_tests.py
```

单独运行：

```powershell
# 配置、双相机映射和 URDF 路径
venv\Scripts\python.exe scripts\pick_place_tests\test_config.py

# 合成 ArUco 图像：检测四个外框角点并测试 27.1 mm x 19.2 mm 单应变换
venv\Scripts\python.exe scripts\pick_place_tests\test_aruco_transform.py

# 桌面坐标到机械臂坐标，以及抓取/放置五个笛卡尔路点
venv\Scripts\python.exe scripts\pick_place_tests\test_geometry.py

# 动作插值（纯计算，不连接机械臂）
venv\Scripts\python.exe scripts\pick_place_tests\test_motion.py

# 使用 robot/so101.urdf 做 IKPy FK -> IK -> FK
venv\Scripts\python.exe scripts\pick_place_tests\test_kinematics.py

# 只构造 wrist=2、fixed=1 的 LeRobot 相机配置
venv\Scripts\python.exe scripts\pick_place_tests\test_cameras.py

# 明确打开两个相机并保存各自一帧（不会连接机械臂）
venv\Scripts\python.exe scripts\pick_place_tests\test_cameras.py --open

# 只加载训练后的 YOLO 权重
venv\Scripts\python.exe scripts\pick_place_tests\test_yolo.py --weights models\cube\best.pt

# 在单张图片上运行 YOLO 选择逻辑
venv\Scripts\python.exe scripts\pick_place_tests\test_yolo.py --weights models\cube\best.pt --image path\to\image.jpg

# 录制器默认只测禁用时的 no-op 行为；下面命令会写一个三帧合成数据集
venv\Scripts\python.exe scripts\pick_place_tests\test_recording.py --write-synthetic

# 使用假机械臂/假录制器检查最终抓取流程的调用顺序
venv\Scripts\python.exe scripts\pick_place_tests\test_pipeline.py
```

实时可视化 ArUco 与 YOLO 仍使用：

```powershell
venv\Scripts\python.exe scripts\visualize_aruco_yolo.py --aruco-only
```

配置中的权重、桌面到机械臂外参、动作高度/夹爪值和数据集名称确认后，完整流程入口是：

```powershell
venv\Scripts\python.exe scripts\aruco_yolo_pick_place.py --check-config
venv\Scripts\python.exe scripts\aruco_yolo_pick_place.py
```
