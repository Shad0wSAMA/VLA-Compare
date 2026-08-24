@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."

if not defined POLICY_PATH set "POLICY_PATH=outputs\train\smolvla_manual_data2_fk"
if not defined ROBOT_PORT set "ROBOT_PORT=COM3"
if not defined ROBOT_ID set "ROBOT_ID=my_follower"

venv\Scripts\python.exe scripts\infer_so101_fk_policy.py ^
  --policy "%POLICY_PATH%" ^
  --dataset-root training\manual_data2_fk ^
  --port "%ROBOT_PORT%" ^
  --robot-id "%ROBOT_ID%" ^
  --fixed-camera 0 ^
  --wrist-camera 1 ^
  %*

endlocal
