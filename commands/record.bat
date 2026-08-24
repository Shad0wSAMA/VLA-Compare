@echo off
cd /d "%~dp0.."

set "DATASET_NAME=manual_data2"

.\venv\Scripts\lerobot-record.exe ^
    --robot.type=so101_follower ^
    --robot.port=COM3 ^
    --robot.id=my_awesome_follower_arm ^
    --robot.use_degrees=true ^
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 60, fourcc: MJPG}, fixed: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" ^
    --teleop.type=so101_leader ^
    --teleop.port=COM4 ^
    --teleop.id=my_awesome_leader_arm ^
    --teleop.use_degrees=true ^
    --dataset.repo_id=%DATASET_NAME% ^
    --dataset.root=./training/%DATASET_NAME% ^
    --dataset.single_task="Pick up one cube and place it in the red area." ^
    --dataset.fps=30 ^
    --dataset.num_episodes=50 ^
    --dataset.episode_time_s=30 ^
    --dataset.reset_time_s=10 ^
    --dataset.streaming_encoding=true ^
    --dataset.rgb_encoder.vcodec=auto ^
    --dataset.rgb_encoder.g=null ^
    --dataset.encoder_threads=2 ^
    --dataset.push_to_hub=false ^
    --resume=true ^
    --display_data=false ^
    --play_sounds=true
