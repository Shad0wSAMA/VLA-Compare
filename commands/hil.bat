@echo off
cd /d "%~dp0.."

.\venv\Scripts\lerobot-rollout.exe ^
    --strategy.type=dagger ^
    --strategy.record_autonomous=false ^
    --strategy.num_episodes=50 ^
    --strategy.wrist_roll_offset_deg=-90 ^
    --policy.path=./models/model2 ^
    --rename_map="{ observation.images.fixed: observation.images.camera1, observation.images.wrist: observation.images.camera2 }" ^
    --robot.type=so101_follower ^
    --robot.port=COM3 ^
    --robot.id=my_awesome_follower_arm ^
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 60, fourcc: MJPG}, fixed: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" ^
    --teleop.type=so101_leader ^
    --teleop.port=COM4 ^
    --teleop.id=my_awesome_leader_arm ^
    --dataset.repo_id=rollout_stack_cube_data2_hil ^
    --dataset.root=./training/rollout_stack_cube_data2_hil_run1 ^
    --dataset.single_task="Pick up one cube and place it in the red area." ^
    --dataset.fps=30 ^
    --dataset.streaming_encoding=true ^
    --dataset.rgb_encoder.vcodec=auto ^
    --dataset.rgb_encoder.g=null ^
    --dataset.encoder_threads=2 ^
    --dataset.push_to_hub=false ^
    --inference.type=rtc ^
    --inference.rtc.execution_horizon=10 ^
    --inference.rtc.max_guidance_weight=10.0 ^
    --resume=false ^
    --fps=30 ^
    --duration=0 ^
    --display_data=false ^  
    --play_sounds=true
