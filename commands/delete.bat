.\venv\Scripts\lerobot-edit-dataset.exe ^
  --repo_id=rollout_stack_cube_hil ^
  --root=./training/rollout_stack_cube_hil_run2 ^
  --new_root=./training/rollout_stack_cube_hil_run2 ^
  --operation.type=delete_episodes ^
  --operation.episode_indices="[20, 21]"