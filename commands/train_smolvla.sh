#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve the workspace from this script, so it works from any current directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${DATASET_NAME:-manual_data1}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE}/training/${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/outputs/train/smolvla_${DATASET_NAME}}"
LOG_DIR="${WORKSPACE}/outputs/train_logs"
LOG_FILE="${LOG_DIR}/smolvla_${DATASET_NAME}.log"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-20000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Dataset not found: ${DATASET_ROOT}/meta/info.json" >&2
    echo "Copy training/${DATASET_NAME} into this workspace, or set DATASET_ROOT." >&2
    exit 1
fi

if [[ -x "${WORKSPACE}/venv/bin/lerobot-train" ]]; then
    LEROBOT_TRAIN="${WORKSPACE}/venv/bin/lerobot-train"
elif [[ -x "${WORKSPACE}/.venv/bin/lerobot-train" ]]; then
    LEROBOT_TRAIN="${WORKSPACE}/.venv/bin/lerobot-train"
elif command -v lerobot-train >/dev/null 2>&1; then
    LEROBOT_TRAIN="$(command -v lerobot-train)"
else
    echo "lerobot-train was not found. Activate/install the AutoDL LeRobot environment first." >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_DIR}")" "${LOG_DIR}" "${WORKSPACE}/.cache/huggingface"

export PYTHONPATH="${WORKSPACE}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Dataset: ${DATASET_ROOT}"
echo "Output:  ${OUTPUT_DIR}"
echo "Log:     ${LOG_FILE}"

TRAIN_ARGS=(
    "--policy.path=lerobot/smolvla_base"
    "--policy.device=cuda"
    "--policy.use_amp=true"
    "--policy.push_to_hub=false"
    "--dataset.repo_id=${DATASET_NAME}"
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.video_backend=pyav"
    '--rename_map={"observation.images.fixed":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
    "--batch_size=${BATCH_SIZE}"
    "--steps=${STEPS}"
    "--num_workers=${NUM_WORKERS}"
    "--save_freq=${SAVE_FREQ}"
    "--log_freq=50"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=smolvla_${DATASET_NAME}"
    "--wandb.enable=false"
    "--save_checkpoint_to_hub=false"
)

# Re-running the same command resumes the latest checkpoint automatically.
LATEST_CHECKPOINT=""
if [[ -d "${OUTPUT_DIR}/checkpoints" ]]; then
    LATEST_CHECKPOINT="$({
        find "${OUTPUT_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null || true
    } | awk '/^[0-9]+$/' | sort -n | tail -1)"
fi

if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    CONFIG_PATH="${OUTPUT_DIR}/checkpoints/${LATEST_CHECKPOINT}/pretrained_model/train_config.json"
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "Checkpoint exists but train_config.json is missing: ${CONFIG_PATH}" >&2
        exit 1
    fi
    echo "Resuming checkpoint: ${LATEST_CHECKPOINT}"
    # On resume, load the policy from the checkpoint instead of smolvla_base.
    TRAIN_ARGS=("${TRAIN_ARGS[@]:1}")
    TRAIN_ARGS+=("--resume=true" "--config_path=${CONFIG_PATH}")
elif [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output directory already exists but contains no resumable checkpoint: ${OUTPUT_DIR}" >&2
    echo "Set OUTPUT_DIR to a new directory before retrying." >&2
    exit 1
fi

"${LEROBOT_TRAIN}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
