#!/bin/bash
# Wait until 8 GPUs are idle, then run formal 7B vanilla DA3 training in tmux.
# After successful training, automatically run VSIbench with the trained model.
# Usage:
#   bash scripts/train/watch_gpu_and_run_train_vanilla_qwen25vl_7b_da3.sh

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${PROJECT_ROOT}/scripts/train/train_vanilla_qwen25vl_7b_da3.sh"}
CONDA_SH=${CONDA_SH:-"/usr/local/anaconda3/etc/profile.d/conda.sh"}
CONDA_ENV=${CONDA_ENV:-"geothinker"}
UTIL_THRESHOLD=${UTIL_THRESHOLD:-10}
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-2000}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-60}
SESSION_NAME=${SESSION_NAME:-"train_vanilla_qwen25vl_7b_da3"}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/vanilla_qwen25vl_7b_da3"}
TRAIN_LOG=${TRAIN_LOG:-"${PROJECT_ROOT}/logs/train_vanilla_qwen25vl_7b_da3.log"}
TRAIN_STATUS_FILE=${TRAIN_STATUS_FILE:-"${PROJECT_ROOT}/logs/train_vanilla_qwen25vl_7b_da3.status"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/eval/vsibench_vanilla_qwen25vl_7b_da3"}
EVAL_LOG=${EVAL_LOG:-"${PROJECT_ROOT}/logs/eval_vanilla_qwen25vl_7b_da3_vsibench.log"}
EVAL_STATUS_FILE=${EVAL_STATUS_FILE:-"${PROJECT_ROOT}/logs/eval_vanilla_qwen25vl_7b_da3_vsibench.status"}
PID_FILE="${PROJECT_ROOT}/logs/.watch_gpu_and_run_train_vanilla_qwen25vl_7b_da3.pid"
RUNNER_SCRIPT="${PROJECT_ROOT}/logs/.run_train_vanilla_qwen25vl_7b_da3.sh"

usage() {
    cat <<EOF
Wait until all 8 GPUs are idle, then launch formal training in tmux session '${SESSION_NAME}'.
If training succeeds, VSIbench is launched automatically in the same tmux session.

Idle condition:
  utilization.gpu < UTIL_THRESHOLD (default ${UTIL_THRESHOLD})
  memory.used < MEM_THRESHOLD_MB (default ${MEM_THRESHOLD_MB} MB)

Example:
  bash scripts/train/watch_gpu_and_run_train_vanilla_qwen25vl_7b_da3.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "${PROJECT_ROOT}/logs" "${TRAIN_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[ERROR] Another train watcher is already running (PID=${OLD_PID})."
        exit 1
    fi
fi
echo $$ > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found."
    exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found."
    exit 1
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] TRAIN_SCRIPT not found: ${TRAIN_SCRIPT}"
    exit 1
fi
if [ ! -f "${CONDA_SH}" ]; then
    echo "[ERROR] conda.sh not found: ${CONDA_SH}"
    exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[ERROR] tmux session '${SESSION_NAME}' already exists."
    exit 1
fi

all_eight_idle() {
    mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
    if [ "${#GPU_LINES[@]}" -lt 8 ]; then
        echo "[ERROR] Fewer than 8 GPUs detected by nvidia-smi."
        return 1
    fi
    for ((i=0; i<8; i++)); do
        IFS=',' read -r idx util mem <<< "${GPU_LINES[i]}"
        idx=$(echo "${idx}" | xargs)
        util=$(echo "${util}" | xargs)
        mem=$(echo "${mem}" | xargs)
        if [ "${util}" -ge "${UTIL_THRESHOLD}" ] || [ "${mem}" -ge "${MEM_THRESHOLD_MB}" ]; then
            return 1
        fi
    done
    return 0
}

echo "[INFO] Waiting for all 8 GPUs to become idle..."
while true; do
    if all_eight_idle; then
        break
    fi
    sleep "${POLL_INTERVAL_SEC}"
done
echo "[INFO] All 8 GPUs are idle. Launching training."

rm -f "${TRAIN_STATUS_FILE}" "${EVAL_STATUS_FILE}"
cat > "${RUNNER_SCRIPT}" <<EOF
#!/bin/bash
set +e
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:\${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SKIP_PIP_INSTALL=1 \\
OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \\
LOG_DIR="${PROJECT_ROOT}/logs" \\
TRAIN_LOG="${TRAIN_LOG}" \\
bash "${TRAIN_SCRIPT}"
TRAIN_EXIT_CODE=\$?
echo "\${TRAIN_EXIT_CODE}" > "${TRAIN_STATUS_FILE}"
if [ "\${TRAIN_EXIT_CODE}" -ne 0 ]; then
    exit \${TRAIN_EXIT_CODE}
fi

MODEL_PATH="${TRAIN_OUTPUT_DIR}"
LATEST_CHECKPOINT=\$(find "${TRAIN_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
if [ -n "\${LATEST_CHECKPOINT}" ] && [ -f "\${LATEST_CHECKPOINT}/config.json" ]; then
    MODEL_PATH="\${LATEST_CHECKPOINT}"
fi

accelerate launch --num_processes=8 --main_process_ip 127.0.0.1 --main_process_port 29605 -m lmms_eval \\
    --verbosity INFO \\
    --model geothinker \\
    --model_args pretrained=\${MODEL_PATH},use_flash_attention_2=true,max_num_frames=32,max_length=12800,max_pixels=451584,min_pixels=12544 \\
    --tasks vsibench \\
    --batch_size 1 \\
    --output_path "${EVAL_OUTPUT_DIR}" \\
    > "${EVAL_LOG}" 2>&1
EVAL_EXIT_CODE=\$?
echo "\${EVAL_EXIT_CODE}" > "${EVAL_STATUS_FILE}"
exit \${EVAL_EXIT_CODE}
EOF
chmod +x "${RUNNER_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" "bash '${RUNNER_SCRIPT}'"
echo "[INFO] Training watcher launched tmux session '${SESSION_NAME}'."
echo "[INFO] Training log: ${TRAIN_LOG}"
echo "[INFO] Eval log: ${EVAL_LOG}"
echo "[INFO] Attach with: tmux attach -t ${SESSION_NAME}"
