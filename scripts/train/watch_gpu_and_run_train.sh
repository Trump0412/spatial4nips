#!/bin/bash
# Wait until 8 GPUs are all idle, then start formal training in tmux.
# Usage:
#   bash scripts/train/watch_gpu_and_run_train.sh
# Optional env:
#   PROJECT_ROOT, TRAIN_SCRIPT, UTIL_THRESHOLD, MEM_THRESHOLD_MB, POLL_INTERVAL_SEC,
#   SESSION_NAME, LOG_FILE, STATUS_FILE

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${PROJECT_ROOT}/scripts/train/train_vanilla_qwen25vl_3b_da3.sh"}
UTIL_THRESHOLD=${UTIL_THRESHOLD:-10}
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-2000}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-60}
SESSION_NAME=${SESSION_NAME:-"train_vanilla_qwen25vl_3b_da3"}
LOG_FILE=${LOG_FILE:-"${PROJECT_ROOT}/logs/train_vanilla_qwen25vl_3b_da3.log"}
STATUS_FILE=${STATUS_FILE:-"${PROJECT_ROOT}/logs/train_vanilla_qwen25vl_3b_da3.status"}
PID_FILE="${PROJECT_ROOT}/logs/.watch_gpu_and_run_train.pid"
RUNNER_SCRIPT="${PROJECT_ROOT}/logs/.run_train_vanilla_qwen25vl_3b_da3.sh"

usage() {
    cat <<EOF
Wait until all 8 GPUs are idle, then launch formal training in tmux session '${SESSION_NAME}'.

Idle condition (for each GPU 0-7, configurable):
  utilization.gpu < UTIL_THRESHOLD (default ${UTIL_THRESHOLD})
  memory.used < MEM_THRESHOLD_MB (default ${MEM_THRESHOLD_MB} MB)

Example:
  POLL_INTERVAL_SEC=30 bash scripts/train/watch_gpu_and_run_train.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[ERROR] Another watch_gpu_and_run_train.sh is already running (PID=${OLD_PID})."
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
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[ERROR] tmux session '${SESSION_NAME}' already exists. Refusing duplicate launch."
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
    echo "[INFO] GPUs not all idle yet. Sleep ${POLL_INTERVAL_SEC}s."
    sleep "${POLL_INTERVAL_SEC}"
done
echo "[INFO] All 8 GPUs are idle. Launching formal training."

rm -f "${STATUS_FILE}"
cat > "${RUNNER_SCRIPT}" <<EOF
#!/bin/bash
set +e
cd "${PROJECT_ROOT}"
SKIP_PIP_INSTALL=1 \\
TRAIN_LOG="${LOG_FILE}" \\
bash "${TRAIN_SCRIPT}"
EXIT_CODE=\$?
echo "\${EXIT_CODE}" > "${STATUS_FILE}"
exit \${EXIT_CODE}
EOF
chmod +x "${RUNNER_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" "bash '${RUNNER_SCRIPT}'"
echo "[INFO] Training launched in tmux session '${SESSION_NAME}'."
echo "[INFO] Training log: ${LOG_FILE}"
echo "[INFO] Attach with: tmux attach -t ${SESSION_NAME}"
