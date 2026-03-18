#!/bin/bash
# Wait for one idle GPU, then start 1-card smoke test in tmux.
# Usage:
#   bash scripts/train/watch_gpu_and_run_smoke.sh
# Optional env:
#   PROJECT_ROOT, TRAIN_SCRIPT, UTIL_THRESHOLD, MEM_THRESHOLD_MB, POLL_INTERVAL_SEC,
#   SESSION_NAME, OUTPUT_DIR, LOG_FILE, STATUS_FILE

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${PROJECT_ROOT}/scripts/train/train_vanilla_qwen25vl_3b_da3.sh"}
UTIL_THRESHOLD=${UTIL_THRESHOLD:-10}
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-2000}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-30}
SESSION_NAME=${SESSION_NAME:-"smoke_vanilla_qwen25vl_3b_da3"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/smoke/vanilla_qwen25vl_3b_da3"}
LOG_FILE=${LOG_FILE:-"${PROJECT_ROOT}/logs/smoke_vanilla_qwen25vl_3b_da3.log"}
STATUS_FILE=${STATUS_FILE:-"${PROJECT_ROOT}/logs/smoke_vanilla_qwen25vl_3b_da3.status"}
PID_FILE="${PROJECT_ROOT}/logs/.watch_gpu_and_run_smoke.pid"
RUNNER_SCRIPT="${PROJECT_ROOT}/logs/.run_smoke_vanilla_qwen25vl_3b_da3.sh"

usage() {
    cat <<EOF
Wait for one idle GPU, then run smoke test in tmux session '${SESSION_NAME}'.

Idle condition (configurable):
  utilization.gpu < UTIL_THRESHOLD (default ${UTIL_THRESHOLD})
  memory.used < MEM_THRESHOLD_MB (default ${MEM_THRESHOLD_MB} MB)

Example:
  UTIL_THRESHOLD=5 MEM_THRESHOLD_MB=1500 bash scripts/train/watch_gpu_and_run_smoke.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[ERROR] Another watch_gpu_and_run_smoke.sh is already running (PID=${OLD_PID})."
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
    echo "[ERROR] tmux session '${SESSION_NAME}' already exists. Please finish/kill it first."
    exit 1
fi

find_idle_gpu() {
    while IFS=',' read -r idx util mem; do
        idx=$(echo "${idx}" | xargs)
        util=$(echo "${util}" | xargs)
        mem=$(echo "${mem}" | xargs)
        if [ "${util}" -lt "${UTIL_THRESHOLD}" ] && [ "${mem}" -lt "${MEM_THRESHOLD_MB}" ]; then
            echo "${idx}"
            return 0
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
    return 1
}

echo "[INFO] Waiting for an idle GPU..."
GPU_ID=""
while true; do
    GPU_ID=$(find_idle_gpu || true)
    if [ -n "${GPU_ID}" ]; then
        break
    fi
    echo "[INFO] No idle GPU yet. Sleep ${POLL_INTERVAL_SEC}s."
    sleep "${POLL_INTERVAL_SEC}"
done
echo "[INFO] Selected GPU ${GPU_ID} for smoke test."

rm -f "${STATUS_FILE}"
cat > "${RUNNER_SCRIPT}" <<EOF
#!/bin/bash
set +e
cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} \\
NPROC_PER_NODE=1 \\
TOTAL_BATCH_SIZE=1 \\
SKIP_PIP_INSTALL=1 \\
OUTPUT_DIR="${OUTPUT_DIR}" \\
LOG_DIR="${PROJECT_ROOT}/logs" \\
TRAIN_LOG="${LOG_FILE}" \\
EXTRA_TRAIN_ARGS="--max_steps 1 --logging_steps 1 --save_steps 1000000 --save_total_limit 1" \\
bash "${TRAIN_SCRIPT}"
EXIT_CODE=\$?
echo "\${EXIT_CODE}" > "${STATUS_FILE}"
exit \${EXIT_CODE}
EOF
chmod +x "${RUNNER_SCRIPT}"

echo "[INFO] Starting tmux session '${SESSION_NAME}'."
tmux new-session -d -s "${SESSION_NAME}" "bash '${RUNNER_SCRIPT}'"
echo "[INFO] Smoke test log: ${LOG_FILE}"
echo "[INFO] Attach with: tmux attach -t ${SESSION_NAME}"

while tmux has-session -t "${SESSION_NAME}" 2>/dev/null; do
    sleep 5
done

if [ ! -f "${STATUS_FILE}" ]; then
    echo "[ERROR] Smoke test finished but status file not found: ${STATUS_FILE}"
    exit 1
fi

SMOKE_EXIT_CODE=$(cat "${STATUS_FILE}")
if [ "${SMOKE_EXIT_CODE}" -ne 0 ]; then
    echo "[ERROR] Smoke test failed with exit code ${SMOKE_EXIT_CODE}. See ${LOG_FILE}"
    exit "${SMOKE_EXIT_CODE}"
fi

echo "[INFO] Smoke test succeeded."
