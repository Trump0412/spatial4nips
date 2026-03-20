#!/bin/bash
# Wait for one tmux training session to finish, then launch another training
# script on a fixed GPU pair in a new tmux session.
# Usage example:
#   WAIT_SESSION=train_vanilla_qwen25vl_3b_da3_sgf \
#   NEXT_SESSION_NAME=train_vanilla_qwen25vl_3b_da3_msgf_base \
#   GPU_PAIR=2,4 \
#   TRAIN_SCRIPT=/data3/yeyuanhao/sp_re_cbp/GeoThinker/scripts/train/train_vanilla_qwen25vl_3b_da3_msgf_base.sh \
#   bash scripts/train/watch_session_pair_then_run.sh

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
WAIT_SESSION=${WAIT_SESSION:-""}
NEXT_SESSION_NAME=${NEXT_SESSION_NAME:-""}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-""}
GPU_PAIR=${GPU_PAIR:-""}
CONDA_ENV=${CONDA_ENV:-"geothinker"}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-64}
SKIP_PIP_INSTALL=${SKIP_PIP_INSTALL:-1}
MASTER_PORT=${MASTER_PORT:-""}
UTIL_THRESHOLD=${UTIL_THRESHOLD:-10}
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-2000}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-30}

if [ -z "${WAIT_SESSION}" ] || [ -z "${NEXT_SESSION_NAME}" ] || [ -z "${TRAIN_SCRIPT}" ] || [ -z "${GPU_PAIR}" ]; then
    echo "[ERROR] WAIT_SESSION, NEXT_SESSION_NAME, TRAIN_SCRIPT, and GPU_PAIR must be set."
    exit 1
fi

LOG_FILE="${PROJECT_ROOT}/logs/watch_${NEXT_SESSION_NAME}.log"
PID_FILE="${PROJECT_ROOT}/logs/.watch_${NEXT_SESSION_NAME}.pid"
RUNNER_SCRIPT="${PROJECT_ROOT}/logs/.run_${NEXT_SESSION_NAME}.sh"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[ERROR] Watcher already running for ${NEXT_SESSION_NAME} (PID=${OLD_PID})." | tee -a "${LOG_FILE}"
        exit 1
    fi
fi
echo $$ > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found." | tee -a "${LOG_FILE}"
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found." | tee -a "${LOG_FILE}"
    exit 1
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] TRAIN_SCRIPT not found: ${TRAIN_SCRIPT}" | tee -a "${LOG_FILE}"
    exit 1
fi
if tmux has-session -t "${NEXT_SESSION_NAME}" 2>/dev/null; then
    echo "[ERROR] tmux session '${NEXT_SESSION_NAME}' already exists." | tee -a "${LOG_FILE}"
    exit 1
fi

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

wait_session_is_busy() {
    if ! tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; then
        return 1
    fi

    while read -r pane_dead pane_cmd; do
        if [ "${pane_dead}" = "1" ]; then
            continue
        fi
        if [ "${pane_cmd}" = "sleep" ]; then
            continue
        fi
        return 0
    done < <(tmux list-panes -t "${WAIT_SESSION}" -F '#{pane_dead} #{pane_current_command}')

    return 1
}

pair_is_idle() {
    local pair_csv="$1"
    IFS=',' read -r -a TARGET_GPUS <<< "${pair_csv}"
    while IFS=',' read -r idx util mem; do
        idx=$(echo "${idx}" | xargs)
        util=$(echo "${util}" | xargs)
        mem=$(echo "${mem}" | xargs)
        for gpu in "${TARGET_GPUS[@]}"; do
            if [ "${idx}" = "${gpu}" ]; then
                if [ "${util}" -ge "${UTIL_THRESHOLD}" ] || [ "${mem}" -ge "${MEM_THRESHOLD_MB}" ]; then
                    return 1
                fi
            fi
        done
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
    return 0
}

echo "[$(timestamp)] Waiting for training session '${WAIT_SESSION}' to finish." | tee -a "${LOG_FILE}"
echo "[$(timestamp)] A session is considered finished when it disappears or its active pane falls back to 'sleep'." | tee -a "${LOG_FILE}"
while wait_session_is_busy; do
    sleep "${POLL_INTERVAL_SEC}"
done

echo "[$(timestamp)] Session '${WAIT_SESSION}' finished. Waiting for GPU pair ${GPU_PAIR} to become idle." | tee -a "${LOG_FILE}"
while ! pair_is_idle "${GPU_PAIR}"; do
    sleep "${POLL_INTERVAL_SEC}"
done

if [ -z "${MASTER_PORT}" ]; then
    MASTER_PORT=$(shuf -i 20000-29999 -n 1)
fi

cat > "${RUNNER_SCRIPT}" <<EOF
#!/bin/bash
set +e
source ~/.bashrc >/dev/null 2>&1
conda activate "${CONDA_ENV}" >/dev/null 2>&1
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_PAIR}"
export NPROC_PER_NODE="${NPROC_PER_NODE}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE}"
export MASTER_PORT="${MASTER_PORT}"
export SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL}"
bash "${TRAIN_SCRIPT}"
EXIT_CODE=\$?
echo "\$(date '+%Y-%m-%d %H:%M:%S') EXIT_CODE=\${EXIT_CODE}" >> "${LOG_FILE}"
exit \${EXIT_CODE}
EOF
chmod +x "${RUNNER_SCRIPT}"

echo "[$(timestamp)] Launching '${NEXT_SESSION_NAME}' on GPU pair ${GPU_PAIR}." | tee -a "${LOG_FILE}"
tmux new-session -d -s "${NEXT_SESSION_NAME}" "bash '${RUNNER_SCRIPT}'"
echo "[$(timestamp)] Attach with: tmux attach -t ${NEXT_SESSION_NAME}" | tee -a "${LOG_FILE}"
