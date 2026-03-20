#!/bin/bash
# Wait for any eligible idle GPU pair, launch stage-1 training, then after it
# finishes launch stage-2 training on the next eligible idle GPU pair.
# Usage example:
#   STAGE1_SESSION=train_vanilla_qwen25vl_3b_da3_hmsgf \
#   STAGE1_SCRIPT=/data3/yeyuanhao/sp_re_cbp/GeoThinker/scripts/train/train_vanilla_qwen25vl_3b_da3_hmsgf.sh \
#   STAGE2_SESSION=train_vanilla_qwen25vl_3b_vggt_sgf \
#   STAGE2_SCRIPT=/data3/yeyuanhao/sp_re_cbp/GeoThinker/scripts/train/train_vanilla_qwen25vl_3b_vggt_sgf.sh \
#   GPU_EXCLUDE=2,4,5,6 \
#   bash scripts/train/watch_any_pair_run_queue.sh

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
STAGE1_SESSION=${STAGE1_SESSION:-""}
STAGE1_SCRIPT=${STAGE1_SCRIPT:-""}
STAGE2_SESSION=${STAGE2_SESSION:-""}
STAGE2_SCRIPT=${STAGE2_SCRIPT:-""}
GPU_EXCLUDE=${GPU_EXCLUDE:-""}
CONDA_ENV=${CONDA_ENV:-"geothinker"}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-64}
SKIP_PIP_INSTALL=${SKIP_PIP_INSTALL:-1}
UTIL_THRESHOLD=${UTIL_THRESHOLD:-10}
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-2000}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-30}

if [ -z "${STAGE1_SESSION}" ] || [ -z "${STAGE1_SCRIPT}" ] || [ -z "${STAGE2_SESSION}" ] || [ -z "${STAGE2_SCRIPT}" ]; then
    echo "[ERROR] STAGE1_SESSION, STAGE1_SCRIPT, STAGE2_SESSION, and STAGE2_SCRIPT must be set."
    exit 1
fi

LOG_FILE="${PROJECT_ROOT}/logs/watch_queue_${STAGE1_SESSION}_then_${STAGE2_SESSION}.log"
PID_FILE="${PROJECT_ROOT}/logs/.watch_queue_${STAGE1_SESSION}_then_${STAGE2_SESSION}.pid"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[ERROR] Queue watcher already running (PID=${OLD_PID})." | tee -a "${LOG_FILE}"
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

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

gpu_is_excluded() {
    local idx="$1"
    IFS=',' read -r -a EXCLUDED <<< "${GPU_EXCLUDE}"
    for gpu in "${EXCLUDED[@]}"; do
        if [ -n "${gpu}" ] && [ "${idx}" = "${gpu}" ]; then
            return 0
        fi
    done
    return 1
}

find_idle_pair() {
    local idle_gpus=()
    while IFS=',' read -r idx util mem; do
        idx=$(echo "${idx}" | xargs)
        util=$(echo "${util}" | xargs)
        mem=$(echo "${mem}" | xargs)
        if gpu_is_excluded "${idx}"; then
            continue
        fi
        if [ "${util}" -lt "${UTIL_THRESHOLD}" ] && [ "${mem}" -lt "${MEM_THRESHOLD_MB}" ]; then
            idle_gpus+=("${idx}")
            if [ "${#idle_gpus[@]}" -ge 2 ]; then
                echo "${idle_gpus[0]},${idle_gpus[1]}"
                return 0
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
    return 1
}

launch_stage() {
    local session_name="$1"
    local train_script="$2"
    local gpu_pair="$3"
    local master_port
    local runner_script

    if tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "[ERROR] tmux session '${session_name}' already exists." | tee -a "${LOG_FILE}"
        exit 1
    fi
    if [ ! -f "${train_script}" ]; then
        echo "[ERROR] TRAIN_SCRIPT not found: ${train_script}" | tee -a "${LOG_FILE}"
        exit 1
    fi

    master_port=$(shuf -i 20000-29999 -n 1)
    runner_script="${PROJECT_ROOT}/logs/.run_${session_name}.sh"
    cat > "${runner_script}" <<EOF
#!/bin/bash
set +e
source ~/.bashrc >/dev/null 2>&1
conda activate "${CONDA_ENV}" >/dev/null 2>&1
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu_pair}"
export NPROC_PER_NODE="${NPROC_PER_NODE}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE}"
export MASTER_PORT="${master_port}"
export SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL}"
bash "${train_script}"
EXIT_CODE=\$?
echo "\$(date '+%Y-%m-%d %H:%M:%S') ${session_name} EXIT_CODE=\${EXIT_CODE}" >> "${LOG_FILE}"
exit \${EXIT_CODE}
EOF
    chmod +x "${runner_script}"

    echo "[$(timestamp)] Launching '${session_name}' on GPUs ${gpu_pair}." | tee -a "${LOG_FILE}"
    tmux new-session -d -s "${session_name}" "bash '${runner_script}'"
}

wait_and_launch_stage() {
    local session_name="$1"
    local train_script="$2"
    local pair=""
    while true; do
        pair=$(find_idle_pair || true)
        if [ -n "${pair}" ]; then
            break
        fi
        sleep "${POLL_INTERVAL_SEC}"
    done
    launch_stage "${session_name}" "${train_script}" "${pair}"
}

echo "[$(timestamp)] Waiting for an extra idle GPU pair for stage 1 (${STAGE1_SESSION})." | tee -a "${LOG_FILE}"
wait_and_launch_stage "${STAGE1_SESSION}" "${STAGE1_SCRIPT}"

while tmux has-session -t "${STAGE1_SESSION}" 2>/dev/null; do
    sleep "${POLL_INTERVAL_SEC}"
done

echo "[$(timestamp)] Stage 1 finished. Waiting for an idle GPU pair for stage 2 (${STAGE2_SESSION})." | tee -a "${LOG_FILE}"
wait_and_launch_stage "${STAGE2_SESSION}" "${STAGE2_SCRIPT}"
echo "[$(timestamp)] Queue watcher finished launching both stages." | tee -a "${LOG_FILE}"
