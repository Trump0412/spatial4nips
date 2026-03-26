#!/bin/bash
# Wait for an optional training tmux session to finish and for any 2 GPUs to
# become idle, then run VSIbench and MMSI-Video-Bench for a GeoThinker model.
#
# Usage examples:
#   MODEL_ROOT=/data3/yeyuanhao/sp_re_cbp/GeoThinker/outputs/vanilla_qwen25vl_3b_da3_sgf \
#   MODEL_TAG=vanilla_qwen25vl_3b_da3_sgf \
#   bash scripts/evaluation/watch_gpu_pair_and_run_benchmarks.sh
#
#   WAIT_SESSION=train_vanilla_qwen25vl_3b_da3_hmsgf \
#   MODEL_ROOT=/data3/yeyuanhao/sp_re_cbp/GeoThinker/outputs/vanilla_qwen25vl_3b_da3_hmsgf \
#   MODEL_TAG=vanilla_qwen25vl_3b_da3_hmsgf \
#   bash scripts/evaluation/watch_gpu_pair_and_run_benchmarks.sh

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
WAIT_SESSION=${WAIT_SESSION:-""}
MODEL_ROOT=${MODEL_ROOT:-""}
MODEL_TAG=${MODEL_TAG:-"model_eval"}
SESSION_NAME=${SESSION_NAME:-"eval_${MODEL_TAG}"}
CONDA_ENV=${CONDA_ENV:-"geothinker"}
UTIL_THRESHOLD=${UTIL_THRESHOLD:-10}
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-2000}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-30}
GPU_COUNT=${GPU_COUNT:-2}
VSI_NUM_PROCESSES=${VSI_NUM_PROCESSES:-2}
VSI_MAIN_PROCESS_PORT=${VSI_MAIN_PROCESS_PORT:-29605}
MMSI_TASK=${MMSI_TASK:-"MMSI_Video_Bench"}
VSI_TASK=${VSI_TASK:-"vsibench"}
PID_FILE="${PROJECT_ROOT}/logs/.watch_${SESSION_NAME}.pid"
WATCH_LOG="${PROJECT_ROOT}/logs/watch_${SESSION_NAME}.log"
RUNNER_SCRIPT="${PROJECT_ROOT}/logs/.run_${SESSION_NAME}.sh"
OUTPUT_ROOT_DEFAULT="${PROJECT_ROOT}/outputs/eval/${MODEL_TAG}"
OUTPUT_ROOT=${OUTPUT_ROOT:-"${OUTPUT_ROOT_DEFAULT}"}

usage() {
    cat <<USAGE
Wait for an optional training session to finish, then wait for ${GPU_COUNT} idle GPUs
and run VSIbench followed by ${MMSI_TASK}.

Required env:
  MODEL_ROOT    Path to the trained model directory or output directory.
  MODEL_TAG     Short tag used in tmux/log/output naming.

Optional env:
  WAIT_SESSION  tmux training session name to wait for before evaluation.
  SESSION_NAME  tmux session name used for the evaluation run.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [ -z "${MODEL_ROOT}" ]; then
    echo "[ERROR] MODEL_ROOT must be set." >&2
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"

if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[ERROR] Watcher already running for ${SESSION_NAME} (PID=${OLD_PID})." | tee -a "${WATCH_LOG}"
        exit 1
    fi
fi

echo $$ > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found." | tee -a "${WATCH_LOG}"
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found." | tee -a "${WATCH_LOG}"
    exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[ERROR] tmux session '${SESSION_NAME}' already exists." | tee -a "${WATCH_LOG}"
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

find_idle_pair() {
    local idle_gpus=()
    while IFS=',' read -r idx util mem; do
        idx=$(echo "${idx}" | xargs)
        util=$(echo "${util}" | xargs)
        mem=$(echo "${mem}" | xargs)
        if [ "${util}" -lt "${UTIL_THRESHOLD}" ] && [ "${mem}" -lt "${MEM_THRESHOLD_MB}" ]; then
            idle_gpus+=("${idx}")
            if [ "${#idle_gpus[@]}" -ge "${GPU_COUNT}" ]; then
                local joined
                joined=$(IFS=,; echo "${idle_gpus[*]}")
                echo "${joined}"
                return 0
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
    return 1
}

resolve_model_path() {
    local root="$1"
    local latest
    if [ -f "${root}/config.json" ]; then
        echo "${root}"
        return 0
    fi
    latest=$(find "${root}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
    if [ -n "${latest}" ] && [ -f "${latest}/config.json" ]; then
        echo "${latest}"
        return 0
    fi
    echo ""
    return 1
}

if [ -n "${WAIT_SESSION}" ]; then
    echo "[$(timestamp)] Waiting for training session '${WAIT_SESSION}' to finish." | tee -a "${WATCH_LOG}"
    while wait_session_is_busy; do
        sleep "${POLL_INTERVAL_SEC}"
    done
    echo "[$(timestamp)] Training session '${WAIT_SESSION}' finished." | tee -a "${WATCH_LOG}"
fi

MODEL_PATH=$(resolve_model_path "${MODEL_ROOT}" || true)
if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] Could not resolve a usable model path under ${MODEL_ROOT}." | tee -a "${WATCH_LOG}"
    exit 1
fi

echo "[$(timestamp)] Resolved MODEL_PATH=${MODEL_PATH}" | tee -a "${WATCH_LOG}"
echo "[$(timestamp)] Waiting for ${GPU_COUNT} idle GPUs." | tee -a "${WATCH_LOG}"
GPU_CSV=""
while [ -z "${GPU_CSV}" ]; do
    GPU_CSV=$(find_idle_pair || true)
    if [ -n "${GPU_CSV}" ]; then
        break
    fi
    sleep "${POLL_INTERVAL_SEC}"
done

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d_%H%M%S')
VSI_OUTPUT_PATH="${OUTPUT_ROOT}/vsibench_${TS}"
MMSI_OUTPUT_PATH="${OUTPUT_ROOT}/mmsi_video_bench_${TS}"
VSI_LOG="${PROJECT_ROOT}/logs/eval_${MODEL_TAG}_vsibench_${TS}.log"
MMSI_LOG="${PROJECT_ROOT}/logs/eval_${MODEL_TAG}_mmsi_${TS}.log"
STATUS_FILE="${PROJECT_ROOT}/logs/eval_${MODEL_TAG}_${TS}.status"
mkdir -p "${VSI_OUTPUT_PATH}" "${MMSI_OUTPUT_PATH}"

cat > "${RUNNER_SCRIPT}" <<RUNEOF
#!/bin/bash
set +e
source ~/.bashrc >/dev/null 2>&1
if command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)" >/dev/null 2>&1
elif [ -x "\${HOME}/.conda/bin/conda" ]; then
    eval "\$("\${HOME}/.conda/bin/conda" shell.bash hook)" >/dev/null 2>&1
elif [ -f "\${HOME}/.conda/etc/profile.d/conda.sh" ]; then
    source "\${HOME}/.conda/etc/profile.d/conda.sh" >/dev/null 2>&1
elif [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/usr/local/anaconda3/etc/profile.d/conda.sh" >/dev/null 2>&1
fi
conda activate "${CONDA_ENV}" >/dev/null 2>&1 || exit 127
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:\${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE=0
export LMMS_EVAL_LAUNCHER=accelerate
export CUDA_VISIBLE_DEVICES="${GPU_CSV}"

accelerate launch --num_processes=${VSI_NUM_PROCESSES} --main_process_ip 127.0.0.1 --main_process_port ${VSI_MAIN_PROCESS_PORT} -m lmms_eval \
    --verbosity INFO \
    --model geothinker \
    --model_args pretrained=${MODEL_PATH},use_flash_attention_2=true,max_num_frames=32,max_length=12800,max_pixels=451584,min_pixels=12544 \
    --tasks ${VSI_TASK} \
    --batch_size 1 \
    --output_path "${VSI_OUTPUT_PATH}" \
    > "${VSI_LOG}" 2>&1
VSI_EXIT_CODE=\$?
if [ "\${VSI_EXIT_CODE}" -ne 0 ]; then
    echo "VSI_EXIT_CODE=\${VSI_EXIT_CODE}" > "${STATUS_FILE}"
    exit \${VSI_EXIT_CODE}
fi

python -m lmms_eval \
    --model geothinker \
    --model_args pretrained=${MODEL_PATH},use_flash_attention_2=true,max_num_frames=32,max_length=12800,max_pixels=451584,min_pixels=12544 \
    --tasks ${MMSI_TASK} \
    --batch_size 1 \
    --output_path "${MMSI_OUTPUT_PATH}" \
    > "${MMSI_LOG}" 2>&1
MMSI_EXIT_CODE=\$?
echo "VSI_EXIT_CODE=\${VSI_EXIT_CODE} MMSI_EXIT_CODE=\${MMSI_EXIT_CODE}" > "${STATUS_FILE}"
exit \${MMSI_EXIT_CODE}
RUNEOF
chmod +x "${RUNNER_SCRIPT}"

echo "[$(timestamp)] Launching '${SESSION_NAME}' on GPUs ${GPU_CSV}." | tee -a "${WATCH_LOG}"
echo "[$(timestamp)] VSI output: ${VSI_OUTPUT_PATH}" | tee -a "${WATCH_LOG}"
echo "[$(timestamp)] MMSI output: ${MMSI_OUTPUT_PATH}" | tee -a "${WATCH_LOG}"
tmux new-session -d -s "${SESSION_NAME}" "bash '${RUNNER_SCRIPT}'"
echo "[$(timestamp)] Attach with: tmux attach -t ${SESSION_NAME}" | tee -a "${WATCH_LOG}"
