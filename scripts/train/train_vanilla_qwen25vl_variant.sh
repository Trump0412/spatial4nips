#!/bin/bash
# Shared vanilla-regime launcher for Qwen2.5-VL variants.
# Expected env vars:
#   PROJECT_ROOT, MODEL_PATH, GEOMETRY_ENCODER_TYPE, GEOMETRY_ENCODER_PATH,
#   GEO_INJECT_VERSION, OUTPUT_DIR, LOG_DIR, TRAIN_LOG.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"/data3/yeyuanhao/checkpoints"}
MODEL_PATH=${MODEL_PATH:-""}
GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE:-"vggt"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-""}
GEO_INJECT_VERSION=${GEO_INJECT_VERSION:-"v2_flash"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/vanilla_qwen25vl_variant"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/vanilla_qwen25vl_variant"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
CACHE_DIR=${CACHE_DIR:-"./cache"}

DATASETS=${DATASETS:-"llava_hound_64k,spar_234k"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-64}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LR=${LR:-1e-5}

TUNE_MM_LLM=${TUNE_MM_LLM:-"True"}
TUNE_MM_VISION=${TUNE_MM_VISION:-"False"}
TUNE_MM_MLP=${TUNE_MM_MLP:-"False"}
SKIP_PIP_INSTALL=${SKIP_PIP_INSTALL:-1}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-""}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [ "${SKIP_PIP_INSTALL}" != "1" ]; then
    pip install transformers==4.50.0
fi

resolve_geometry_encoder_path() {
    if [ -n "${GEOMETRY_ENCODER_PATH}" ]; then
        echo "${GEOMETRY_ENCODER_PATH}"
        return 0
    fi

    if [ "${GEOMETRY_ENCODER_TYPE}" = "da3" ]; then
        for path in \
            "/data3/yeyuanhao/checkpoints/DA3-GIANT" \
            "${CHECKPOINT_ROOT}/DA3-GIANT" \
            "${PROJECT_ROOT}/checkpoints/DA3-GIANT"; do
            if [ -d "${path}" ]; then
                echo "${path}"
                return 0
            fi
        done
        echo "/data3/yeyuanhao/checkpoints/DA3-GIANT"
        return 0
    fi

    for path in \
        "/data3/yeyuanhao/checkpoints/VGGT-1B" \
        "${CHECKPOINT_ROOT}/VGGT-1B" \
        "${PROJECT_ROOT}/checkpoints/VGGT-1B" \
        "${PROJECT_ROOT}/models/VGGT-1B" \
        "/data3/yeyuanhao/sp_re_cbp/TRASE/models/VGGT-1B"; do
        if [ -d "${path}" ]; then
            echo "${path}"
            return 0
        fi
    done

    echo "facebook/VGGT-1B"
    return 0
}

if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] MODEL_PATH must be set by the wrapper script or environment."
    exit 1
fi

GEOMETRY_ENCODER_PATH=$(resolve_geometry_encoder_path)
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}"
    exit 1
fi
if [ "${GEOMETRY_ENCODER_TYPE}" = "da3" ] && [ ! -d "${GEOMETRY_ENCODER_PATH}" ]; then
    echo "[ERROR] GEOMETRY_ENCODER_PATH does not exist: ${GEOMETRY_ENCODER_PATH}"
    exit 1
fi
if [ "${TOTAL_BATCH_SIZE}" -lt "${NPROC_PER_NODE}" ]; then
    echo "[ERROR] TOTAL_BATCH_SIZE (${TOTAL_BATCH_SIZE}) must be >= NPROC_PER_NODE (${NPROC_PER_NODE})."
    exit 1
fi
if [ $((TOTAL_BATCH_SIZE % NPROC_PER_NODE)) -ne 0 ]; then
    echo "[ERROR] TOTAL_BATCH_SIZE (${TOTAL_BATCH_SIZE}) must be divisible by NPROC_PER_NODE (${NPROC_PER_NODE})."
    exit 1
fi

GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / NPROC_PER_NODE))
EXTRA_ARGS_ARRAY=()
if [ -n "${EXTRA_TRAIN_ARGS}" ]; then
    read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_TRAIN_ARGS}"
fi

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE}"
echo "[INFO] GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH}"
echo "[INFO] GEO_INJECT_VERSION=${GEO_INJECT_VERSION}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] TRAIN_LOG=${TRAIN_LOG}"

cd "${PROJECT_ROOT}"
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/qwen_vl/train/train_qwen.py \
    --model_name_or_path "${MODEL_PATH}" \
    --tune_mm_llm "${TUNE_MM_LLM}" \
    --tune_mm_vision "${TUNE_MM_VISION}" \
    --tune_mm_mlp "${TUNE_MM_MLP}" \
    --dataset_use "${DATASETS}" \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LR}" \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 12800 \
    --data_flatten False \
    --max_pixels $((576*28*28)) \
    --min_pixels $((16*28*28)) \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels $((1664*28*28)) \
    --video_min_frame_pixels $((256*28*28)) \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 50 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --deepspeed "scripts/zero2_opt.json" \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --group_by_modality_length true \
    --seed 0 \
    --report_to "none" \
    --use_geometry_encoder true \
    --geometry_encoder_type "${GEOMETRY_ENCODER_TYPE}" \
    --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
    --feature_fusion_method "zero" \
    --geo_cross_attn True \
    --geo_inject_version "${GEO_INJECT_VERSION}" \
    --geo_importance_gate True \
    --depart_smi_token True \
    "${EXTRA_ARGS_ARRAY[@]}" \
    > "${TRAIN_LOG}" 2>&1
