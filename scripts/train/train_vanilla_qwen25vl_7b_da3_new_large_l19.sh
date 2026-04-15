#!/bin/bash
# 7B Vanilla ablation: DA3-LARGE-1.1 + out_layer_index=0 (layer 11, shallowest out_layer for vitl)
# vs baseline: DA3-GIANT + layer 39 (deepest)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
MODEL_PATH=${MODEL_PATH:-"/data3/yeyuanhao/sp_re_cbp/thirdparty/models/Qwen2.5-VL-7B-Instruct"}
GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE:-"da3_new"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/sp_re_cbp/thirdparty/DA3-LARGE-1.1"}
GEO_INJECT_VERSION=${GEO_INJECT_VERSION:-"da3_msgf_temporal_refine"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/vanilla_qwen25vl_7b_da3_new_large_l19"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/vanilla_qwen25vl_7b_da3_new_large_l19"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}

BASE_EXTRA_ARGS="--geo_inject_version da3_msgf_temporal_refine --msgf_topr 32 --msgf_frame_topk_max 3 --msgf_atom_topk_max 8 --msgf_use_bidirectional True --temporal_bonus_lambda 0.10 --geo_encoder_out_layer_index 0"

if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS} ${EXTRA_TRAIN_ARGS}"
else
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS}"
fi

export PROJECT_ROOT MODEL_PATH
export GEOMETRY_ENCODER_TYPE GEOMETRY_ENCODER_PATH GEO_INJECT_VERSION
export OUTPUT_DIR LOG_DIR TRAIN_LOG EXTRA_TRAIN_ARGS

exec bash "${SCRIPT_DIR}/train_vanilla_qwen25vl_variant.sh"
