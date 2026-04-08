#!/bin/bash
# 7B Vanilla Regime: DA3-new (blocks_to_take features, 3072-dim for vitg)
# Uses DA3-GIANT backbone, SGF interaction, Qwen2.5-VL-7B-Instruct base
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
MODEL_PATH=${MODEL_PATH:-"/data3/yeyuanhao/sp_re_cbp/TRASE/models/Qwen2.5-VL-7B-Instruct"}
GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE:-"da3_new"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/checkpoints/DA3-GIANT"}
GEO_INJECT_VERSION=${GEO_INJECT_VERSION:-"da3_new"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/vanilla_qwen25vl_7b_da3_new"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/vanilla_qwen25vl_7b_da3_new"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
BASE_EXTRA_ARGS="--geo_inject_version da3_new"

if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS} ${EXTRA_TRAIN_ARGS}"
else
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS}"
fi

export PROJECT_ROOT MODEL_PATH
export GEOMETRY_ENCODER_TYPE GEOMETRY_ENCODER_PATH GEO_INJECT_VERSION
export OUTPUT_DIR LOG_DIR TRAIN_LOG EXTRA_TRAIN_ARGS

exec bash "${SCRIPT_DIR}/train_vanilla_qwen25vl_variant.sh"
