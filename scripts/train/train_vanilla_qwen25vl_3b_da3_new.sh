#!/bin/bash
# DA3-new: uses blocks_to_take features (local+global, 3072-dim for vitg)
# instead of aux features (global-only, 1536-dim).
# Uses DA3-GIANT-1.1 backbone, same SGF interaction as da3_sgf_baseline.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
VARIANT_NAME=${VARIANT_NAME:-"vanilla_qwen25vl_3b_da3_new"}
MODEL_PATH=${MODEL_PATH:-"${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"}
GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE:-"da3_new"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/checkpoints/DA3-GIANT"}
GEO_INJECT_VERSION=${GEO_INJECT_VERSION:-"da3_new"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/${VARIANT_NAME}"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/${VARIANT_NAME}"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
BASE_EXTRA_ARGS="--geo_inject_version da3_new"

if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS} ${EXTRA_TRAIN_ARGS}"
else
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS}"
fi

export PROJECT_ROOT
export VARIANT_NAME
export MODEL_PATH
export GEOMETRY_ENCODER_TYPE
export GEOMETRY_ENCODER_PATH
export GEO_INJECT_VERSION
export OUTPUT_DIR
export LOG_DIR
export TRAIN_LOG
export EXTRA_TRAIN_ARGS

exec bash "${SCRIPT_DIR}/train_vanilla_qwen25vl_variant.sh"
