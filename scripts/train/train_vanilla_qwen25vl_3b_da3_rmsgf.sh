#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
VARIANT_NAME=${VARIANT_NAME:-"vanilla_qwen25vl_3b_da3_rmsgf"}
MODEL_PATH=${MODEL_PATH:-"${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"}
GEOMETRY_ENCODER_TYPE=${GEOMETRY_ENCODER_TYPE:-"da3"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/checkpoints/DA3-GIANT"}
GEO_INJECT_VERSION=${GEO_INJECT_VERSION:-"da3_rmsgf"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/${VARIANT_NAME}"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/${VARIANT_NAME}"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
BASE_EXTRA_ARGS="--rmsgf_topr 32 --rmsgf_atom_topk_max 8 --rmsgf_refine_gate True --rmsgf_refine_residual True"

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
