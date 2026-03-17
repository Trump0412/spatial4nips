#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"/data3/yeyuanhao/sp_re_cbp/GeoThinker"}
GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH:-"/data3/yeyuanhao/checkpoints/DA3-GIANT"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/vanilla_qwen25vl_3b_da3_mmr"}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/vanilla_qwen25vl_3b_da3_mmr"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_DIR}/train.log"}
BASE_EXTRA_ARGS="--geo_inject_version da3_mmr --geo_importance_gate False --mmr_use_region_memory False --mmr_frame_topk_max 3 --mmr_region_topk_max 8 --mmr_write_start 8 --mmr_write_end 15 --mmr_read_start 16 --mmr_read_end 20 --mmr_use_view_continuity True --mmr_use_temporal_continuity True --mmr_region_atoms_per_frame 8 --mmr_query_use_text True --mmr_query_use_visual_summary True"

if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS} ${EXTRA_TRAIN_ARGS}"
else
    EXTRA_TRAIN_ARGS="${BASE_EXTRA_ARGS}"
fi

exec bash "${SCRIPT_DIR}/train_vanilla_qwen25vl_3b_da3.sh"
