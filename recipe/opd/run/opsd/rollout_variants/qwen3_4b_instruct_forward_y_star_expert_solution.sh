#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 OPSD forward KL on raw DeepScaleR y*.

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

export TASK="math"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-4B-Instruct-2507}"

export TEACHER_MODEL_PATH=""
export TEACHER_MODEL=""
export Y_MODE="y_o"
export Y_O_ROLLOUT_MODE="expert"
export TRAJECTORY_MODEL_PATH=""
export TRAJECTORY_MODEL=""

RAW_DEEPSCALER_PATH="${RAW_DEEPSCALER_PATH:-/data2/data/jiangli/data/DeepScaleR-Preview-Dataset}"
EXPERT_COT_DATASET="${EXPERT_COT_DATASET:-$VERL_ROOT/data/train_dataset/deepscaler/opsd_expert_cot_only}"
EXPERT_TRAJECTORY_PARQUET="${EXPERT_TRAJECTORY_PARQUET:-$VERL_ROOT/data/train_dataset/deepscaler/train_opsd_y_star_solution_cot_only.parquet}"
if [ ! -s "$EXPERT_COT_DATASET/state.json" ]; then
    python3 "$SCRIPT_DIR/prepare_expert_cot_dataset.py" \
        --input "$RAW_DEEPSCALER_PATH" \
        --output "$EXPERT_COT_DATASET"
fi
if [ ! -s "$EXPERT_TRAJECTORY_PARQUET" ]; then
    python3 "$SCRIPT_DIR/prepare_expert_trajectory_parquet.py" \
        --input "$EXPERT_COT_DATASET" \
        --output "$EXPERT_TRAJECTORY_PARQUET"
fi
export TRAIN_DATA_PATH="$EXPERT_COT_DATASET"
export PRECOMPUTED_Y_O_TRAJECTORY_PATH="$EXPERT_TRAJECTORY_PARQUET"

exec bash "$SCRIPT_DIR/../forward_y_o.sh" "$@"
