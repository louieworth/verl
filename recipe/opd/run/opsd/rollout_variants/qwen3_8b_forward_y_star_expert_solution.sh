#!/usr/bin/env bash
# Qwen3-8B OPSD forward KL on raw DeepScaleR y*.

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

export TASK="math"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-8B}"
export MULTI_STEP=1

export TEACHER_MODEL_PATH=""
export TEACHER_MODEL=""
export Y_MODE="y_o"
export Y_O_ROLLOUT_MODE="expert"
export TRAJECTORY_MODEL_PATH=""
export TRAJECTORY_MODEL=""
export PRECOMPUTED_STAGE1_PROMPTS_PATH=""
export PRECOMPUTED_Y_O_TRAJECTORY_PATH=""

OPSD_Y_STAR_DATA="${OPSD_Y_STAR_DATA:-$VERL_ROOT/data/train_dataset/deepscaler/train_opsd_y_star_solution_cot_only.parquet}"
if [ ! -s "$OPSD_Y_STAR_DATA" ]; then
    echo "ERROR: missing repo-local CoT-only OPSD y* parquet: $OPSD_Y_STAR_DATA" >&2
    exit 1
fi
export TRAIN_DATA_PATH="$OPSD_Y_STAR_DATA"
export DATA_PATH="$OPSD_Y_STAR_DATA"

exec bash "$SCRIPT_DIR/../forward_y_o.sh" "$@"
