#!/usr/bin/env bash
# Qwen3-8B OPSD forward KL on
# y_t ~ pi_8B(. | x, y*), using an expert-solution rewrite prompt.

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

# OPSD's KL teacher and the trajectory policy are both the corresponding
# student's fixed base model.  The trajectory is generated before each OPSD
# update from a prompt containing x and the expert solution y*.
export TEACHER_MODEL_PATH=""
export TEACHER_MODEL=""
export Y_MODE="y_o"
export Y_O_ROLLOUT_MODE="teacher"
export TEACHER_TRAINING_PROMPT="vanilla"
# Pin the rollout policy to this experiment's own model, even if the caller's
# shell still exports trajectory variables from an older teacher experiment.
export TRAJECTORY_MODEL_PATH="$MODEL_PATH"
export TRAJECTORY_MODEL="$MODEL_NAME"
export TEACHER_TRAJECTORY_CACHE_MODE="${TEACHER_TRAJECTORY_CACHE_MODE:-read_write}"

CLEANED_DEEPSCALER_PATH="${CLEANED_DEEPSCALER_PATH:-/data2/data/jiangli/data/DeepScaleR-Cleaned}"
EXPERT_TRAJECTORY_PARQUET="${EXPERT_TRAJECTORY_PARQUET:-$VERL_ROOT/data/train_dataset/deepscaler/train_opsd_y_star_solution_or_answer.parquet}"
export TEACHER_TRAJECTORY_PROMPT_PATH="${TEACHER_TRAJECTORY_PROMPT_PATH:-$VERL_ROOT/data/train_dataset/deepscaler/train_opsd_y_t_expert_rewrite_prompts.parquet}"

if [ ! -s "$CLEANED_DEEPSCALER_PATH/state.json" ]; then
    echo "ERROR: missing DeepScaleR-Cleaned dataset: $CLEANED_DEEPSCALER_PATH" >&2
    exit 1
fi
if [ ! -s "$EXPERT_TRAJECTORY_PARQUET" ]; then
    python3 "$SCRIPT_DIR/prepare_expert_trajectory_parquet.py" \
        --input "$CLEANED_DEEPSCALER_PATH" \
        --output "$EXPERT_TRAJECTORY_PARQUET"
fi
if [ ! -s "$TEACHER_TRAJECTORY_PROMPT_PATH" ]; then
    python3 -m recipe.opd.generation.teacher_y_t_prepare \
        --input "$EXPERT_TRAJECTORY_PARQUET" \
        --output "$TEACHER_TRAJECTORY_PROMPT_PATH" \
        --task math
fi
export TRAIN_DATA_PATH="$CLEANED_DEEPSCALER_PATH"

exec bash "$SCRIPT_DIR/../forward_y_o.sh" "$@"
