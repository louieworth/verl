#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 OPSD forward KL on
# y_t ~ Qwen3-14B(. | x, y*), using an expert-solution rewrite prompt.

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

# Keep OPSD's KL reference model unchanged; Qwen3-14B is used only to generate
# trajectories from prompts containing x and the expert solution y*.
export TEACHER_MODEL_PATH=""
export TEACHER_MODEL=""
export Y_MODE="y_o"
export Y_O_ROLLOUT_MODE="teacher"
export TEACHER_TRAINING_PROMPT="vanilla"
# Pin the rollout policy even if the caller's shell exports stale trajectory
# variables from another experiment.
export TRAJECTORY_MODEL_PATH="Qwen/Qwen3-14B"
export TRAJECTORY_MODEL="Qwen3-14B"
export TEACHER_TRAJECTORY_CACHE_MODE="${TEACHER_TRAJECTORY_CACHE_MODE:-read_write}"
# Qwen3-14B sampling is stochastic. Keep this student's generated y_t parquet
# separate from the 8B experiment so concurrent runs never share or overwrite it.
TEACHER_TRAJECTORY_CACHE_BASE_ROOT="${TEACHER_TRAJECTORY_CACHE_BASE_ROOT:-$VERL_ROOT/gen_results/fixed_teacher_trajectory_cache}"
export TEACHER_TRAJECTORY_CACHE_ROOT="$TEACHER_TRAJECTORY_CACHE_BASE_ROOT/qwen3_4b_instruct_forward_y_t_qwen3_14b"

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
