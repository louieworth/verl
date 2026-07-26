#!/usr/bin/env bash
# Qwen3-8B OPSD forward KL on
# y_t ~ pi_T(. | x), with the fixed trajectory teacher pi_T = Qwen3-14B.
#
# The filename is retained for compatibility with existing launch commands;
# this experiment does not condition the trajectory teacher on y*.

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

# Keep OPSD's KL reference model unchanged; Qwen3-14B is trajectory-only.
export TEACHER_MODEL_PATH=""
export TEACHER_MODEL=""
export Y_MODE="y_o"
export Y_O_ROLLOUT_MODE="teacher"
export TEACHER_TRAINING_PROMPT="vanilla"
# Pin the rollout policy even if the caller's shell exports stale trajectory
# variables from another experiment.
export TRAJECTORY_MODEL_PATH="Qwen/Qwen3-14B"
export TRAJECTORY_MODEL="Qwen3-14B"
export TEACHER_TRAJECTORY_CONDITIONING="pi_T_x_only_v1"
export TEACHER_TRAJECTORY_PROMPT_PATH=""
export TEACHER_TRAJECTORY_CACHE_MODE="${TEACHER_TRAJECTORY_CACHE_MODE:-read_write}"
# Qwen3-14B sampling is stochastic. Keep this experiment's generated y_t
# separate from the 4B student experiment.
TEACHER_TRAJECTORY_CACHE_BASE_ROOT="${TEACHER_TRAJECTORY_CACHE_BASE_ROOT:-$VERL_ROOT/data/train_dataset/gen_y_t}"
export TEACHER_TRAJECTORY_CACHE_ROOT="$TEACHER_TRAJECTORY_CACHE_BASE_ROOT/qwen3_8b_forward_y_t_qwen3_14b_x_only"

Y_T_X_ONLY_PROMPTS="$VERL_ROOT/data/train_dataset/deepscaler/train_grpo.parquet"
if [ ! -s "$Y_T_X_ONLY_PROMPTS" ]; then
    echo "ERROR: missing repo-local x-only y_t prompt parquet: $Y_T_X_ONLY_PROMPTS" >&2
    exit 1
fi
export PRECOMPUTED_STAGE1_PROMPTS_PATH="$Y_T_X_ONLY_PROMPTS"
export PRECOMPUTED_Y_O_TRAJECTORY_PATH=""
export TRAIN_DATA_PATH="$Y_T_X_ONLY_PROMPTS"
export DATA_PATH=""

exec bash "$SCRIPT_DIR/../forward_y_o.sh" "$@"
