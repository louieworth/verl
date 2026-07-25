#!/usr/bin/env bash
# Qwen3-8B OPSD forward KL on y_t ~ Qwen3-14B(.|x).

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

# Same trajectory model/cache as the 4B run; training outputs remain separate.
export TEACHER_MODEL_PATH=""
export TEACHER_MODEL=""
export Y_MODE="y_o"
export Y_O_ROLLOUT_MODE="teacher"
export TRAJECTORY_MODEL_PATH="${TRAJECTORY_MODEL_PATH:-Qwen/Qwen3-14B}"
export TRAJECTORY_MODEL="${TRAJECTORY_MODEL:-}"
export TEACHER_TRAJECTORY_CACHE_MODE="${TEACHER_TRAJECTORY_CACHE_MODE:-read_write}"

export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data2/data/jiangli/data/DeepScaleR-Cleaned}"

exec bash "$SCRIPT_DIR/../forward_y_o.sh" "$@"
