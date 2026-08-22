#!/usr/bin/env bash
set -euo pipefail

# Explicit math OPD clip: forward full-vocab KL clipped per token at 0.05.
export OPD_TASK="math"
export OPD_FAMILY="opd"
export OPD_VARIANT="clip"
export OPD_MODEL_SIZE="1B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}"
export TEACHER_ENABLE_THINKING="${TEACHER_ENABLE_THINKING:-false}"
export MULTI_STEP="${MULTI_STEP:-0}"

export DISTILL_MODE="opd"
export KL_METHOD="full_vocab"
export BETA=0
export KL_TYPE="forward"
export Y_MODE="y_o"
export TEACHER_TRAINING_PROMPT="vanilla"
export Y_O_ROLLOUT_MODE="student"
export TOP_K=0
export KL_TOKEN_CLIP=0.05

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
