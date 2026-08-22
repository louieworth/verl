#!/usr/bin/env bash
set -euo pipefail

# Explicit code OPD top-k: reverse full-vocab KL on teacher support K=32.
export OPD_TASK="code"
export OPD_FAMILY="opd"
export OPD_VARIANT="top_k"
export OPD_MODEL_SIZE="8B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B-Base}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}"
export TEACHER_ENABLE_THINKING="${TEACHER_ENABLE_THINKING:-false}"
export MULTI_STEP="${MULTI_STEP:-0}"

export DISTILL_MODE="opd"
export KL_METHOD="full_vocab"
export BETA=0
export KL_TYPE="reverse"
export Y_MODE="y_o"
export TEACHER_TRAINING_PROMPT="vanilla"
export Y_O_ROLLOUT_MODE="student"
export TOP_K=32
export KL_TOKEN_CLIP=0

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
