#!/usr/bin/env bash
set -euo pipefail

# Math OPD vanilla with Qwen3-14B teacher thinking enabled by default.
export OPD_TASK="math"
export OPD_FAMILY="opd"
export OPD_VARIANT="vanilla"
export OPD_MODEL_SIZE="1B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}"
export TEACHER_ENABLE_THINKING="${TEACHER_ENABLE_THINKING:-true}"
export MULTI_STEP="${MULTI_STEP:-0}"

export DISTILL_MODE="opd"
export KL_METHOD="full_vocab"
export BETA=0
export KL_TYPE="reverse"
export Y_MODE="y_o"
export TEACHER_TRAINING_PROMPT="vanilla"
export Y_O_ROLLOUT_MODE="student"
export TOP_K=0
export KL_TOKEN_CLIP=0

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
