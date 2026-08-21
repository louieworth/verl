#!/usr/bin/env bash
set -euo pipefail

# Explicit code OPD TRD: refined y_r targets plus forward full-vocab KL.
export OPD_TASK="code"
export OPD_FAMILY="opd"
export OPD_VARIANT="trd"
export OPD_MODEL_SIZE="1B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}"
export MULTI_STEP="${MULTI_STEP:-0}"

export DISTILL_MODE="opd"
export KL_METHOD="full_vocab"
export BETA=0
export KL_TYPE="forward"
export Y_MODE="y_r"
export TEACHER_TRAINING_PROMPT="refine"
export Y_O_ROLLOUT_MODE="student"
export TOP_K=0
export KL_TOKEN_CLIP=0

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
