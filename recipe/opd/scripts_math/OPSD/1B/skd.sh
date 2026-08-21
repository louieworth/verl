#!/usr/bin/env bash
set -euo pipefail

# Explicit math OPSD SKD: teacher-corrected vLLM rollouts plus forward full-vocab KL.
export OPD_TASK="math"
export OPD_FAMILY="opsd"
export OPD_VARIANT="skd"
export OPD_MODEL_SIZE="1B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$MODEL_PATH}"
export MULTI_STEP="${MULTI_STEP:-0}"

export DISTILL_MODE="opsd"
export KL_METHOD="full_vocab"
export BETA=0
export KL_TYPE="forward"
export Y_MODE="y_o"
export TEACHER_TRAINING_PROMPT="vanilla"
export Y_O_ROLLOUT_MODE="skd_vllm"
export TOP_K=0
export KL_TOKEN_CLIP=0
export SKD_GAMMA=5
export SKD_ACCEPT_TOP_K=25
export SKD_ACCEPT_TOP_P=1.0
export SKD_STUDENT_TEMPERATURE=1.0
export SKD_STUDENT_TOP_P=1.0
export SKD_TEACHER_TEMPERATURE=0.2
export SKD_TEACHER_TOP_P=1.0
export SKD_ROLLOUT_BATCH_SIZE=128
export SKD_PIPELINE_LANES=2
export SKD_PARALLEL_STUDENT_TEACHER=true

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
