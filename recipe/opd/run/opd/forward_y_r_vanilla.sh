#!/bin/bash
# =============================================================================
# OPD recipe 2b — Forward KL on y_r, vanilla training prompt (original OPD spec).
# =============================================================================
#
# Generation:  y_r ~ π_T(·|x, y_o)            (teacher uses refine prompt)
# Training:    teacher sees only x            → KL(π_S(·|x), π_T(·|x)) over y_r
#
# Same as forward_y_r.sh except TEACHER_TRAINING_PROMPT=vanilla — teacher's
# training-time conditioning drops y_o, matching the original OPD spec where
# both student and teacher condition only on x and y_r is just the rollout
# sequence the KL is summed over.
#
# Applicable knobs (override via env): same as forward_y_r.sh except
# TEACHER_TRAINING_PROMPT defaults to "vanilla" here.
#   MODEL_PATH              : student model (default Qwen/Qwen3-1.7B)
#   TEACHER_MODEL_PATH      : teacher model (default Qwen/Qwen3-14B)
#   MULTI_STEP              : default 40 offline on-policy updates
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export DISTILL_MODE="opd"
export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_r}"
# OPD y_r launches a separate teacher vLLM after stage1; keep resident y_o off
# by default because a sleeping Ray actor still holds GPU scheduler resources.
export RESIDENT_STUDENT_ROLLOUT="${RESIDENT_STUDENT_ROLLOUT:-false}"
export KL_TOKEN_CLIP=0
export TOP_K=0

# Vanilla training prompt: teacher conditions only on x, no y_o (original OPD spec).
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"

export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-40}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
