#!/bin/bash
# =============================================================================
# OPD recipe 2 — Forward KL on y_r (teacher ≠ student, refine training prompt).
# =============================================================================
#
# Generation:  y_r ~ π_T(·|x, y_o)   (teacher uses refine prompt, no expert)
# Training:    teacher sees (x, y_o)  → KL(π_S(·|x), π_T(·|x, y_o)) over y_r
#
# For the "vanilla" training-prompt variant (teacher sees only x at training,
# matching the original OPD spec), use forward_y_r_vanilla.sh.
#
# Applicable knobs (override via env):
#   MODEL_PATH              : student model (default Qwen/Qwen3-1.7B)
#   TEACHER_MODEL_PATH      : teacher model (default Qwen/Qwen3-8B, must differ from student)
#   Y_MODE                  : "y_r" (default) | "y_o"
#   MAX_PROMPT_LENGTH       : auto-derived teacher prompt budget
#   MAX_RESPONSE_LENGTH     : default 16384
#   TEMPERATURE             : default 1.0
#   LEARNING_RATE           : default 2e-6
#   TOTAL_EPOCHS            : default 1
#   MULTI_STEP              : default 40 offline on-policy updates
#   TEACHER_TRAINING_PROMPT : "refine" (default for y_r) | "vanilla"
#   FORWARD_FILTER_STAGE2   : score y_r after gen and keep reward>=threshold
#
# Inert / not applicable:
#   KL_TOKEN_CLIP           : pinned to 0.
#   TOP_K                   : reverse-KL only.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"
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

export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-40}"
# In multi-step mode, run_kl_training.sh auto-computes accumulation so each
# chunk performs one optimizer update by default.
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-refine}"

# OOM mitigation (y_r refine prompts are long).
# y_r refine default MAX_LENGTH = (1024 + 16384) + 16384 = 33792; single-sample
# logits are still large plus fp32 softmax intermediates. With Qwen3-8B teacher
# resident alongside Qwen3-1.7B student on A100-40G, this OOMs unless we
# either split the sequence (SP) or offload teacher params.
#
# NOTE: do NOT export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True at top
# level — it propagates into the stage1/stage2 vLLM subprocesses and crashes
# them with "Expandable segments are not compatible with memory pool".
# run_kl_training.sh already enables it inside the inner torchrun subshell.
export SP_SIZE="${SP_SIZE:-1}"                  # halve per-GPU token count → logits peak ~6 GiB
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-true}"   # 8B teacher params → CPU when idle, ~2 GiB/GPU back

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
