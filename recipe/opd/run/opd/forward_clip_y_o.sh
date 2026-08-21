#!/bin/bash
# =============================================================================
# OPD recipe 1 — Forward KL + per-token clip, trained on y_o (teacher ≠ student).
# =============================================================================
#
# Applicable knobs (override via env):
#   MODEL_PATH              : student model (default Qwen/Qwen3-1.7B)
#   TEACHER_MODEL_PATH      : teacher model (default Qwen/Qwen3-14B, must differ from student)
#   Y_MODE                  : "y_o" (default) | "y_r"
#   MAX_PROMPT_LENGTH       : auto-derived teacher prompt budget
#   MAX_RESPONSE_LENGTH     : default 8192
#   TEMPERATURE             : default 1.0
#   TOTAL_EPOCHS            : default 1
#   MULTI_STEP              : default 40 offline on-policy updates
#   KL_TOKEN_CLIP           : default 0.06 (forward-only knob)
#   TEACHER_TRAINING_PROMPT : "vanilla" (auto for y_o) | "refine"
#
# Inert / not applicable:
#   TOP_K                   : reverse-KL only.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export DISTILL_MODE="opd"
export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.06}"
export TOP_K=0

export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-40}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
