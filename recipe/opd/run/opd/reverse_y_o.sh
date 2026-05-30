#!/bin/bash
# =============================================================================
# OPD recipe 4 — Reverse KL (no top-K) trained on y_o (teacher ≠ student).
# =============================================================================
#
# Baseline reverse-KL recipe in OPD mode. Use as a control against the top-K
# variant (reverse_topk_y_o.sh) and the forward variants.
#
# Applicable knobs (override via env):
#   MODEL_PATH              : student model (default Qwen/Qwen3-1.7B)
#   TEACHER_MODEL_PATH      : teacher model (default Qwen/Qwen3-8B, must differ from student)
#   Y_MODE                  : "y_o" (default) | "y_r"
#   MAX_PROMPT_LENGTH       : auto-derived teacher prompt budget
#   MAX_RESPONSE_LENGTH     : default 16384
#   TEMPERATURE             : default 1.0
#   TOTAL_EPOCHS            : default 1
#   MULTI_STEP              : default 40 offline on-policy updates
#   TEACHER_TRAINING_PROMPT : "vanilla" (auto for y_o) | "refine"
#
# Inert / not applicable:
#   KL_TOKEN_CLIP           : forward only (pinned to 0).
#   TOP_K                   : pinned to 0 — use reverse_topk_y_o.sh for top-K.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export DISTILL_MODE="opd"
export KL_TYPE="reverse"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export TOP_K=0
export KL_TOKEN_CLIP=0

export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-40}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
