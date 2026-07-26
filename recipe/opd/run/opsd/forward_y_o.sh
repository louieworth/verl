#!/bin/bash
# =============================================================================
# OPSD recipe — Forward KL (no clip) on y_o.
# =============================================================================
#
# Forward KL on stage1 student rollouts, no per-token clip. Baseline for the
# forward-KL family — control against forward_clip_y_o.sh (with clip).
#
# Applicable knobs (override via env):
#   Y_MODE                  : "y_o" (default) | "y_r"
#   MAX_PROMPT_LENGTH       : auto-derived teacher prompt budget
#   MAX_RESPONSE_LENGTH     : default 16384
#   TEMPERATURE             : default 1.0
#   TOTAL_EPOCHS            : default 1
#   MULTI_STEP              : default 1 (generate the full trajectory set, then train)
#   TEACHER_TRAINING_PROMPT : "vanilla" (auto for y_o) | "refine"
#
# Inert / not applicable:
#   KL_TOKEN_CLIP           : pinned to 0 — use forward_clip_y_o.sh for clip.
#   TOP_K                   : reverse-KL only.
# =============================================================================

set -e
set -o pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"

export DISTILL_MODE="opsd"
export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export KL_TOKEN_CLIP=0
export TOP_K=0

export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
