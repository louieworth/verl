#!/bin/bash
# =============================================================================
# Canonical recipe 1/4 — Forward KL + per-token clip, trained on y_o.
# =============================================================================
#
# Forward KL (teacher-leading) on stage1 student rollouts. The per-token KL
# clip (OPSD-style) caps each token's KL contribution to prevent outlier
# tokens — where the teacher places mass on something the student has near-
# zero probability for — from dominating the gradient.
#
# Applicable knobs (override via env):
#   Y_MODE                : "y_o" (default) | "y_r" — data routing
#   MAX_PROMPT_LENGTH     : auto-derived teacher prompt budget
#   MAX_RESPONSE_LENGTH   : default 16384
#   TEMPERATURE           : default 1.0 — distillation softmax temperature
#   LEARNING_RATE         : default 5e-6
#   TOTAL_EPOCHS          : default 1
#   MULTI_STEP              : default 40 offline on-policy updates
#   KL_TOKEN_CLIP         : default 0.06 — per-token KL clamp (forward-only knob)
#
# Inert / not applicable to this recipe:
#   TOP_K                 : reverse-KL only (KL_TYPE=jsd rejects top_k anyway).
# =============================================================================

set -e
set -o pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"

export DISTILL_MODE="opsd"
export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.06}"
export TOP_K=0

export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-40}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
