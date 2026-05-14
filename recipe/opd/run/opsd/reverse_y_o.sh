#!/bin/bash
# =============================================================================
# Canonical recipe 4/4 — Reverse KL (no top-K), trained on y_o.
# =============================================================================
#
# Reverse KL on stage1 student rollouts, computed over the full vocabulary
# distribution. The baseline reverse-KL recipe — useful as a control against
# both the top-K variant (reverse_topk_y_o.sh) and the forward variants.
#
# Applicable knobs (override via env):
#   Y_MODE                : "y_o" (default) | "y_r"
#   MAX_PROMPT_LENGTH     : default depends on Y_MODE
#   MAX_RESPONSE_LENGTH   : default 16384
#   TEMPERATURE           : default 1.0
#   LEARNING_RATE         : default 2e-6
#   TOTAL_EPOCHS          : default 1
#
# Inert / not applicable to this recipe:
#   KL_TOKEN_CLIP         : forward/JSD only (pinned to 0 here).
#   TOP_K                 : pinned to 0 — use reverse_topk_y_o.sh for top-K.
# =============================================================================

set -e
set -o pipefail

export KL_TYPE="reverse"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export TOP_K=0
export KL_TOKEN_CLIP=0

export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export GRADIENT_ACCUMULATION_STEPS=16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
