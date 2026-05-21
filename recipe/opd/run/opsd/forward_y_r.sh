#!/bin/bash
# =============================================================================
# Canonical recipe 2/4 — Forward KL without clip, trained on y_r.
# =============================================================================
#
# Forward KL on stage2 teacher rewrites (y_r). The teacher rewrites the
# student's failed attempt with expert guidance; the student learns to
# reproduce the teacher's corrected response directly. No per-token clip:
# y_r already biases the distribution toward solvable tokens, so clipping
# would suppress the signal we actually want.
#
# Applicable knobs (override via env):
#   Y_MODE                : "y_r" (default) | "y_o" — data routing
#   MAX_PROMPT_LENGTH     : auto-derived teacher prompt budget
#   MAX_RESPONSE_LENGTH   : default 16384
#   TEMPERATURE           : default 1.0
#   LEARNING_RATE         : default 2e-6
#   TOTAL_EPOCHS          : default 1
#   MULTI_STEP              : default 40 offline on-policy updates
#   FORWARD_STAGE2_MODE   : y_r only — "rewrite_all" (default) | "reward0_only"
#   FORWARD_FILTER_STAGE2 : y_r only — score y_1 after stage2 gen and keep reward>=threshold
#
# Inert / not applicable to this recipe:
#   KL_TOKEN_CLIP         : pinned to 0 (no clip in this recipe).
#   TOP_K                 : reverse-KL only.
# =============================================================================

set -e
set -o pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"

export DISTILL_MODE="opsd"
export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_r}"
export KL_TOKEN_CLIP=0
export TOP_K=0

export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-40}"

# OOM mitigation (must be set before python starts; y_r prompts are long).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
