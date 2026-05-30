#!/bin/bash
# =============================================================================
# Canonical recipe 3/4 — Reverse KL with teacher top-K, trained on y_o.
# =============================================================================
#
# Reverse KL (student-leading) with the truncated objective on the teacher's
# top-K local support (Fu et al. 2026, arXiv:2603.25562, §3.1 Eq. 8):
# at each prefix, support S = TopK_q(c_t); both teacher and student are
# renormalized over S; reverse KL is computed inside S. The renormalization
# naturally bounds per-position KL, so no additional KL_TOKEN_CLIP is needed.
#
# Applicable knobs (override via env):
#   Y_MODE                : "y_o" (default) | "y_r"
#   MAX_PROMPT_LENGTH     : auto-derived teacher prompt budget
#   MAX_RESPONSE_LENGTH   : default 16384
#   TEMPERATURE           : default 1.0
#   TOTAL_EPOCHS          : default 1
#   MULTI_STEP              : default 40 offline on-policy updates
#   TOP_K                 : default 32 (paper) — reverse-only knob
#
# Inert / not applicable to this recipe:
#   KL_TOKEN_CLIP         : top-K renormalization already bounds per-position KL.
# =============================================================================

set -e
set -o pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"

export DISTILL_MODE="opsd"
export KL_TYPE="reverse"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export TOP_K="${TOP_K:-32}"
export KL_TOKEN_CLIP=0

export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-40}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
