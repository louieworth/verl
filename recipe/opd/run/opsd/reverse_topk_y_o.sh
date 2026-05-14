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
#   MAX_PROMPT_LENGTH     : default depends on Y_MODE
#   MAX_RESPONSE_LENGTH   : default 16384
#   TEMPERATURE           : default 1.0
#   LEARNING_RATE         : default 2e-6
#   TOTAL_EPOCHS          : default 1
#   TOP_K                 : default 32 (paper) — reverse-only knob
#
# Inert / not applicable to this recipe:
#   KL_TOKEN_CLIP         : top-K renormalization already bounds per-position KL.
# =============================================================================

set -e
set -o pipefail

export KL_TYPE="reverse"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export TOP_K="${TOP_K:-32}"
export KL_TOKEN_CLIP=0

export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export GRADIENT_ACCUMULATION_STEPS=8   # effective batch 128 (paper Table A1)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
