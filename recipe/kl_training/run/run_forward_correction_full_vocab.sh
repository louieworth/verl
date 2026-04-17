#!/bin/bash
# =============================================================================
# Forward KL + Correction + Full Vocabulary (thin wrapper)
# =============================================================================
#
# Forward KL on stage2 correction data (teacher rewrites the student's failed
# attempt with expert guidance), computed over the full vocabulary distribution
# instead of monte_carlo samples.
#
# Memory note: full_vocab is much heavier than monte_carlo (loss head
# materializes |V|≈152K logits per kept token). Correction prompts are also
# longer (~24K). To stay under 80 GB/GPU we:
#   - keep SP=2 to halve attention activation
#   - halve MAX_TOKEN_LEN_PER_GPU vs the MC correction wrapper
#   - smaller micro-batch
#
# Usage:
#   bash run_forward_correction_full_vocab.sh
#   KL_TOKEN_CLIP=0.3 bash run_forward_correction_full_vocab.sh
# =============================================================================

set -e
set -o pipefail

# KL Settings
export KL_TYPE="${KL_TYPE:-forward}"
export KL_METHOD="${KL_METHOD:-full_vocab}"
export TEMPERATURE="${TEMPERATURE:-1}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.1}"
export USE_INITIAL_RESPONSE="${USE_INITIAL_RESPONSE:-true}"
export FORWARD_STAGE2_MODE="${FORWARD_STAGE2_MODE:-rewrite_all}"

# Training Settings (full_vocab is memory-heavy; smaller micro-batch + more accum)
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"

# FSDP / Length (correction mode carries longer contexts)
export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20480}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-40960}"
export STAGE2_PROMPT_LENGTH="${STAGE2_PROMPT_LENGTH:-24576}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Forward KL + Correction + Full Vocabulary"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  KL_METHOD=$KL_METHOD  CLIP=$KL_TOKEN_CLIP  TEMP=$TEMPERATURE"
echo "  USE_INITIAL_RESPONSE=$USE_INITIAL_RESPONSE  STAGE2=$FORWARD_STAGE2_MODE"
echo "  BATCH=$TRAIN_BATCH_SIZE  GRAD_ACCUM=$GRADIENT_ACCUMULATION_STEPS"
echo "  MAX_LEN=$MAX_LENGTH  MAX_TOKEN_LEN_PER_GPU=$MAX_TOKEN_LEN_PER_GPU  SP=$SP_SIZE"
echo "  STAGE2_PROMPT_LENGTH=$STAGE2_PROMPT_LENGTH"
echo "  DATA: stage2 correction responses (teacher rewrite of student failure)"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
