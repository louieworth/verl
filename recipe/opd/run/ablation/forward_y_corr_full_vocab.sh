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
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export Y_MODE="${Y_MODE:-y_cor}"
export FORWARD_STAGE2_MODE="${FORWARD_STAGE2_MODE:-rewrite_all}"
# Do NOT reward-filter the stage2 output (keep full corrected set).
export FORWARD_FILTER_STAGE2="${FORWARD_FILTER_STAGE2:-false}"

# Training Settings (full_vocab is memory-heavy; smaller micro-batch + more accum)
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"

# FSDP / Length (correction mode carries longer contexts)
export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20480}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"
export STAGE2_PROMPT_LENGTH="${STAGE2_PROMPT_LENGTH:-24576}"

# OOM mitigation (must be set before python starts)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Diagnostic metrics (T1–T4 from the y vs y' study) ---
# This script trains on y' (stage2 teacher rewrite). Enable the same metrics
# as run_opsd_jsd_full_vocab_8b.sh so the two runs are directly comparable
# on wandb.
export GRAD_COSINE_INTERVAL="${GRAD_COSINE_INTERVAL:-50}"
export CORRECTION_TOKEN_PHRASES="${CORRECTION_TOKEN_PHRASES:-Wait,Actually,However,Alternatively,Oops,Wrong,Error,Incorrect,Correction,Sorry,Hmm,Oh,Hold,Pause,Uh,Um}"
export LOG_DIFFICULTY_BUCKETS="${LOG_DIFFICULTY_BUCKETS:-true}"
# T4 on stage2 needs stage1 reward backfilled into the stage2 parquet
# (run_kl_training.sh handles this automatically when forward + T4).
export SCORE_STAGE1="${SCORE_STAGE1:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Forward KL + Correction + Full Vocabulary"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  KL_METHOD=$KL_METHOD  CLIP=$KL_TOKEN_CLIP  TEMP=$TEMPERATURE"
echo "  Y_MODE=$Y_MODE  STAGE2=$FORWARD_STAGE2_MODE  STAGE2_FILTER=$FORWARD_FILTER_STAGE2"
echo "  BATCH=$TRAIN_BATCH_SIZE  GRAD_ACCUM=$GRADIENT_ACCUMULATION_STEPS"
echo "  MAX_LEN=$MAX_LENGTH  MAX_TOKEN_LEN_PER_GPU=$MAX_TOKEN_LEN_PER_GPU  SP=$SP_SIZE"
echo "  STAGE2_PROMPT_LENGTH=$STAGE2_PROMPT_LENGTH"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  DATA: stage2 correction responses (teacher rewrite of student failure)"
echo "  Metrics: T1(auto) T2(every $GRAD_COSINE_INTERVAL) T3 T4=$LOG_DIFFICULTY_BUCKETS (auto-backfill stage1 reward into stage2)"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
