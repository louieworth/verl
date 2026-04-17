#!/bin/bash
# =============================================================================
# Reverse KL + Rewrite + Full Vocabulary (thin wrapper)
# =============================================================================
#
# Reverse KL on stage1 (student rollout) data, computed over the full
# vocabulary distribution instead of monte_carlo samples.
#
# Memory note: full_vocab forces the loss to materialize a |V|-sized softmax
# over every kept token (Qwen3 |V| ≈ 152K). Compared to monte_carlo, this
# multiplies activation memory for the loss head by O(|V|), so we halve the
# per-GPU token budget and drop micro-batch size relative to the MC defaults.
#
# Usage:
#   bash run_reverse_rewrite_full_vocab.sh
#   KL_TOKEN_CLIP=0.1 bash run_reverse_rewrite_full_vocab.sh
# =============================================================================

set -e
set -o pipefail

# KL Settings
export KL_TYPE="${KL_TYPE:-reverse}"
export KL_METHOD="${KL_METHOD:-full_vocab}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.06}"
export USE_INITIAL_RESPONSE="${USE_INITIAL_RESPONSE:-false}"

# Training Settings (full_vocab is memory-heavy; smaller micro-batch + more accum)
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

# FSDP / Length
export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-18432}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Reverse KL + Rewrite + Full Vocabulary"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  KL_METHOD=$KL_METHOD  CLIP=$KL_TOKEN_CLIP  TEMP=$TEMPERATURE"
echo "  BATCH=$TRAIN_BATCH_SIZE  GRAD_ACCUM=$GRADIENT_ACCUMULATION_STEPS"
echo "  MAX_LEN=$MAX_LENGTH  MAX_TOKEN_LEN_PER_GPU=$MAX_TOKEN_LEN_PER_GPU  SP=$SP_SIZE"
echo "  DATA: stage1 responses (student rollout)"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
