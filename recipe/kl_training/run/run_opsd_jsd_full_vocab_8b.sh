#!/bin/bash
# =============================================================================
# OPSD JSD + Full Vocabulary — Qwen3-8B preset
# =============================================================================
#
# Same OPSD 8B preset as run_opsd_jsd_8b.sh, but pinned to KL_METHOD=full_vocab
# (matching what OPSD's reference implementation actually uses — full logits,
# not monte_carlo samples). All other OPSD hyperparameters are kept identical.
#
# Memory note: full_vocab on 8B with |V|≈152K needs more headroom than the MC
# default. We keep SP=1 and a conservative token budget; bump if your GPU
# memory observation shows headroom.
#
# Usage:
#   bash run_opsd_jsd_full_vocab_8b.sh                    # β=0 (forward KL endpoint)
#   BETA=0.5 bash run_opsd_jsd_full_vocab_8b.sh           # JSD mixture
#   BETA=1   bash run_opsd_jsd_full_vocab_8b.sh           # reverse KL endpoint
# =============================================================================

set -e
set -o pipefail

# --- OPSD 8B pinned hyperparameters (full_vocab variant) ---
export KL_TYPE="${KL_TYPE:-jsd}"
export KL_METHOD="${KL_METHOD:-full_vocab}"        # OPSD reference uses full logits
export BETA="${BETA:-0}"                           # OPSD 8B: --beta 0 (forward KL endpoint)
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.06}"      # OPSD 8B: --jsd_token_clip 0.06
export TEMPERATURE="${TEMPERATURE:-1.1}"           # OPSD 8B: --temperature 1.1

export LEARNING_RATE="${LEARNING_RATE:-5e-6}"      # OPSD 8B: --learning_rate 5e-6
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"   # OPSD 8B: --per_device_train_batch_size 2
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"  # OPSD 8B: --gradient_accumulation_steps 2
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

export LORA_RANK="${LORA_RANK:-64}"                # OPSD 8B: --lora_r 64
export LORA_ALPHA="${LORA_ALPHA:-128}"             # OPSD 8B: --lora_alpha 128
export USE_LORA="${USE_LORA:-true}"

# max_length: OPSD uses 20000, we match the MC variant
export MAX_LENGTH="${MAX_LENGTH:-18432}"

# Data source: stage1 (student rollout). USE_INITIAL_RESPONSE only tags the
# run name and does not affect which data we load (stage1 is always stage1).
export USE_INITIAL_RESPONSE="${USE_INITIAL_RESPONSE:-false}"

# FSDP / memory — full_vocab on 8B is heavier than MC; keep token budget tight.
export SP_SIZE="${SP_SIZE:-1}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"

# Epochs: OPSD's 30 is on-policy; offline SFT diverges from OPSD here.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "OPSD JSD 8B preset — Full Vocabulary"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  BETA=$BETA  KL_METHOD=$KL_METHOD"
echo "  CLIP=$KL_TOKEN_CLIP  TEMP=$TEMPERATURE  LR=$LEARNING_RATE"
echo "  BATCH=$TRAIN_BATCH_SIZE  GRAD_ACCUM=$GRADIENT_ACCUMULATION_STEPS"
echo "  LORA=$LORA_RANK/$LORA_ALPHA  MAX_LEN=$MAX_LENGTH"
echo "  MAX_TOKEN_LEN_PER_GPU=$MAX_TOKEN_LEN_PER_GPU  SP=$SP_SIZE"
echo "  DATA: stage1 responses (student rollout)"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
