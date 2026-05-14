#!/bin/bash
# =============================================================================
# OPSD JSD Training — Qwen3-8B preset
# =============================================================================
#
# Thin wrapper around run_kl_training.sh that pins every portable
# hyperparameter to OPSD's scripts/run_opsd_8b.sh
# (https://github.com/siyan-zhao/OPSD/blob/main/scripts/run_opsd_8b.sh).
#
# Data source: stage1 responses only (student's original rollout on the math
# problem). This is the closest offline analogue to OPSD's on-policy student
# samples. JSD has nothing to do with stage2 rewrites.
#
# On-policy-only params from OPSD that do NOT port to this offline SFT
# pipeline (documented, not set):
#   - lmbda=1                 (student-sample ratio; no student rollout here)
#   - max_completion_length=1024  (our stage1 already generated 16384 tokens)
#   - num_train_epochs=30    (on-policy resamples every epoch; offline SFT
#                              starts overfitting quickly — we default to 1
#                              and recommend 1–3)
#
# Usage:
#   bash run_opsd_jsd_8b.sh                           # full defaults (β=0)
#   BETA=0.5 bash run_opsd_jsd_8b.sh                  # JSD mixture
#   KL_METHOD=monte_carlo BETA=0.5 bash run_opsd_jsd_8b.sh  # MC JSD
# =============================================================================

set -e
set -o pipefail

# --- OPSD 8B pinned hyperparameters ---
export KL_TYPE="${KL_TYPE:-jsd}"
export KL_METHOD="${KL_METHOD:-monte_carlo}"       # OPSD uses full logits
export BETA="${BETA:-0}"                          # OPSD 8B: --beta 0 (forward KL endpoint)
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.06}"     # OPSD 8B: --jsd_token_clip 0.06
export TEMPERATURE="${TEMPERATURE:-1.1}"          # OPSD 8B: --temperature 1.1

export LEARNING_RATE="${LEARNING_RATE:-5e-6}"     # OPSD 8B: --learning_rate 5e-6
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"  # OPSD 8B: --per_device_train_batch_size 2
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"  # OPSD 8B: --gradient_accumulation_steps 2
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

export LORA_RANK="${LORA_RANK:-64}"               # OPSD 8B: --lora_r 64
export LORA_ALPHA="${LORA_ALPHA:-128}"            # OPSD 8B: --lora_alpha 128
export USE_LORA="${USE_LORA:-true}"

# max_length: OPSD uses 20000, we match
export MAX_LENGTH="${MAX_LENGTH:-18432}"

# y_raw: train on stage1 student rollouts; teacher prompt = π(·|x, y*) (no initial response).
export Y_MODE="${Y_MODE:-y_raw}"

# FSDP / memory — full_vocab on 8B needs SP and enough token budget per GPU
export SP_SIZE="${SP_SIZE:-1}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-40960}"

# Epochs: OPSD's 30 is on-policy (every epoch rolls out fresh trajectories);
# offline SFT diverges from OPSD here — keep small.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "OPSD JSD 8B preset"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  BETA=$BETA  KL_METHOD=$KL_METHOD"
echo "  CLIP=$KL_TOKEN_CLIP  TEMP=$TEMPERATURE  LR=$LEARNING_RATE"
echo "  BATCH=$TRAIN_BATCH_SIZE  GRAD_ACCUM=$GRADIENT_ACCUMULATION_STEPS"
echo "  LORA=$LORA_RANK/$LORA_ALPHA  MAX_LEN=$MAX_LENGTH  SP=$SP_SIZE"
echo "  DATA: stage1 responses (student rollout)"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
