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
export KL_METHOD="${KL_METHOD:-monte_carlo}"        # OPSD reference uses full logits
export BETA="${BETA:-0}"                           # OPSD 8B: --beta 0 (forward KL endpoint)
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"      # OPSD 8B: --jsd_token_clip 0.06
export TEMPERATURE="${TEMPERATURE:-1}"           # OPSD 8B: --temperature 1.1

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"   # OPSD 8B: --per_device_train_batch_size 2
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"  # OPSD 8B: --gradient_accumulation_steps 2
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

export LORA_RANK="${LORA_RANK:-64}"                # OPSD 8B: --lora_r 64
export LORA_ALPHA="${LORA_ALPHA:-128}"             # OPSD 8B: --lora_alpha 128
export USE_LORA="${USE_LORA:-true}"

# max_length: OPSD uses 20000, we match the MC variant
export MAX_LENGTH="${MAX_LENGTH:-18432}"

# y_raw: train on stage1 student rollouts; teacher prompt = π(·|x, y*) (no initial response).
export Y_MODE="${Y_MODE:-y_raw}"

# FSDP / memory — full_vocab on 8B is heavier than MC; keep token budget tight.
export SP_SIZE="${SP_SIZE:-1}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"

# Epochs: OPSD's 30 is on-policy; offline SFT diverges from OPSD here.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND:-1}"

# --- Diagnostic metrics (T1–T4 from the y vs y' study) ---
# T1 (per-token KL) is unconditional — already logged as train/kl_loss and
# train/kl_loss_per_traj. T2/T3/T4 are opt-in but enabled by default in this
# preset because the study explicitly compares y (stage1) vs y' (stage2)
# training dynamics.
export GRAD_COSINE_INTERVAL="${GRAD_COSINE_INTERVAL:-50}"          # T2: cos(g_t, g_{t-1}) every 50 steps
# T3: hesitation/correction phrases. Trainer takes the *first* token of each
# phrase (bare + leading-space variant). Each entry below was verified to
# tokenize as a self-contained word in the Qwen3 tokenizer — phrases that
# collapse to a generic stem (e.g. Reconsider→"Re", Mistake→"M") are excluded
# because the stem fires on unrelated text and inflates the metric.
#
# Three tiers from the candidate sweep (run grep "candidates" in this repo
# to reproduce). Default = strong + filler hesitation, ~16 phrases / ~32 ids.
#   Strong correction:  Wait Actually However Alternatively Oops
#                       Wrong Error Incorrect Correction Sorry
#   Hesitation/halt:    Hmm Oh Hold Pause Uh Um
#   (Optional, noisier in math text — opt in via override:
#    But Yet Still Though Maybe Perhaps Probably Correct Thinking Wow)
export CORRECTION_TOKEN_PHRASES="${CORRECTION_TOKEN_PHRASES:-Wait,Actually,However,Alternatively,Oops,Wrong,Error,Incorrect,Correction,Sorry,Hmm,Oh,Hold,Pause,Uh,Um}"
export LOG_DIFFICULTY_BUCKETS="${LOG_DIFFICULTY_BUCKETS:-true}"     # T4: split kl_loss by stage1 reward bucket
export SCORE_STAGE1="${SCORE_STAGE1:-true}"                         # required for T4 to bucket data

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
echo "  Metrics: T1(auto) T2(every $GRAD_COSINE_INTERVAL) T3(\"$CORRECTION_TOKEN_PHRASES\") T4=$LOG_DIFFICULTY_BUCKETS"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
