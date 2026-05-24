#!/bin/bash
# =============================================================================
# Ablation: Forward KL FV on y_cor (stage2 rewrite) filtered to stage1 r=1
# =============================================================================
#
# Mirrors forward_y_corr_full_vocab.sh, but trains only on those stage2
# rewrites whose stage1 student rollout was correct (extra_info.reward==1).
# Data parquet was pre-generated at:
#   gen_results/Qwen3-8B/epoch1/deepscaleR_stage2_reward1_y_cor_responses_s1reward.parquet
# (already has stage1 reward backfilled into extra_info.reward).
#
# We pass it via DATA_PATH because run_kl_training.sh's FORWARD_STAGE2_MODE
# only natively supports rewrite_all and reward0_only for y_cor; reward1_only
# is unsupported. LOG_DIFFICULTY_BUCKETS=false avoids a redundant backfill
# (reward field is already populated).
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
export FORWARD_FILTER_STAGE2="${FORWARD_FILTER_STAGE2:-false}"

# Training Settings (full_vocab is memory-heavy; smaller micro-batch + more accum)
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"

# FSDP / Length (correction mode carries longer contexts)
export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20480}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"
export STAGE2_PROMPT_LENGTH="${STAGE2_PROMPT_LENGTH:-24576}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pre-generated, pre-backfilled parquet (s1 reward==1 only).
export DATA_PATH="${DATA_PATH:-/data/verl/gen_results/Qwen3-8B/epoch1/deepscaleR_stage2_reward1_y_cor_responses_s1reward.parquet}"

# T4 backfill is unnecessary (reward already in parquet) and would create a
# redundant ..._s1reward_s1reward.parquet next to it.
export LOG_DIFFICULTY_BUCKETS="${LOG_DIFFICULTY_BUCKETS:-false}"
export SCORE_STAGE1="${SCORE_STAGE1:-false}"
export GRAD_COSINE_INTERVAL="${GRAD_COSINE_INTERVAL:-50}"
export CORRECTION_TOKEN_PHRASES="${CORRECTION_TOKEN_PHRASES:-Wait,Actually,However,Alternatively,Oops,Wrong,Error,Incorrect,Correction,Sorry,Hmm,Oh,Hold,Pause,Uh,Um}"

# Custom MODEL_SAVE_DIR / WANDB_RUN_NAME so naming reflects the s1r1 split,
# since EXPERIMENT_TAG (auto) does not encode reward1_only for y_cor.
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
TAG="kl_forward_full_vocab_y_cor_clip0_reward1_${RUN_DATE}"
export MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-/data/data/jiangli/models/Qwen3-8B_${TAG}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TAG}}"

# Eval after training
export PASS_K="${PASS_K:-16}"
export EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25,beyondaime,amobench}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "ABLATION: Forward KL FV on y_cor (stage1 r=1)"
echo "=========================================="
echo "  DATA_PATH=$DATA_PATH"
echo "  MODEL_SAVE_DIR=$MODEL_SAVE_DIR"
echo "  WANDB_RUN_NAME=$WANDB_RUN_NAME"
echo "  PASS_K=$PASS_K"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
