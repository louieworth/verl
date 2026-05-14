#!/bin/bash
# =============================================================================
# Ablation: Forward KL FV on y_cor (stage2 rewrite) filtered to stage1 r=0
# =============================================================================
#
# Trains forward KL FV on stage2 rewrites generated only from stage1 prompts
# the student got WRONG (reward==0). Data already exists on disk via the
# script's natural FORWARD_STAGE2_MODE=reward0_only path:
#   gen_results/Qwen3-8B/epoch1/deepscaleR_stage2_reward0_y_cor_responses.parquet
# (12318 rows; the script will auto-backfill stage1 reward into a *_s1reward
# variant if LOG_DIFFICULTY_BUCKETS=true.)
# =============================================================================

set -e
set -o pipefail

export KL_TYPE="${KL_TYPE:-forward}"
export KL_METHOD="${KL_METHOD:-full_vocab}"
export TEMPERATURE="${TEMPERATURE:-1}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export Y_MODE="${Y_MODE:-y_cor}"
export FORWARD_STAGE2_MODE="${FORWARD_STAGE2_MODE:-reward0_only}"
export FORWARD_FILTER_STAGE2="${FORWARD_FILTER_STAGE2:-false}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"

export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20480}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"
export STAGE2_PROMPT_LENGTH="${STAGE2_PROMPT_LENGTH:-24576}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export GRAD_COSINE_INTERVAL="${GRAD_COSINE_INTERVAL:-50}"
export CORRECTION_TOKEN_PHRASES="${CORRECTION_TOKEN_PHRASES:-Wait,Actually,However,Alternatively,Oops,Wrong,Error,Incorrect,Correction,Sorry,Hmm,Oh,Hold,Pause,Uh,Um}"
export LOG_DIFFICULTY_BUCKETS="${LOG_DIFFICULTY_BUCKETS:-true}"
export SCORE_STAGE1="${SCORE_STAGE1:-true}"

# Custom save dir/name to keep tag distinct from rewrite_all variants.
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
TAG="kl_forward_full_vocab_y_cor_clip0_reward0only_${RUN_DATE}"
export MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-/data/data/jiangli/models/Qwen3-8B_${TAG}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TAG}}"

export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export PASS_K="${PASS_K:-16}"
export EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25,beyondaime,amobench}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "ABLATION: Forward KL FV on y_cor (stage1 r=0)"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  Y_MODE=$Y_MODE  STAGE2_MODE=$FORWARD_STAGE2_MODE  FILTER=$FORWARD_FILTER_STAGE2"
echo "  MODEL_SAVE_DIR=$MODEL_SAVE_DIR"
echo "  PASS_K=$PASS_K"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
