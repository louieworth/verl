#!/bin/bash
# =============================================================================
# Ablation: Forward KL FV on y_cor (stage1 r=0 AND stage2/y_cor reward==1)
# =============================================================================
#
# Trains forward KL FV on stage2 rewrites that satisfy BOTH:
#   stage1 student got it WRONG (reward==0) AND teacher's rewrite is CORRECT
#   (stage2_reward >= 1.0).
# After the y_r_prepare.py generation-mode simplification (always generates
# for every stage1 row), both constraints are enforced post-generation:
#   FORWARD_FILTER_STAGE2=true + FORWARD_FILTER_THRESHOLD=1.0
#   + FORWARD_FILTER_REQUIRE_STAGE1_FAILED=true
# Resulting parquet:
#   gen_results/Qwen3-8B/epoch1/deepscaleR_stage2_y_cor_responses_filtered_s1fail.parquet
# Cost: extra GPU time generating rewrites for reward==1 rows that are then dropped.
# =============================================================================

set -e
set -o pipefail

export KL_TYPE="${KL_TYPE:-forward}"
export KL_METHOD="${KL_METHOD:-full_vocab}"
export TEMPERATURE="${TEMPERATURE:-1}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export Y_MODE="${Y_MODE:-y_cor}"
export FORWARD_FILTER_STAGE2="${FORWARD_FILTER_STAGE2:-true}"
export FORWARD_FILTER_THRESHOLD="${FORWARD_FILTER_THRESHOLD:-1.0}"
export FORWARD_FILTER_REQUIRE_STAGE1_FAILED="${FORWARD_FILTER_REQUIRE_STAGE1_FAILED:-true}"

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

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
TAG="kl_forward_full_vocab_y_cor_clip0_reward0only_filtered_${RUN_DATE}"
export MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-/data/data/jiangli/models/Qwen3-8B_${TAG}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TAG}}"

export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export PASS_K="${PASS_K:-16}"
export EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25,beyondaime,amobench}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "ABLATION: Forward KL FV on y_cor (stage1 r=0 AND stage2 r=1)"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  Y_MODE=$Y_MODE  POST_FILTER=$FORWARD_FILTER_STAGE2 (th=$FORWARD_FILTER_THRESHOLD, s1_failed=$FORWARD_FILTER_REQUIRE_STAGE1_FAILED)"
echo "  MODEL_SAVE_DIR=$MODEL_SAVE_DIR"
echo "  PASS_K=$PASS_K"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
