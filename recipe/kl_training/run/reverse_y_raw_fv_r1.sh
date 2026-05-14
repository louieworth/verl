#!/bin/bash
# =============================================================================
# Ablation: Reverse KL FV on y_raw, filtered to stage1 reward==1
# =============================================================================
set -e
set -o pipefail

export KL_TYPE="${KL_TYPE:-reverse}"
export KL_METHOD="${KL_METHOD:-full_vocab}"
export TEMPERATURE="${TEMPERATURE:-1}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export Y_MODE="${Y_MODE:-y_raw}"
export FORWARD_STAGE2_MODE="${FORWARD_STAGE2_MODE:-stage1_reward_1_only}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-18432}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"

export LOG_DIFFICULTY_BUCKETS="${LOG_DIFFICULTY_BUCKETS:-true}"
export SCORE_STAGE1="${SCORE_STAGE1:-true}"

# Eval after training
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export PASS_K="${PASS_K:-16}"
export EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25,beyondaime,amobench}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "ABLATION: Reverse KL FV on y_raw (s1 r=1)"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  Y_MODE=$Y_MODE  FORWARD_STAGE2_MODE=$FORWARD_STAGE2_MODE"
echo "  PASS_K=$PASS_K"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
