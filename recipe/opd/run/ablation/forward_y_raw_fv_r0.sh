#!/bin/bash
# =============================================================================
# Ablation: Forward KL FV on y_raw, filtered to stage1 reward==0
# =============================================================================
# Trains on deepscaleR_stage1_responses_reward0.parquet (samples the student
# got wrong). Forward KL is implemented via JSD with BETA=0 (forward KL endpoint).
# =============================================================================

set -e
set -o pipefail

export KL_TYPE="${KL_TYPE:-jsd}"
export KL_METHOD="${KL_METHOD:-full_vocab}"
export BETA="${BETA:-0}"                       # β=0 = forward KL
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export TEMPERATURE="${TEMPERATURE:-1}"
export Y_MODE="${Y_MODE:-y_raw}"
export FORWARD_STAGE2_MODE="${FORWARD_STAGE2_MODE:-stage1_reward_0_only}"

export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export USE_LORA="${USE_LORA:-true}"

export MAX_LENGTH="${MAX_LENGTH:-18432}"
export SP_SIZE="${SP_SIZE:-1}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND:-1}"

export GRAD_COSINE_INTERVAL="${GRAD_COSINE_INTERVAL:-50}"
export CORRECTION_TOKEN_PHRASES="${CORRECTION_TOKEN_PHRASES:-Wait,Actually,However,Alternatively,Oops,Wrong,Error,Incorrect,Correction,Sorry,Hmm,Oh,Hold,Pause,Uh,Um}"
export LOG_DIFFICULTY_BUCKETS="${LOG_DIFFICULTY_BUCKETS:-true}"
export SCORE_STAGE1="${SCORE_STAGE1:-true}"

# Eval after training
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export PASS_K="${PASS_K:-16}"
export EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25,beyondaime,amobench}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "ABLATION: Forward KL FV on y_raw (s1 r=0)"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE BETA=$BETA  Y_MODE=$Y_MODE  FORWARD_STAGE2_MODE=$FORWARD_STAGE2_MODE"
echo "  PASS_K=$PASS_K"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
