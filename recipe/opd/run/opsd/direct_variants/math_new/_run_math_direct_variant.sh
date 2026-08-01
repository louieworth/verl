#!/usr/bin/env bash
# Shared launcher for the math_new OPSD direct variants.
#
# Model/variant wrappers set MODEL_PATH, MODEL_NAME, STUDENT_MODEL,
# DIRECT_VARIANT, and (for the clipped variant) DIRECT_DEFAULT_CLIP.
# Everything else defaults to data and output directories inside this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$VERL_ROOT"

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL_PATH:?The model wrapper must set MODEL_PATH}"
: "${MODEL_NAME:?The model wrapper must set MODEL_NAME}"
: "${DIRECT_VARIANT:?The model wrapper must set DIRECT_VARIANT}"

case "$DIRECT_VARIANT" in
    forward_kl)
        export KL_TYPE="forward"
        export KL_METHOD="full_vocab"
        export Y_MODE="y_o"
        export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
        export KL_TOKEN_CLIP="0"
        export TOP_K="0"
        ;;
    forward_kl_clip)
        : "${DIRECT_DEFAULT_CLIP:?The clipped variant wrapper must set DIRECT_DEFAULT_CLIP}"
        export KL_TYPE="forward"
        export KL_METHOD="full_vocab"
        export Y_MODE="y_o"
        export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
        export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-$DIRECT_DEFAULT_CLIP}"
        export TOP_K="0"
        ;;
    forward_kl_y_r)
        export KL_TYPE="forward"
        export KL_METHOD="full_vocab"
        export Y_MODE="y_r"
        export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-refine}"
        export KL_TOKEN_CLIP="0"
        export TOP_K="0"
        ;;
    reverse_kl)
        export KL_TYPE="reverse"
        export KL_METHOD="full_vocab"
        export Y_MODE="y_o"
        export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
        export KL_TOKEN_CLIP="0"
        export TOP_K="0"
        ;;
    reverse_kl_topk)
        export KL_TYPE="reverse"
        export KL_METHOD="full_vocab"
        export Y_MODE="y_o"
        export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
        export KL_TOKEN_CLIP="0"
        export TOP_K="${TOP_K:-32}"
        ;;
    skd)
        export KL_TYPE="forward"
        export KL_METHOD="full_vocab"
        export Y_MODE="y_o"
        export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
        export KL_TOKEN_CLIP="0"
        export TOP_K="0"
        ;;
    *)
        echo "ERROR: unsupported DIRECT_VARIANT=$DIRECT_VARIANT" >&2
        exit 2
        ;;
esac

# Task and OPSD model relationship.
export TASK="math"
export DISTILL_MODE="opsd"
export STUDENT_MODEL="${STUDENT_MODEL:-$MODEL_NAME}"
export TEACHER_MODEL_PATH=""
unset TEACHER_MODEL

if [ "$DIRECT_VARIANT" = "skd" ]; then
    export Y_O_ROLLOUT_MODE="skd_vllm"
    export SKD_GAMMA="${SKD_GAMMA:-5}"
    export SKD_ACCEPT_TOP_K="${SKD_ACCEPT_TOP_K:-25}"
    export SKD_ACCEPT_TOP_P="${SKD_ACCEPT_TOP_P:-1.0}"
    export SKD_SHARED_GPUS="${SKD_SHARED_GPUS:-0,1,2,3,4,5,6,7}"
    export SKD_STUDENT_GPUS="${SKD_STUDENT_GPUS:-$SKD_SHARED_GPUS}"
    export SKD_TEACHER_GPUS="${SKD_TEACHER_GPUS:-$SKD_SHARED_GPUS}"
    export SKD_ROLLOUT_BATCH_SIZE="${SKD_ROLLOUT_BATCH_SIZE:-64}"
    export SKD_VLLM_MAX_NUM_SEQS="${SKD_VLLM_MAX_NUM_SEQS:-$SKD_ROLLOUT_BATCH_SIZE}"
    export SKD_PIPELINE_LANES="${SKD_PIPELINE_LANES:-2}"
    export STEP1_STAGE1_RESPONSE_REUSE_PATH="${STEP1_STAGE1_RESPONSE_REUSE_PATH:-${SKD_STAGE1_RESPONSE_REUSE_PATH:-auto}}"
    export PIPELINE_CLEANUP_BATCH_DATA="${PIPELINE_CLEANUP_BATCH_DATA:-false}"
else
    export Y_O_ROLLOUT_MODE="student"
fi

# Repo-local prepared data. MULTI_STEP=1 is the one-update equivalent of the
# historical one-step pipeline, while allowing the already-prepared parquet to
# bypass y_o_prepare.py's raw Dataset/save_to_disk input adapter.
export TRAIN_DATA_PATH="${MATH_NEW_TRAIN_DATA_PATH:-$VERL_ROOT/data/train_dataset/deepscaler/train_grpo.parquet}"
export PRECOMPUTED_STAGE1_PROMPTS_PATH="${MATH_NEW_PRECOMPUTED_STAGE1_PROMPTS_PATH:-$TRAIN_DATA_PATH}"
export TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-deepscaleR}"
export MULTI_STEP="${MATH_NEW_MULTI_STEP:-1}"
export DATA_PATH=""
export CORRECTED_RESPONSES_PATH=""
export PRECOMPUTED_Y_O_TRAJECTORY_PATH=""

# Repo-local evaluation data.
export EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25,beyondaime,amobench}"
export EVAL_DATASETS_DIR="${MATH_NEW_EVAL_DATASETS_DIR:-$VERL_ROOT/data/eval_dataset/math}"
export PASS_K="${PASS_K:-16}"

# Preserve the current math_new length defaults.
export BASE_PROMPT_LENGTH="${BASE_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export EXPERT_SOLUTION_PROMPT_LENGTH="${EXPERT_SOLUTION_PROMPT_LENGTH:-4096}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
if [ "$Y_MODE" = "y_r" ]; then
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-22528}"
    export STAGE2_PROMPT_LENGTH="${STAGE2_PROMPT_LENGTH:-22528}"
    export MAX_LENGTH="${MAX_LENGTH:-38912}"
    export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-38912}"
else
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
    export STAGE2_PROMPT_LENGTH="${STAGE2_PROMPT_LENGTH:-22528}"
    export MAX_LENGTH="${MAX_LENGTH:-22528}"
    export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-24576}"
fi

# Preserve the current shared training defaults, but make them visible and
# independently overrideable from the command line.
export TEMPERATURE="${TEMPERATURE:-1.0}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND:-1}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.005}"
export USE_LORA="${USE_LORA:-true}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"

# Distributed and evaluation defaults are unchanged from the existing wrappers.
export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export GEN_TP="${GEN_TP:-$NGPUS_PER_NODE}"
export EVAL_GEN_TP="${EVAL_GEN_TP:-$NGPUS_PER_NODE}"
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"

# Keep all persistent outputs and caches under the repository.
export RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M%S)}"
export GEN_RESULTS_ROOT="${MATH_NEW_GEN_RESULTS_ROOT:-$VERL_ROOT/gen_results/opsd/direct_variants/math_new}"
export GEN_RESULTS_RUN_PREFIX="${GEN_RESULTS_RUN_PREFIX:-math_new_${MODEL_NAME}_${DIRECT_VARIANT}}"
OUTPUT_ROOT="${MATH_NEW_OUTPUT_ROOT:-$VERL_ROOT/outputs/opsd/direct_variants/math_new}"
RESULTS_ROOT="${MATH_NEW_RESULTS_ROOT:-$VERL_ROOT/results/opsd/direct_variants/math_new}"
export OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_NAME/$DIRECT_VARIANT/$RUN_DATE"
export MODEL_SAVE_DIR="${MATH_NEW_MODEL_SAVE_DIR:-$VERL_ROOT/model/trained/opsd/direct_variants/math_new}"
export PIPELINE_ARCHIVE_MODEL_ROOT="${MATH_NEW_ARCHIVE_MODEL_ROOT:-$VERL_ROOT/model/archive/opsd/direct_variants/math_new}"
export PIPELINE_ARCHIVE_MODEL_DIR="${MATH_NEW_ARCHIVE_MODEL_DIR:-}"
export PIPELINE_TEMP_MODEL_DIR="$OUTPUT_DIR/pipeline_tmp_checkpoints"
export RESULTS_FILE="$RESULTS_ROOT/$MODEL_NAME/$DIRECT_VARIANT/$RUN_DATE/results.json"
export HF_HOME="${MATH_NEW_HF_HOME:-$VERL_ROOT/model/cache/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="${MATH_NEW_TORCH_HOME:-$VERL_ROOT/model/cache/torch}"
export TRITON_CACHE_DIR="${MATH_NEW_TRITON_CACHE_DIR:-$VERL_ROOT/model/cache/triton}"
export WANDB_DIR="${MATH_NEW_WANDB_DIR:-$VERL_ROOT/outputs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"
export NTFY_ENABLED="${NTFY_ENABLED:-false}"
export EVAL_OUTPUT_DIR=""
export EVAL_RESULTS_FILE=""
unset TOKENIZER_PATH

for required_file in "$TRAIN_DATA_PATH" "$PRECOMPUTED_STAGE1_PROMPTS_PATH"; do
    if [ ! -s "$required_file" ]; then
        echo "ERROR: required repo-local training parquet is missing: $required_file" >&2
        exit 1
    fi
done

IFS=',' read -r -a eval_dataset_names <<< "$EVAL_DATASETS"
for dataset_name in "${eval_dataset_names[@]}"; do
    dataset_name="${dataset_name//[[:space:]]/}"
    [ -n "$dataset_name" ] || continue
    eval_file="$EVAL_DATASETS_DIR/$dataset_name/${dataset_name}_test.parquet"
    if [ ! -s "$eval_file" ]; then
        echo "ERROR: required repo-local eval parquet is missing: $eval_file" >&2
        exit 1
    fi
done

dry_run="${DIRECT_VARIANT_DRY_RUN:-false}"
if [ "${1:-}" = "--dry-run" ]; then
    dry_run="true"
    shift
fi

if [ "$dry_run" = "true" ]; then
    clip_tag="clip${KL_TOKEN_CLIP//./}"
    topk_tag=""
    [ "$TOP_K" -le 0 ] || topk_tag="_topk${TOP_K}"
    rollout_tag=""
    [ "$Y_O_ROLLOUT_MODE" != "skd_vllm" ] || rollout_tag="_skd"
    run_name="${Y_MODE}${rollout_tag}_kl_${KL_TYPE}_${KL_METHOD}_${clip_tag}${topk_tag}_${TEACHER_TRAINING_PROMPT}_ms${MULTI_STEP}_${RUN_DATE}"
    cat <<EOF
math_new direct variant preflight: OK
  variant:                  $DIRECT_VARIANT
  run name:                 $run_name
  model:                    $MODEL_NAME ($MODEL_PATH)
  train parquet:            $TRAIN_DATA_PATH
  prepared prompt parquet:  $PRECOMPUTED_STAGE1_PROMPTS_PATH
  eval root:                $EVAL_DATASETS_DIR
  KL:                       $KL_TYPE/$KL_METHOD, y=$Y_MODE, clip=$KL_TOKEN_CLIP, top_k=$TOP_K
  temperature:              $TEMPERATURE
  learning rate:            $LEARNING_RATE
  batch / grad accum:       $TRAIN_BATCH_SIZE / $GRADIENT_ACCUMULATION_STEPS
  lengths:                  prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=$MAX_LENGTH
  policy updates:           $MULTI_STEP
  y_o rollout:              $Y_O_ROLLOUT_MODE
  GPUs / generation TP:     $NGPUS_PER_NODE / $GEN_TP
  evaluation:               after_train=$RUN_EVAL_AFTER_TRAINING, avg@${PASS_K} / pass@${PASS_K}
  model root:               $MODEL_SAVE_DIR
  gen root:                 $GEN_RESULTS_ROOT
  output:                   $OUTPUT_DIR
  results:                  $RESULTS_FILE
  Hugging Face cache:       $HF_HOME
EOF
    exit 0
fi

mkdir -p \
    "$GEN_RESULTS_ROOT" \
    "$OUTPUT_DIR" \
    "$MODEL_SAVE_DIR" \
    "$PIPELINE_ARCHIVE_MODEL_ROOT" \
    "$(dirname "$RESULTS_FILE")" \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$HF_HUB_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$WANDB_DIR"

exec bash "$VERL_ROOT/recipe/opd/run/run_kl_training.sh" "$@"
