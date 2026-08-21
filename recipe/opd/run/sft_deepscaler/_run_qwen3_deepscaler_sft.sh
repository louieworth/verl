#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/recipe/opd/run/hf_export_validation.sh"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_ALIAS="${MODEL_ALIAS:?MODEL_ALIAS is required}"
if [ -x /data2/conda/envs/verl/bin/python ]; then
    DEFAULT_PYTHON_BIN=/data2/conda/envs/verl/bin/python
else
    DEFAULT_PYTHON_BIN=python3
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

SOURCE_SFT_DATA="${SOURCE_SFT_DATA:-}"
SFT_DATA_TAG="${SFT_DATA_TAG:-openthoughts_math_30k_opsd}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/train_dataset/openthoughts_math_30k_opsd/train_sft.parquet}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
RUN_ROOT="${RUN_ROOT:-outputs/trd/math/baseline/sft/${MODEL_ALIAS}_${SFT_DATA_TAG}_lr${LEARNING_RATE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
MODELS_DIR="${MODELS_DIR:-$RUN_ROOT/models}"
FINAL_MODEL_LINK="${FINAL_MODEL_LINK:-$RUN_ROOT/final_model}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"

NORMALIZED_VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$NORMALIZED_VISIBLE_GPUS"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPU_IDS[@]}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
MAX_LENGTH="${MAX_LENGTH:-18432}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-18432}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
STOP_AT_STEP="${STOP_AT_STEP:-null}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
FSDP_SIZE="${FSDP_SIZE:--1}"
SP_SIZE="${SP_SIZE:-1}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"
RESUME_MODE="${RESUME_MODE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/data2/tmp/sft_gpu_locks}"
GPU_PREFLIGHT_MAX_USED_MIB="${GPU_PREFLIGHT_MAX_USED_MIB:-1024}"

RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
EVAL_DATASETS="${EVAL_DATASETS:-aime25 aime26 hmmt26 amobench}"
EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-$REPO_ROOT/data/eval_dataset/math}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-results/SFT/math/${MODEL_ALIAS}_${SFT_DATA_TAG}_lr${LEARNING_RATE}}"
EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-$EVAL_RESULTS_DIR/results.json}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EVAL_RESULTS_DIR/generations}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-${MODEL_ALIAS}_OpenThoughts_SFT_lr${LEARNING_RATE}_pass16}"

require_file() {
    if [ ! -s "$1" ]; then
        echo "ERROR: missing $2: $1" >&2
        exit 1
    fi
}

require_source() {
    if [ ! -e "$1" ]; then
        echo "ERROR: missing $2: $1" >&2
        exit 1
    fi
}

require_model() {
    if [ -d "$1" ]; then
        require_file "$1/config.json" "model config"
    elif [[ "$1" != */* ]]; then
        echo "ERROR: model must be a local path or Hugging Face repo ID: $1" >&2
        exit 1
    fi
}

find_final_hf_model() {
    local tracker_file="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
    require_file "$tracker_file" "checkpoint tracker"
    local latest_step
    latest_step="$(tr -d '[:space:]' < "$tracker_file")"
    case "$latest_step" in
        *[!0-9]*|"") echo "ERROR: invalid checkpoint step: $latest_step" >&2; exit 1 ;;
    esac
    local checkpoint_dir="$CHECKPOINT_DIR/global_step_${latest_step}"
    local hf_model="$MODELS_DIR/step_${latest_step}"
    if ! hf_export_complete "$hf_model"; then
        "$PYTHON_BIN" -m recipe.opd.export_checkpoint \
            --local-dir "$checkpoint_dir" \
            --target-dir "$hf_model" \
            --base-model "$MODEL_PATH" \
            --lora-rank "$LORA_RANK" \
            --lora-alpha "$LORA_ALPHA" \
            --trust-remote-code >&2
    fi
    require_file "$hf_model/config.json" "strict merged HuggingFace config"
    require_file "$hf_model/opd_export.json" "strict export marker"
    if ! hf_export_complete "$hf_model"; then
        echo "ERROR: exported HuggingFace weights not found: $hf_model" >&2
        exit 1
    fi
    printf '%s\n' "$hf_model"
}

validate_gpu_config() {
    if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
        exit 1
    fi
    if [ "$NUM_GPUS" -ne "${#VISIBLE_GPU_IDS[@]}" ]; then
        echo "ERROR: NUM_GPUS=$NUM_GPUS does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        exit 1
    fi
    if [ $((TRAIN_BATCH_SIZE % NUM_GPUS)) -ne 0 ]; then
        echo "ERROR: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by NUM_GPUS=$NUM_GPUS" >&2
        exit 1
    fi
    local gpu_id
    for gpu_id in "${VISIBLE_GPU_IDS[@]}"; do
        if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
            echo "ERROR: this runner requires numeric CUDA_VISIBLE_DEVICES entries, got: $gpu_id" >&2
            exit 1
        fi
    done
}

acquire_gpu_locks() {
    mkdir -p "$GPU_LOCK_DIR"
    GPU_LOCK_FDS=()
    local gpu_id lock_fd
    for gpu_id in "${VISIBLE_GPU_IDS[@]}"; do
        exec {lock_fd}>"$GPU_LOCK_DIR/gpu_${gpu_id}.lock"
        if ! flock -n "$lock_fd"; then
            echo "ERROR: GPU $gpu_id is reserved by another SFT pipeline." >&2
            exit 1
        fi
        GPU_LOCK_FDS+=("$lock_fd")
    done
}

check_selected_gpus_are_free() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi
    local busy_gpus
    busy_gpus="$(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
            awk -F',' -v selected=",$NORMALIZED_VISIBLE_GPUS," -v limit="$GPU_PREFLIGHT_MAX_USED_MIB" '
                {
                    gpu_index = $1
                    used = $2
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu_index)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", used)
                    if (index(selected, "," gpu_index ",") && (used + 0) > limit) {
                        printf "%s(%s MiB) ", gpu_index, used
                    }
                }
            '
    )"
    if [ -n "$busy_gpus" ]; then
        echo "ERROR: refusing to start because selected GPUs are not free: $busy_gpus" >&2
        exit 1
    fi
}

require_model "$MODEL_PATH"
if [ ! -s "$TRAIN_FILE" ] && [ -n "$SOURCE_SFT_DATA" ]; then
    require_source "$SOURCE_SFT_DATA" "raw OpenThoughts dataset"
fi
validate_gpu_config

if [ "${SFT_DRY_RUN:-false}" = "true" ]; then
    echo "Model:              $MODEL_PATH"
    echo "Raw SFT source:     ${SOURCE_SFT_DATA:-<canonical parquet>}"
    echo "SFT data tag:       $SFT_DATA_TAG"
    echo "Training data:      $TRAIN_FILE"
    echo "Checkpoint dir:     $CHECKPOINT_DIR"
    echo "Milestone models:   $MODELS_DIR/step_<N>"
    echo "Visible GPUs:       $CUDA_VISIBLE_DEVICES"
    echo "Number of GPUs:     $NUM_GPUS"
    echo "GPU lock dir:       $GPU_LOCK_DIR"
    echo "Global batch:       $TRAIN_BATCH_SIZE"
    echo "Max length:         $MAX_LENGTH"
    echo "Max tokens/GPU:     $MAX_TOKEN_LEN_PER_GPU"
    echo "Micro/dynamic:      $MICRO_BATCH_SIZE_PER_GPU/$USE_DYNAMIC_BSZ"
    echo "Learning rate:      $LEARNING_RATE"
    echo "Epochs:             $TOTAL_EPOCHS"
    echo "Full/stop steps:    $TOTAL_TRAINING_STEPS/$STOP_AT_STEP"
    echo "Eval datasets:      $EVAL_DATASETS"
    echo "Eval pass@k:        16"
    echo "Eval results:       $EVAL_RESULTS_FILE"
    echo "Eval generations:   $EVAL_OUTPUT_DIR"
    exit 0
fi

mkdir -p "$(dirname "$TRAIN_FILE")" "$RUN_ROOT" "$CHECKPOINT_DIR" "$MODELS_DIR" "$LOG_DIR"
acquire_gpu_locks
check_selected_gpus_are_free

if [ ! -s "$TRAIN_FILE" ]; then
    echo "ERROR: canonical OpenThoughts SFT parquet is missing: $TRAIN_FILE" >&2
    echo "Run recipe/opd/dataset/prepare_experiment_data.py first." >&2
    exit 1
else
    echo "Reusing prepared OpenThoughts Base-completion SFT parquet: $TRAIN_FILE"
fi

if [ -L "$FINAL_MODEL_LINK" ] && hf_export_complete "$FINAL_MODEL_LINK" && \
   { [ "$STOP_AT_STEP" = "null" ] || [ "$STOP_AT_STEP" = "$TOTAL_TRAINING_STEPS" ]; }; then
    echo "Reusing completed SFT model: $FINAL_MODEL_LINK"
    require_checkpoint_step "$CHECKPOINT_DIR" "$STOP_AT_STEP"
    current_hf_model="$FINAL_MODEL_LINK"
else
    echo "Starting OpenThoughts Math SFT for $MODEL_ALIAS"
    "$PYTHON_BIN" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc-per-node="$NUM_GPUS" \
        -m verl.trainer.sft_trainer \
        data.train_files="$TRAIN_FILE" \
        data.val_files=null \
        data.train_batch_size="$TRAIN_BATCH_SIZE" \
        data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
        data.max_length="$MAX_LENGTH" \
        data.pad_mode=no_padding \
        data.truncation=error \
        data.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
        data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
        data.custom_cls.path="file://$REPO_ROOT/recipe/opd/base_completion.py" \
        data.custom_cls.name=BaseCompletionSFTDataset \
        data.train_max_samples="$TRAIN_MAX_SAMPLES" \
        data.num_workers="$NUM_WORKERS" \
        model.path="$MODEL_PATH" \
        model.trust_remote_code=true \
        model.use_remove_padding=true \
        model.enable_gradient_checkpointing=true \
        model.lora_rank="$LORA_RANK" \
        model.lora_alpha="$LORA_ALPHA" \
        engine=fsdp \
        optim=fsdp \
        engine.strategy="$FSDP_STRATEGY" \
        engine.fsdp_size="$FSDP_SIZE" \
        engine.ulysses_sequence_parallel_size="$SP_SIZE" \
        engine.use_torch_compile="$USE_TORCH_COMPILE" \
        optim.lr="$LEARNING_RATE" \
        optim.lr_warmup_steps_ratio="$WARMUP_RATIO" \
        optim.weight_decay="$WEIGHT_DECAY" \
        optim.betas='[0.9,0.95]' \
        optim.clip_grad=1.0 \
        optim.min_lr_ratio=0.1 \
        optim.lr_scheduler_type=cosine \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
        trainer.stop_at_step="$STOP_AT_STEP" \
        trainer.logger="['console','wandb']" \
        trainer.project_name="${WANDB_PROJECT:-trd}" \
        trainer.experiment_name="${WANDB_RUN_NAME:-$MODEL_ALIAS}" \
        trainer.default_local_dir="$CHECKPOINT_DIR" \
        trainer.resume_mode="$RESUME_MODE" \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.max_ckpt_to_keep=1 \
        checkpoint.save_contents='[model,optimizer,extra]' \
        checkpoint.load_contents='[model,optimizer,extra]' \
        "$@" \
        2>&1 | tee "$LOG_DIR/train_step_${STOP_AT_STEP}.log"

    require_checkpoint_step "$CHECKPOINT_DIR" "$STOP_AT_STEP"
    current_hf_model="$(find_final_hf_model)"
    latest_step="$(tr -d '[:space:]' < "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")"
    if [ "$TOTAL_TRAINING_STEPS" = "null" ] || [ "$latest_step" -ge "$TOTAL_TRAINING_STEPS" ]; then
        ln -sfn "$(realpath "$current_hf_model")" "$FINAL_MODEL_LINK"
        echo "Final SFT model: $FINAL_MODEL_LINK -> $current_hf_model"
    else
        echo "SFT milestone model: step $latest_step -> $current_hf_model"
    fi
fi

if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
    mkdir -p "$(dirname "$EVAL_RESULTS_FILE")" "$EVAL_OUTPUT_DIR"
    echo "Starting pass@16 / Avg@16 evaluation for $MODEL_ALIAS"
    PYTHON_BIN="$PYTHON_BIN" \
    NGPUS_PER_NODE="$NUM_GPUS" \
    NNODES=1 \
    GEN_TP=1 \
    EVAL_DATASETS_DIR="$EVAL_DATASETS_DIR" \
    DATASETS="$EVAL_DATASETS" \
    PASS_K=16 \
    EVAL_BASE_MODEL_NAME="$MODEL_ALIAS" \
    EVAL_MODEL_NAME="$EVAL_MODEL_NAME" \
    EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
    EVAL_RESULTS_FILE="$EVAL_RESULTS_FILE" \
    EVAL_RESULTS_CSV_FILE="${EVAL_RESULTS_FILE%.json}.csv" \
    EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-64}" \
    EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}" \
    WRITE_PASS16_AGGREGATES=true \
    WRITE_RESULTS_CSV=true \
        bash "$REPO_ROOT/recipe/math_evaluation/benchmark_kl_model.sh" "$current_hf_model" \
        2>&1 | tee "$LOG_DIR/eval_step_${STOP_AT_STEP}_pass16.log"
fi

latest_step="$(tr -d '[:space:]' < "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")"
prune_old_step_checkpoints "$CHECKPOINT_DIR" "$latest_step"

echo "OpenThoughts Math SFT pipeline complete: $MODEL_ALIAS"
