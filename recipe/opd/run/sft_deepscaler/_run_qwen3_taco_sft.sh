#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_ALIAS="${MODEL_ALIAS:?MODEL_ALIAS is required}"
if [ -x /data2/conda/envs/verl/bin/python ]; then
    DEFAULT_PYTHON_BIN=/data2/conda/envs/verl/bin/python
else
    DEFAULT_PYTHON_BIN=python3
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

SOURCE_SFT_DATA="${SOURCE_SFT_DATA:-/data2/data/jiangli/huggingface/datasets/TACO}"
SFT_DATA_TAG="${SFT_DATA_TAG:-taco_solution_only_max8192}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/train_dataset/taco/train_sft_${SFT_DATA_TAG}.parquet}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
RUN_ROOT="${RUN_ROOT:-/data2/tmp/taco_sft_runs/${MODEL_ALIAS}_${SFT_DATA_TAG}_lr${LEARNING_RATE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
FINAL_MODEL_LINK="${FINAL_MODEL_LINK:-$RUN_ROOT/final_model}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"

NORMALIZED_VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$NORMALIZED_VISIBLE_GPUS"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPU_IDS[@]}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-16384}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
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
EVAL_DATASETS="${EVAL_DATASETS:-humaneval_plus mbpp_plus livecodebench_v6}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-results/SFT/code/${MODEL_ALIAS}_${SFT_DATA_TAG}_lr${LEARNING_RATE}}"
EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-$EVAL_RESULTS_DIR/results.json}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EVAL_RESULTS_DIR/generations}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-${MODEL_ALIAS}_TACO_Solution_Only_SFT_lr${LEARNING_RATE}_pass16}"
CODE_EVAL_GEN_TP="${CODE_EVAL_GEN_TP:-1}"

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
    if [ ! -d "$1" ] || [ ! -s "$1/config.json" ]; then
        echo "ERROR: missing local model or config.json: $1" >&2
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
    local hf_model="$CHECKPOINT_DIR/global_step_${latest_step}/huggingface"
    require_file "$hf_model/config.json" "exported HuggingFace config"
    if ! find "$hf_model" -maxdepth 1 -type f -name 'model*.safetensors' -print -quit | grep -q .; then
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
    if ! [[ "$CODE_EVAL_GEN_TP" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: CODE_EVAL_GEN_TP must be a positive integer, got: $CODE_EVAL_GEN_TP" >&2
        exit 1
    fi
    if [ "$CODE_EVAL_GEN_TP" -gt "$NUM_GPUS" ] || [ $((NUM_GPUS % CODE_EVAL_GEN_TP)) -ne 0 ]; then
        echo "ERROR: CODE_EVAL_GEN_TP=$CODE_EVAL_GEN_TP must divide NUM_GPUS=$NUM_GPUS" >&2
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
            echo "ERROR: GPU $gpu_id is reserved by another TACO SFT pipeline." >&2
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
require_source "$SOURCE_SFT_DATA" "local TACO dataset"
validate_gpu_config

if [ "${SFT_DRY_RUN:-false}" = "true" ]; then
    echo "Model:              $MODEL_PATH"
    echo "Raw SFT source:     $SOURCE_SFT_DATA"
    echo "SFT data tag:       $SFT_DATA_TAG"
    echo "Training data:      $TRAIN_FILE"
    echo "Checkpoint dir:     $CHECKPOINT_DIR"
    echo "Visible GPUs:       $CUDA_VISIBLE_DEVICES"
    echo "Number of GPUs:     $NUM_GPUS"
    echo "Global batch:       $TRAIN_BATCH_SIZE"
    echo "Max length:         $MAX_LENGTH"
    echo "Max tokens/GPU:     $MAX_TOKEN_LEN_PER_GPU"
    echo "Learning rate:      $LEARNING_RATE"
    echo "Epochs:             $TOTAL_EPOCHS"
    echo "Eval datasets:      $EVAL_DATASETS"
    echo "Eval pass@k:        16"
    echo "Eval tensor parallel:$CODE_EVAL_GEN_TP"
    echo "Eval results:       $EVAL_RESULTS_FILE"
    echo "Eval generations:   $EVAL_OUTPUT_DIR"
    exit 0
fi

mkdir -p "$(dirname "$TRAIN_FILE")" "$RUN_ROOT" "$CHECKPOINT_DIR" "$LOG_DIR"
acquire_gpu_locks
check_selected_gpus_are_free

if [ ! -s "$TRAIN_FILE" ]; then
    "$PYTHON_BIN" -m recipe.opd.run.sft_deepscaler.prepare_taco_sft \
        --input "$SOURCE_SFT_DATA" \
        --output "$TRAIN_FILE" \
        --model-path "$MODEL_PATH" \
        --max-length "$MAX_LENGTH"
else
    echo "Reusing prepared length-safe TACO SFT parquet: $TRAIN_FILE"
fi

if [ -L "$FINAL_MODEL_LINK" ] && [ -s "$FINAL_MODEL_LINK/config.json" ]; then
    echo "Reusing completed TACO SFT model: $FINAL_MODEL_LINK"
else
    echo "Starting TACO solution-only SFT for $MODEL_ALIAS"
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
        data.use_dynamic_bsz=true \
        data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
        data.messages_key=messages \
        data.ignore_input_ids_mismatch=true \
        data.train_max_samples="$TRAIN_MAX_SAMPLES" \
        data.num_workers="$NUM_WORKERS" \
        model.path="$MODEL_PATH" \
        model.trust_remote_code=true \
        model.use_remove_padding=true \
        model.enable_gradient_checkpointing=true \
        model.lora_rank=0 \
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
        trainer.logger="['console']" \
        trainer.project_name=taco-solution-sft \
        trainer.experiment_name="$MODEL_ALIAS" \
        trainer.default_local_dir="$CHECKPOINT_DIR" \
        trainer.resume_mode="$RESUME_MODE" \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.max_ckpt_to_keep=1 \
        checkpoint.save_contents='[model,extra,hf_model]' \
        checkpoint.load_contents='[model,extra]' \
        2>&1 | tee "$LOG_DIR/train.log"

    final_hf_model="$(find_final_hf_model)"
    ln -sfn "$final_hf_model" "$FINAL_MODEL_LINK"
    echo "Final TACO SFT model: $FINAL_MODEL_LINK -> $final_hf_model"
fi

if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
    mkdir -p "$(dirname "$EVAL_RESULTS_FILE")" "$EVAL_OUTPUT_DIR"
    echo "Starting code Avg@16 / Pass@16 evaluation for $MODEL_ALIAS"
    PYTHON_BIN="$PYTHON_BIN" \
    NGPUS_PER_NODE="$NUM_GPUS" \
    GEN_TP="$CODE_EVAL_GEN_TP" \
    DATASETS="$EVAL_DATASETS" \
    PASS_K=16 \
    EVAL_BASE_MODEL_NAME="$MODEL_ALIAS" \
    EVAL_MODEL_NAME="$EVAL_MODEL_NAME" \
    EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
    EVAL_RESULTS_FILE="$EVAL_RESULTS_FILE" \
    EVAL_RESULTS_CSV_FILE="${EVAL_RESULTS_FILE%.json}.csv" \
    WRITE_RESULTS_CSV=true \
        bash "$REPO_ROOT/recipe/code_evaluation/benchmark_code_model.sh" "$FINAL_MODEL_LINK" \
        2>&1 | tee "$LOG_DIR/eval_pass16.log"
fi

echo "TACO solution-only SFT pipeline complete: $MODEL_ALIAS"
