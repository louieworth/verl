#!/bin/bash
# =============================================================================
# KL Divergence Training for Math Reasoning
# =============================================================================
#
# This script runs token-level KL divergence training with 4 variants:
#   1. Reverse KL + Monte Carlo
#   2. Reverse KL + Full Vocabulary
#   3. Forward KL + Monte Carlo
#   4. Forward KL + Full Vocabulary
#
# Multi-epoch behavior:
#   - TOTAL_EPOCHS controls the outer pipeline epochs
#   - Each epoch generates its own stage1/stage2 data under gen_results/${MODEL_NAME}/epochN/
#   - Each epoch trains from the current model and saves to an epoch-specific checkpoint dir
#   - Epoch N+1 loads the model saved by epoch N
# =============================================================================

set -e
set -o pipefail

# =============================================================================
# Configuration
# =============================================================================

# KL Training Settings
KL_TYPE=${KL_TYPE:-"forward"}          # reverse or forward
KL_METHOD=${KL_METHOD:-"monte_carlo"}  # monte_carlo or full_vocab
TEMPERATURE=${TEMPERATURE:-0.7}         # Softmax temperature
USE_INITIAL_RESPONSE=${USE_INITIAL_RESPONSE:-"false"}  # Stage 2 / forward teacher prompt: false=rewrite, true=correct initial response
FORWARD_STAGE2_MODE=${FORWARD_STAGE2_MODE:-"rewrite_all"}  # rewrite_all or reward0_only

# Model Settings
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B-Instruct-2507"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-""}  # Empty means same as student
USE_LORA=${USE_LORA:-true}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}

# Training Settings
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}  # Outer pipeline epochs
TRAIN_EPOCHS_PER_ROUND=${TRAIN_EPOCHS_PER_ROUND:-1}  # Trainer epochs for each pipeline epoch
MAX_LENGTH=${MAX_LENGTH:-20480}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

# Data Settings
DATA_PATH=${DATA_PATH:-""}  # Optional override; reused across epochs if set
CORRECTED_RESPONSES_PATH=${CORRECTED_RESPONSES_PATH:-""}  # Optional legacy two-file mode
MAX_SAMPLES=${MAX_SAMPLES:-""}  # For testing, leave empty for full data

# Distributed Training
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
GEN_TP=${GEN_TP:-1}  # Tensor parallel for generation

# verl FSDP Settings
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}
FSDP_SIZE=${FSDP_SIZE:--1}
SP_SIZE=${SP_SIZE:-1}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-40960}
NUM_WORKERS=${NUM_WORKERS:-4}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-"true"}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-"false"}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-"false"}
OFFLOAD_POLICY=${OFFLOAD_POLICY:-"false"}

# Output Settings
OUTPUT_DIR=${OUTPUT_DIR:-""}  # Base output dir; epoch subdirs are appended
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"/data/data/jiangli/models"}  # Base model save dir; epoch subdirs are appended
WANDB_PROJECT=${WANDB_PROJECT:-"verl-kl-training"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}  # Base wandb run name; _epochN is appended
SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-"true"}  # Merge LoRA after training

# Evaluation Settings
RUN_EVAL_AFTER_TRAINING=${RUN_EVAL_AFTER_TRAINING:-"true"}
EVAL_DATASETS=${EVAL_DATASETS:-"aime24,aime25,math500,hmmt25"}
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-"/data/data/jiangli/huggingface/datasets"}
PASS_K=${PASS_K:-1}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RECIPE_DIR="$SCRIPT_DIR"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")/open_math_reasoning"

# Training data path (for prompts)
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"$VERL_ROOT/data/deepscaleR_train.parquet"}

# Model name from the original base model, not from epoch checkpoints
MODEL_NAME="${MODEL_PATH##*/}"

if [ "$KL_TYPE" = "forward" ] && [ "$FORWARD_STAGE2_MODE" != "rewrite_all" ] && [ "$FORWARD_STAGE2_MODE" != "reward0_only" ]; then
    echo "ERROR: FORWARD_STAGE2_MODE must be one of: rewrite_all, reward0_only"
    exit 1
fi

if [ "$TOTAL_EPOCHS" -lt 1 ]; then
    echo "ERROR: TOTAL_EPOCHS must be >= 1"
    exit 1
fi

if [ "$TOTAL_EPOCHS" -gt 1 ] && [ "$SAVE_MERGED_MODEL" != "true" ]; then
    echo "ERROR: Multi-epoch training requires SAVE_MERGED_MODEL=true so epoch N+1 can load epoch N as HuggingFace weights."
    exit 1
fi

# Base directories
GEN_RESULTS_BASE_DIR="$VERL_ROOT/gen_results/${MODEL_NAME}"
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_BASE_DIR="$VERL_ROOT/outputs/${MODEL_NAME}_kl_${KL_TYPE}_${KL_METHOD}"
else
    OUTPUT_BASE_DIR="$OUTPUT_DIR"
fi

if [ -z "$MODEL_SAVE_DIR" ] || [ "$MODEL_SAVE_DIR" = "/data/data/jiangli/models" ]; then
    MODEL_SAVE_BASE_DIR="/data/data/jiangli/models/${MODEL_NAME}_kl_${KL_TYPE}_${KL_METHOD}"
else
    MODEL_SAVE_BASE_DIR="$MODEL_SAVE_DIR"
fi

if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME_BASE="kl_${KL_TYPE}_${KL_METHOD}"
else
    WANDB_RUN_NAME_BASE="$WANDB_RUN_NAME"
fi

# =============================================================================
# Helper Functions
# =============================================================================

file_exists_and_nonempty() {
    local file="$1"
    [ -f "$file" ] && [ -s "$file" ]
}

epoch_gen_results_dir() {
    local epoch="$1"
    echo "$GEN_RESULTS_BASE_DIR/epoch${epoch}"
}

epoch_output_dir() {
    local epoch="$1"
    echo "$OUTPUT_BASE_DIR/epoch${epoch}"
}

epoch_model_save_dir() {
    local epoch="$1"
    echo "$MODEL_SAVE_BASE_DIR/epoch${epoch}"
}

resolve_epoch_data_path() {
    local epoch="$1"

    if [ -n "$DATA_PATH" ]; then
        echo "$DATA_PATH"
        return
    fi

    local epoch_dir
    epoch_dir="$(epoch_gen_results_dir "$epoch")"

    if [ "$KL_TYPE" = "forward" ]; then
        if [ "$FORWARD_STAGE2_MODE" = "reward0_only" ]; then
            echo "$epoch_dir/deepscaleR_stage2_reward0_responses.parquet"
        else
            echo "$epoch_dir/deepscaleR_stage2_responses.parquet"
        fi
    else
        echo "$epoch_dir/deepscaleR_stage1_responses.parquet"
    fi
}

resolve_epoch_model_path() {
    local epoch="$1"

    if [ "$epoch" -eq 1 ]; then
        echo "$MODEL_PATH"
        return
    fi

    local prev_epoch=$((epoch - 1))
    local prev_model_dir
    prev_model_dir="$(epoch_model_save_dir "$prev_epoch")"

    local prev_model_path
    if [ "$SAVE_MERGED_MODEL" = "true" ]; then
        prev_model_path="$prev_model_dir/hf_merged"
    else
        prev_model_path="$prev_model_dir/final"
    fi

    if [ ! -d "$prev_model_path" ]; then
        echo "ERROR: Previous epoch model not found: $prev_model_path"
        exit 1
    fi

    echo "$prev_model_path"
}

print_base_configuration() {
    echo "=========================================="
    echo "KL Divergence Training Configuration"
    echo "=========================================="
    echo ""
    echo "KL Settings:"
    echo "  Type:     $KL_TYPE"
    echo "  Method:   $KL_METHOD"
    echo "  Temp:     $TEMPERATURE"
    if [ "$KL_TYPE" = "forward" ]; then
        echo "  Stage2 Mode: $FORWARD_STAGE2_MODE"
        echo "  Stage2 Prompt: $( [ "$USE_INITIAL_RESPONSE" = "true" ] && echo "correct_initial_response" || echo "rewrite_from_expert" )"
    fi
    echo ""
    echo "Model Settings:"
    echo "  Base Model:      $MODEL_PATH"
    echo "  Teacher:         ${TEACHER_MODEL_PATH:-<same as student>}"
    echo "  LoRA:            $USE_LORA (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
    echo ""
    echo "Training Settings:"
    echo "  Pipeline Epochs: $TOTAL_EPOCHS"
    echo "  Train Epochs/Round: $TRAIN_EPOCHS_PER_ROUND"
    echo "  LR:             $LEARNING_RATE"
    echo "  Batch:          $TRAIN_BATCH_SIZE"
    echo "  Max Len:        $MAX_LENGTH"
    echo "  Grad Accum:     $GRADIENT_ACCUMULATION_STEPS"
    echo "  FSDP:           $FSDP_STRATEGY (size=$FSDP_SIZE, sp=$SP_SIZE)"
    echo "  Token/GPU:      $MAX_TOKEN_LEN_PER_GPU"
    echo ""
    echo "Data Settings:"
    echo "  Train Data:     $TRAIN_DATA_PATH"
    echo "  Manual Data Override: ${DATA_PATH:-<auto>}"
    if [ "$KL_TYPE" = "forward" ] && [ -n "$CORRECTED_RESPONSES_PATH" ]; then
        echo "  Legacy Rewrite Targets: $CORRECTED_RESPONSES_PATH"
    fi
    echo "  Max Samples:    ${MAX_SAMPLES:-all}"
    echo ""
    echo "Output Base:"
    echo "  Output Dir:     $OUTPUT_BASE_DIR"
    echo "  Model Save:     $MODEL_SAVE_BASE_DIR"
    echo "  Gen Results:    $GEN_RESULTS_BASE_DIR"
    echo "  Wandb:          $WANDB_PROJECT / $WANDB_RUN_NAME_BASE"
    echo ""
    echo "Evaluation:"
    echo "  Run After Training: $RUN_EVAL_AFTER_TRAINING"
    if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
        echo "  Datasets:           $EVAL_DATASETS"
        echo "  Dataset Root:       $EVAL_DATASETS_DIR"
    fi
    echo ""
    echo "=========================================="
}

run_epoch() {
    local epoch="$1"
    local current_model_path="$2"
    local current_teacher_model_path="$3"
    local current_gen_results_dir
    current_gen_results_dir="$(epoch_gen_results_dir "$epoch")"
    local current_output_dir
    current_output_dir="$(epoch_output_dir "$epoch")"
    local current_model_save_dir
    current_model_save_dir="$(epoch_model_save_dir "$epoch")"
    local current_wandb_run_name="${WANDB_RUN_NAME_BASE}_epoch${epoch}"
    local current_data_path
    current_data_path="$(resolve_epoch_data_path "$epoch")"

    mkdir -p "$current_gen_results_dir"
    mkdir -p "$current_output_dir"
    mkdir -p "$current_output_dir/logs"
    mkdir -p "$current_model_save_dir"

    echo ""
    echo "=========================================="
    echo "Epoch ${epoch}/${TOTAL_EPOCHS}"
    echo "=========================================="
    echo "Student Model: $current_model_path"
    echo "Teacher Model: ${current_teacher_model_path:-<same as student>}"
    echo "Gen Results Dir: $current_gen_results_dir"
    echo "Output Dir: $current_output_dir"
    echo "Model Save Dir: $current_model_save_dir"
    echo "Training Data: $current_data_path"
    echo ""

    echo "=========================================="
    echo "Checking Training Data"
    echo "=========================================="

    if [ "$KL_TYPE" = "forward" ]; then
        if file_exists_and_nonempty "$current_data_path"; then
            echo "Found existing Stage 2 rewrite data: $current_data_path"
        else
            echo "Stage 2 rewrite data not found. Generating..."

            local stage1_output="$current_gen_results_dir/deepscaleR_stage1_responses.parquet"
            local stage1_prompts="$current_gen_results_dir/deepscaleR_stage1_prompts.parquet"

            if file_exists_and_nonempty "$stage1_output"; then
                echo "  Stage 1 already done: $stage1_output"
            else
                echo "  [Stage 1] Generating initial responses..."
                python3 "$PIPELINE_DIR/stage1_prepare.py" \
                    --train_file "$TRAIN_DATA_PATH" \
                    --output_file "$stage1_prompts"

                python3 -m verl.trainer.main_generation_server \
                    trainer.nnodes="${NNODES}" \
                    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                    actor_rollout_ref.model.path="${current_model_path}" \
                    actor_rollout_ref.model.trust_remote_code=true \
                    actor_rollout_ref.rollout.temperature=0.6 \
                    actor_rollout_ref.rollout.top_p=0.95 \
                    actor_rollout_ref.rollout.top_k=20 \
                    actor_rollout_ref.rollout.prompt_length=4096 \
                    actor_rollout_ref.rollout.response_length=16384 \
                    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
                    actor_rollout_ref.rollout.name=vllm \
                    actor_rollout_ref.rollout.n=1 \
                    data.train_files="['${stage1_prompts}']" \
                    data.prompt_key=prompt \
                    +data.output_path="${stage1_output}"
            fi

            local stage2_prompts
            if [ "$FORWARD_STAGE2_MODE" = "reward0_only" ]; then
                echo "  [Stage 2] Generating reward==0 prompts..."
                stage2_prompts="$current_gen_results_dir/deepscaleR_stage2_reward0_prompts.parquet"
                if file_exists_and_nonempty "$stage2_prompts"; then
                    echo "  Stage 2 reward==0 prompts already prepared: $stage2_prompts"
                else
                    python3 "$PIPELINE_DIR/stage2_prepare_rewrite_reward0.py" \
                        --stage1_output "$stage1_output" \
                        --use_initial_response "$USE_INITIAL_RESPONSE" \
                        --output_file "$stage2_prompts"
                fi
            else
                echo "  [Stage 2] Generating prompts over all samples..."
                stage2_prompts="$current_gen_results_dir/deepscaleR_stage2_prompts.parquet"
                if file_exists_and_nonempty "$stage2_prompts"; then
                    echo "  Stage 2 prompts already prepared: $stage2_prompts"
                else
                    python3 "$PIPELINE_DIR/stage2_prepare_rewrite_all.py" \
                        --stage1_output "$stage1_output" \
                        --use_initial_response "$USE_INITIAL_RESPONSE" \
                        --output_file "$stage2_prompts"
                fi
            fi

            echo "  [Stage 2] Generating rewritten responses..."
            python3 -m verl.trainer.main_generation_server \
                trainer.nnodes="${NNODES}" \
                trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                actor_rollout_ref.model.path="${current_model_path}" \
                actor_rollout_ref.model.trust_remote_code=true \
                actor_rollout_ref.rollout.temperature=0.6 \
                actor_rollout_ref.rollout.top_p=0.95 \
                actor_rollout_ref.rollout.top_k=20 \
                actor_rollout_ref.rollout.prompt_length=20480 \
                actor_rollout_ref.rollout.response_length=16384 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.n=1 \
                data.train_files="['${stage2_prompts}']" \
                data.prompt_key=prompt \
                +data.output_path="${current_data_path}"
        fi
    else
        if file_exists_and_nonempty "$current_data_path"; then
            echo "Found existing Stage 1 data: $current_data_path"
        else
            echo "Stage 1 data not found. Generating..."

            local stage1_prompts="$current_gen_results_dir/deepscaleR_stage1_prompts.parquet"

            python3 "$PIPELINE_DIR/stage1_prepare.py" \
                --train_file "$TRAIN_DATA_PATH" \
                --output_file "$stage1_prompts"

            python3 -m verl.trainer.main_generation_server \
                trainer.nnodes="${NNODES}" \
                trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                actor_rollout_ref.model.path="${current_model_path}" \
                actor_rollout_ref.model.trust_remote_code=true \
                actor_rollout_ref.rollout.temperature=0.6 \
                actor_rollout_ref.rollout.top_p=0.95 \
                actor_rollout_ref.rollout.top_k=20 \
                actor_rollout_ref.rollout.prompt_length=4096 \
                actor_rollout_ref.rollout.response_length=16384 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.n=1 \
                data.train_files="['${stage1_prompts}']" \
                data.prompt_key=prompt \
                +data.output_path="${current_data_path}"
        fi
    fi

    echo ""
    echo "Training data ready: $current_data_path"

    cat > "$current_output_dir/training_config.yaml" << EOF
pipeline_epoch: $epoch
pipeline_total_epochs: $TOTAL_EPOCHS
trainer_total_epochs: $TRAIN_EPOCHS_PER_ROUND
kl_type: $KL_TYPE
kl_method: $KL_METHOD
temperature: $TEMPERATURE
use_initial_response: $USE_INITIAL_RESPONSE
forward_stage2_mode: $FORWARD_STAGE2_MODE
base_model_name: $MODEL_NAME
student_model_path: $current_model_path
teacher_model_path: ${current_teacher_model_path:-$current_model_path}
use_lora: $USE_LORA
lora_rank: $LORA_RANK
lora_alpha: $LORA_ALPHA
learning_rate: $LEARNING_RATE
train_batch_size: $TRAIN_BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
max_length: $MAX_LENGTH
warmup_ratio: $WARMUP_RATIO
weight_decay: $WEIGHT_DECAY
num_workers: $NUM_WORKERS
fsdp_strategy: $FSDP_STRATEGY
fsdp_size: $FSDP_SIZE
ulysses_sequence_parallel_size: $SP_SIZE
max_token_len_per_gpu: $MAX_TOKEN_LEN_PER_GPU
use_torch_compile: $USE_TORCH_COMPILE
param_offload: $PARAM_OFFLOAD
optimizer_offload: $OPTIMIZER_OFFLOAD
offload_policy: $OFFLOAD_POLICY
data_path: $current_data_path
corrected_responses_path: ${CORRECTED_RESPONSES_PATH:-null}
max_samples: ${MAX_SAMPLES:-null}
output_dir: $current_output_dir
model_save_dir: $current_model_save_dir
gen_results_dir: $current_gen_results_dir
wandb_project: $WANDB_PROJECT
wandb_run_name: $current_wandb_run_name
save_merged_model: $SAVE_MERGED_MODEL
EOF

    export PYTHONPATH="$VERL_ROOT:$PYTHONPATH"

    local cmd="torchrun \
        --nproc-per-node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        $RECIPE_DIR/run_training.py \
        --nnodes $NNODES \
        --n_gpus_per_node $NGPUS_PER_NODE \
        --kl_type $KL_TYPE \
        --kl_method $KL_METHOD \
        --temperature $TEMPERATURE \
        --student_model_path $current_model_path \
        ${current_teacher_model_path:+--teacher_model_path $current_teacher_model_path} \
        --base_model_name $MODEL_NAME \
        --use_lora $USE_LORA \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --learning_rate $LEARNING_RATE \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --total_epochs $TRAIN_EPOCHS_PER_ROUND \
        --max_length $MAX_LENGTH \
        --warmup_steps_ratio $WARMUP_RATIO \
        --weight_decay $WEIGHT_DECAY \
        --min_lr_ratio 0.1 \
        --data_path $current_data_path \
        ${CORRECTED_RESPONSES_PATH:+--corrected_responses_path $CORRECTED_RESPONSES_PATH} \
        --num_workers $NUM_WORKERS \
        --fsdp_strategy $FSDP_STRATEGY \
        --fsdp_size $FSDP_SIZE \
        --ulysses_sequence_parallel_size $SP_SIZE \
        --max_token_len_per_gpu $MAX_TOKEN_LEN_PER_GPU \
        --use_torch_compile $USE_TORCH_COMPILE \
        --param_offload $PARAM_OFFLOAD \
        --optimizer_offload $OPTIMIZER_OFFLOAD \
        --offload_policy $OFFLOAD_POLICY \
        --epoch_index $epoch \
        --output_dir $current_output_dir \
        --model_save_dir $current_model_save_dir \
        --gen_results_dir $current_gen_results_dir \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name $current_wandb_run_name \
        --save_merged_model $SAVE_MERGED_MODEL \
        --run_eval_after_training $RUN_EVAL_AFTER_TRAINING \
        --eval_datasets $EVAL_DATASETS \
        --eval_datasets_dir $EVAL_DATASETS_DIR \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        --use_initial_response $USE_INITIAL_RESPONSE"

    echo "Running command:"
    echo "$cmd"
    echo ""

    eval "$cmd" 2>&1 | tee "$current_output_dir/logs/training_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "Epoch ${epoch} complete"
    echo "  Logs & Config: $current_output_dir"
    echo "  Model Checkpoints: $current_model_save_dir"
    echo "  Gen Results: $current_gen_results_dir"
    if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
        echo "  Eval Results: results/${MODEL_NAME}/results.json"
    fi
    if [ "$SAVE_MERGED_MODEL" = "true" ]; then
        echo "  Merged Model: $current_model_save_dir/hf_merged"
    else
        echo "  FSDP Checkpoints: $current_model_save_dir/global_step_*"
    fi
}

# =============================================================================
# Main
# =============================================================================

print_base_configuration

if [ "$TOTAL_EPOCHS" -gt 1 ] && [ -n "$DATA_PATH" ]; then
    echo "WARNING: DATA_PATH is manually set and will be reused for every epoch: $DATA_PATH"
fi

if [ "$TOTAL_EPOCHS" -gt 1 ] && [ -n "$CORRECTED_RESPONSES_PATH" ]; then
    echo "WARNING: CORRECTED_RESPONSES_PATH is manually set and will be reused for every epoch: $CORRECTED_RESPONSES_PATH"
fi

mkdir -p "$GEN_RESULTS_BASE_DIR"
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$MODEL_SAVE_BASE_DIR"

for EPOCH in $(seq 1 "$TOTAL_EPOCHS"); do
    CURRENT_MODEL_PATH="$(resolve_epoch_model_path "$EPOCH")"
    CURRENT_TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH"
    run_epoch "$EPOCH" "$CURRENT_MODEL_PATH" "$CURRENT_TEACHER_MODEL_PATH"
done

FINAL_OUTPUT_DIR="$(epoch_output_dir "$TOTAL_EPOCHS")"
FINAL_MODEL_SAVE_DIR="$(epoch_model_save_dir "$TOTAL_EPOCHS")"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Final Epoch: $TOTAL_EPOCHS"
echo "Logs & Config: $FINAL_OUTPUT_DIR"
echo "Model Checkpoints: $FINAL_MODEL_SAVE_DIR"
echo "Gen Results Base: $GEN_RESULTS_BASE_DIR"
if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
    echo "Eval Results: results/${MODEL_NAME}/results.json"
fi
if [ "$SAVE_MERGED_MODEL" = "true" ]; then
    echo "Merged Model: $FINAL_MODEL_SAVE_DIR/hf_merged"
    echo "Benchmark: bash $PIPELINE_DIR/benchmark_kl_model.sh $FINAL_MODEL_SAVE_DIR/hf_merged"
else
    echo "FSDP Checkpoints: $FINAL_MODEL_SAVE_DIR/global_step_*"
fi
echo "Wandb Dashboard: https://wandb.ai/$WANDB_PROJECT"
