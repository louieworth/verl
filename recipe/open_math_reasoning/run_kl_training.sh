#!/bin/bash
# =============================================================================
# KL Divergence Training for Math Reasoning
# =============================================================================
#
# This script runs token-level KL divergence training with 4 variants:
#   1. Reverse KL + Monte Carlo (mode-seeking, efficient)
#   2. Reverse KL + Full Vocabulary (mode-seeking, more accurate)
#   3. Forward KL + Monte Carlo (distribution-covering, efficient)
#   4. Forward KL + Full Vocabulary (distribution-covering, more accurate)
#
# Usage:
#   bash run_kl_training.sh [options]
#
# Examples:
#   # Reverse KL with Monte Carlo (default)
#   KL_TYPE=reverse KL_METHOD=monte_carlo bash run_kl_training.sh
#
#   # Forward KL with Full Vocabulary
#   KL_TYPE=forward KL_METHOD=full_vocab bash run_kl_training.sh
#
#   # Quick test with small data
#   MAX_SAMPLES=100 bash run_kl_training.sh
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# KL Training Settings
KL_TYPE=${KL_TYPE:-"reverse"}          # reverse or forward
KL_METHOD=${KL_METHOD:-"monte_carlo"}  # monte_carlo or full_vocab
KL_COEF=${KL_COEF:-0.1}                 # KL loss coefficient
TEMPERATURE=${TEMPERATURE:-1.0}         # Softmax temperature
USE_INITIAL_RESPONSE=${USE_INITIAL_RESPONSE:-"false"}  # For reverse KL: Variant 1 (false) or Variant 2 (true)

# Model Settings
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-""}  # Empty means same as student
USE_LORA=${USE_LORA:-true}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}

# Training Settings
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
MAX_LENGTH=${MAX_LENGTH:-20480}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

# Data Settings
DATA_PATH=${DATA_PATH:-""}  # Will be auto-generated if empty
CORRECTED_RESPONSES_PATH=${CORRECTED_RESPONSES_PATH:-""}
MAX_SAMPLES=${MAX_SAMPLES:-""}  # For testing, leave empty for full data

# Distributed Training
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
GEN_TP=${GEN_TP:-1}  # Tensor parallel for generation

# Output Settings
OUTPUT_DIR=${OUTPUT_DIR:-""}  # Will be auto-generated if empty
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"/data/data/jiangli/models"}  # Models saved here
WANDB_PROJECT=${WANDB_PROJECT:-"verl-kl-training"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}
SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-"true"}  # Merge LoRA after training

# Evaluation Settings
RUN_EVAL_AFTER_TRAINING=${RUN_EVAL_AFTER_TRAINING:-"false"}  # Set to true to run evaluation
EVAL_DATASETS=${EVAL_DATASETS:-"aime24,aime25,math500"}  # Comma-separated list
PASS_K=${PASS_K:-1}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RECIPE_DIR="$SCRIPT_DIR"

# Training data path (for prompts)
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"$VERL_ROOT/data/deepscaleR_train.parquet"}

# Model name (extract from path)
MODEL_NAME="${MODEL_PATH##*/}"

# Generate unique run ID for this training run
RUN_ID="kl_${KL_TYPE}_${KL_METHOD}_$(date +%Y%m%d_%H%M%S)"

# Generation results directory (intermediate files, like gen_results/${MODEL_NAME}/epoch${EPOCH}/)
GEN_RESULTS_DIR="$VERL_ROOT/gen_results/${MODEL_NAME}_${RUN_ID}"

# Default data paths based on KL type
if [ -z "$DATA_PATH" ]; then
    if [ "$KL_TYPE" = "forward" ]; then
        DATA_PATH="$GEN_RESULTS_DIR/stage2_correction.parquet"
    else
        DATA_PATH="$GEN_RESULTS_DIR/stage1_generation.parquet"
    fi
fi

if [ -z "$CORRECTED_RESPONSES_PATH" ] && [ "$KL_TYPE" = "forward" ]; then
    CORRECTED_RESPONSES_PATH="$GEN_RESULTS_DIR/stage2_correction.parquet"
fi

# Auto-generate output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$VERL_ROOT/outputs/${MODEL_NAME}_${RUN_ID}"
fi

# Auto-generate model save directory
if [ -z "$MODEL_SAVE_DIR" ] || [ "$MODEL_SAVE_DIR" = "/data/data/jiangli/models" ]; then
    MODEL_SAVE_DIR="/data/data/jiangli/models/${MODEL_NAME}_${RUN_ID}"
fi

# Auto-generate wandb run name
if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME="kl_${KL_TYPE}_${KL_METHOD}_$(date +%m%d_%H%M)"
fi

# =============================================================================
# Print Configuration
# =============================================================================

echo "=========================================="
echo "KL Divergence Training Configuration"
echo "=========================================="
echo ""
echo "KL Settings:"
echo "  Type:     $KL_TYPE"
echo "  Method:   $KL_METHOD"
echo "  Coef:     $KL_COEF"
echo "  Temp:     $TEMPERATURE"
if [ "$KL_TYPE" = "reverse" ]; then
echo "  Use Initial Response: $USE_INITIAL_RESPONSE"
fi
echo ""
echo "Model Settings:"
echo "  Model:    $MODEL_PATH"
echo "  Teacher:  ${TEACHER_MODEL_PATH:-$MODEL_PATH}"
echo "  LoRA:     $USE_LORA (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
echo ""
echo "Training Settings:"
echo "  LR:       $LEARNING_RATE"
echo "  Batch:    $TRAIN_BATCH_SIZE"
echo "  Epochs:   $TOTAL_EPOCHS"
echo "  Max Len:  $MAX_LENGTH"
echo ""
echo "Data Settings:"
echo "  Data:     $DATA_PATH"
if [ "$KL_TYPE" = "forward" ]; then
echo "  Corrected: $CORRECTED_RESPONSES_PATH"
fi
echo "  Max Samples: ${MAX_SAMPLES:-all}"
echo ""
echo "Output:"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Model Save:   $MODEL_SAVE_DIR"
echo "  Gen Results:  $GEN_RESULTS_DIR"
echo "  Wandb:        $WANDB_PROJECT / $WANDB_RUN_NAME"
echo "  Save Merged:  $SAVE_MERGED_MODEL"
echo ""
echo "Evaluation:"
echo "  Run After Training: $RUN_EVAL_AFTER_TRAINING"
if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
echo "  Datasets:           $EVAL_DATASETS"
fi
echo ""
echo "=========================================="

# =============================================================================
# Check and Generate Data
# =============================================================================

# Helper function to check if file exists and is non-empty
file_exists_and_nonempty() {
    local file="$1"
    [ -f "$file" ] && [ -s "$file" ]
}

# Create directories
mkdir -p "$GEN_RESULTS_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$MODEL_SAVE_DIR"

# Generate training data if not exists
echo ""
echo "=========================================="
echo "Checking Training Data"
echo "=========================================="

if [ "$KL_TYPE" = "forward" ]; then
    # Forward KL needs Stage 2 corrected responses
    if file_exists_and_nonempty "$DATA_PATH"; then
        echo "Found existing Stage 2 data: $DATA_PATH"
    else
        echo "Stage 2 data not found. Generating..."

        # Stage 1: Generate initial responses
        STAGE1_OUTPUT="$GEN_RESULTS_DIR/stage1_generation.parquet"
        STAGE1_PROMPTS="$GEN_RESULTS_DIR/stage1_prompts.parquet"

        if file_exists_and_nonempty "$STAGE1_OUTPUT"; then
            echo "  Stage 1 already done: $STAGE1_OUTPUT"
        else
            echo "  [Stage 1] Generating initial responses..."

            # Prepare prompts
            python3 "$RECIPE_DIR/stage1_prepare.py" \
                --train_file "$TRAIN_DATA_PATH" \
                --output_file "$STAGE1_PROMPTS"

            # Generate responses
            python3 -m verl.trainer.main_generation_server \
                trainer.nnodes="${NNODES}" \
                trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                actor_rollout_ref.model.path="${MODEL_PATH}" \
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
                data.train_files="['${STAGE1_PROMPTS}']" \
                data.prompt_key=prompt \
                +data.output_path="${STAGE1_OUTPUT}"

            if [ $? -ne 0 ]; then
                echo "ERROR: Stage 1 generation failed"
                exit 1
            fi
        fi

        # Stage 2: Generate corrections
        echo "  [Stage 2] Generating corrections..."

        STAGE2_REWARD0="$GEN_RESULTS_DIR/stage2_reward0_correction.parquet"

        # Prepare correction data
        python3 "$RECIPE_DIR/stage2_prepare.py" \
            --stage1_output "$STAGE1_OUTPUT" \
            --output_reward0 "$STAGE2_REWARD0" \
            --output_reward1 "$GEN_RESULTS_DIR/stage2_reward1.parquet"

        # Generate corrections
        python3 -m verl.trainer.main_generation_server \
            trainer.nnodes="${NNODES}" \
            trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
            actor_rollout_ref.model.path="${MODEL_PATH}" \
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
            data.train_files="['${STAGE2_REWARD0}']" \
            data.prompt_key=prompt \
            +data.output_path="${DATA_PATH}"

        if [ $? -ne 0 ]; then
            echo "ERROR: Stage 2 correction failed"
            exit 1
        fi

        echo "  Stage 2 data generated: $DATA_PATH"
    fi
else
    # Reverse KL needs Stage 1 initial responses
    if file_exists_and_nonempty "$DATA_PATH"; then
        echo "Found existing Stage 1 data: $DATA_PATH"
    else
        echo "Stage 1 data not found. Generating..."

        STAGE1_PROMPTS="$GEN_RESULTS_DIR/stage1_prompts.parquet"

        # Prepare prompts
        python3 "$RECIPE_DIR/stage1_prepare.py" \
            --train_file "$TRAIN_DATA_PATH" \
            --output_file "$STAGE1_PROMPTS"

        # Generate responses
        python3 -m verl.trainer.main_generation_server \
            trainer.nnodes="${NNODES}" \
            trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
            actor_rollout_ref.model.path="${MODEL_PATH}" \
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
            data.train_files="['${STAGE1_PROMPTS}']" \
            data.prompt_key=prompt \
            +data.output_path="${DATA_PATH}"

        if [ $? -ne 0 ]; then
            echo "ERROR: Stage 1 generation failed"
            exit 1
        fi

        echo "  Stage 1 data generated: $DATA_PATH"
    fi
fi

echo ""
echo "Training data ready: $DATA_PATH"

# =============================================================================
# Create Output Directory
# =============================================================================

# Note: Directories already created in the "Check and Generate Data" section

# Save configuration
cat > "$OUTPUT_DIR/training_config.yaml" << EOF
kl_type: $KL_TYPE
kl_method: $KL_METHOD
kl_coef: $KL_COEF
temperature: $TEMPERATURE
use_initial_response: $USE_INITIAL_RESPONSE
model_path: $MODEL_PATH
teacher_model_path: ${TEACHER_MODEL_PATH:-$MODEL_PATH}
use_lora: $USE_LORA
lora_rank: $LORA_RANK
lora_alpha: $LORA_ALPHA
learning_rate: $LEARNING_RATE
train_batch_size: $TRAIN_BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
total_epochs: $TOTAL_EPOCHS
max_length: $MAX_LENGTH
warmup_ratio: $WARMUP_RATIO
weight_decay: $WEIGHT_DECAY
data_path: $DATA_PATH
corrected_responses_path: ${CORRECTED_RESPONSES_PATH:-null}
max_samples: ${MAX_SAMPLES:-null}
output_dir: $OUTPUT_DIR
model_save_dir: $MODEL_SAVE_DIR
gen_results_dir: $GEN_RESULTS_DIR
wandb_project: $WANDB_PROJECT
wandb_run_name: $WANDB_RUN_NAME
save_merged_model: $SAVE_MERGED_MODEL
EOF

# =============================================================================
# Run Training
# =============================================================================

# Set Python path
export PYTHONPATH="$VERL_ROOT:$PYTHONPATH"

# Distributed training settings
DISTRIBUTED_ARGS=""
if [ "$NGPUS_PER_NODE" -gt 1 ] || [ "$NNODES" -gt 1 ]; then
    DISTRIBUTED_ARGS="--local_rank \$LOCAL_RANK"
fi

# Build command
CMD="python3 -m torch.distributed.launch \
    --nproc_per_node=$NGPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $RECIPE_DIR/kl_training/run_training.py \
    --kl_type $KL_TYPE \
    --kl_method $KL_METHOD \
    --kl_coef $KL_COEF \
    --temperature $TEMPERATURE \
    --student_model_path $MODEL_PATH \
    ${TEACHER_MODEL_PATH:+--teacher_model_path $TEACHER_MODEL_PATH} \
    --use_lora $USE_LORA \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --learning_rate $LEARNING_RATE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --total_epochs $TOTAL_EPOCHS \
    --max_length $MAX_LENGTH \
    --warmup_steps_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --data_path $DATA_PATH \
    ${CORRECTED_RESPONSES_PATH:+--corrected_responses_path $CORRECTED_RESPONSES_PATH} \
    --output_dir $OUTPUT_DIR \
    --model_save_dir $MODEL_SAVE_DIR \
    --gen_results_dir $GEN_RESULTS_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --save_merged_model $SAVE_MERGED_MODEL \
    --run_eval_after_training $RUN_EVAL_AFTER_TRAINING \
    --eval_datasets $EVAL_DATASETS \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
    --use_initial_response $USE_INITIAL_RESPONSE \
    $DISTRIBUTED_ARGS"

echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD 2>&1 | tee "$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# Post-training
# =============================================================================

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Output saved to:"
echo "  Logs & Config:    $OUTPUT_DIR"
echo "  Model Checkpoints: $MODEL_SAVE_DIR"
echo "  Gen Results:       $GEN_RESULTS_DIR"
if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
echo "  Eval Results:      results/${MODEL_NAME}/results.json"
fi
if [ "$USE_LORA" = "true" ] && [ "$SAVE_MERGED_MODEL" = "true" ]; then
echo "  Merged Model:      $MODEL_SAVE_DIR/hf_merged"
fi
echo "  Wandb Dashboard:   https://wandb.ai/$WANDB_PROJECT"
echo ""
echo "To benchmark the trained model:"
if [ "$USE_LORA" = "true" ] && [ "$SAVE_MERGED_MODEL" = "true" ]; then
echo "  bash $RECIPE_DIR/benchmark_kl_model.sh $MODEL_SAVE_DIR/hf_merged"
else
echo "  bash $RECIPE_DIR/benchmark_kl_model.sh $MODEL_SAVE_DIR/final"
fi
echo ""
