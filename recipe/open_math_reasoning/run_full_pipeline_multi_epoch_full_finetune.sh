#!/usr/bin/env bash
#
# Multi-Epoch Three-Stage Policy Correction Pipeline with Stage 3 Evaluation
# Features:
#   - Multiple epochs: each epoch runs full Stage 1-3
#   - Resume from previous epoch's SFT checkpoint
#   - Stage 3 evaluation on AIME24/25, AMC23, MATH500 after each epoch
#   - Auto-resume from breakpoints based on generated files
#   - FULL PARAMETER TRAINING (No LoRA)
#
# Usage:
#   bash run_full_pipeline_multi_epoch_full_finetune.sh
#
# Environment Variables:
#   FORCE_RERUN=false    # Set to 'true' to disable auto-resume and rerun everything
#
# Auto-Resume Behavior:
#   - The script automatically detects completed stages by checking for:
#     * Stage 1: stage1_generation.parquet + stage1_eval_results.json
#     * Stage 2: stage2_correction.parquet + stage2_eval_results.json
#     * Stage 3 SFT: deepscaleR_stage3_sft.parquet + checkpoint directory
#     * Stage 3 Eval: evaluation directory with results
#   - Fully completed epochs (4/4 stages) are automatically skipped
#   - Partially completed epochs continue from the last completed stage
#
# To Re-run a Specific Epoch:
#   Delete the corresponding output directories:
#   - rm -rf results/<MODEL_NAME>/epoch<N>/
#   - rm -rf gen_results/epoch<N>/
#   - rm -rf /path/to/checkpoint/<MODEL_NAME>_epoch<N>/
#   - rm -rf results/<MODEL_NAME>/epoch<N>_stage3_eval/

set -e  # Exit on error

################################################################################
# Environment Variables for vLLM
################################################################################

# Allow vLLM to use model length beyond the default max_position_embeddings
# Qwen3-1.7B has max_position_embeddings=40960, but we need 4096+16384=36864
# This is safe as long as we stay within RoPE limits
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

################################################################################
# Resume/Checkpoint State Management
################################################################################

# Function to check if a file exists and is non-empty
file_exists_and_nonempty() {
    [ -f "$1" ] && [ -s "$1" ]
}

# Function to check if a directory exists and has content
dir_exists_and_nonempty() {
    [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]
}

# Function to detect completed stages for an epoch
detect_epoch_progress() {
    local EPOCH=$1

    # If FORCE_RERUN is enabled, always return empty (no stages completed)
    if [ "${FORCE_RERUN}" == "true" ]; then
        echo ""
        return
    fi

    local EPOCH_OUTPUT_DIR="${OUTPUT_BASE_DIR}/epoch${EPOCH}"
    local STAGE1_OUTPUT="${EPOCH_OUTPUT_DIR}/stage1_generation.parquet"
    local STAGE1_EVAL="${EPOCH_OUTPUT_DIR}/stage1_eval_results.json"
    local STAGE2_OUTPUT="${EPOCH_OUTPUT_DIR}/stage2_correction.parquet"
    local STAGE2_EVAL="${EPOCH_OUTPUT_DIR}/stage2_eval_results.json"
    local STAGE3_SFT_DATA="gen_results/epoch${EPOCH}/deepscaleR_stage3_sft.parquet"
    local CKPT_HOME="${CKPT_HOME_BASE}_epoch${EPOCH}"
    local STAGE3_EVAL_DIR="${OUTPUT_BASE_DIR}/epoch${EPOCH}_stage3_eval"
    local STAGE3_EVAL_MARKER="${STAGE3_EVAL_DIR}/.evaluation_completed"

    local completed_stages=()

    # Check Stage 1 completion
    if file_exists_and_nonempty "${STAGE1_OUTPUT}" && file_exists_and_nonempty "${STAGE1_EVAL}"; then
        completed_stages+=("stage1")
    fi

    # Check Stage 2 completion
    if file_exists_and_nonempty "${STAGE2_OUTPUT}" && file_exists_and_nonempty "${STAGE2_EVAL}"; then
        completed_stages+=("stage2")
    fi

    # Check Stage 3 SFT completion
    # Look for either HuggingFace format (config.json + safetensors) or FSDP format (any files)
    local has_valid_ckpt=false
    if dir_exists_and_nonempty "${CKPT_HOME}"; then
        # First check for hf_merged directory at top level (merged model from model_merger)
        local HF_MERGED="${CKPT_HOME}/hf_merged"
        if dir_exists_and_nonempty "${HF_MERGED}"; then
            # Check if merged model has weights
            if [ -f "${HF_MERGED}/model.safetensors" ] || [ -f "${HF_MERGED}/pytorch_model.bin" ] || [ -f "${HF_MERGED}/config.json" ]; then
                has_valid_ckpt=true
            fi
        fi

        # If not found, check global_step_* directories
        if [ "${has_valid_ckpt}" = "false" ]; then
            local latest_step=$(ls -td "${CKPT_HOME}"/global_step_* 2>/dev/null | head -1)
            if [ -n "${latest_step}" ]; then
                # HuggingFace format: config.json in global_step_X/
                if [ -f "${latest_step}/config.json" ]; then
                    has_valid_ckpt=true
                # FSDP format: config.json in global_step_X/huggingface/
                elif [ -f "${latest_step}/huggingface/config.json" ]; then
                    # Check if model weights exist (safetensors or bin)
                    if [ -n "$(ls -A "${latest_step}/huggingface/"*.safetensors 2>/dev/null)" ] || [ -n "$(ls -A "${latest_step}/huggingface/"*.bin 2>/dev/null)" ]; then
                        has_valid_ckpt=true
                    fi
                fi
            fi
        fi
    fi

    if ${has_valid_ckpt} && file_exists_and_nonempty "${STAGE3_SFT_DATA}"; then
        completed_stages+=("stage3_sft")
    fi

    # Check Stage 3 Evaluation completion
    if [ -f "${STAGE3_EVAL_MARKER}" ] || dir_exists_and_nonempty "${STAGE3_EVAL_DIR}"; then
        completed_stages+=("stage3_eval")
    fi

    # Return the list as a string
    echo "${completed_stages[@]}"
}

# Function to mark a stage as completed
mark_stage_completed() {
    local EPOCH=$1
    local STAGE=$2

    if [ "${STAGE}" == "stage3_eval" ]; then
        local STAGE3_EVAL_DIR="${OUTPUT_BASE_DIR}/epoch${EPOCH}_stage3_eval"
        mkdir -p "${STAGE3_EVAL_DIR}"
        touch "${STAGE3_EVAL_DIR}/.evaluation_completed"
    fi
}

# Function to display overall pipeline progress
show_progress() {
    echo ""
    echo "=========================================="
    echo "Pipeline Progress Status"
    echo "=========================================="
    echo "Output Base: ${OUTPUT_BASE_DIR}"
    echo "Checkpoint Base: ${CKPT_HOME_BASE}"
    echo ""

    for EPOCH in $(seq 1 ${NUM_EPOCHS}); do
        COMPLETED_STAGES=($(detect_epoch_progress ${EPOCH}))
        NUM_COMPLETED=${#COMPLETED_STAGES[@]}

        echo "Epoch ${EPOCH}: ${NUM_COMPLETED}/4 stages completed"

        if [ ${NUM_COMPLETED} -gt 0 ]; then
            echo "  Completed stages: ${COMPLETED_STAGES[@]}"
        fi

        if [ ${NUM_COMPLETED} -eq 4 ]; then
            echo "  Status: ✓ FULLY COMPLETED"
        else
            echo "  Status: ○ INCOMPLETE"
        fi
        echo ""
    done
    echo "=========================================="
    echo ""
}

################################################################################
# Configuration
################################################################################

# Model and GPU settings
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}
PASS_K=${PASS_K:-1}
# TEST_PASS_K_VALUES: Space-separated list of pass@k values to test (e.g., "1 32")
# If not set, defaults to "1 32" for both pass@1 and pass@32 evaluation
TEST_PASS_K_VALUES=${TEST_PASS_K_VALUES:-"1"}

# Model paths
# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
MODEL_NAME=$(basename "${MODEL_PATH}")

# Data paths
DEEPSCALE_PATH=${DEEPSCALE_PATH:-/data/data/jiangli/huggingface/datasets/DeepScaleR-Cleaned}

# Eval datasets (prepared by prepare_aime.py, prepare_math500.py, prepare_hmmt.py, prepare_usamo.py)
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}
AIME24_PATH="${EVAL_DATASETS_DIR}/aime24/aime24_test.parquet"
AIME25_PATH="${EVAL_DATASETS_DIR}/aime25/aime25_test.parquet"
AMC23_PATH="${EVAL_DATASETS_DIR}/amc23/amc23_test.parquet"
MATH500_PATH="${EVAL_DATASETS_DIR}/math500/math500_test.parquet"
HMMT25_PATH="${EVAL_DATASETS_DIR}/hmmt25/hmmt25_test.parquet"
HMMT24_PATH="${EVAL_DATASETS_DIR}/hmmt24/hmmt24_test.parquet"
USAMO25_PATH="${EVAL_DATASETS_DIR}/usamo25/usamo25_test.parquet"
USAMO24_PATH="${EVAL_DATASETS_DIR}/usamo24/usamo24_test.parquet"

# Output directories
OUTPUT_BASE_DIR="results/${MODEL_NAME}"
mkdir -p "${OUTPUT_BASE_DIR}"

# Stage 3 SFT settings
BACKEND=${BACKEND:-fsdp}
CKPT_HOME_BASE=/data/data/jiangli/ckpt/${MODEL_NAME}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-4}
FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}

# Multi-epoch settings
NUM_EPOCHS=${NUM_EPOCHS:-3}

# Testing mode: limit samples for quick testing (set to 100 for testing, None for full run)
MAX_SAMPLES=${MAX_SAMPLES:-}

################################################################################
# Helper Functions
################################################################################

# Function: Run Stage 3 evaluation on benchmark datasets
run_stage3_evaluation() {
    local MODEL_TO_EVAL=$1
    local EPOCH=$2
    local EVAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/epoch${EPOCH}_stage3_eval"
    mkdir -p "${EVAL_OUTPUT_DIR}"

    echo ""
    echo "========================================"
    echo "Stage 3 Evaluation - Epoch ${EPOCH}"
    echo "Model: ${MODEL_TO_EVAL}"
    echo "Testing pass@k values: ${TEST_PASS_K_VALUES}"
    echo "========================================"

    # Array of eval datasets
    declare -A EVAL_DATASETS=(
        ["aime24"]="${AIME24_PATH}"
        ["aime25"]="${AIME25_PATH}"
        ["amc23"]="${AMC23_PATH}"
        ["math500"]="${MATH500_PATH}"
        ["hmmt25"]="${HMMT25_PATH}"
        ["hmmt24"]="${HMMT24_PATH}"
        ["usamo25"]="${USAMO25_PATH}"
        ["usamo24"]="${USAMO24_PATH}"
    )

    # Loop over each pass@k value
    for CURRENT_PASS_K in ${TEST_PASS_K_VALUES}; do
        echo ""
        echo "=========================================="
        echo "Testing with pass@${CURRENT_PASS_K}"
        echo "=========================================="

        # for DATASET_NAME in aime24 aime25 amc23 math500 hmmt25 hmmt24 usamo25 usamo24; do
        # TODO comments what needs to be evaluated
        for DATASET_NAME in aime24 aime25 amc23 math500 hmmt25 hmmt24; do
            local DATASET_PATH="${EVAL_DATASETS[$DATASET_NAME]}"

            if [ ! -f "${DATASET_PATH}" ]; then
                echo "Warning: ${DATASET_NAME} not found at ${DATASET_PATH}, skipping..."
                continue
            fi

            echo ""
            echo "[Evaluating on ${DATASET_NAME} with pass@${CURRENT_PASS_K}]"

            local GEN_OUTPUT="${EVAL_OUTPUT_DIR}/${DATASET_NAME}_pass${CURRENT_PASS_K}_generation.parquet"

            # Generate responses
            python3 -m verl.trainer.main_generation_server \
                trainer.nnodes="${NNODES}" \
                trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                actor_rollout_ref.model.path="${MODEL_TO_EVAL}" \
                actor_rollout_ref.model.trust_remote_code=true \
                actor_rollout_ref.rollout.temperature=0.6 \
                actor_rollout_ref.rollout.top_p=0.95 \
                actor_rollout_ref.rollout.prompt_length=4096 \
                actor_rollout_ref.rollout.response_length=32768 \
                actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.n="${CURRENT_PASS_K}" \
                data.train_files="['${DATASET_PATH}']" \
                data.prompt_key=prompt \
                +data.output_path="${GEN_OUTPUT}"

            # Evaluate and save results directly to unified results.json
            python3 -m verl.trainer.main_eval \
                data.path="${GEN_OUTPUT}" \
                data.prompt_key=prompt \
                custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
                custom_reward_function.name=compute_score_data_source \
                +output_json_path="${OUTPUT_BASE_DIR}/results.json" \
                +model_name="${MODEL_NAME}_epoch${EPOCH}" \
                +pass_k=${CURRENT_PASS_K}

            echo "${DATASET_NAME} pass@${CURRENT_PASS_K} evaluation completed!"
        done
    done

    echo ""
    echo "Stage 3 Evaluation Summary - Epoch ${EPOCH}:"
    echo "Results saved to: ${OUTPUT_BASE_DIR}/{dataset}_results.json"
}

# Function: Run single epoch
run_epoch() {
    local EPOCH=$1
    local MODEL_FOR_GENERATION=$2  # Base model or previous epoch checkpoint
    local IS_FIRST_EPOCH=$3

    echo ""
    echo "################################################################################"
    echo "# EPOCH ${EPOCH}/${NUM_EPOCHS}"
    echo "################################################################################"

    local EPOCH_OUTPUT_DIR="${OUTPUT_BASE_DIR}/epoch${EPOCH}"
    mkdir -p "${EPOCH_OUTPUT_DIR}"

    # Detect current progress
    COMPLETED_STAGES=($(detect_epoch_progress ${EPOCH}))
    echo "Detected completed stages: ${COMPLETED_STAGES[@]:-none}"

    # Determine if we need to resume from checkpoint
    local RESUME_MODE="disable"
    local RESUME_FROM_PATH="null"
    if [ "${IS_FIRST_EPOCH}" == "false" ]; then
        # For epoch 2+, resume from previous epoch's checkpoint
        local PREV_EPOCH=$((EPOCH - 1))
        local PREV_CKPT="${CKPT_HOME_BASE}_epoch${PREV_EPOCH}"
        if [ -d "${PREV_CKPT}" ]; then
            echo "Resuming from Epoch ${PREV_EPOCH} checkpoint: ${PREV_CKPT}"
            RESUME_MODE="resume_path"
            RESUME_FROM_PATH="${PREV_CKPT}"
        else
            echo "Warning: Previous checkpoint not found at ${PREV_CKPT}, starting from scratch"
        fi
    fi

    ################################################################################
    # Stage 1: Initial Response Generation
    ################################################################################

    local STAGE1_OUTPUT="${EPOCH_OUTPUT_DIR}/stage1_generation.parquet"
    local STAGE1_EVAL="${EPOCH_OUTPUT_DIR}/stage1_eval_results.json"

    # Check if Stage 1 is already completed
    if [[ " ${COMPLETED_STAGES[@]} " =~ " stage1 " ]]; then
        echo ""
        echo "========================================"
        echo "EPOCH ${EPOCH} - STAGE 1: SKIPPED (already completed)"
        echo "========================================"
        echo "Found existing outputs:"
        echo "  - ${STAGE1_OUTPUT}"
        echo "  - ${STAGE1_EVAL}"
    else
        echo ""
        echo "========================================"
        echo "EPOCH ${EPOCH} - STAGE 1: Initial Response Generation"
        echo "========================================"

        # Step 1.1: Prepare Stage 1 data
        echo "[Step 1.1] Preparing Stage 1 data..."
        if [ -n "${MAX_SAMPLES}" ]; then
            echo "Testing mode: limiting to ${MAX_SAMPLES} samples"
            python3 recipe/open_math_reasoning/stage1_prepare.py \
                --input_path "${DEEPSCALE_PATH}" \
                --output_file "gen_results/epoch${EPOCH}/deepscaleR_stage1.parquet" \
                --data_source deepscaleR \
                --max_samples ${MAX_SAMPLES}
        else
            python3 recipe/open_math_reasoning/stage1_prepare.py \
                --input_path "${DEEPSCALE_PATH}" \
                --output_file "gen_results/epoch${EPOCH}/deepscaleR_stage1.parquet" \
                --data_source deepscaleR
        fi

        # Step 1.2: Generate initial responses
        echo "[Step 1.2] Generating initial responses..."

        python3 -m verl.trainer.main_generation_server \
            trainer.nnodes="${NNODES}" \
            trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
            actor_rollout_ref.model.path="${MODEL_FOR_GENERATION}" \
            actor_rollout_ref.model.trust_remote_code=true \
            actor_rollout_ref.rollout.temperature=0.6 \
            actor_rollout_ref.rollout.top_p=0.95 \
            actor_rollout_ref.rollout.prompt_length=4096 \
            actor_rollout_ref.rollout.response_length=16384 \
            actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.n="${PASS_K}" \
            data.train_files="['gen_results/epoch${EPOCH}/deepscaleR_stage1.parquet']" \
            data.prompt_key=prompt \
            +data.output_path="${STAGE1_OUTPUT}"

        if [ $? -ne 0 ]; then
            echo "Epoch ${EPOCH} Stage 1 generation failed. Exiting."
            exit 1
        fi

        # Step 1.3: Evaluate
        echo "[Step 1.3] Evaluating responses..."
        python3 -m verl.trainer.main_eval \
            data.path="${STAGE1_OUTPUT}" \
            custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
            custom_reward_function.name=compute_score_data_source \
            +output_json_path="${EPOCH_OUTPUT_DIR}/stage1_eval_results.json" \
            +model_name="${MODEL_NAME}_epoch${EPOCH}" \
            +pass_k=${PASS_K}

        echo "Stage 1 completed!"
    fi

    ################################################################################
    # Stage 2: Answer Correction
    ################################################################################

    local STAGE2_OUTPUT="${EPOCH_OUTPUT_DIR}/stage2_correction.parquet"
    local STAGE2_EVAL="${EPOCH_OUTPUT_DIR}/stage2_eval_results.json"

    # Check if Stage 2 is already completed
    if [[ " ${COMPLETED_STAGES[@]} " =~ " stage2 " ]]; then
        echo ""
        echo "========================================"
        echo "EPOCH ${EPOCH} - STAGE 2: SKIPPED (already completed)"
        echo "========================================"
        echo "Found existing outputs:"
        echo "  - ${STAGE2_OUTPUT}"
        echo "  - ${STAGE2_EVAL}"
    else
        echo ""
        echo "========================================"
        echo "EPOCH ${EPOCH} - STAGE 2: Answer Correction"
        echo "========================================"

        # Step 2.1: Prepare correction data
        echo "[Step 2.1] Preparing Stage 2 correction data..."
        python3 recipe/open_math_reasoning/stage2_prepare.py \
            --stage1_output "${STAGE1_OUTPUT}" \
            --output_reward0 "gen_results/epoch${EPOCH}/deepscaleR_stage2_reward0_correction.parquet" \
            --output_reward1 "gen_results/epoch${EPOCH}/deepscaleR_stage2_reward1.parquet"

        # Step 2.2: Generate corrections
        echo "[Step 2.2] Generating corrections for incorrect responses..."

        python3 -m verl.trainer.main_generation_server \
            trainer.nnodes="${NNODES}" \
            trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
            actor_rollout_ref.model.path="${MODEL_FOR_GENERATION}" \
            actor_rollout_ref.model.trust_remote_code=true \
            actor_rollout_ref.rollout.temperature=0.6 \
            actor_rollout_ref.rollout.top_p=0.95 \
            actor_rollout_ref.rollout.prompt_length=32768 \
            actor_rollout_ref.rollout.response_length=16384 \
            actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.n="${PASS_K}" \
            data.train_files="['gen_results/epoch${EPOCH}/deepscaleR_stage2_reward0_correction.parquet']" \
            data.prompt_key=prompt \
            +data.output_path="${STAGE2_OUTPUT}"

        if [ $? -ne 0 ]; then
            echo "Epoch ${EPOCH} Stage 2 correction failed. Exiting."
            exit 1
        fi

        # Step 2.3: Evaluate corrections
        echo "[Step 2.3] Evaluating corrected responses..."
        python3 -m verl.trainer.main_eval \
            data.path="${STAGE2_OUTPUT}" \
            custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
            custom_reward_function.name=compute_score_data_source \
            +output_json_path="${EPOCH_OUTPUT_DIR}/stage2_eval_results.json" \
            +model_name="${MODEL_NAME}_epoch${EPOCH}_corrected" \
            +pass_k=${PASS_K}

        echo "Stage 2 completed!"
    fi

    ################################################################################
    # Stage 3: SFT Training (Full Parameter Training - No LoRA)
    ################################################################################

    local SFT_DATASET="gen_results/epoch${EPOCH}/deepscaleR_stage3_sft.parquet"
    local CKPT_HOME="${CKPT_HOME_BASE}_epoch${EPOCH}"

    # Check if Stage 3 SFT is already completed
    if [[ " ${COMPLETED_STAGES[@]} " =~ " stage3_sft " ]]; then
        echo ""
        echo "========================================"
        echo "EPOCH ${EPOCH} - STAGE 3 SFT: SKIPPED (already completed)"
        echo "========================================"
        echo "Found existing outputs:"
        echo "  - ${SFT_DATASET}"
        echo "  - ${CKPT_HOME}"
    else
        echo ""
        echo "========================================"
        echo "EPOCH ${EPOCH} - STAGE 3: Policy SFT Training (Full Parameter)"
        echo "========================================"

        # Step 3.1: Prepare SFT dataset
        echo "[Step 3.1] Preparing SFT dataset..."

        python3 recipe/open_math_reasoning/stage3_prepare.py \
            --stage2_corrected "${STAGE2_OUTPUT}" \
            --output_file "${SFT_DATASET}"

        # Step 3.2: Run SFT training
        echo "[Step 3.2] Running SFT training (FULL PARAMETER TRAINING - No LoRA)..."
        mkdir -p "${CKPT_HOME}"

        # Build SFT command
        # Training config: 16K context, Full Parameter Training, effective batch size 64
        local SFT_CMD="torchrun --standalone --nnodes=1 --nproc-per-node=${NGPUS_PER_NODE} \
            -m verl.trainer.sft_trainer \
            data.train_files=\"${SFT_DATASET}\" \
            data.train_batch_size=96 \
            data.max_length=16000 \
            data.pad_mode=no_padding \
            data.truncation=error \
            data.use_dynamic_bsz=true \
            data.max_token_len_per_gpu=32000 \
            data.messages_key=messages"

        # Set model path for SFT training.
        # For epoch 2+, we will load the weights from the previous epoch's model
        # (which is provided by MODEL_FOR_GENERATION), but explicitly disable resume
        # of optimizer states or training progress.
        if [ "${RESUME_MODE}" == "resume_path" ] && [ -d "${RESUME_FROM_PATH}" ]; then
            # Although a previous checkpoint path is available,
            # we are explicitly disabling full resume (optimizer, etc.)
            # and only loading model weights from MODEL_FOR_GENERATION.
            echo "Starting SFT training with weights from ${MODEL_FOR_GENERATION} (resume disabled)."
        else
            echo "Starting SFT training from base model: ${MODEL_FOR_GENERATION} (resume disabled)."
        fi
        SFT_CMD="${SFT_CMD} model.path=\"${MODEL_FOR_GENERATION}\""
        SFT_CMD="${SFT_CMD} trainer.resume_mode=disable"

        SFT_CMD="${SFT_CMD} model.use_remove_padding=true \
            model.trust_remote_code=true \
            model.enable_gradient_checkpointing=true \
            engine=${BACKEND} \
            optim=${BACKEND} \
            optim.lr=2e-5 \
            optim.lr_warmup_steps_ratio=0.1 \
            optim.weight_decay=0.01 \
            optim.betas=\"[0.9,0.95]\" \
            optim.clip_grad=1.0 \
            optim.min_lr_ratio=0.1 \
            optim.warmup_style=cosine \
            engine.ulysses_sequence_parallel_size=${SP_SIZE} \
            engine.strategy=${FSDP_STRATEGY} \
            engine.fsdp_size=${FSDP_SIZE} \
            trainer.total_epochs=1 \
            trainer.logger=['console','wandb'] \
            trainer.project_name=\"policy_correction\" \
            trainer.experiment_name=\"${MODEL_NAME}_epoch${EPOCH}_sft_full_finetune\" \
            trainer.default_local_dir=\"${CKPT_HOME}\" \
            trainer.save_freq=100 \
            trainer.test_freq=-1 \
            trainer.max_ckpt_to_keep=3 \
            +trainer.checkpoint.save_contents='["model","optimizer","extra"]'"

        eval ${SFT_CMD}

        echo "Epoch ${EPOCH} Stage 3 SFT completed! Checkpoint: ${CKPT_HOME}"

        # Convert FSDP checkpoint to HuggingFace format for vLLM evaluation
        echo ""
        echo "[Converting FSDP checkpoint to HuggingFace format...]"

        # Find the latest checkpoint
        local LATEST_STEP_DIR=$(ls -td "${CKPT_HOME}"/global_step_* 2>/dev/null | head -1)

        if [ -z "${LATEST_STEP_DIR}" ]; then
            echo "WARNING: No checkpoint found in ${CKPT_HOME}"
            echo "This may happen if training didn't save any checkpoints (e.g., empty training loop)."
            echo "Attempting to use previous epoch's checkpoint as fallback..."

            if [ "${IS_FIRST_EPOCH}" == "false" ] && [ -d "${PREV_CKPT}" ]; then
                # Use previous epoch's checkpoint directory
                local PREV_LATEST_STEP=$(ls -td "${PREV_CKPT}"/global_step_* 2>/dev/null | head -1)
                if [ -n "${PREV_LATEST_STEP}" ]; then
                    echo "Creating symlink from ${PREV_LATEST_STEP} to ${CKPT_HOME}/"
                    mkdir -p "${CKPT_HOME}"
                    ln -sf "${PREV_LATEST_STEP}" "${CKPT_HOME}/global_step_$(basename ${PREV_LATEST_STEP} | grep -oP 'global_step_\K\d+')"
                    LATEST_STEP_DIR=$(ls -td "${CKPT_HOME}"/global_step_* 2>/dev/null | head -1)
                    echo "Using checkpoint: ${LATEST_STEP_DIR}"
                else
                    echo "ERROR: No checkpoint found in previous epoch ${PREV_CKPT}"
                    exit 1
                fi
            else
                echo "ERROR: No checkpoint available and no previous epoch to fallback to"
                exit 1
            fi
        fi

        # Output directory for merged HF model
        local HF_MERGED_DIR="${CKPT_HOME}/hf_merged"

        # Check if already converted
        if [ -f "${HF_MERGED_DIR}/model.safetensors" ] || [ -f "${HF_MERGED_DIR}/pytorch_model.bin" ]; then
            echo "HuggingFace model already exists at ${HF_MERGED_DIR}, skipping conversion."
        else
            echo "Merging FSDP checkpoint from ${LATEST_STEP_DIR} to ${HF_MERGED_DIR}..."
            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${LATEST_STEP_DIR}" \
                --target_dir "${HF_MERGED_DIR}"

            if [ $? -ne 0 ]; then
                echo "ERROR: model_merger conversion failed!"
                exit 1
            fi

            echo "✓ Checkpoint converted successfully!"
        fi
    fi

    ################################################################################
    # Stage 3 Evaluation
    ################################################################################

    # Check if Stage 3 Evaluation is already completed
    if [[ " ${COMPLETED_STAGES[@]} " =~ " stage3_eval " ]]; then
        echo ""
        echo "========================================"
        echo "EPOCH ${EPOCH} - STAGE 3 EVALUATION: SKIPPED (already completed)"
        echo "========================================"
        echo "Found existing evaluation results in: ${OUTPUT_BASE_DIR}/epoch${EPOCH}_stage3_eval/"
    else
        # Use the merged HuggingFace model for evaluation
        local HF_MERGED_DIR="${CKPT_HOME}/hf_merged"

        if [ ! -d "${HF_MERGED_DIR}" ]; then
            echo "ERROR: Merged HuggingFace model not found at ${HF_MERGED_DIR}"
            echo ""
            echo "This should have been created by the model_merger step after training."
            echo "Please ensure training completed successfully."
            exit 1
        fi

        # Verify the merged model has weights
        if ! ls "${HF_MERGED_DIR}"/model*.safetensors "${HF_MERGED_DIR}"/pytorch_model*.bin 2>/dev/null | grep -q .; then
            echo "ERROR: No model weights found in merged directory ${HF_MERGED_DIR}"
            ls -la "${HF_MERGED_DIR}"
            echo ""
            echo "The model_merger conversion may have failed."
            echo "To fix:"
            echo "  1. Delete the checkpoint: rm -rf ${CKPT_HOME}"
            echo "  2. Delete stage3 SFT marker: rm -f gen_results/epoch${EPOCH}/deepscaleR_stage3_sft.parquet"
            echo "  3. Re-run the script"
            exit 1
        fi

        echo "Using merged HuggingFace model for evaluation: ${HF_MERGED_DIR}"

        run_stage3_evaluation "${HF_MERGED_DIR}" "${EPOCH}"
        mark_stage_completed "${EPOCH}" "stage3_eval"
    fi

    # Aggregate all evaluation results for this epoch into a single JSON
    aggregate_epoch_results "${EPOCH}"

    echo ""
    echo "Epoch ${EPOCH} completed!"
}

# Function: Aggregate all evaluation results for an epoch into unified results.json
aggregate_epoch_results() {
    local EPOCH=$1

    echo ""
    echo "Aggregating results for Epoch ${EPOCH}..."

    # Results are already saved by main_eval.py to unified results.json
    # Just display summary here

    echo ""
    echo "Results Summary for Epoch ${EPOCH}:"

    python3 << EOF
import json
import os

results_file = "${OUTPUT_BASE_DIR}/results.json"
model_key = "${MODEL_NAME}_epoch${EPOCH}"

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)

    if model_key in data:
        print(f"\n  {model_key}:")
        results = data[model_key]
        for dataset_key in sorted(results.keys()):
            if dataset_key.endswith("_stats"):
                continue
            score = results[dataset_key]
            if isinstance(score, (int, float)):
                print(f"    {dataset_key}: {score:.2%}")
            else:
                print(f"    {dataset_key}: {score}")
    else:
        print(f"\n  No results found for {model_key}")
else:
    print(f"\n  Results file not found: {results_file}")
EOF
}

################################################################################
# Main Pipeline
################################################################################

echo "################################################################################"
echo "# Multi-Epoch Policy Correction Pipeline (Full Parameter Training - No LoRA)"
echo "# Total Epochs: ${NUM_EPOCHS}"
echo "# Base Model: ${MODEL_PATH}"
echo "# Auto-resume: Enabled (will skip completed stages)"
echo "################################################################################"

# Prepare eval datasets if needed
echo ""
echo "Preparing evaluation datasets..."
python3 recipe/open_math_reasoning/prepare_aime.py --local_dataset_path /data/data/jiangli/huggingface/datasets/ 2>/dev/null || echo "AIME datasets may already exist"
python3 recipe/open_math_reasoning/prepare_math500.py --local_dataset_path /data/data/jiangli/huggingface/datasets/ 2>/dev/null || echo "MATH500/AMC23 datasets may already exist"
python3 recipe/open_math_reasoning/prepare_hmmt.py --local_dataset_path /data/data/jiangli/huggingface/datasets/ 2>/dev/null || echo "HMMT datasets may already exist"
python3 recipe/open_math_reasoning/prepare_usamo.py --local_dataset_path /data/data/jiangli/huggingface/datasets/ 2>/dev/null || echo "USAMO datasets may already exist"

# Show current progress
show_progress

# Check if force rerun is enabled
if [ "${FORCE_RERUN}" == "true" ]; then
    echo ""
    echo "=========================================="
    echo "WARNING: FORCE_RERUN is enabled!"
    echo "All stages will be re-run regardless of existing files."
    echo "=========================================="
    echo ""
fi

# Run epochs
for EPOCH in $(seq 1 ${NUM_EPOCHS}); do
    echo ""
    echo "=========================================="
    echo "Checking Epoch ${EPOCH} status..."
    echo "=========================================="

    # Check if this epoch is fully completed
    COMPLETED_STAGES=($(detect_epoch_progress ${EPOCH}))
    NUM_COMPLETED=${#COMPLETED_STAGES[@]}

    # If all 4 stages are completed (stage1, stage2, stage3_sft, stage3_eval), skip this epoch
    if [ ${NUM_COMPLETED} -eq 4 ]; then
        echo "Epoch ${EPOCH} is fully completed (all 4 stages done). Skipping..."
        echo "Completed stages: ${COMPLETED_STAGES[@]}"
        echo "To re-run this epoch, delete the corresponding output files:"
        echo "  - ${OUTPUT_BASE_DIR}/epoch${EPOCH}/"
        echo "  - gen_results/epoch${EPOCH}/"
        echo "  - ${CKPT_HOME_BASE}_epoch${EPOCH}/"
        echo "  - ${OUTPUT_BASE_DIR}/epoch${EPOCH}_stage3_eval/"
        continue
    fi

    echo "Epoch ${EPOCH} is incomplete (${NUM_COMPLETED}/4 stages completed). Proceeding..."
    echo "Completed stages: ${COMPLETED_STAGES[@]:-none}"

    if [ ${EPOCH} -eq 1 ]; then
        # First epoch: use base model
        run_epoch ${EPOCH} "${MODEL_PATH}" "true"
    else
        # Subsequent epochs: use previous epoch's checkpoint
        PREV_EPOCH=$((EPOCH - 1))
        PREV_CKPT="${CKPT_HOME_BASE}_epoch${PREV_EPOCH}"

        # Verify previous epoch's checkpoint exists
        if [ ! -d "${PREV_CKPT}" ]; then
            echo "ERROR: Previous epoch's checkpoint not found at ${PREV_CKPT}"
            echo "Cannot proceed with Epoch ${EPOCH}. Please complete Epoch ${PREV_EPOCH} first."
            exit 1
        fi

        # Use the merged HuggingFace model for generation (same as Stage 3 Evaluation)
        PREV_HF_MERGED="${PREV_CKPT}/hf_merged"
        if [ ! -d "${PREV_HF_MERGED}" ]; then
            echo "ERROR: Merged HuggingFace model not found at ${PREV_HF_MERGED}"
            echo "Please ensure Epoch ${PREV_EPOCH} Stage 3 SFT completed successfully."
            exit 1
        fi

        run_epoch ${EPOCH} "${PREV_HF_MERGED}" "false"
    fi
done
