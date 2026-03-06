#!/usr/bin/env bash
#
# Ablation Study: Train only on successfully corrected samples
# Purpose: Compare training on all corrections vs only successful corrections
#
# This script:
# 1. Evaluates stage2_correction responses to find which corrections are correct
# 2. Filters to only reward=1 (successfully corrected)
# 3. Trains SFT on this filtered dataset
# 4. Evaluates on benchmark datasets
#
# Usage:
#   bash recipe/open_math_reasoning/ablation_stage3_only_corrected.sh

set -e  # Exit on error

################################################################################
# Configuration
################################################################################

# Model and GPU settings
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}

# Model paths
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
MODEL_NAME=$(basename "${MODEL_PATH}")

# Data paths
# Stage2 correction output (from Epoch 1) - contains corrections that need evaluation
STAGE2_CORRECTION="results/${MODEL_NAME}/epoch1/stage2_correction.parquet"

# Output directories
OUTPUT_BASE_DIR="results/${MODEL_NAME}"
ABLATION_DIR="${OUTPUT_BASE_DIR}/ablation_only_corrected"
mkdir -p "${ABLATION_DIR}"

# SFT settings
BACKEND=${BACKEND:-fsdp}
CKPT_HOME="${ABLATION_DIR}/sft_checkpoint"
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-4}
FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}

# Eval datasets
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}
AIME24_PATH="${EVAL_DATASETS_DIR}/aime24/aime24_test.parquet"
AIME25_PATH="${EVAL_DATASETS_DIR}/aime25/aime25_test.parquet"
AMC23_PATH="${EVAL_DATASETS_DIR}/amc23/amc23_test.parquet"
MATH500_PATH="${EVAL_DATASETS_DIR}/math500/math500_test.parquet"
HMMT25_PATH="${EVAL_DATASETS_DIR}/hmmt25/hmmt25_test.parquet"
HMMT24_PATH="${EVAL_DATASETS_DIR}/hmmt24/hmmt24_test.parquet"

PASS_K=${PASS_K:-1}

################################################################################
# Helper Functions
################################################################################

echo ""
echo "================================================================================"
echo "ABLATION STUDY: Train Only on Successfully Corrected Samples"
echo "================================================================================"
echo ""
echo "Stage2 Correction File: ${STAGE2_CORRECTION}"
echo "Output Directory: ${ABLATION_DIR}"
echo ""

################################################################################
# Step 1: Evaluate stage2 corrections and filter to reward=1
################################################################################

echo ""
echo "================================================================================"
echo "Step 1: Evaluate Stage2 Corrections & Filter to reward=1"
echo "================================================================================"
echo ""

# First, evaluate the stage2 corrections to get rewards
echo "Evaluating stage2 correction responses..."
python3 -m verl.trainer.main_eval \
    data.path="${STAGE2_CORRECTION}" \
    data.prompt_key=prompt \
    custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
    custom_reward_function.name=compute_score_data_source \
    +output_json_path="${ABLATION_DIR}/stage2_eval.json"

if [ $? -ne 0 ]; then
    echo "Error: Failed to evaluate stage2 corrections"
    exit 1
fi

# Now prepare filtered dataset
echo ""
echo "Preparing filtered dataset (only reward=1)..."

python3 << EOF
import pandas as pd
import datasets
import json
import numpy as np

stage2_path = "${STAGE2_CORRECTION}"
eval_result_path = "${ABLATION_DIR}/stage2_eval.json"
output_path = "${ABLATION_DIR}/sft_only_corrected.parquet"

print(f"Loading stage2 correction data: {stage2_path}")
stage2_df = pd.read_parquet(stage2_path)
print(f"  Total samples: {len(stage2_df)}")

# Load evaluation results
print(f"\nLoading evaluation results: {eval_result_path}")
with open(eval_result_path, 'r') as f:
    eval_results = json.load(f)

# Extract rewards from evaluation results
# The eval results have model_key -> dataset_key -> score
# We need to map each sample to its reward

# Since main_eval writes aggregated results, we need to re-evaluate per-sample
# Let's use the reward computation directly

from verl.trainer.ppo.reward import get_custom_reward_fn
import hydra
from omegaconf import OmegaConf

# Create config for reward function
config = {
    'custom_reward_function': {
        'path': 'recipe/open_math_reasoning/compute_score.py',
        'name': 'compute_score_data_source'
    }
}

# Get reward function
reward_fn = get_custom_reward_fn(config)

# Evaluate each sample and collect rewards
rewards = []
for idx in range(len(stage2_df)):
    data_source = stage2_df.iloc[idx]['data_source']
    responses = stage2_df.iloc[idx]['responses']
    response_text = responses[0] if isinstance(responses, (list, np.ndarray)) and len(responses) > 0 else ""

    # Get ground truth from extra_info
    extra_info = stage2_df.iloc[idx].get('extra_info', {})
    if isinstance(extra_info, dict):
        ground_truth = extra_info.get('answer', '')
    else:
        ground_truth = ''

    # Compute reward
    reward = reward_fn(data_source, response_text, ground_truth)
    rewards.append(reward)

# Now filter to reward=1
filtered_indices = [i for i, r in enumerate(rewards) if r == 1.0]

print(f"\nReward distribution:")
print(f"  Reward=1 (successfully corrected): {len(filtered_indices)}")
print(f"  Reward=0 (still incorrect): {len(rewards) - len(filtered_indices)}")
print(f"  Success rate: {100 * len(filtered_indices) / len(rewards):.1f}%")

# Create filtered dataset
filtered_df = stage2_df.iloc[filtered_indices].copy()

# Convert to SFT format
def to_sft_format(example):
    """Convert example to SFT message format."""
    responses_val = example.get("responses", [""])
    response_text = responses_val[0] if isinstance(responses_val, (list, np.ndarray)) else responses_val

    # Extract prompt content
    prompt_val = example.get("prompt", [])
    if isinstance(prompt_val, (list, np.ndarray)) and len(prompt_val) > 0:
        prompt_dict = prompt_val[0]
        if isinstance(prompt_dict, dict):
            prompt_content = prompt_dict.get('content', '')
        else:
            prompt_content = str(prompt_dict)
    else:
        prompt_content = str(prompt_val)

    return {
        "messages": [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": response_text}
        ]
    }

# Convert to SFT format
ds = datasets.Dataset.from_pandas(filtered_df)
ds_sft = ds.map(to_sft_format, remove_columns=ds.column_names)

# Save
print(f"\nSaving filtered SFT dataset to: {output_path}")
ds_sft.to_parquet(output_path)
print(f"  Saved {len(ds_sft)} samples")

print(f"\n{'='*80}")
print(f"Summary: Filtered to {len(ds_sft)} successfully corrected samples")
print(f"{'='*80}")

EOF

if [ $? -ne 0 ]; then
    echo "Error: Failed to prepare filtered dataset"
    exit 1
fi

################################################################################
# Step 2: Run SFT Training
################################################################################

echo ""
echo "================================================================================"
echo "Step 2: Run SFT Training"
echo "================================================================================"
echo ""

SFT_DATASET="${ABLATION_DIR}/sft_only_corrected.parquet"

echo "Training on: ${SFT_DATASET}"
echo "Checkpoint will be saved to: ${CKPT_HOME}"
echo ""

# Build SFT command
SFT_CMD="torchrun --standalone --nnodes=1 --nproc-per-node=${NGPUS_PER_NODE} \
    -m verl.trainer.sft_trainer \
    data.train_files=\"${SFT_DATASET}\" \
    data.train_batch_size=96 \
    data.max_length=16000 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu=32000 \
    data.messages_key=messages \
    model.path=\"${MODEL_PATH}\" \
    model.use_remove_padding=true \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.lora_rank=64 \
    model.lora_alpha=128 \
    model.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
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
    trainer.resume_mode=disable \
    trainer.logger=['console','wandb'] \
    trainer.project_name=\"ablation_only_corrected\" \
    trainer.experiment_name=\"${MODEL_NAME}_only_corrected\" \
    trainer.default_local_dir=\"${CKPT_HOME}\" \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=3 \
    +trainer.checkpoint.save_contents='[\"model\",\"optimizer\",\"extra\"]'"

echo "Running SFT training..."
eval ${SFT_CMD}

if [ $? -ne 0 ]; then
    echo "Error: SFT training failed"
    exit 1
fi

echo ""
echo "✓ SFT training completed!"

################################################################################
# Step 3: Convert FSDP checkpoint to HuggingFace format
################################################################################

echo ""
echo "================================================================================"
echo "Step 3: Convert Checkpoint to HuggingFace Format"
echo "================================================================================"
echo ""

# Find the latest checkpoint
LATEST_STEP_DIR=$(ls -td "${CKPT_HOME}"/global_step_* 2>/dev/null | head -1)

if [ -z "${LATEST_STEP_DIR}" ]; then
    echo "Error: No checkpoint found in ${CKPT_HOME}"
    exit 1
fi

echo "Latest checkpoint: ${LATEST_STEP_DIR}"

# Output directory for merged HF model
HF_MERGED_DIR="${CKPT_HOME}/hf_merged"

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
        echo "Error: model_merger conversion failed!"
        exit 1
    fi

    echo "✓ Checkpoint converted successfully!"
fi

################################################################################
# Step 4: Evaluate on Benchmarks
################################################################################

echo ""
echo "================================================================================"
echo "Step 4: Evaluate on Benchmark Datasets"
echo "================================================================================"
echo ""

EVAL_OUTPUT_DIR="${ABLATION_DIR}/evaluation"
mkdir -p "${EVAL_OUTPUT_DIR}"

echo "Model to evaluate: ${HF_MERGED_DIR}"
echo "Output directory: ${EVAL_OUTPUT_DIR}"
echo ""

# Array of eval datasets
declare -A EVAL_DATASETS=(
    ["aime24"]="${AIME24_PATH}"
    ["aime25"]="${AIME25_PATH}"
    ["amc23"]="${AMC23_PATH}"
    ["math500"]="${MATH500_PATH}"
    ["hmmt25"]="${HMMT25_PATH}"
    ["hmmt24"]="${HMMT24_PATH}"
)

# Initialize results.json
RESULT_FILE="${ABLATION_DIR}/results.json"
echo '{}' > "${RESULT_FILE}"

for DATASET_NAME in aime24 aime25 amc23 math500 hmmt25 hmmt24; do
    DATASET_PATH="${EVAL_DATASETS[$DATASET_NAME]}"

    if [ ! -f "${DATASET_PATH}" ]; then
        echo "Warning: ${DATASET_NAME} not found at ${DATASET_PATH}, skipping..."
        continue
    fi

    echo ""
    echo "[Evaluating on ${DATASET_NAME}]"

    GEN_OUTPUT="${EVAL_OUTPUT_DIR}/${DATASET_NAME}_pass${PASS_K}_generation.parquet"

    # Generate responses
    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="${NNODES}" \
        trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
        actor_rollout_ref.model.path="${HF_MERGED_DIR}" \
        actor_rollout_ref.model.trust_remote_code=true \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.prompt_length=4096 \
        actor_rollout_ref.rollout.response_length=32768 \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n="${PASS_K}" \
        data.train_files="['${DATASET_PATH}']" \
        data.prompt_key=prompt \
        +data.output_path="${GEN_OUTPUT}"

    if [ $? -ne 0 ]; then
        echo "Warning: Generation failed for ${DATASET_NAME}"
        continue
    fi

    # Evaluate
    python3 -m verl.trainer.main_eval \
        data.path="${GEN_OUTPUT}" \
        data.prompt_key=prompt \
        custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
        custom_reward_function.name=compute_score_data_source \
        +output_json_path="${RESULT_FILE}" \
        +model_name="${MODEL_NAME}_only_corrected" \
        +pass_k=${PASS_K}

    echo "${DATASET_NAME} evaluation completed!"
done

################################################################################
# Step 5: Summary
################################################################################

echo ""
echo "================================================================================"
echo "ABLATION STUDY COMPLETED"
echo "================================================================================"
echo ""
echo "Training Configuration:"
echo "  - Filtered dataset: Only reward=1 (successfully corrected)"
SAMPLE_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${ABLATION_DIR}/sft_only_corrected.parquet'))))" 2>/dev/null || echo "N/A")
echo "  - Sample count: ${SAMPLE_COUNT}"
echo "  - Checkpoint: ${HF_MERGED_DIR}"
echo ""
echo "Results saved to: ${RESULT_FILE}"
echo ""
echo "To compare with baseline (all corrections):"
echo "  Baseline: results/${MODEL_NAME}/results.json (model: ${MODEL_NAME}_epoch1)"
echo "  This run: ${RESULT_FILE} (model: ${MODEL_NAME}_only_corrected)"
echo ""
