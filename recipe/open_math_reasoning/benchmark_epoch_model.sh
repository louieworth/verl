#!/usr/bin/env bash
#
# Benchmark Epoch Model on Math Datasets (AIME24, AIME25, MATH500, HMMT25)
# Results are saved to results/${MODEL_NAME}/${DATASET}_results.json
#
# Usage:
#   bash benchmark_epoch_model.sh <epoch_number>
#   # Example: bash benchmark_epoch_model.sh 1
#
# Environment Variables:
#   MODEL_PATH     - Path to model (default: /data/data/jiangli/ckpt/${BASE_MODEL_NAME}_epoch${EPOCH}/hf_merged)
#   DATASETS       - Datasets to test (default: "aime24 aime25 math500 hmmt25")
#   PASS_K_VALUES  - Pass@k values (default: "1")
#
# Examples:
#   # Epoch 1 checkpoint
#   bash benchmark_epoch_model.sh 1
#
#   # Specific datasets with pass@1 and pass@32
#   DATASETS="aime24 aime25" PASS_K_VALUES="1 32" bash benchmark_epoch_model.sh 1
#

set -e

################################################################################
# Configuration
################################################################################

# Epoch number (required, can be set as environment variable or first argument)
EPOCH=${EPOCH:-$1}
if [ -z "${EPOCH}" ]; then
    echo "ERROR: Please specify epoch number"
    echo "Usage: bash benchmark_epoch_model.sh <epoch_number>"
    echo "Example: bash benchmark_epoch_model.sh 1"
    exit 1
fi

# Base model name
BASE_MODEL_NAME="Qwen3-4B-Instruct-2507"

# Model path - epoch checkpoint
DEFAULT_CKPT_BASE="/data/data/jiangli/ckpt/${BASE_MODEL_NAME}_epoch${EPOCH}"
MODEL_PATH=${MODEL_PATH:-"${DEFAULT_CKPT_BASE}/hf_merged"}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

# Model name for results
MODEL_NAME="${BASE_MODEL_NAME}_epoch${EPOCH}"

# GPU settings
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}

# Output directory - same format as benchmark.sh
OUTPUT_DIR="results/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Eval datasets (same as benchmark.sh)
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}
declare -A DATASETS=(
    ["aime24"]="${EVAL_DATASETS_DIR}/aime24/aime24_test.parquet"
    ["aime25"]="${EVAL_DATASETS_DIR}/aime25/aime25_test.parquet"
    ["math500"]="${EVAL_DATASETS_DIR}/math500/math500_test.parquet"
    ["hmmt25"]="${EVAL_DATASETS_DIR}/hmmt25/hmmt25_test.parquet"
)

# Datasets to test (can be overridden by DATASETS env var)
DEFAULT_DATASETS="aime24 aime25 math500 hmmt25"
DATASETS_TO_TEST=${DATASETS:-"${DEFAULT_DATASETS}"}

# Pass@k values to test (space-separated, e.g., "1 32")
PASS_K_VALUES=${PASS_K_VALUES:-"1"}

################################################################################
# Function: Run generation and evaluation for a dataset
################################################################################

benchmark_dataset() {
    local DATASET_NAME=$1
    local DATASET_PATH=$2
    local CURRENT_PASS_K=$3

    echo ""
    echo "=========================================="
    echo "Benchmarking on ${DATASET_NAME} with pass@${CURRENT_PASS_K}"
    echo "=========================================="
    echo "Model: ${MODEL_PATH}"
    echo "Dataset: ${DATASET_PATH}"
    echo ""

    if [ ! -f "${DATASET_PATH}" ]; then
        echo "WARNING: Dataset not found at ${DATASET_PATH}, skipping..."
        return
    fi

    # Generation output
    local GEN_OUTPUT="${OUTPUT_DIR}/${DATASET_NAME}_pass${CURRENT_PASS_K}_generation.parquet"

    # Generate responses (same parameters as benchmark.sh)
    echo "[1/2] Generating responses..."
    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="${NNODES}" \
        trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.trust_remote_code=true \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.prompt_length=4096 \
        actor_rollout_ref.rollout.response_length=38912 \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=${CURRENT_PASS_K} \
        data.train_files="['${DATASET_PATH}']" \
        data.prompt_key=prompt \
        +data.output_path="${GEN_OUTPUT}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Generation failed for ${DATASET_NAME}"
        return 1
    fi

    # Evaluate - save to per-dataset results file (same as benchmark.sh)
    echo "[2/2] Evaluating responses..."
    python3 -m verl.trainer.main_eval \
        data.path="${GEN_OUTPUT}" \
        custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
        custom_reward_function.name=compute_score_data_source \
        +output_json_path="${OUTPUT_DIR}/${DATASET_NAME}_results.json" \
        +model_name="${MODEL_NAME}" \
        +pass_k=${CURRENT_PASS_K}

    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for ${DATASET_NAME}"
        return 1
    fi

    echo "✓ ${DATASET_NAME} pass@${CURRENT_PASS_K} benchmark completed!"
    echo "  Results saved to: ${OUTPUT_DIR}/${DATASET_NAME}_results.json"
}

################################################################################
# Main
################################################################################

echo "################################################################################"
echo "# Epoch ${EPOCH} Model Benchmark"
echo "# Model: ${MODEL_PATH}"
echo "# Model Name: ${MODEL_NAME}"
echo "# Datasets: ${DATASETS_TO_TEST}"
echo "# Pass@k values: ${PASS_K_VALUES}"
echo "# Output: ${OUTPUT_DIR}"
echo "################################################################################"

# Loop over each pass@k value
for CURRENT_PASS_K in ${PASS_K_VALUES}; do
    echo ""
    echo "=========================================="
    echo "Testing with pass@${CURRENT_PASS_K}"
    echo "=========================================="

    # Run benchmarks for each dataset
    for DATASET_NAME in ${DATASETS_TO_TEST}; do
        DATASET_PATH="${DATASETS[$DATASET_NAME]}"
        if [ -n "${DATASET_PATH}" ]; then
            benchmark_dataset "${DATASET_NAME}" "${DATASET_PATH}" "${CURRENT_PASS_K}"
        else
            echo "WARNING: Unknown dataset '${DATASET_NAME}', skipping..."
        fi
    done
done

# Display summary
echo ""
echo "=========================================="
echo "Benchmark Summary - Epoch ${EPOCH}"
echo "=========================================="
echo ""

python3 << EOF
import json
import os

output_dir = "${OUTPUT_DIR}"
model_name = "${MODEL_NAME}"
dataset_names = "${DATASETS_TO_TEST}".split()

print(f"Results for {model_name}:")
for dataset_name in dataset_names:
    result_file = f"{output_dir}/{dataset_name}_results.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
        # Extract score (same format as benchmark.sh)
        for model_key, scores in data.items():
            if model_name in model_key:
                for key, value in scores.items():
                    if '_generation_pass_' in key:
                        if isinstance(value, (int, float)):
                            print(f"  {dataset_name}: {value:.2%}")
                        else:
                            print(f"  {dataset_name}: {value}")
                        break
                break
    else:
        print(f"  {dataset_name}: (not found)")
EOF

echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
