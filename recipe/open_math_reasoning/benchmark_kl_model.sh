#!/usr/bin/env bash
#
# Benchmark KL Training Model on Math Datasets
# Results are saved to results/${BASE_MODEL_NAME}/results.json (same format as run_full_pipeline_multi_epoch.sh)
#
# Usage:
#   bash benchmark_kl_model.sh <model_path>
#
# Examples:
#   bash benchmark_kl_model.sh /data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/hf_merged
#
#   DATASETS="aime24 aime25 math500" bash benchmark_kl_model.sh /path/to/model
#

set -e

################################################################################
# Configuration
################################################################################

# Model path (required)
MODEL_PATH=${MODEL_PATH:-$1}
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: Please specify model path"
    echo "Usage: bash benchmark_kl_model.sh <model_path>"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

# Extract model names
# e.g., /data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/hf_merged
# -> BASE_MODEL_NAME = Qwen3-1.7B
# -> MODEL_NAME = Qwen3-1.7B_kl_reverse_monte_carlo
FULL_MODEL_DIR=$(dirname "${MODEL_PATH}")
FULL_MODEL_NAME=$(basename "${FULL_MODEL_DIR}")
BASE_MODEL_NAME=$(echo "${FULL_MODEL_NAME}" | cut -d'_' -f1)
MODEL_NAME="${FULL_MODEL_NAME}"

# GPU settings
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}

# Temporary output directory (will be deleted after evaluation)
GEN_OUTPUT_DIR="gen_results/${FULL_MODEL_NAME}/evaluate"
mkdir -p "${GEN_OUTPUT_DIR}"

# Eval datasets
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}
declare -A DATASETS=(
    ["aime24"]="${EVAL_DATASETS_DIR}/aime24/aime24_test.parquet"
    ["aime25"]="${EVAL_DATASETS_DIR}/aime25/aime25_test.parquet"
    ["math500"]="${EVAL_DATASETS_DIR}/math500/math500_test.parquet"
    ["hmmt25"]="${EVAL_DATASETS_DIR}/hmmt25/hmmt25_test.parquet"
)

# Datasets to test
DEFAULT_DATASETS="aime24 aime25 math500"
DATASETS_TO_TEST=${DATASETS:-"${DEFAULT_DATASETS}"}

# Pass@k
PASS_K=${PASS_K:-1}

# VERL root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

################################################################################
# Main
################################################################################

echo "################################################################################"
echo "# KL Model Benchmark"
echo "# Model: ${MODEL_PATH}"
echo "# Model Name: ${MODEL_NAME}"
echo "# Datasets: ${DATASETS_TO_TEST}"
echo "################################################################################"

# Results file path (same format as run_full_pipeline_multi_epoch.sh)
RESULTS_FILE="results/${BASE_MODEL_NAME}/results.json"
mkdir -p "results/${BASE_MODEL_NAME}"

# Load existing results if file exists
if [ -f "${RESULTS_FILE}" ]; then
    echo "Loading existing results from: ${RESULTS_FILE}"
fi

# Run evaluation for each dataset
for DATASET_NAME in ${DATASETS_TO_TEST}; do
    DATASET_PATH="${DATASETS[$DATASET_NAME]}"

    if [ -z "${DATASET_PATH}" ]; then
        echo "WARNING: Unknown dataset '${DATASET_NAME}', skipping..."
        continue
    fi

    if [ ! -f "${DATASET_PATH}" ]; then
        echo "WARNING: Dataset not found at ${DATASET_PATH}, skipping..."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating on ${DATASET_NAME}"
    echo "=========================================="

    # Generation output
    GEN_OUTPUT="${GEN_OUTPUT_DIR}/${DATASET_NAME}_pass${PASS_K}_generation.parquet"

    # Generate responses
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
        actor_rollout_ref.rollout.n=${PASS_K} \
        data.train_files="['${DATASET_PATH}']" \
        data.prompt_key=prompt \
        +data.output_path="${GEN_OUTPUT}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Generation failed for ${DATASET_NAME}"
        continue
    fi

    # Evaluate and save to results.json
    echo "[2/2] Evaluating responses..."
    python3 -m verl.trainer.main_eval \
        data.path="${GEN_OUTPUT}" \
        custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
        custom_reward_function.name=compute_score_data_source \
        +output_json_path="${RESULTS_FILE}" \
        +model_name="${MODEL_NAME}" \
        +pass_k=${PASS_K}

    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for ${DATASET_NAME}"
        continue
    fi

    echo "✓ ${DATASET_NAME} completed!"
done

# Display summary
echo ""
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="
echo ""

python3 << EOF
import json
import os

results_file = "${RESULTS_FILE}"
model_name = "${MODEL_NAME}"
dataset_names = "${DATASETS_TO_TEST}".split()

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)

    if model_name in data:
        print(f"Results for {model_name}:")
        for key, value in data[model_name].items():
            if '_generation_pass_' in key:
                # Extract dataset name from key
                dataset = key.split('_pass')[0]
                if isinstance(value, (int, float)):
                    print(f"  {dataset}: {value:.2%}")
                else:
                    print(f"  {dataset}: {value}")
    else:
        print(f"No results found for {model_name}")
else:
    print(f"Results file not found: {results_file}")
EOF

echo ""
echo "Results saved to:"
echo "  Accuracy: ${RESULTS_FILE}"
echo "  Generation files: ${GEN_OUTPUT_DIR}/"
echo ""
