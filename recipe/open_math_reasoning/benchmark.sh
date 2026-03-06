#!/usr/bin/env bash
#
# Benchmark Qwen3-1.7B Base Model on Math Datasets
# Tests performance on AIME24, AIME25, AMC23, MATH500, HMMT25, HMMT24
#
# Usage:
#   bash benchmark_qwen3_1.7b.sh

set -e

################################################################################
# Configuration
################################################################################

# Model settings
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B-Instruct-2507"}
MODEL_NAME=$(basename "${MODEL_PATH}")

# GPU settings
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}

# Output directory - fixed path without timestamp
OUTPUT_DIR="results/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Unified results file (same structure as run_full_pipeline_multi_epoch.sh)
RESULT_JSON="${OUTPUT_DIR}/results.json"

# Eval datasets
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}
declare -A DATASETS=(
    ["aime24"]="${EVAL_DATASETS_DIR}/aime24/aime24_test.parquet"
    ["aime25"]="${EVAL_DATASETS_DIR}/aime25/aime25_test.parquet"
    ["amc23"]="${EVAL_DATASETS_DIR}/amc23/amc23_test.parquet"
    ["math500"]="${EVAL_DATASETS_DIR}/math500/math500_test.parquet"
    ["hmmt25"]="${EVAL_DATASETS_DIR}/hmmt25/hmmt25_test.parquet"
    ["hmmt24"]="${EVAL_DATASETS_DIR}/hmmt24/hmmt24_test.parquet"
)

################################################################################
# Function: Run generation and evaluation for a dataset
################################################################################

benchmark_dataset() {
    local DATASET_NAME=$1
    local DATASET_PATH=$2

    echo ""
    echo "=========================================="
    echo "Benchmarking on ${DATASET_NAME}"
    echo "=========================================="
    echo "Model: ${MODEL_PATH}"
    echo "Dataset: ${DATASET_PATH}"
    echo ""

    if [ ! -f "${DATASET_PATH}" ]; then
        echo "WARNING: Dataset not found at ${DATASET_PATH}, skipping..."
        return
    fi

    # Generation output
    local GEN_OUTPUT="${OUTPUT_DIR}/${DATASET_NAME}_generation.parquet"

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
        actor_rollout_ref.rollout.response_length=38912  \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=1 \
        data.train_files="['${DATASET_PATH}']" \
        data.prompt_key=prompt \
        +data.output_path="${GEN_OUTPUT}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Generation failed for ${DATASET_NAME}"
        return 1
    fi

    # Evaluate
    echo "[2/2] Evaluating responses..."
    python3 -m verl.trainer.main_eval \
        data.path="${GEN_OUTPUT}" \
        custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
        custom_reward_function.name=compute_score_data_source \
        +output_json_path="${RESULT_JSON}" \
        +model_name="${MODEL_NAME}" \
        +pass_k=1

    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for ${DATASET_NAME}"
        return 1
    fi

    echo "✓ ${DATASET_NAME} benchmark completed!"
    echo "  Results saved to: ${RESULT_JSON}"
}

################################################################################
# Main
################################################################################

echo "################################################################################"
echo "# Qwen3-1.7B Base Model Benchmark"
echo "# Model: ${MODEL_PATH}"
echo "# Output: ${OUTPUT_DIR}"
echo "################################################################################"

# Run benchmarks for each dataset
for DATASET_NAME in aime24 aime25 math500 hmmt25; do
    DATASET_PATH="${DATASETS[$DATASET_NAME]}"
    benchmark_dataset "${DATASET_NAME}" "${DATASET_PATH}"
done

echo ""
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="
echo ""
echo "Results file: ${RESULT_JSON}"
echo ""
echo "To view results:"
echo "  cat ${RESULT_JSON}"
echo ""
echo "To view summary:"
echo "  python3 -c \"import json; r=json.load(open('${RESULT_JSON}')); m=r.get('${MODEL_NAME}', {}); [print(f'{k}: {v:.2%}') for k,v in sorted(m.items()) if isinstance(v, (int, float))]\""
