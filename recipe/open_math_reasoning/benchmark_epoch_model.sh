#!/usr/bin/env bash
#
# Benchmark checkpoint models on math datasets
# (AIME24, AIME25, MATH500, HMMT25, BeyondAIME, AMO-Bench, GSM8K)
# Results are saved to results/${MODEL_NAME}/${DATASET}_results.json
#
# Usage:
#   bash benchmark_epoch_model.sh <epoch_or_checkpoint_path> [more...]
#   # Example: bash benchmark_epoch_model.sh 1
#
# Environment Variables:
#   MODEL_PATH     - Single model path or checkpoint root
#   MODEL_PATHS    - Space-separated model paths or epoch numbers
#   EPOCH / EPOCHS - Alternative single or multiple epoch numbers
#   DATASETS       - Datasets to test
#                    (default: "aime24 aime25 math500 hmmt25 beyondaime amobench gsm8k")
#   PASS_K_VALUES  - Pass@k values (default: "1")
#
# Examples:
#   # Epoch 1 checkpoint
#   bash benchmark_epoch_model.sh 1
#
#   # Epoch 1 and epoch 2 checkpoints
#   bash benchmark_epoch_model.sh 1 2
#
#   # Explicit checkpoint directories
#   bash benchmark_epoch_model.sh \
#       /data/data/jiangli/ckpt/Qwen3-4B-Instruct-2507_epoch1 \
#       /data/data/jiangli/ckpt/Qwen3-4B-Instruct-2507_epoch2
#
#   # Specific datasets with pass@1 and pass@32
#   DATASETS="beyondaime amobench gsm8k" PASS_K_VALUES="1 32" bash benchmark_epoch_model.sh 1
#

set -e

################################################################################
# Configuration
################################################################################

# Base model name
BASE_MODEL_NAME="Qwen3-4B-Instruct-2507"

# Model specs to benchmark
MODEL_SPECS=()
if [ $# -gt 0 ]; then
    MODEL_SPECS=("$@")
elif [ -n "${MODEL_PATHS:-}" ]; then
    read -r -a MODEL_SPECS <<< "${MODEL_PATHS}"
elif [ -n "${MODEL_PATH:-}" ]; then
    MODEL_SPECS=("${MODEL_PATH}")
elif [ -n "${EPOCHS:-}" ]; then
    read -r -a MODEL_SPECS <<< "${EPOCHS}"
elif [ -n "${EPOCH:-}" ]; then
    MODEL_SPECS=("${EPOCH}")
fi

if [ ${#MODEL_SPECS[@]} -eq 0 ]; then
    echo "ERROR: Please specify at least one epoch number or checkpoint path"
    echo "Usage: bash benchmark_epoch_model.sh <epoch_or_checkpoint_path> [more...]"
    echo "Example: bash benchmark_epoch_model.sh 1 2"
    exit 1
fi

# GPU settings
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}

# Eval datasets (same as benchmark.sh)
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}
declare -A DATASET_PATHS=(
    ["aime24"]="${EVAL_DATASETS_DIR}/aime24/aime24_test.parquet"
    ["aime25"]="${EVAL_DATASETS_DIR}/aime25/aime25_test.parquet"
    ["math500"]="${EVAL_DATASETS_DIR}/math500/math500_test.parquet"
    ["hmmt25"]="${EVAL_DATASETS_DIR}/hmmt25/hmmt25_test.parquet"
    ["beyondaime"]="${EVAL_DATASETS_DIR}/beyondaime/beyondaime_test.parquet"
    ["amobench"]="${EVAL_DATASETS_DIR}/amobench/amobench_test.parquet"
    ["gsm8k"]="${EVAL_DATASETS_DIR}/gsm8k/gsm8k_test.parquet"
)

# Datasets to test (can be overridden by DATASETS env var)
DEFAULT_DATASETS="beyondaime amobench gsm8k"
DATASETS_TO_TEST=${DATASETS:-"${DEFAULT_DATASETS}"}

# Pass@k values to test (space-separated, e.g., "1 32")
PASS_K_VALUES=${PASS_K_VALUES:-"1"}

dir_has_model_artifacts() {
    local MODEL_DIR=$1

    if [ ! -d "${MODEL_DIR}" ]; then
        return 1
    fi

    if [ -f "${MODEL_DIR}/config.json" ]; then
        return 0
    fi

    if compgen -G "${MODEL_DIR}/*.safetensors" > /dev/null; then
        return 0
    fi

    if compgen -G "${MODEL_DIR}/*.bin" > /dev/null; then
        return 0
    fi

    return 1
}

resolve_model_spec() {
    local MODEL_SPEC=$1
    local CKPT_BASE=""
    local MODEL_PATH=""
    local MODEL_NAME=""
    local LATEST_STEP=""

    if [[ "${MODEL_SPEC}" =~ ^[0-9]+$ ]]; then
        CKPT_BASE="/data/data/jiangli/ckpt/${BASE_MODEL_NAME}_epoch${MODEL_SPEC}"
        MODEL_NAME="${BASE_MODEL_NAME}_epoch${MODEL_SPEC}"
    else
        CKPT_BASE="${MODEL_SPEC%/}"
        case "$(basename "${CKPT_BASE}")" in
            hf_merged|huggingface)
                MODEL_NAME="$(basename "$(dirname "${CKPT_BASE}")")"
                ;;
            *)
                MODEL_NAME="$(basename "${CKPT_BASE}")"
                ;;
        esac
    fi

    if [ -z "${MODEL_PATH}" ] && dir_has_model_artifacts "${CKPT_BASE}/hf_merged"; then
        MODEL_PATH="${CKPT_BASE}/hf_merged"
    fi

    if [ -z "${MODEL_PATH}" ] && dir_has_model_artifacts "${CKPT_BASE}"; then
        MODEL_PATH="${CKPT_BASE}"
    fi

    if [ -z "${MODEL_PATH}" ]; then
        LATEST_STEP=$(ls -td "${CKPT_BASE}"/global_step_* 2>/dev/null | head -1 || true)
        if [ -n "${LATEST_STEP}" ] && dir_has_model_artifacts "${LATEST_STEP}/huggingface"; then
            MODEL_PATH="${LATEST_STEP}/huggingface"
        elif [ -n "${LATEST_STEP}" ] && dir_has_model_artifacts "${LATEST_STEP}"; then
            MODEL_PATH="${LATEST_STEP}"
        fi
    fi

    if [ -z "${MODEL_PATH}" ]; then
        echo "ERROR: Model not found for '${MODEL_SPEC}'"
        echo "  Checked: ${CKPT_BASE}"
        echo "  Expected a valid model dir, ${CKPT_BASE}/hf_merged, or latest global_step_*/huggingface"
        return 1
    fi

    RESOLVED_MODEL_PATH="${MODEL_PATH}"
    RESOLVED_MODEL_NAME="${MODEL_NAME}"
}

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

show_summary() {
    local MODEL_NAME=$1
    local OUTPUT_DIR=$2

    echo ""
    echo "=========================================="
    echo "Benchmark Summary - ${MODEL_NAME}"
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
}

benchmark_model() {
    local MODEL_SPEC=$1

    resolve_model_spec "${MODEL_SPEC}"

    local MODEL_PATH="${RESOLVED_MODEL_PATH}"
    local MODEL_NAME="${RESOLVED_MODEL_NAME}"
    local OUTPUT_DIR="results/${MODEL_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    echo "################################################################################"
    echo "# Benchmark Model"
    echo "# Spec: ${MODEL_SPEC}"
    echo "# Model: ${MODEL_PATH}"
    echo "# Model Name: ${MODEL_NAME}"
    echo "# Datasets: ${DATASETS_TO_TEST}"
    echo "# Pass@k values: ${PASS_K_VALUES}"
    echo "# Output: ${OUTPUT_DIR}"
    echo "################################################################################"

    for CURRENT_PASS_K in ${PASS_K_VALUES}; do
        echo ""
        echo "=========================================="
        echo "Testing with pass@${CURRENT_PASS_K}"
        echo "=========================================="

        for DATASET_NAME in ${DATASETS_TO_TEST}; do
            DATASET_PATH="${DATASET_PATHS[$DATASET_NAME]}"
            if [ -n "${DATASET_PATH}" ]; then
                benchmark_dataset "${DATASET_NAME}" "${DATASET_PATH}" "${CURRENT_PASS_K}"
            else
                echo "WARNING: Unknown dataset '${DATASET_NAME}', skipping..."
            fi
        done
    done

    show_summary "${MODEL_NAME}" "${OUTPUT_DIR}"
}

################################################################################
# Main
################################################################################

for MODEL_SPEC in "${MODEL_SPECS[@]}"; do
    benchmark_model "${MODEL_SPEC}"
done
