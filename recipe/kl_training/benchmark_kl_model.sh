#!/usr/bin/env bash
#
# Benchmark KL Training Model on Math Datasets
# Results are saved to results/${BASE_MODEL_NAME}/results.json (same format as run_full_pipeline_multi_epoch.sh)
#
# Usage:
#   bash benchmark_kl_model.sh <model_path> [tokenizer_path]
#   bash benchmark_kl_model.sh --config <config_json>
#
# Examples:
#   bash benchmark_kl_model.sh /data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/hf_merged
#
#   bash benchmark_kl_model.sh /path/to/hf_merged /data/data/jiangli/models/Qwen3-4B-Instruct-2507
#
#   DATASETS="aime24 aime25 math500" bash benchmark_kl_model.sh /path/to/model
#   bash benchmark_kl_model.sh --config recipe/kl_training/benchmark_batch_config.json
#

set -e

################################################################################
# Configuration
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_BIN=${PYTHON_BIN:-python3}

BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-""}
DRY_RUN_FLAG=""

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN_FLAG="--dry_run"
    shift
fi

if [ "${1:-}" = "--config" ]; then
    BENCHMARK_CONFIG="$2"
    shift 2
fi

if [ -n "${BENCHMARK_CONFIG}" ]; then
    if [ ! -f "${BENCHMARK_CONFIG}" ]; then
        echo "ERROR: Config file not found at ${BENCHMARK_CONFIG}"
        exit 1
    fi

    "${PYTHON_BIN}" "$SCRIPT_DIR/run_benchmark_batch.py" --config "${BENCHMARK_CONFIG}" ${DRY_RUN_FLAG}
    exit $?
fi

# Model path (required)
MODEL_PATH=${MODEL_PATH:-$1}
TOKENIZER_PATH=${TOKENIZER_PATH:-$2}
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: Please specify model path"
    echo "Usage: bash benchmark_kl_model.sh <model_path> [tokenizer_path]"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

if [ -n "${TOKENIZER_PATH}" ] && [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "ERROR: Tokenizer path not found at ${TOKENIZER_PATH}"
    exit 1
fi

# Extract model names
# Supported layouts:
#   /.../<base>_kl_<type>_<sample>/hf_merged
#   /.../<base>_kl_<type>_<sample>/epochN/hf_merged
FULL_MODEL_DIR=$(dirname "${MODEL_PATH}")
FULL_MODEL_NAME=$(basename "${FULL_MODEL_DIR}")

if [[ "${FULL_MODEL_NAME}" =~ ^epoch([0-9]+)$ ]]; then
    EPOCH_SUFFIX="_${FULL_MODEL_NAME}"
    FULL_MODEL_DIR=$(dirname "${FULL_MODEL_DIR}")
    FULL_MODEL_NAME=$(basename "${FULL_MODEL_DIR}")
else
    EPOCH_SUFFIX=""
fi

BASE_MODEL_NAME=$(echo "${FULL_MODEL_NAME}" | sed -E 's/_kl_.*$//')
MODEL_NAME="${FULL_MODEL_NAME}${EPOCH_SUFFIX}"

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
    ["beyondaime"]="${EVAL_DATASETS_DIR}/beyondaime/beyondaime_test.parquet"
    ["amobench"]="${EVAL_DATASETS_DIR}/amobench/amobench_test.parquet"
    ["gsm8k"]="${EVAL_DATASETS_DIR}/gsm8k/gsm8k_test.parquet"
)

# Datasets to test
DEFAULT_DATASETS="beyondaime amobench gsm8k"
DATASETS_TO_TEST=${DATASETS:-"${DEFAULT_DATASETS}"}

# Pass@k
PASS_K=${PASS_K:-1}

################################################################################
# Main
################################################################################

echo "################################################################################"
echo "# KL Model Benchmark"
echo "# Model: ${MODEL_PATH}"
if [ -n "${TOKENIZER_PATH}" ]; then
    echo "# Tokenizer: ${TOKENIZER_PATH}"
fi
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

DATASETS_TO_TEST_CSV=$(echo "${DATASETS_TO_TEST}" | tr ' ' ',')

echo ""
echo "=========================================="
echo "Evaluating with one model load"
echo "=========================================="
echo ""

PY_CMD=(
    "${PYTHON_BIN}" "$SCRIPT_DIR/run_eval_suite.py"
    --model_path "${MODEL_PATH}"
    --model_name "${MODEL_NAME}"
    --output_dir "${GEN_OUTPUT_DIR}"
    --results_file "${RESULTS_FILE}"
    --datasets "${DATASETS_TO_TEST_CSV}"
    --datasets_dir "${EVAL_DATASETS_DIR}"
    --pass_k "${PASS_K}"
    --temperature 0.6
    --top_p 0.95
    --nnodes "${NNODES}"
    --n_gpus_per_node "${NGPUS_PER_NODE}"
    --gen_tp "${GEN_TP}"
)

if [ -n "${TOKENIZER_PATH}" ]; then
    PY_CMD+=(--tokenizer_path "${TOKENIZER_PATH}")
fi

"${PY_CMD[@]}"

# Display summary
echo ""
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="
echo ""

"${PYTHON_BIN}" << EOF
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
