#!/usr/bin/env bash
#
# Benchmark KL Training Model on Math Datasets
# Results are saved to results/${BASE_MODEL_NAME}/results.json (same format as run_full_pipeline_multi_epoch.sh)
#
# Usage:
#   bash benchmark_kl_model.sh <model_path> [<model_path2> ...]
#   bash benchmark_kl_model.sh --config <config_json>
#
# Positional args: one or more local model directories. Each model is evaluated
# sequentially on all configured GPUs. Tokenizer (optional) goes through the
# TOKENIZER_PATH env var and is shared across all models.
#
# Examples:
#   # Single model
#   bash benchmark_kl_model.sh /opt/dlami/nvme/jiangli/models/Qwen3-8B_kl_forward_monte_carlo/epoch1/hf_merged
#
#   # Two (or more) models evaluated back-to-back, each using all GPUs
#   bash benchmark_kl_model.sh \
#       /opt/dlami/nvme/jiangli/models/Qwen3-8B_kl_forward_monte_carlo/epoch1/hf_merged \
#       /opt/dlami/nvme/jiangli/models/Qwen3-8B_kl_reverse_monte_carlo/epoch1/hf_merged
#
#   # Override GPU count / datasets / pass@k via env vars
#   NGPUS_PER_NODE=8 DATASETS="aime24 aime25 math500" PASS_K=1 \
#       bash benchmark_kl_model.sh /path/to/model1/hf_merged /path/to/model2/hf_merged
#
#   # Custom tokenizer (applies to all models in the run)
#   TOKENIZER_PATH=/path/to/tokenizer bash benchmark_kl_model.sh /path/to/model
#
#   # Batch config mode (unchanged): per-model overrides live in JSON
#   bash benchmark_kl_model.sh --config recipe/opd/benchmark_batch_config.json
#

set -e

################################################################################
# Configuration
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # recipe/math_evaluation (where run_eval_suite.py and run_benchmark_batch.py live)
VERL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"             # repo root
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

# Model paths (one or more positional args; or env var MODEL_PATH for a single model)
MODEL_PATHS=("$@")
if [ ${#MODEL_PATHS[@]} -eq 0 ] && [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATHS=("${MODEL_PATH}")
fi
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "ERROR: Please specify at least one model path"
    echo "Usage: bash benchmark_kl_model.sh <model_path> [<model_path2> ...]"
    echo "       TOKENIZER_PATH=<dir> bash benchmark_kl_model.sh <model_path>   # optional"
    exit 1
fi

for mp in "${MODEL_PATHS[@]}"; do
    if [ ! -d "${mp}" ]; then
        echo "ERROR: Model not found at ${mp}"
        exit 1
    fi
done

TOKENIZER_PATH=${TOKENIZER_PATH:-""}
if [ -n "${TOKENIZER_PATH}" ] && [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "ERROR: Tokenizer path not found at ${TOKENIZER_PATH}"
    exit 1
fi

# GPU settings
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}

# Eval datasets
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}

# Datasets to test
# DEFAULT_DATASETS="math500 hmmt25 beyondaime amobench gsm8k"
DEFAULT_DATASETS="aime24 aime25 hmmt25 beyondaime amobench"
DATASETS_TO_TEST=${DATASETS:-"${DEFAULT_DATASETS}"}
DATASETS_TO_TEST_CSV=$(echo "${DATASETS_TO_TEST}" | tr ' ' ',')

# Pass@k
PASS_K=${PASS_K:-1}

################################################################################
# Per-model evaluation
################################################################################

evaluate_one_model() {
    local MODEL_PATH="$1"

    # Extract model names from path.
    # Supported layouts:
    #   /.../<base>_kl_<type>_<sample>/hf_merged
    #   /.../<base>_kl_<type>_<sample>/epochN/hf_merged
    local FULL_MODEL_DIR FULL_MODEL_NAME EPOCH_SUFFIX BASE_MODEL_NAME MODEL_NAME
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

    local GEN_OUTPUT_DIR="gen_results/${FULL_MODEL_NAME}/evaluate"
    mkdir -p "${GEN_OUTPUT_DIR}"

    local RESULTS_FILE="results/${BASE_MODEL_NAME}/results.json"
    mkdir -p "results/${BASE_MODEL_NAME}"

    echo "################################################################################"
    echo "# KL Model Benchmark"
    echo "# Model:      ${MODEL_PATH}"
    if [ -n "${TOKENIZER_PATH}" ]; then
        echo "# Tokenizer:  ${TOKENIZER_PATH}"
    fi
    echo "# Model Name: ${MODEL_NAME}"
    echo "# Datasets:   ${DATASETS_TO_TEST}"
    echo "# Results:    ${RESULTS_FILE}"
    echo "################################################################################"

    if [ -f "${RESULTS_FILE}" ]; then
        echo "Loading existing results from: ${RESULTS_FILE}"
    fi

    local PY_CMD=(
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

    echo ""
    echo "=========================================="
    echo "Benchmark Summary: ${MODEL_NAME}"
    echo "=========================================="

    "${PYTHON_BIN}" - "${RESULTS_FILE}" "${MODEL_NAME}" "${DATASETS_TO_TEST}" << 'PYEOF'
import json, os, sys
results_file, model_name, datasets_str = sys.argv[1], sys.argv[2], sys.argv[3]
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
    if model_name in data:
        print(f"Results for {model_name}:")
        for key, value in data[model_name].items():
            if '_generation_pass_' in key:
                dataset = key.split('_pass')[0]
                if isinstance(value, (int, float)):
                    print(f"  {dataset}: {value:.2%}")
                else:
                    print(f"  {dataset}: {value}")
    else:
        print(f"No results found for {model_name}")
else:
    print(f"Results file not found: {results_file}")
PYEOF

    echo ""
    echo "Results saved to:"
    echo "  Accuracy: ${RESULTS_FILE}"
    echo "  Generation files: ${GEN_OUTPUT_DIR}/"
    echo ""
}

################################################################################
# Main: loop over all model paths
################################################################################

echo "Will evaluate ${#MODEL_PATHS[@]} model(s):"
for mp in "${MODEL_PATHS[@]}"; do
    echo "  - ${mp}"
done
echo ""

for mp in "${MODEL_PATHS[@]}"; do
    evaluate_one_model "${mp}"
done

echo "All ${#MODEL_PATHS[@]} model(s) evaluated."
