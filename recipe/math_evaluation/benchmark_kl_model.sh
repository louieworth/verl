#!/usr/bin/env bash
#
# Benchmark KL Training Model on Math Datasets
# Results are saved to ${EVAL_RESULTS_FILE}, or results/${BASE_MODEL_NAME}/results.json by default.
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

# Default Qwen3 math evaluation context allocation.
export EVAL_PROMPT_LENGTH="${EVAL_PROMPT_LENGTH:-2048}"
export EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-16384}"
export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-$((EVAL_PROMPT_LENGTH + EVAL_RESPONSE_LENGTH))}"

# Canonical training launchers provide a persisted W&B run ID and explicit
# global step. Keep that run active for the entire benchmark so normal terminal
# output appears in W&B Logs and structured dataset progress is visible live.
if [ -n "${WANDB_RUN_ID:-}" ] && [ "${EVAL_WANDB_WRAPPED:-0}" != 1 ] && \
   [ "${1:-}" != "--dry-run" ] && [ "${1:-}" != "--config" ]; then
    export EVAL_WANDB_WRAPPED=1
    exec "${PYTHON_BIN}" "$SCRIPT_DIR/run_eval_with_wandb.py" -- \
        bash "$SCRIPT_DIR/benchmark_kl_model.sh" "$@"
fi

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

is_hf_repo_id() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

for mp in "${MODEL_PATHS[@]}"; do
    if [ ! -d "${mp}" ] && ! is_hf_repo_id "${mp}"; then
        echo "ERROR: Model must be an existing local directory or a Hugging Face repo ID: ${mp}"
        exit 1
    fi
done

TOKENIZER_PATH=${TOKENIZER_PATH:-""}
if [ -n "${TOKENIZER_PATH}" ] && [ ! -d "${TOKENIZER_PATH}" ] && ! is_hf_repo_id "${TOKENIZER_PATH}"; then
    echo "ERROR: Tokenizer must be an existing local directory or a Hugging Face repo ID: ${TOKENIZER_PATH}"
    exit 1
fi

# GPU settings
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-${NGPUS_PER_NODE:-1}}
if ! [[ "$NGPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "$GEN_TP" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NGPUS_PER_NODE, NNODES, and GEN_TP must be positive integers" >&2
    exit 1
fi
TOTAL_GPUS=$((NGPUS_PER_NODE * NNODES))
if [ "$GEN_TP" -gt "$TOTAL_GPUS" ] || [ $((TOTAL_GPUS % GEN_TP)) -ne 0 ]; then
    echo "ERROR: GEN_TP=$GEN_TP must divide total GPUs=$TOTAL_GPUS" >&2
    exit 1
fi

# Eval datasets
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-data/eval_dataset/math}

# Datasets to test
DEFAULT_DATASETS="aime25 aime26 hmmt26 amobench"
DATASETS_TO_TEST=${DATASETS:-"${DEFAULT_DATASETS}"}
DATASETS_TO_TEST_CSV=$(echo "${DATASETS_TO_TEST}" | tr ' ' ',')
METRICS_DATASETS=${EVAL_ALL_DATASETS:-${EVAL_DATASETS:-${DATASETS_TO_TEST}}}
METRICS_DATASETS_CSV=$(echo "${METRICS_DATASETS}" | tr ' ' ',')

# Pass@k
PASS_K=${PASS_K:-16}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
EVAL_TOP_P=${EVAL_TOP_P:-0.7}
EVAL_SEED=${EVAL_SEED:-42}
EVAL_SIGNATURE="n${PASS_K}_t${EVAL_TEMPERATURE}_p${EVAL_TOP_P}_prompt${EVAL_PROMPT_LENGTH}_response${EVAL_RESPONSE_LENGTH}_seed${EVAL_SEED}_base"
GEN_OUTPUT_BASE_DIR=${GEN_OUTPUT_BASE_DIR:-gen_results/eval}
RESULTS_BASE_DIR=${RESULTS_BASE_DIR:-results}
WRITE_PASS16_AGGREGATES=${WRITE_PASS16_AGGREGATES:-true}
WRITE_RESULTS_CSV=${WRITE_RESULTS_CSV:-true}

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
    if [ "$(basename "${MODEL_PATH}")" = "hf_merged" ]; then
        FULL_MODEL_DIR=$(dirname "${MODEL_PATH}")
        FULL_MODEL_NAME=$(basename "${FULL_MODEL_DIR}")
    else
        FULL_MODEL_DIR="${MODEL_PATH}"
        FULL_MODEL_NAME=$(basename "${MODEL_PATH}")
    fi

    if [[ "${FULL_MODEL_NAME}" =~ ^epoch([0-9]+)$ ]]; then
        EPOCH_SUFFIX="_${FULL_MODEL_NAME}"
        FULL_MODEL_DIR=$(dirname "${FULL_MODEL_DIR}")
        FULL_MODEL_NAME=$(basename "${FULL_MODEL_DIR}")
    else
        EPOCH_SUFFIX=""
    fi

    BASE_MODEL_NAME="${EVAL_BASE_MODEL_NAME:-$(echo "${FULL_MODEL_NAME}" | sed -E 's/_kl_.*$//')}"
    MODEL_NAME="${EVAL_MODEL_NAME:-${FULL_MODEL_NAME}${EPOCH_SUFFIX}}"

    local GEN_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${GEN_OUTPUT_BASE_DIR}/${FULL_MODEL_NAME}/${EVAL_SIGNATURE}}"
    mkdir -p "${GEN_OUTPUT_DIR}"

    local RESULTS_FILE="${EVAL_RESULTS_FILE:-${RESULTS_BASE_DIR}/${BASE_MODEL_NAME}/results.json}"
    mkdir -p "$(dirname "${RESULTS_FILE}")"

    echo "################################################################################"
    echo "# KL Model Benchmark"
    echo "# Model:      ${MODEL_PATH}"
    if [ -n "${TOKENIZER_PATH}" ]; then
        echo "# Tokenizer:  ${TOKENIZER_PATH}"
    fi
    echo "# Model Name: ${MODEL_NAME}"
    echo "# Datasets:   ${DATASETS_TO_TEST}"
    echo "# Lengths:    prompt=${EVAL_PROMPT_LENGTH}, response=${EVAL_RESPONSE_LENGTH}, max_model=${EVAL_MAX_MODEL_LEN}"
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
        --temperature "${EVAL_TEMPERATURE}"
        --top_p "${EVAL_TOP_P}"
        --seed "${EVAL_SEED}"
        --prompt_length "${EVAL_PROMPT_LENGTH}"
        --response_length "${EVAL_RESPONSE_LENGTH}"
        --max_model_len "${EVAL_MAX_MODEL_LEN}"
        --force_base_prompt
        --nnodes "${NNODES}"
        --n_gpus_per_node "${NGPUS_PER_NODE}"
        --gen_tp "${GEN_TP}"
    )
    if [ -n "${TOKENIZER_PATH}" ]; then
        PY_CMD+=(--tokenizer_path "${TOKENIZER_PATH}")
    fi

    "${PY_CMD[@]}"

    if [ "${PASS_K}" = "16" ] && [ "${WRITE_PASS16_AGGREGATES}" = "true" ] && [ -f "${RESULTS_FILE}" ]; then
        echo ""
        echo "Computing avg@16/pass@16 aggregates from generated responses..."
        local METRICS_FILE="${EVAL_METRICS_FILE:-${GEN_OUTPUT_DIR}/metrics.json}"
        local METRICS_CMD=(
            "${PYTHON_BIN}" "$SCRIPT_DIR/compute_pass_at_k_from_gen.py"
            --gen_dir "${GEN_OUTPUT_DIR}"
            --results_file "${RESULTS_FILE}"
            --metrics_file "${METRICS_FILE}"
            --model_name "${MODEL_NAME}"
            --model_path "${MODEL_PATH}"
            --datasets "${METRICS_DATASETS_CSV}"
            --n_samples "${PASS_K}"
            --prompt_length "${EVAL_PROMPT_LENGTH}"
            --response_length "${EVAL_RESPONSE_LENGTH}"
            --temperature "${EVAL_TEMPERATURE}"
            --top_p "${EVAL_TOP_P}"
            --seed "${EVAL_SEED}"
        )
        if [ -n "${EVAL_STEP:-${WANDB_GLOBAL_STEP:-}}" ]; then
            METRICS_CMD+=(--step "${EVAL_STEP:-${WANDB_GLOBAL_STEP}}")
        fi
        if [ -n "${EVAL_MILESTONE_FRACTION:-}" ]; then
            METRICS_CMD+=(--milestone_fraction "${EVAL_MILESTONE_FRACTION}")
        fi
        "${METRICS_CMD[@]}"

        if [ -n "${WANDB_RUN_ID:-}" ]; then
            if [ -z "${WANDB_PROJECT:-}" ] || [ -z "${WANDB_GLOBAL_STEP:-}" ]; then
                echo "ERROR: WANDB_RUN_ID requires WANDB_PROJECT and WANDB_GLOBAL_STEP" >&2
                exit 1
            fi
            if [ "${EVAL_KIND:-milestone}" != "base" ] && [ -z "${EVAL_MILESTONE_FRACTION:-}" ]; then
                echo "ERROR: milestone W&B eval logging requires EVAL_MILESTONE_FRACTION" >&2
                exit 1
            fi
            if [ "${EVAL_WANDB_WRAPPED:-0}" != 1 ]; then
                "${PYTHON_BIN}" "$SCRIPT_DIR/log_metrics_wandb.py" --metrics_file "${METRICS_FILE}"
            fi
        fi
    fi

    if [ -n "${EVAL_METADATA_FILE:-}" ] && [ -f "${EVAL_METADATA_FILE}" ] && [ -f "${RESULTS_FILE}" ]; then
        "${PYTHON_BIN}" - "${RESULTS_FILE}" "${MODEL_NAME}" "${EVAL_METADATA_FILE}" <<'PY'
import json
import os
import sys

results_file, model_name, metadata_file = sys.argv[1], sys.argv[2], sys.argv[3]
with open(results_file) as f:
    results = json.load(f)
with open(metadata_file) as f:
    metadata = json.load(f)
entry = results.setdefault(model_name, {})
entry["_metadata"] = metadata
tmp = results_file + ".tmp"
with open(tmp, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
os.replace(tmp, results_file)
PY
    fi

    if [ "${WRITE_RESULTS_CSV}" = "true" ] && [ -f "${RESULTS_FILE}" ]; then
        local CSV_RESULTS_FILE="${EVAL_RESULTS_CSV_FILE:-${RESULTS_FILE%.json}.csv}"
        local BASE_RESULTS_FILE="${EVAL_BASE_RESULTS_FILE:-${RESULTS_BASE_DIR}/base/${BASE_MODEL_NAME}.json}"
        "${PYTHON_BIN}" "$SCRIPT_DIR/results_json_to_csv.py" \
            --results_file "${RESULTS_FILE}" \
            --output_file "${CSV_RESULTS_FILE}" \
            --base_results_file "${BASE_RESULTS_FILE}" \
            --base_model_name "${BASE_MODEL_NAME}"
    fi

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
echo "GPU layout: NNODES=$NNODES NGPUS_PER_NODE=$NGPUS_PER_NODE GEN_TP=$GEN_TP replicas=$((TOTAL_GPUS / GEN_TP))"
echo ""

if [ -n "$DRY_RUN_FLAG" ]; then
    echo "Datasets: $DATASETS_TO_TEST"
    echo "Sampling: pass_k=$PASS_K temperature=$EVAL_TEMPERATURE top_p=$EVAL_TOP_P seed=$EVAL_SEED prompt_format=base_completion"
    echo "Lengths: prompt=$EVAL_PROMPT_LENGTH response=$EVAL_RESPONSE_LENGTH max_model=$EVAL_MAX_MODEL_LEN"
    echo "Dry run: configuration validated; no model was loaded."
    exit 0
fi

for mp in "${MODEL_PATHS[@]}"; do
    evaluate_one_model "${mp}"
done

echo "All ${#MODEL_PATHS[@]} model(s) evaluated."
