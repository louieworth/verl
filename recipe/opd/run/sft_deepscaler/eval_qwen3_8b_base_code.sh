#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

if [ -x /data2/conda/envs/verl/bin/python ]; then
    DEFAULT_PYTHON_BIN=/data2/conda/envs/verl/bin/python
else
    DEFAULT_PYTHON_BIN=python3
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/data2/.huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
RESULTS_FILE="${RESULTS_FILE:-$REPO_ROOT/results/base/code/Qwen3-8B_code_avg16_pass16.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/gen_results/eval/code/base/Qwen3-8B}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/code_eval/base/Qwen3-8B}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/eval_avg16_pass16_local_v6.log}"
GEN_TP="${GEN_TP:-1}"
BENCHMARK_ARGS=()
if [ "${1:-}" = "--dry-run" ]; then
    BENCHMARK_ARGS+=(--dry-run)
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

mkdir -p "$LOG_DIR" "$(dirname "$RESULTS_FILE")" "$OUTPUT_DIR"

LCB_AGGREGATE_ALL="${LCB_AGGREGATE_ALL:-false}"
LCB_EXPECTED_TASKS="${LCB_EXPECTED_TASKS:-}"
if [ -d "$OUTPUT_DIR/livecodebench/shards" ]; then
    LCB_AGGREGATE_ALL=true
    LCB_EXPECTED_TASKS=1055
fi

PYTHON_BIN="$PYTHON_BIN" \
NGPUS_PER_NODE="$GEN_TP" \
GEN_TP="$GEN_TP" \
DATASETS="${DATASETS:-humaneval_plus mbpp_plus livecodebench_v6}" \
PASS_K=16 \
EVAL_BASE_MODEL_NAME=Qwen3-8B \
EVAL_MODEL_NAME=Qwen3-8B \
EVAL_OUTPUT_DIR="$OUTPUT_DIR" \
EVAL_RESULTS_FILE="$RESULTS_FILE" \
EVAL_RESULTS_CSV_FILE="${RESULTS_FILE%.json}.csv" \
LCB_AGGREGATE_ALL="$LCB_AGGREGATE_ALL" \
LCB_EXPECTED_TASKS="$LCB_EXPECTED_TASKS" \
WRITE_RESULTS_CSV=true \
    bash "$REPO_ROOT/recipe/code_evaluation/benchmark_code_model.sh" "${BENCHMARK_ARGS[@]}" "$MODEL_PATH" \
    2>&1 | tee -a "$LOG_FILE"
