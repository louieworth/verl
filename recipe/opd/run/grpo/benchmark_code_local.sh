#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

CODE_EVAL_ROOT="${CODE_EVAL_ROOT:-data/eval_dataset/code}"
export HUMANEVAL_OVERRIDE_PATH="${HUMANEVAL_OVERRIDE_PATH:-$CODE_EVAL_ROOT/evalplus/HumanEvalPlus-v0.1.10.jsonl}"
export MBPP_OVERRIDE_PATH="${MBPP_OVERRIDE_PATH:-$CODE_EVAL_ROOT/evalplus/MbppPlus-v0.2.0.jsonl}"
export LCB_REPO="${LCB_REPO:-$CODE_EVAL_ROOT/LiveCodeBench}"
export LCB_CODEGEN_LITE_DIR="${LCB_CODEGEN_LITE_DIR:-$CODE_EVAL_ROOT/livecodebench/code_generation_lite}"
export HF_HOME="${GRPO_CODE_EVAL_HF_HOME:-$CODE_EVAL_ROOT/huggingface_cache}"
export HF_DATASETS_CACHE="${GRPO_CODE_EVAL_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTHONPATH="$LCB_REPO:.:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

for dataset_file in "$HUMANEVAL_OVERRIDE_PATH" "$MBPP_OVERRIDE_PATH"; do
    if [ ! -s "$dataset_file" ]; then
        echo "ERROR: missing code evaluation dataset: $dataset_file" >&2
        echo "Run: bash recipe/opd/run/grpo/prepare/ready_to_train.sh eval-data" >&2
        exit 1
    fi
done
if [ ! -f "$LCB_REPO/lcb_runner/runner/main.py" ]; then
    echo "ERROR: missing LiveCodeBench runtime under $LCB_REPO" >&2
    exit 1
fi
for index in 1 2 3 4 5 6; do
    file_name="test${index}.jsonl"
    [ "$index" = "1" ] && file_name="test.jsonl"
    if [ ! -s "$LCB_CODEGEN_LITE_DIR/$file_name" ]; then
        echo "ERROR: missing LiveCodeBench release_v6 file: $LCB_CODEGEN_LITE_DIR/$file_name" >&2
        exit 1
    fi
done

exec bash recipe/code_evaluation/benchmark_code_model.sh "$@"
