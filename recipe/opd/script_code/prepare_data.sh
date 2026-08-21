#!/usr/bin/env bash
# Prepare the canonical cleaned-TACO train parquet for every code recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -x /data2/conda/envs/verl/bin/python ]; then
    PYTHON_BIN="${PYTHON_BIN:-/data2/conda/envs/verl/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
if [ -d /data2/.huggingface/Qwen3-4B-Base ] && [ "$MODEL_PATH" = Qwen/Qwen3-4B-Base ]; then
    MODEL_PATH=/data2/.huggingface/Qwen3-4B-Base
fi
CODE_TRAIN_SOURCE="${CODE_TRAIN_SOURCE:-}"
if [ -z "$CODE_TRAIN_SOURCE" ] && [ -d /data2/OPSD/data/train/taco_code_clean/data ]; then
    CODE_TRAIN_SOURCE=/data2/OPSD/data/train/taco_code_clean/data
elif [ -z "$CODE_TRAIN_SOURCE" ] && [ -d data/train_dataset/taco/raw/ALL ]; then
    CODE_TRAIN_SOURCE=data/train_dataset/taco/raw/ALL
fi

args=(
    --task code
    --output-dir data/train_dataset/taco/canonical
    --cache-dir data/download_cache/opd_training/code
    --model-path "$MODEL_PATH"
    --max-prompt-length 2048
    --max-response-length 16384
    --min-prompt-coverage 0.95
    --pad-to-multiple 512
)
[ -z "$CODE_TRAIN_SOURCE" ] || args+=(--input "$CODE_TRAIN_SOURCE")

exec "$PYTHON_BIN" -m recipe.opd.dataset.prepare_experiment_data "${args[@]}" "$@"
