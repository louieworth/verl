#!/usr/bin/env bash
# Prepare the canonical OpenThoughts math train parquet for every recipe.
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
MATH_TRAIN_SOURCE="${MATH_TRAIN_SOURCE:-}"
if [ -z "$MATH_TRAIN_SOURCE" ] && \
   [ -d /data2/OPSD/data/train/openthoughts_math_30k_opsd/data ]; then
    MATH_TRAIN_SOURCE=/data2/OPSD/data/train/openthoughts_math_30k_opsd/data
fi

args=(
    --task math
    --output-dir data/train_dataset/openthoughts_math_30k_opsd
    --cache-dir data/download_cache/opd_training/math
    --model-path "$MODEL_PATH"
    --max-prompt-length 2048
    --max-response-length 16384
    --min-prompt-coverage 0.95
    --pad-to-multiple 512
)
[ -z "$MATH_TRAIN_SOURCE" ] || args+=(--input "$MATH_TRAIN_SOURCE")

exec "$PYTHON_BIN" -m recipe.opd.dataset.prepare_experiment_data "${args[@]}" "$@"
