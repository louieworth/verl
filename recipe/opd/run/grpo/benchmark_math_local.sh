#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-data/eval_dataset/math}"
export HF_HOME="${GRPO_MATH_EVAL_HF_HOME:-$EVAL_DATASETS_DIR/huggingface_cache}"
export HF_DATASETS_CACHE="${GRPO_MATH_EVAL_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${GRPO_MATH_EVAL_HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
exec bash recipe/math_evaluation/benchmark_kl_model.sh "$@"
