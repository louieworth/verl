#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=math
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-1.7B}"
# User-selected math budget: 4h30m.
export MAX_TRAIN_DURATION_SECONDS="${MAX_TRAIN_DURATION_SECONDS:-16200}"
export PASS_K="${PASS_K:-16}"
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"

exec "$SCRIPT_DIR/_run_qwen3_grpo_8h100.sh" "$@"
