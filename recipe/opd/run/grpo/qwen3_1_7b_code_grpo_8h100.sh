#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=code
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-1.7B}"
# Matched to the recovered TRD code sampling + training time: 2h15m06s.
export MAX_TRAIN_DURATION_SECONDS="${MAX_TRAIN_DURATION_SECONDS:-8106}"
export PASS_K="${PASS_K:-16}"
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"

exec "$SCRIPT_DIR/_run_qwen3_grpo_8h100.sh" "$@"
