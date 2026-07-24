#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=code
export MODEL_PATH="${MODEL_PATH:-model/base/Qwen3-4B-Instruct-2507}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-4B-Instruct-2507}"
export MAX_TRAIN_DURATION_SECONDS="${MAX_TRAIN_DURATION_SECONDS:-9000}"
exec "$SCRIPT_DIR/_run_qwen3_grpo_8h100.sh" "$@"
