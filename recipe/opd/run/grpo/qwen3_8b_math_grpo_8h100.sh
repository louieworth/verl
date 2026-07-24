#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=math
export MODEL_PATH="${MODEL_PATH:-model/base/Qwen3-8B}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-8B}"
export MAX_TRAIN_DURATION_SECONDS="${MAX_TRAIN_DURATION_SECONDS:-34200}"
exec "$SCRIPT_DIR/_run_qwen3_grpo_8h100.sh" "$@"
