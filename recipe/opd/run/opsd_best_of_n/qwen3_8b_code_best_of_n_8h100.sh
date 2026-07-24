#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=code
export MODEL_PATH="${MODEL_PATH:-model/base/Qwen3-8B}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-8B}"
export SAMPLING_BUDGET_SECONDS="${SAMPLING_BUDGET_SECONDS:-13800}"
exec "$SCRIPT_DIR/_run_qwen3_best_of_n_8h100.sh" "$@"
