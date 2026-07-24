#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=math
export MODEL_PATH="${MODEL_PATH:-model/base/Qwen3-4B-Instruct-2507}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-4B-Instruct-2507}"
export SAMPLING_BUDGET_SECONDS="${SAMPLING_BUDGET_SECONDS:-15000}"
exec "$SCRIPT_DIR/_run_qwen3_best_of_n_8h100.sh" "$@"
