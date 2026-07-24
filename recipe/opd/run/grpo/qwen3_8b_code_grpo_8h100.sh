#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=code
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-8B}"
exec "$SCRIPT_DIR/_run_qwen3_grpo_8h100.sh" "$@"
