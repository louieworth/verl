#!/usr/bin/env bash
# Qwen3-8B: reverse KL on y_o.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-8B}"
export DIRECT_VARIANT="reverse_kl"

exec bash "$SCRIPT_DIR/../_run_math_direct_variant.sh" "$@"
