#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507: reverse KL on y_o with teacher top-k support.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-4B-Instruct-2507}"
export DIRECT_VARIANT="reverse_kl_topk"

exec bash "$SCRIPT_DIR/../_run_math_direct_variant.sh" "$@"
