#!/usr/bin/env bash
# Qwen3-8B: forward KL on y_o with the same current clip default as 4B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-8B}"
export DIRECT_VARIANT="forward_kl_clip"
export DIRECT_DEFAULT_CLIP="0.06"

exec bash "$SCRIPT_DIR/../_run_math_direct_variant.sh" "$@"
