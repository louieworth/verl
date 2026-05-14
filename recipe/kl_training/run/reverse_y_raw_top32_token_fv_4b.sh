#!/bin/bash
# =============================================================================
# 4B variant of reverse_y_raw_top32_token_fv.sh — Qwen3-4B-Instruct-2507
# student/teacher. Only difference vs the 8B wrapper is MODEL_PATH; the paper-
# pinned hyperparameters (TOP_K=32, LR=2e-6, WARMUP=0, TEMPERATURE=1, no clip,
# effective batch=128) inherit from the 8B wrapper.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/reverse_y_raw_top32_token_fv.sh" "$@"
