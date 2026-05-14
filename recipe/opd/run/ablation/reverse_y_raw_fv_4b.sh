#!/bin/bash
# =============================================================================
# 4B variant of reverse_y_raw_fv.sh — Qwen3-4B-Instruct-2507 student/teacher.
# Only difference vs the 8B wrapper is MODEL_PATH; all KL settings, FSDP,
# memory budget, eval config inherit from reverse_y_raw_fv.sh.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/reverse_y_raw_fv.sh" "$@"
