#!/bin/bash
# =============================================================================
# 4B variant of forward_y_raw_fv.sh — Qwen3-4B-Instruct-2507 student/teacher.
# Forward KL on raw y is implemented as JSD with β=0 (the codebase routes
# `Y_MODE=y_raw` through stage1 only for non-forward KL_TYPE; β=0 jsd is the
# mathematical equivalent of forward KL but keeps the y_raw data path).
# Only difference vs the 8B wrapper is MODEL_PATH.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/forward_y_raw_fv.sh" "$@"
