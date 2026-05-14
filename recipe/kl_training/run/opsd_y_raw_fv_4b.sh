#!/bin/bash
# =============================================================================
# 4B variant of opsd_y_raw_fv.sh — Qwen3-4B-Instruct-2507 student/teacher.
# OPSD JSD β=0 preset (clip=0.06, T=1.1, LR=5e-6). All other hyperparameters
# inherit from opsd_y_raw_fv.sh.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/opsd_y_raw_fv.sh" "$@"
