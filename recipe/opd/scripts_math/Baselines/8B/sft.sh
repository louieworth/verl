#!/usr/bin/env bash
set -euo pipefail

# Explicit math baseline/sft launcher using the 8B Qwen3 Base model.
export OPD_TASK="math"
export OPD_FAMILY="baseline"
export OPD_VARIANT="sft"
export OPD_MODEL_SIZE="8B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B-Base}"
export MULTI_STEP="${MULTI_STEP:-0}"

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
