#!/usr/bin/env bash
set -euo pipefail

# Explicit math baseline/grpo launcher using the 4B Qwen3 Base model.
export OPD_TASK="math"
export OPD_FAMILY="baseline"
export OPD_VARIANT="grpo"
export OPD_MODEL_SIZE="4B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export MULTI_STEP="${MULTI_STEP:-0}"
export ROLLOUT_N="${ROLLOUT_N:-8}"

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
