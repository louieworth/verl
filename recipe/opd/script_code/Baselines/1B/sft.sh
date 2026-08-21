#!/usr/bin/env bash
set -euo pipefail

# Explicit code baseline/sft launcher using the 1B Qwen3 Base model.
export OPD_TASK="code"
export OPD_FAMILY="baseline"
export OPD_VARIANT="sft"
export OPD_MODEL_SIZE="1B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"
export MULTI_STEP="${MULTI_STEP:-0}"

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
