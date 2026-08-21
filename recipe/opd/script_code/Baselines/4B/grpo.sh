#!/usr/bin/env bash
set -euo pipefail

# Explicit code baseline/grpo launcher using the 4B Qwen3 Base model.
export OPD_TASK="code"
export OPD_FAMILY="baseline"
export OPD_VARIANT="grpo"
export OPD_MODEL_SIZE="4B"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export MULTI_STEP="${MULTI_STEP:-0}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export CODE_GRPO_REWARD_CONTRACT="deepcoder_binary_15_longest_v1"
export CODE_GRPO_MAX_TEST_CASES="15"
export CODE_GRPO_EXEC_TIMEOUT_SECONDS="10"

OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/launch_common.sh"
