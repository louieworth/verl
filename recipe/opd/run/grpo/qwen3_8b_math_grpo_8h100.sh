#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=math
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-8B}"
export MAX_TRAIN_DURATION_SECONDS="${MAX_TRAIN_DURATION_SECONDS:-34200}"

# 8B/16K GRPO memory profile for 8 x 80GB GPUs.
#
# The shared runner already uses the minimum possible PPO micro-batch (1/GPU).
# Reduce the batches that surround it, cap each actor/ref/log-prob dynamic batch
# at one maximum-length sequence, and halve vLLM sequence concurrency. Keep
# ROLLOUT_N=8 and all prompt/response lengths unchanged.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-18432}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"

exec "$SCRIPT_DIR/_run_qwen3_grpo_8h100.sh" "$@"
