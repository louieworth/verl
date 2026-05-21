#!/bin/bash
set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"
export DISTILL_MODE="opd"
export KL_TYPE="jsd"
export Y_MODE="${Y_MODE:-y_o}"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export TOP_K="${TOP_K:-0}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-${TRAIN_EPOCHS_PER_ROUND:-1}}"
export OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-one_step}"
export SCHEDULER="${SCHEDULER:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_opd_megatron.sh" "$@"
