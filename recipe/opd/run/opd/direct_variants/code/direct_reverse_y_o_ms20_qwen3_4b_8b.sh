#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export TASK="code"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-4B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen3-8B}"

export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"

export MULTI_STEP="${MULTI_STEP:-20}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PIPELINE_CLEANUP_BATCH_DATA="${PIPELINE_CLEANUP_BATCH_DATA:-true}"

exec bash "$SCRIPT_DIR/../../reverse_y_o.sh" "$@"
