#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 OPSD forward KL on raw DeepScaleR y*.

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

export TASK="math"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-4B-Instruct-2507}"
export MULTI_STEP=1
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export GEN_TP="${GEN_TP:-$NGPUS_PER_NODE}"
export EVAL_GEN_TP="${EVAL_GEN_TP:-$NGPUS_PER_NODE}"

# Keep all non-model inputs and generated artifacts inside this repository.
export EVAL_DATASETS_DIR="$VERL_ROOT/data/eval_dataset/math"
export MODEL_SAVE_DIR="$VERL_ROOT/model/trained/opsd_rollout_variants"
export PIPELINE_ARCHIVE_MODEL_ROOT="$MODEL_SAVE_DIR/archive"
export GEN_RESULTS_ROOT="$VERL_ROOT/gen_results/opsd_rollout_variants"
export HF_HOME="$VERL_ROOT/data/download_cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export NTFY_ENABLED=false

export TEACHER_MODEL_PATH=""
export TEACHER_MODEL=""
export Y_MODE="y_o"
export Y_O_ROLLOUT_MODE="expert"
export TRAJECTORY_MODEL_PATH=""
export TRAJECTORY_MODEL=""
export PRECOMPUTED_STAGE1_PROMPTS_PATH=""
export PRECOMPUTED_Y_O_TRAJECTORY_PATH=""

OPSD_Y_STAR_DATA="$VERL_ROOT/data/train_dataset/deepscaler/train_opsd_y_star_solution_cot_only.parquet"
SFT_COT_SOURCE="$VERL_ROOT/data/train_dataset/deepscaler/train_sft_solution_cot_only.parquet"
if [ ! -s "$OPSD_Y_STAR_DATA" ]; then
    if [ ! -s "$SFT_COT_SOURCE" ]; then
        echo "ERROR: missing repo-local CoT-only SFT source parquet: $SFT_COT_SOURCE" >&2
        exit 1
    fi
    python3 "$SCRIPT_DIR/prepare_sft_cot_as_opsd_y_star.py" \
        --input "$SFT_COT_SOURCE" \
        --output "$OPSD_Y_STAR_DATA"
fi
export TRAIN_DATA_PATH="$OPSD_Y_STAR_DATA"
export DATA_PATH="$OPSD_Y_STAR_DATA"

exec bash "$SCRIPT_DIR/../forward_y_o.sh" "$@"
