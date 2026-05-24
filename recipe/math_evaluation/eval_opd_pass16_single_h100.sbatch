#!/bin/bash
#SBATCH --job-name=opd-extra-pass16-h100
#SBATCH --partition=gpubase_bynode_b3
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --output=/scratch/l/luli/src/verl/recipe/math_evaluation/logs/%x-%j.out
#SBATCH --error=/scratch/l/luli/src/verl/recipe/math_evaluation/logs/%x-%j.err

set -euo pipefail

cd /scratch/l/luli/src/verl

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${EVAL_TAG:?EVAL_TAG must be set}"
: "${EVAL_MODEL_NAME:?EVAL_MODEL_NAME must be set}"

export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-/scratch/l/luli/conda/envs/verl/bin/python}"
export PYTHONPATH="/scratch/l/luli/src/verl:${PYTHONPATH:-}"

export DATASETS="${DATASETS:-hmmt25 beyondaime amobench}"
export PASS_K="${PASS_K:-16}"
export NGPUS_PER_NODE=4
export NNODES=1
export GEN_TP=1
export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-/scratch/l/luli/jiangli/datasets/eval}"
export EVAL_BASE_MODEL_NAME="${EVAL_BASE_MODEL_NAME:-Qwen3-1.7B}"

export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval_diag/opd_pass16_extra_default/gen/${EVAL_TAG}}"
export EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-eval_diag/opd_pass16_extra_default/results/${EVAL_TAG}_pass16.json}"
export EVAL_RESULTS_CSV_FILE="${EVAL_RESULTS_CSV_FILE:-eval_diag/opd_pass16_extra_default/results/${EVAL_TAG}_pass16.csv}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r${SLURM_JOB_ID}_${EVAL_TAG}_ray}"
export TMPDIR="${TMPDIR:-/tmp/r${SLURM_JOB_ID}_${EVAL_TAG}_tmp}"

mkdir -p "$EVAL_OUTPUT_DIR" "$(dirname "$EVAL_RESULTS_FILE")" "$(dirname "$EVAL_RESULTS_CSV_FILE")" "$RAY_TMPDIR" "$TMPDIR"

echo "MODEL_PATH=$MODEL_PATH"
echo "EVAL_MODEL_NAME=$EVAL_MODEL_NAME"
echo "DATASETS=$DATASETS"
echo "PASS_K=$PASS_K"
echo "EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR"
echo "EVAL_RESULTS_FILE=$EVAL_RESULTS_FILE"

bash recipe/math_evaluation/benchmark_kl_model.sh "$MODEL_PATH"
