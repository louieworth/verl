#!/bin/bash
# Run one model's math eval inside an existing Slurm allocation on one node.
#
# Intended to be launched with:
#   srun --jobid=<job> --overlap --nodes=1 --ntasks=1 --nodelist=<node> \
#     --gres=gpu:h100:4 bash recipe/math_evaluation/run_single_existing_alloc_eval.sh

set -Eeuo pipefail

cd /scratch/l/luli/src/verl

export PYTHON_BIN="${PYTHON_BIN:-/scratch/l/luli/conda/envs/verl/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="/scratch/l/luli/src/verl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/l/luli/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${MODEL_LABEL:?MODEL_LABEL is required}"

export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-/scratch/l/luli/jiangli/datasets/eval}"
export DATASETS="${DATASETS:-aime24 aime25}"
export PASS_K="${PASS_K:-16}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export GEN_TP="${GEN_TP:-1}"
export WRITE_RESULTS_CSV="${WRITE_RESULTS_CSV:-false}"
export WRITE_PASS16_AGGREGATES="${WRITE_PASS16_AGGREGATES:-true}"

RUN_ROOT="${RUN_ROOT:-/scratch/l/luli/jiangli/eval_diag/full_pass16_parallel/job${SLURM_JOB_ID:-manual}-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="$RUN_ROOT/$MODEL_LABEL"
mkdir -p "$OUT_DIR/gen" "$OUT_DIR/results"

export EVAL_MODEL_NAME="$MODEL_LABEL"
export EVAL_OUTPUT_DIR="$OUT_DIR/gen"
export EVAL_RESULTS_FILE="$OUT_DIR/results/results.json"
export EVAL_RESULTS_CSV_FILE="$OUT_DIR/results/results.csv"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_eval_${SLURM_JOB_ID:-manual}_${MODEL_LABEL}}"
export TORCHINDUCTOR_CACHE_DIR="$OUT_DIR/torchinductor"
export TRITON_CACHE_DIR="$OUT_DIR/triton"
export TORCH_EXTENSIONS_DIR="$OUT_DIR/torch_extensions"
export VLLM_CACHE_ROOT="$OUT_DIR/vllm_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT"

LOG="$OUT_DIR/eval.log"
echo "RUN_ROOT=$RUN_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "LOG=$LOG"

exec > "$LOG" 2>&1

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "model_label=$MODEL_LABEL"
echo "model_path=$MODEL_PATH"
echo "datasets=$DATASETS"
echo "pass_k=$PASS_K"
echo "python=$PYTHON_BIN"
date
nvidia-smi || true

bash recipe/math_evaluation/benchmark_kl_model.sh "$MODEL_PATH"

for dataset in ${DATASETS}; do
    gen_parquet="$EVAL_OUTPUT_DIR/${dataset}_pass${PASS_K}_generation.parquet"
    if [ -f "$gen_parquet" ]; then
        "$PYTHON_BIN" recipe/math_evaluation/diagnose_opd_zero_aime.py \
            --stage analyze-parquet \
            --bad-model "$MODEL_PATH" \
            --generation-parquet "$gen_parquet" \
            --output-dir "$OUT_DIR/${dataset}_analysis"
    else
        echo "missing generation parquet: $gen_parquet"
    fi
done

date
nvidia-smi || true
