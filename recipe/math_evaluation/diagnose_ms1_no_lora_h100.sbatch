#!/bin/bash
#SBATCH --job-name=diag-ms1-zero-aime
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=48
#SBATCH --output=/scratch/l/luli/src/verl/recipe/math_evaluation/logs/diag-ms1-zero-aime-%j.out
#SBATCH --error=/scratch/l/luli/src/verl/recipe/math_evaluation/logs/diag-ms1-zero-aime-%j.err

set -Eeuo pipefail

cd /scratch/l/luli/src/verl

export PYTHON_BIN="${PYTHON_BIN:-/scratch/l/luli/conda/envs/verl/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="/scratch/l/luli/src/verl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/l/luli/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

export BAD_MODEL="${BAD_MODEL:-/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/epoch1/ms1/batch00001/hf_merged}"
export REF_MODEL="${REF_MODEL:-/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms40_20260520-234705/epoch1/ms40/batch00040/hf_merged}"
export BASE_MODEL="${BASE_MODEL:-/scratch/l/luli/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-/scratch/l/luli/jiangli/datasets/eval}"
export AIME24_PATH="$EVAL_DATASETS_DIR/aime24/aime24_test.parquet"

RUN_ID="${SLURM_JOB_ID:-manual}-$(date +%Y%m%d-%H%M%S)"
OUT_DIR="/scratch/l/luli/jiangli/eval_diag/ms1_zero_aime24/${RUN_ID}"
mkdir -p "$OUT_DIR" "$OUT_DIR/gen_results" "$OUT_DIR/results"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_diag_ms1_${SLURM_JOB_ID:-manual}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$OUT_DIR/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OUT_DIR/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$OUT_DIR/torch_extensions}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$OUT_DIR/vllm_cache}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT"

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "out_dir=$OUT_DIR"
echo "bad_model=$BAD_MODEL"
echo "ref_model=$REF_MODEL"
echo "base_model=$BASE_MODEL"
echo "python=$PYTHON_BIN"
date
nvidia-smi || true

"$PYTHON_BIN" recipe/math_evaluation/diagnose_opd_zero_aime.py \
    --stage metadata \
    --bad-model "$BAD_MODEL" \
    --ref-model "$REF_MODEL" \
    --base-model "$BASE_MODEL" \
    --dataset "$AIME24_PATH" \
    --output-dir "$OUT_DIR"

"$PYTHON_BIN" recipe/math_evaluation/diagnose_opd_zero_aime.py \
    --stage hf-smoke \
    --bad-model "$BAD_MODEL" \
    --ref-model "$REF_MODEL" \
    --base-model "$BASE_MODEL" \
    --dataset "$AIME24_PATH" \
    --output-dir "$OUT_DIR" \
    --smoke-max-new-tokens "${SMOKE_MAX_NEW_TOKENS:-2048}" \
    --smoke-num-return "${SMOKE_NUM_RETURN:-4}"

ray stop --force >/dev/null 2>&1 || true

export DATASETS="aime24"
export PASS_K="16"
export NGPUS_PER_NODE="4"
export NNODES="1"
export GEN_TP="${GEN_TP:-1}"
export EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-Qwen3-1.7B_ms1_no_lora_diag}"
export EVAL_OUTPUT_DIR="$OUT_DIR/gen_results/ms1_bad"
export EVAL_RESULTS_FILE="$OUT_DIR/results/aime24_ms1_bad.json"
export EVAL_RESULTS_CSV_FILE="$OUT_DIR/results/aime24_ms1_bad.csv"
export WRITE_RESULTS_CSV="false"
export WRITE_PASS16_AGGREGATES="true"

bash recipe/math_evaluation/benchmark_kl_model.sh "$BAD_MODEL"

GEN_PARQUET="$EVAL_OUTPUT_DIR/aime24_pass16_generation.parquet"
if [ -f "$GEN_PARQUET" ]; then
    "$PYTHON_BIN" recipe/math_evaluation/diagnose_opd_zero_aime.py \
        --stage analyze-parquet \
        --bad-model "$BAD_MODEL" \
        --generation-parquet "$GEN_PARQUET" \
        --output-dir "$OUT_DIR"
else
    echo "missing generation parquet: $GEN_PARQUET"
fi

echo "results_json=$EVAL_RESULTS_FILE"
echo "out_dir=$OUT_DIR"
date
