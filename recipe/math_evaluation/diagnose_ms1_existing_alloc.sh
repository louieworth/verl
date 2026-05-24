#!/bin/bash
# Run OPD zero-AIME diagnostics inside an existing Slurm allocation.
#
# This intentionally avoids `ray stop --force` so it does not disturb the
# allocation owner process when invoked via `srun --jobid=<job> --overlap`.

set -Eeuo pipefail

cd /scratch/l/luli/src/verl

export PYTHON_BIN="${PYTHON_BIN:-/scratch/l/luli/conda/envs/verl/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="/scratch/l/luli/src/verl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/l/luli/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export BAD_MODEL="${BAD_MODEL:-/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/epoch1/ms1/batch00001/hf_merged}"
export YO_MS40_REF="${YO_MS40_REF:-/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms40_20260520-234705/epoch1/ms40/batch00040/hf_merged}"
export YR_MS40_FINAL="${YR_MS40_FINAL:-/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms40_20260523-175757/final/hf_merged}"
export BASE_MODEL="${BASE_MODEL:-/scratch/l/luli/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-/scratch/l/luli/jiangli/datasets/eval}"
export AIME24_PATH="$EVAL_DATASETS_DIR/aime24/aime24_test.parquet"

RUN_ID="job${SLURM_JOB_ID:-manual}-$(date +%Y%m%d-%H%M%S)"
OUT_DIR="/scratch/l/luli/jiangli/eval_diag/ms1_zero_aime24/${RUN_ID}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/diagnose.log"

echo "OUT_DIR=$OUT_DIR"
echo "LOG=$LOG"
exec > "$LOG" 2>&1

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "bad_model=$BAD_MODEL"
echo "yo_ms40_ref=$YO_MS40_REF"
echo "yr_ms40_final=$YR_MS40_FINAL"
echo "base_model=$BASE_MODEL"
echo "python=$PYTHON_BIN"
date
nvidia-smi || true

run_metadata_and_smoke() {
    local ref_name="$1"
    local ref_model="$2"
    local subdir="$OUT_DIR/$ref_name"
    mkdir -p "$subdir"
    echo "===== metadata + HF smoke: $ref_name ====="
    "$PYTHON_BIN" recipe/math_evaluation/diagnose_opd_zero_aime.py \
        --stage metadata \
        --bad-model "$BAD_MODEL" \
        --ref-model "$ref_model" \
        --base-model "$BASE_MODEL" \
        --dataset "$AIME24_PATH" \
        --output-dir "$subdir"
    "$PYTHON_BIN" recipe/math_evaluation/diagnose_opd_zero_aime.py \
        --stage hf-smoke \
        --bad-model "$BAD_MODEL" \
        --ref-model "$ref_model" \
        --base-model "$BASE_MODEL" \
        --dataset "$AIME24_PATH" \
        --output-dir "$subdir" \
        --smoke-max-new-tokens "${SMOKE_MAX_NEW_TOKENS:-2048}" \
        --smoke-num-return "${SMOKE_NUM_RETURN:-4}"
}

run_aime24_pass16() {
    local model_name="$1"
    local model_path="$2"
    local subdir="$OUT_DIR/aime24_$model_name"
    mkdir -p "$subdir/gen" "$subdir/results"
    echo "===== AIME24 pass16 eval: $model_name ====="
    export DATASETS="aime24"
    export PASS_K="16"
    export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
    export NNODES="1"
    export GEN_TP="${GEN_TP:-1}"
    export EVAL_MODEL_NAME="$model_name"
    export EVAL_OUTPUT_DIR="$subdir/gen"
    export EVAL_RESULTS_FILE="$subdir/results/results.json"
    export EVAL_RESULTS_CSV_FILE="$subdir/results/results.csv"
    export WRITE_RESULTS_CSV="false"
    export WRITE_PASS16_AGGREGATES="true"
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_diag_${SLURM_JOB_ID:-manual}_$model_name}"
    export TORCHINDUCTOR_CACHE_DIR="$subdir/torchinductor"
    export TRITON_CACHE_DIR="$subdir/triton"
    export TORCH_EXTENSIONS_DIR="$subdir/torch_extensions"
    export VLLM_CACHE_ROOT="$subdir/vllm_cache"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT"

    bash recipe/math_evaluation/benchmark_kl_model.sh "$model_path"

    local gen_parquet="$EVAL_OUTPUT_DIR/aime24_pass16_generation.parquet"
    if [ -f "$gen_parquet" ]; then
        "$PYTHON_BIN" recipe/math_evaluation/diagnose_opd_zero_aime.py \
            --stage analyze-parquet \
            --bad-model "$model_path" \
            --generation-parquet "$gen_parquet" \
            --output-dir "$subdir"
    else
        echo "missing generation parquet: $gen_parquet"
    fi
}

run_metadata_and_smoke "yo_ms40_ref" "$YO_MS40_REF"
run_metadata_and_smoke "yr_ms40_final" "$YR_MS40_FINAL"
run_aime24_pass16 "ms1_no_lora_bad" "$BAD_MODEL"
run_aime24_pass16 "yr_ms40_final" "$YR_MS40_FINAL"

echo "OUT_DIR=$OUT_DIR"
date
nvidia-smi || true
