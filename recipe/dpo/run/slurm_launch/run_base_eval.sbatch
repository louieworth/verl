#!/usr/bin/env bash
#SBATCH --job-name=pens_base_step0_eval
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=80gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --chdir=/scratch/l/luli/src/dpo_pens
#SBATCH --output=/scratch/l/luli/src/dpo_pens/recipe/dpo/run/slurm_launch/logs/%x-%j.out
#SBATCH --error=/scratch/l/luli/src/dpo_pens/recipe/dpo/run/slurm_launch/logs/%x-%j.err

set -euo pipefail
export REPO_ROOT=/scratch/l/luli/src/dpo_pens
source "${REPO_ROOT}/recipe/dpo/run/common.sh"
export PATH="$(dirname -- "${PENS_PYTHON_BIN}"):${PATH}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export WANDB_MODE=offline PENS_WANDB_ENABLED=false
export NNODES=1 NGPUS_PER_NODE=4 GEN_TP=1
export GEN_TEMPERATURE=0.0 VLLM_ENABLE_THINKING=false
export MODEL_PATH="${PENS_MODEL_DIR}"
export MODEL_KEY=qwen3_4b_instruct_base
export DATE_TAG=step_0
export RESULTS_GEN_DIR="${REPO_ROOT}/outputs/pens_base_step_0_${SLURM_JOB_ID}"
export RAW_FILE="${RESULTS_GEN_DIR}/qwen3_4b_instruct_base_step_0.parquet"
export PROMPT_FILE="${PENS_EVAL_PROMPT_FILE}" TEST_FILE="${PENS_EVAL_TEST_FILE}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/pens_base_${SLURM_JOB_ID}"
export RAY_TMPDIR="/tmp/pens_base_ray_${SLURM_JOB_ID}"
export PENS_EVAL_RAY_TMPDIR="${RAY_TMPDIR}" PENS_EVAL_RAY_ADDRESS=local
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton" VLLM_CACHE_ROOT="${TMPDIR}/vllm"
export RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK}"
unset RAY_ADDRESS PENS_EVAL_STATE_FILE RESULT_JSON_FILE
mkdir -p "${RESULTS_GEN_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"

bash "${REPO_ROOT}/recipe/dpo/evaluation/run_pens_personalized_eval.sh" \
    | tee "${RESULTS_GEN_DIR}/evaluation.log"
"${PENS_PYTHON_BIN}" -S "${REPO_ROOT}/paper_plots/code/record_base_results.py" \
    --eval-log "${RESULTS_GEN_DIR}/evaluation.log" --repo-root "${REPO_ROOT}"
