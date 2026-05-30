#!/bin/bash
#SBATCH --job-name=opd-ms40-fwd-yr-h100x8-18h
#SBATCH --partition=gpubase_bynode_b3
#SBATCH --nodes=2
#SBATCH --gres=gpu:h100:4
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --output=/scratch/l/luli/src/verl/recipe/opd/run/opd/sbatch/logs/opd-ms40-fwd-yr-h100x8-18h-%j.out
#SBATCH --error=/scratch/l/luli/src/verl/recipe/opd/run/opd/sbatch/logs/opd-ms40-fwd-yr-h100x8-18h-%j.err

set -E
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/recipe/opd/run/opd/sbatch/_slurm_gpu_common.sh" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/recipe/opd/run/opd/sbatch"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$SCRIPT_DIR/_slurm_gpu_common.sh"
opd_install_failure_trap

export NNODES="${NNODES:-2}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export SLURM_GPU_TYPE="${SLURM_GPU_TYPE:-h100}"
export RUN_SCRIPT="${RUN_SCRIPT:-/scratch/l/luli/src/verl/recipe/opd/run/opd/forward_y_r.sh}"
export Y_MODE="${Y_MODE:-y_r}"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-refine}"
export USE_LORA="${USE_LORA:-false}"
export MULTI_STEP="${MULTI_STEP:-40}"

# Resume by default when a matching prior run exists. Use
# PIPELINE_RESUME_MODE=fresh when intentionally starting a new run.
export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"
export PIPELINE_CLEANUP_BATCH_DATA="${PIPELINE_CLEANUP_BATCH_DATA:-true}"
# y_r refine is materially heavier than y_o; keep the smaller batch/chunk
# defaults used by the no-LoRA H200 y_r preset.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export KL_FULL_VOCAB_CHUNK_SIZE="${KL_FULL_VOCAB_CHUNK_SIZE:-160}"
export PASS_K="${PASS_K:-16}"

opd_run_slurm_gpu "$@"
