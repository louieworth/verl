#!/bin/bash
#SBATCH --job-name=opd-ms40-fwd-yr-h200x8
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=/scratch/l/luli/src/verl/recipe/opd/run/opd/sbatch/logs/opd-ms40-fwd-yr-h200x8-%j.out
#SBATCH --error=/scratch/l/luli/src/verl/recipe/opd/run/opd/sbatch/logs/opd-ms40-fwd-yr-h200x8-%j.err

set -E
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/recipe/opd/run/opd/sbatch/_slurm_gpu_common.sh" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/recipe/opd/run/opd/sbatch"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$SCRIPT_DIR/_slurm_gpu_common.sh"
opd_install_failure_trap

export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export SLURM_GPU_TYPE="${SLURM_GPU_TYPE:-h200}"
export RUN_SCRIPT="${RUN_SCRIPT:-/scratch/l/luli/src/verl/recipe/opd/run/opd/forward_y_r.sh}"
export Y_MODE="${Y_MODE:-y_r}"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-refine}"
export USE_LORA="${USE_LORA:-false}"
export MULTI_STEP="${MULTI_STEP:-40}"

# Resume by default when a matching prior run exists. Use
# PIPELINE_RESUME_MODE=fresh when intentionally starting a new run.
export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"
export PIPELINE_CLEANUP_BATCH_DATA="${PIPELINE_CLEANUP_BATCH_DATA:-true}"
# y_r refine is materially heavier than y_o; 4-way per-GPU batch was hitting
# host RSS OOM in the training step. Use the heavier full-vocab preset's
# smaller batch default here.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export KL_FULL_VOCAB_CHUNK_SIZE="${KL_FULL_VOCAB_CHUNK_SIZE:-320}"
export PASS_K="${PASS_K:-16}"

opd_run_slurm_gpu "$@"
