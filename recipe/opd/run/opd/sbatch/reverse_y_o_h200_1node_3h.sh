#!/bin/bash
#SBATCH --job-name=opd-ms40-rev-yo-h200x8-3h
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=64
#SBATCH --output=/scratch/l/luli/src/verl/recipe/opd/run/opd/sbatch/logs/opd-ms40-rev-yo-h200x8-3h-%j.out
#SBATCH --error=/scratch/l/luli/src/verl/recipe/opd/run/opd/sbatch/logs/opd-ms40-rev-yo-h200x8-3h-%j.err

set -E
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/recipe/opd/run/opd/sbatch/_h100_4node_common.sh" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/recipe/opd/run/opd/sbatch"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$SCRIPT_DIR/_h100_4node_common.sh"
opd_install_keepalive_trap

export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export SLURM_GPU_TYPE="${SLURM_GPU_TYPE:-h200}"
export RUN_SCRIPT="${RUN_SCRIPT:-/scratch/l/luli/src/verl/recipe/opd/run/opd/reverse_y_o.sh}"
export Y_MODE="${Y_MODE:-y_o}"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"

# Resume by default when a matching prior run exists. Use
# PIPELINE_RESUME_MODE=fresh when intentionally starting a new run.
export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PASS_K="${PASS_K:-16}"

opd_run_h100_4node "$@"
