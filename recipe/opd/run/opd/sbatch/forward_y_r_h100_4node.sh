#!/bin/bash
#SBATCH --job-name=opd-ms40-fwd-yr-h100x16
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=48
#SBATCH --output=/scratch/l/luli/src/verl/recipe/opd/run/opd/sbatch/logs/opd-ms40-fwd-yr-h100x16-%j.out
#SBATCH --error=/scratch/l/luli/src/verl/recipe/opd/run/opd/sbatch/logs/opd-ms40-fwd-yr-h100x16-%j.err

set -E
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/recipe/opd/run/opd/sbatch/_h100_4node_common.sh" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/recipe/opd/run/opd/sbatch"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$SCRIPT_DIR/_h100_4node_common.sh"
opd_install_keepalive_trap

export RUN_SCRIPT="${RUN_SCRIPT:-/scratch/l/luli/src/verl/recipe/opd/run/opd/forward_y_r.sh}"
export Y_MODE="${Y_MODE:-y_r}"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-refine}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"

opd_run_h100_4node "$@"
