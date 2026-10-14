#!/usr/bin/env bash
#SBATCH --job-name=pifo_all_kplus3.5_kminus2.5
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=80gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/l/luli/src/dpo_pens
#SBATCH --output=/scratch/l/luli/src/dpo_pens/recipe/dpo/run/slurm_launch/logs/%x-%j.out
#SBATCH --error=/scratch/l/luli/src/dpo_pens/recipe/dpo/run/slurm_launch/logs/%x-%j.err

set -euo pipefail
exec bash /scratch/l/luli/src/dpo_pens/recipe/dpo/run/slurm_launch/run_job.sh all_kplus3.5_kminus2.5 "$@"
