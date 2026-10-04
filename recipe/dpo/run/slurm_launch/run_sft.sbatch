#!/usr/bin/env bash
#SBATCH --job-name=pens_sft_from0_500
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=80gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --chdir=/scratch/l/luli/src/dpo_pens
#SBATCH --output=/scratch/l/luli/src/dpo_pens/recipe/dpo/run/slurm_launch/logs/%x-%j.out
#SBATCH --error=/scratch/l/luli/src/dpo_pens/recipe/dpo/run/slurm_launch/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=/scratch/l/luli/src/dpo_pens
source "${REPO_ROOT}/recipe/dpo/run/common.sh"

# A new output directory starts at zero; auto-resume is needed only for the
# later train/eval segments inside this same job.
export SFT_EXPERIMENT_NAME="sft_click_hist_positive_from0_500_${SLURM_JOB_ID:-dryrun}"
export SFT_DEFAULT_LOCAL_DIR="${PENS_CKPT_ROOT}/${SFT_EXPERIMENT_NAME}"
export SFT_CHECKPOINT_EVAL_GEN_DIR="${PENS_OUTPUT_ROOT}/${SFT_EXPERIMENT_NAME}"
export SFT_CHECKPOINT_EVAL_STATE_FILE="${SFT_DEFAULT_LOCAL_DIR}/interleaved_eval_state.json"
if [[ -e "${SFT_DEFAULT_LOCAL_DIR}" || -e "${PENS_WANDB_ROOT}/${SFT_EXPERIMENT_NAME}" || -e "${SFT_CHECKPOINT_EVAL_GEN_DIR}" ]]; then
  echo "Fresh SFT run requires unused output directories: ${SFT_EXPERIMENT_NAME}" >&2
  exit 1
fi

export SFT_TOTAL_TRAINING_STEPS=500
export SFT_TOTAL_EPOCHS=1
export SFT_LR=5e-6
export SFT_LR_SCHEDULER_TYPE=constant
export SFT_LR_WARMUP_STEPS=0
export SFT_LR_WARMUP_STEPS_RATIO=0.0
export SFT_LR_MIN_RATIO=0.0
export SFT_WEIGHT_DECAY=0.01
export SFT_SEED=42
export SFT_TRAIN_BATCH_SIZE=128
export SFT_MICRO_BATCH_SIZE_PER_GPU=4
export SFT_LORA_RANK=64
export SFT_LORA_ALPHA=128
export SFT_LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj]'
export SFT_RESUME_MODE=auto
export SFT_STOP_AT_STEP=500
export SFT_SAVE_FREQ=100
export SFT_CHECKPOINT_EVAL_INTERVAL=100
export SFT_MAX_CKPT_TO_KEEP=2
export SFT_KEEP_BEST_CHECKPOINT=false
export SFT_RUN_CHECKPOINT_EVAL=true
export SFT_EARLY_STOP_PATIENCE=0
export SFT_CATASTROPHIC_DROP=1.0
export SFT_TARGET_ROUGE_1=1.0
export SFT_MINIMUM_ROUGE_2=0.0
export SFT_MINIMUM_ROUGE_L=0.0
export SFT_MINIMUM_OUTPUT_COVERAGE=0.0
export SFT_CHECKPOINT_EVAL_PROMPT_FILE="${PENS_EVAL_PROMPT_FILE}"
export SFT_CHECKPOINT_EVAL_TEST_FILE="${PENS_EVAL_TEST_FILE}"
export SFT_CHECKPOINT_EVAL_GEN_TP=1
export SFT_CHECKPOINT_EVAL_THINKING_MODE=false
export SFT_LOGGER='[console,wandb]'
export PENS_PROJECT_NAME=PENS
export SFT_PROJECT_NAME=PENS
export WANDB_ENTITY="${WANDB_ENTITY:-li-jiang-research}"
export WANDB_MODE=offline
# Generate a new run ID from the new experiment name.
unset WANDB_RUN_ID WANDB_DIR

echo "Fresh SFT: steps=0..500; LR=${SFT_LR}; eval steps=100,200,300,400,500"
echo "Evaluation data: ${SFT_CHECKPOINT_EVAL_TEST_FILE}"
exec bash "${REPO_ROOT}/recipe/dpo/run/slurm_launch/run_job.sh" sft "$@"
