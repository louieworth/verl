#!/usr/bin/env bash
#SBATCH --job-name=single_wise_click_hist_positive_only_no_weight_resume
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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# Continue job 503785 in its original experiment, including the missing eval.
export SINGLE_WISE_DPO_EXPERIMENT_NAME=prospect_click_hist_positive_without_weight_20261004_031239
export SINGLE_WISE_DPO_RUN_TIMESTAMP=20261004_031239
export SINGLE_WISE_DPO_CKPT_DIR="${PENS_CKPT_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_GEN_DIR="${PENS_OUTPUT_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_STATE_FILE="${SINGLE_WISE_DPO_CKPT_DIR}/interleaved_eval_state.json"
export SINGLE_WISE_DPO_RESUME_MODE=auto
export SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS=500
export SINGLE_WISE_DPO_SAVE_FREQ=50
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_INTERVAL=50
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_PROMPT_FILE="${PENS_EVAL_PROMPT_FILE}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_TEST_FILE="${PENS_EVAL_TEST_FILE}"
export SINGLE_WISE_DPO_PROJECT_NAME=PENS
export WANDB_RUN_ID=eccfa0df388e4e67
export WANDB_DIR="${PENS_WANDB_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
export WANDB_ENTITY=li-jiang-research
export WANDB_RESUME=allow
export WANDB_MODE=offline

# Retry vLLM initialization without CUDA graph capture and keep full errors.
export GEN_ENFORCE_EAGER=true
export HYDRA_FULL_ERROR=1

tracker="${SINGLE_WISE_DPO_CKPT_DIR}/latest_checkpointed_iteration.txt"
if [[ ! -s "${tracker}" ]]; then
  echo "Missing resume checkpoint tracker: ${tracker}" >&2
  exit 1
fi
checkpoint_step="$(<"${tracker}")"
if ! [[ "${checkpoint_step}" =~ ^(400|450|500)$ ]]; then
  echo "Expected a saved step 400, 450, or 500, got: ${checkpoint_step}" >&2
  exit 1
fi
checkpoint="${SINGLE_WISE_DPO_CKPT_DIR}/global_step_${checkpoint_step}"
for required in data.pt actor/fsdp_config.json; do
  if [[ ! -s "${checkpoint}/${required}" ]]; then
    echo "Missing resume state: ${checkpoint}/${required}" >&2
    exit 1
  fi
done
for rank in 0 1 2 3; do
  for kind in model optim extra_state; do
    if [[ ! -s "${checkpoint}/actor/${kind}_world_size_4_rank_${rank}.pt" ]]; then
      echo "Missing rank ${rank} ${kind} resume state in ${checkpoint}" >&2
      exit 1
    fi
  done
done
echo "Resume job 503785 from saved step ${checkpoint_step}; target 500; W&B run ${WANDB_RUN_ID}"
exec bash "${SCRIPT_DIR}/run_job.sh" positive_no_weight "$@"
