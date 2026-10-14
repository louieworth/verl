#!/usr/bin/env bash
# Continue the four ablation runs from their complete step-500 states.
set -euo pipefail

REPO_ROOT=/scratch/l/luli/src/dpo_pens
SCRIPT_DIR="${REPO_ROOT}/recipe/dpo/run/slurm_launch"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/../common.sh"

variant="${1:?Expected negative, negative_no_weight, positive_no_weight, or all_no_weight}"
shift
case "${variant}" in
  negative)
    original=prospect_click_hist_negative_with_weight_20261004_031023
    train_script=negative_only.sh
    use_weights=true
    ;;
  negative_no_weight)
    original=prospect_click_hist_negative_without_weight_20261004_031006
    train_script=ablation/negative_only_no_weight.sh
    use_weights=false
    ;;
  positive_no_weight)
    original=prospect_click_hist_positive_without_weight_20261004_031239
    train_script=ablation/positive_only_no_weight.sh
    use_weights=false
    ;;
  all_no_weight)
    original=prospect_click_hist_all_without_weight_20261004_030847
    train_script=ablation/all_no_weight.sh
    use_weights=false
    ;;
  *) echo "Unsupported PIFO resume variant: ${variant}" >&2; exit 1 ;;
esac

source_checkpoint="${PENS_CKPT_ROOT}/${original}/global_step_500"
for required in data.pt actor/fsdp_config.json; do
  [[ -s "${source_checkpoint}/${required}" ]] || {
    echo "Missing resume state: ${source_checkpoint}/${required}" >&2; exit 1;
  }
done
for rank in 0 1 2 3; do
  for kind in model optim extra_state; do
    [[ -s "${source_checkpoint}/actor/${kind}_world_size_4_rank_${rank}.pt" ]] || {
      echo "Missing rank ${rank} ${kind} resume state" >&2; exit 1;
    }
  done
done

export SINGLE_WISE_DPO_EXPERIMENT_NAME="${original}_resume600_${SLURM_JOB_ID:-inspect}"
export SINGLE_WISE_DPO_CKPT_DIR="${PENS_CKPT_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_GEN_DIR="${PENS_OUTPUT_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_STATE_FILE="${SINGLE_WISE_DPO_CKPT_DIR}/interleaved_eval_state.json"
export SINGLE_WISE_DPO_PROJECT_NAME=PENS
export WANDB_DIR="${PENS_WANDB_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
unset WANDB_RUN_ID
export WANDB_MODE=offline
export WANDB_RESUME=allow

# Match the original four runs, including their microbatch class reduction.
export POINTWISE_DPO_LOSS_TYPE=prospect_dpo
export SINGLE_WISE_DPO_REFERENCE_FREE=true
export SINGLE_WISE_DPO_AVERAGE_LOG_PROB=true
export SINGLE_WISE_DPO_USE_FEEDBACK_WEIGHTS="${use_weights}"
export SINGLE_WISE_DPO_POSITIVE_KAPPA=3.5
export SINGLE_WISE_DPO_NEGATIVE_KAPPA=1.5
export SINGLE_WISE_DPO_BETA=0.1
export SINGLE_WISE_DPO_ACTOR_LR=5e-6
export SINGLE_WISE_DPO_ACTOR_LR_SCHEDULER_TYPE=constant
export SINGLE_WISE_DPO_ACTOR_LR_WARMUP_STEPS=0
export SINGLE_WISE_DPO_ACTOR_LR_WARMUP_STEPS_RATIO=0.0
export SINGLE_WISE_DPO_TRAIN_BATCH_SIZE=128
export SINGLE_WISE_DPO_MICRO_BATCH_SIZE=4
export SINGLE_WISE_DPO_USE_DYNAMIC_BSZ=false
export SINGLE_WISE_DPO_USE_LENGTH_BUCKET_SAMPLER=true
export SINGLE_WISE_DPO_SEED=42
export SINGLE_WISE_DPO_RESUME_MODE=auto
export SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS=600
export SINGLE_WISE_DPO_SAVE_FREQ=50
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_INTERVAL=50
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_PROMPT_FILE="${PENS_EVAL_PROMPT_FILE}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_TEST_FILE="${PENS_EVAL_TEST_FILE}"
export SINGLE_WISE_DPO_EARLY_STOP_PATIENCE=0
export SINGLE_WISE_DPO_TARGET_ROUGE_1=1.0
export SINGLE_WISE_DPO_CATASTROPHIC_DROP=1.0
export SINGLE_WISE_DPO_MINIMUM_OUTPUT_COVERAGE=0.0
export SINGLE_WISE_DPO_KEEP_BEST_CHECKPOINT=false
export HYDRA_FULL_ERROR=1
if [[ "${variant}" == positive_no_weight ]]; then
  # Retain the successful generation setting from its earlier resume job.
  export GEN_ENFORCE_EAGER=true
fi
loss_args=(
  algorithm.prospect_dpo_global_class_mean=false
  algorithm.prospect_dpo_sft_coef=0.0
  algorithm.prospect_dpo_negative_scale=1.0
  algorithm.prospect_dpo_negative_warmup_steps=0
)

if [[ "${PIFO_RESUME_INSPECT:-false}" == true ]]; then
  inspect_root="/tmp/pifo_resume600_inspect_${variant}"
  export SINGLE_WISE_DPO_CKPT_DIR="${inspect_root}/checkpoints"
  export SINGLE_WISE_DPO_CHECKPOINT_EVAL_GEN_DIR="${inspect_root}/generations"
  export SINGLE_WISE_DPO_CHECKPOINT_EVAL_STATE_FILE="${inspect_root}/eval_state.json"
  export WANDB_DIR="${inspect_root}/wandb"
  export SINGLE_WISE_DPO_REFERENCE_LOGPS_MATERIALIZED_DIR="${inspect_root}/reference_logps"
  export TMPDIR="${inspect_root}/tmp"
  export RAY_TMPDIR="${TMPDIR}/ray"
  export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor"
  export TRITON_CACHE_DIR="${TMPDIR}/triton"
  export VLLM_CACHE_ROOT="${TMPDIR}/vllm"
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export SINGLE_WISE_DPO_NNODES=1
  export SINGLE_WISE_DPO_N_GPUS_PER_NODE=4
  export SINGLE_WISE_DPO_CHECKPOINT_EVAL_NNODES=1
  export SINGLE_WISE_DPO_CHECKPOINT_EVAL_NGPUS_PER_NODE=4
  exec bash "${SCRIPT_DIR}/../${train_script}" "${loss_args[@]}" "$@"
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Submit the resume .sbatch file with sbatch." >&2; exit 1;
}

# Continue in a separate namespace because the evaluator prunes old states.
if [[ ! -f "${SINGLE_WISE_DPO_CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
  mkdir -p "${SINGLE_WISE_DPO_CKPT_DIR}"
  staging="${SINGLE_WISE_DPO_CKPT_DIR}/.step500_copy_${BASHPID}"
  mkdir -p "${staging}/actor"
  cp --reflink=auto "${source_checkpoint}/data.pt" "${staging}/"
  cp --reflink=auto "${source_checkpoint}/actor/fsdp_config.json" "${staging}/actor/"
  for rank in 0 1 2 3; do
    for kind in model optim extra_state; do
      cp --reflink=auto "${source_checkpoint}/actor/${kind}_world_size_4_rank_${rank}.pt" "${staging}/actor/"
    done
  done
  mv "${staging}" "${SINGLE_WISE_DPO_CKPT_DIR}/global_step_500"
  printf '500' > "${SINGLE_WISE_DPO_CKPT_DIR}/latest_checkpointed_iteration.txt"
  printf '{"last_evaluated_step":500,"current_step":500,"evaluations":{},"resume_source":"%s"}\n' \
    "${source_checkpoint}" > "${SINGLE_WISE_DPO_CHECKPOINT_EVAL_STATE_FILE}"
fi

echo "PIFO ${variant}: resume from ${source_checkpoint}; target 600; evaluate at 550 and 600."
echo "Continuation checkpoints: ${SINGLE_WISE_DPO_CKPT_DIR}"
exec bash "${SCRIPT_DIR}/run_job.sh" "${variant}" "${loss_args[@]}" "$@"
