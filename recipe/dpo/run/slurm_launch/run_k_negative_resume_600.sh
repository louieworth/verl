#!/usr/bin/env bash
# Resume the interrupted kappa-negative sweep in its original experiment.
set -euo pipefail

REPO_ROOT=/scratch/l/luli/src/dpo_pens
SCRIPT_DIR="${REPO_ROOT}/recipe/dpo/run/slurm_launch"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/../common.sh"

kminus="${1:?Expected kappa-negative 3.5, 4.5, or 5.5}"
shift
case "${kminus}" in
  3.5) original_job=513195; minimum_step=550; wandb_run_id=2998f78056162f51 ;;
  4.5) original_job=513196; minimum_step=300; wandb_run_id=999270ac0f34a40f ;;
  5.5) original_job=513197; minimum_step=200; wandb_run_id=03ac4973d249eade ;;
  *) echo "Unsupported kappa-negative resume: ${kminus}" >&2; exit 1 ;;
esac

export SINGLE_WISE_DPO_RUN_TIMESTAMP=20261008_154851
export SINGLE_WISE_DPO_EXPERIMENT_NAME="prospect_click_hist_all_with_weight_kplus3.5_kminus${kminus}_${SINGLE_WISE_DPO_RUN_TIMESTAMP}"
export SINGLE_WISE_DPO_CKPT_DIR="${PENS_CKPT_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_GEN_DIR="${PENS_OUTPUT_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_STATE_FILE="${SINGLE_WISE_DPO_CKPT_DIR}/interleaved_eval_state.json"
export SINGLE_WISE_DPO_PROJECT_NAME=PENS
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_DIR="${PENS_WANDB_ROOT}/${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
export WANDB_MODE=offline
export WANDB_RESUME=allow

# Restore the full model/optimizer/RNG/dataloader checkpoint, not just HF weights.
export SINGLE_WISE_DPO_RESUME_MODE=auto
export SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS=600
export SINGLE_WISE_DPO_SAVE_FREQ=50
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_INTERVAL=50
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_PROMPT_FILE="${PENS_EVAL_PROMPT_FILE}"
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_TEST_FILE="${PENS_EVAL_TEST_FILE}"

# Keep the original learning rate, batching, and feedback objective.
export POINTWISE_DPO_LOSS_TYPE=prospect_dpo
export SINGLE_WISE_DPO_REFERENCE_FREE=true
export SINGLE_WISE_DPO_AVERAGE_LOG_PROB=true
export SINGLE_WISE_DPO_USE_FEEDBACK_WEIGHTS=true
export SINGLE_WISE_DPO_POSITIVE_KAPPA=3.5
export SINGLE_WISE_DPO_NEGATIVE_KAPPA="${kminus}"
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
export SINGLE_WISE_DPO_EARLY_STOP_PATIENCE=0
export SINGLE_WISE_DPO_TARGET_ROUGE_1=1.0
export SINGLE_WISE_DPO_CATASTROPHIC_DROP=1.0
export SINGLE_WISE_DPO_MINIMUM_OUTPUT_COVERAGE=0.0
export SINGLE_WISE_DPO_KEEP_BEST_CHECKPOINT=false

# Avoid the stalled graph initialization and the corrupted compiler cache.
# Sampling, prompt limits, and scoring retain their original settings.
export GEN_ENFORCE_EAGER=true
export HYDRA_FULL_ERROR=1

tracker="${SINGLE_WISE_DPO_CKPT_DIR}/latest_checkpointed_iteration.txt"
[[ -s "${tracker}" ]] || {
  echo "Missing resume checkpoint tracker: ${tracker}" >&2; exit 1;
}
checkpoint_step="$(<"${tracker}")"
if ! [[ "${checkpoint_step}" =~ ^[1-9][0-9]*$ ]] || \
   (( checkpoint_step < minimum_step || checkpoint_step > 600 || checkpoint_step % 50 != 0 )); then
  echo "Invalid saved checkpoint step ${checkpoint_step}; expected ${minimum_step}..600 in steps of 50" >&2
  exit 1
fi
checkpoint="${SINGLE_WISE_DPO_CKPT_DIR}/global_step_${checkpoint_step}"
for required in data.pt actor/fsdp_config.json actor/hf_merged/config.json actor/hf_merged/model.safetensors.index.json; do
  [[ -s "${checkpoint}/${required}" ]] || {
    echo "Missing resume state: ${checkpoint}/${required}" >&2; exit 1;
  }
done
for rank in 0 1 2 3; do
  for kind in model optim extra_state; do
    [[ -s "${checkpoint}/actor/${kind}_world_size_4_rank_${rank}.pt" ]] || {
      echo "Missing rank ${rank} ${kind} resume state in ${checkpoint}" >&2; exit 1;
    }
  done
done
"${PENS_DEFAULTS_PYTHON}" -S - "${SINGLE_WISE_DPO_CHECKPOINT_EVAL_STATE_FILE}" "${checkpoint}/actor/hf_merged" "${checkpoint_step}" <<'PY'
import json
from pathlib import Path
import sys
state = json.loads(Path(sys.argv[1]).read_text())
hf = Path(sys.argv[2])
step = int(sys.argv[3])
if not 0 <= state['last_evaluated_step'] <= step or state.get('should_stop_trial'):
    raise SystemExit('Evaluation state is incompatible with the saved checkpoint')
index = json.loads((hf / 'model.safetensors.index.json').read_text())
for shard in set(index['weight_map'].values()):
    path = hf / shard
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f'Missing merged model shard: {path}')
print(f"Validated saved step {step}; last successful evaluation {state['last_evaluated_step']}")
PY

echo "Resume original job ${original_job}: kappa+=3.5, kappa-=${kminus}, checkpoint=${checkpoint_step}, target=600"
echo "Original experiment: ${SINGLE_WISE_DPO_EXPERIMENT_NAME}; W&B run ${WANDB_RUN_ID}; eager evaluation=${GEN_ENFORCE_EAGER}"
exec bash "${SCRIPT_DIR}/run_job.sh" "all_kplus3.5_kminus${kminus}" "$@"
