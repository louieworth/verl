#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/common.sh"

is_truthy() {
  local value
  value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

CKPT_DIR="${SINGLE_WISE_DPO_CKPT_DIR:?Set SINGLE_WISE_DPO_CKPT_DIR}"
PYTHON_BIN="${SINGLE_WISE_DPO_PYTHON_BIN:?Set SINGLE_WISE_DPO_PYTHON_BIN}"
EXPERIMENT_NAME="${SINGLE_WISE_DPO_EXPERIMENT_NAME:-$(basename "${CKPT_DIR}")}"
export SINGLE_WISE_DPO_EXPERIMENT_NAME="${EXPERIMENT_NAME}"
pens_init_run "${PENS_RUN_KIND:-dpo}" "$@"
TOTAL_STEPS="${SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS}"
EVAL_INTERVAL="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_INTERVAL:-${PENS_EVAL_INTERVAL}}"
CKPT_DIR="${SINGLE_WISE_DPO_CKPT_DIR}"
EXPERIMENT_NAME="${SINGLE_WISE_DPO_EXPERIMENT_NAME}"

TRAIN_SCRIPT="${SINGLE_WISE_DPO_TRAIN_SCRIPT:-recipe/dpo/run/run_single_wise.sh}"
EVAL_SCRIPT="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_SCRIPT:-recipe/dpo/evaluation/run_pens_personalized_eval.sh}"
CHECKPOINT_MANAGER="${SINGLE_WISE_DPO_CHECKPOINT_MANAGER:-recipe/dpo/evaluation/manage_pens_interleaved_checkpoint.py}"
EVAL_WORKDIR="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_WORKDIR:-${REPO_ROOT}}"
EVAL_TEST_FILE="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_TEST_FILE:-${PENS_EVAL_TEST_FILE}}"
EVAL_PROMPT_FILE="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_PROMPT_FILE:-${PENS_EVAL_PROMPT_FILE}}"
EVAL_GEN_DIR="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_GEN_DIR:-${PENS_OUTPUT_ROOT}/${EXPERIMENT_NAME}}"
EVAL_STATE_FILE="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_STATE_FILE:-${CKPT_DIR}/interleaved_eval_state.json}"
EVAL_NNODES="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_NNODES:-1}"
EVAL_NGPUS_PER_NODE="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_NGPUS_PER_NODE:-8}"
EVAL_GEN_TP="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_GEN_TP:-1}"
EVAL_THINKING_MODE="${SINGLE_WISE_DPO_CHECKPOINT_EVAL_THINKING_MODE:-false}"
EVAL_THINKING_MODE="${EVAL_THINKING_MODE,,}"
case "${EVAL_THINKING_MODE}" in
  true) EVAL_THINKING_TAG="thinkON" ;;
  false) EVAL_THINKING_TAG="thinkOFF" ;;
  skip) EVAL_THINKING_TAG="thinkNA" ;;
  *)
    echo "SINGLE_WISE_DPO_CHECKPOINT_EVAL_THINKING_MODE must be true, false, or skip; got ${EVAL_THINKING_MODE}" >&2
    exit 1
    ;;
esac
RUN_EVAL="${SINGLE_WISE_DPO_RUN_CHECKPOINT_EVAL:-true}"
DRY_RUN="${SINGLE_WISE_DPO_INTERLEAVED_DRY_RUN:-false}"
EARLY_STOP_PATIENCE="${SINGLE_WISE_DPO_EARLY_STOP_PATIENCE:-2}"
EARLY_STOP_MIN_DELTA="${SINGLE_WISE_DPO_EARLY_STOP_MIN_DELTA:-0.0005}"
CATASTROPHIC_DROP="${SINGLE_WISE_DPO_CATASTROPHIC_DROP:-0.01}"
MINIMUM_OUTPUT_COVERAGE="${SINGLE_WISE_DPO_MINIMUM_OUTPUT_COVERAGE:-0.99}"
TARGET_ROUGE_1="${SINGLE_WISE_DPO_TARGET_ROUGE_1:-0.2933888365419484}"
MINIMUM_ROUGE_2="${SINGLE_WISE_DPO_MINIMUM_ROUGE_2:-0.10612834364424847}"
MINIMUM_ROUGE_L="${SINGLE_WISE_DPO_MINIMUM_ROUGE_L:-0.24049149824883745}"
CHECKPOINT_SELECTION_MODE="${SINGLE_WISE_DPO_CHECKPOINT_SELECTION_MODE:-rouge_1}"
SELECTION_BASELINE_ROUGE_1="${SINGLE_WISE_DPO_SELECTION_BASELINE_ROUGE_1:-${TARGET_ROUGE_1}}"
SELECTION_BASELINE_ROUGE_2="${SINGLE_WISE_DPO_SELECTION_BASELINE_ROUGE_2:-${MINIMUM_ROUGE_2}}"
SELECTION_BASELINE_ROUGE_L="${SINGLE_WISE_DPO_SELECTION_BASELINE_ROUGE_L:-${MINIMUM_ROUGE_L}}"
SELECTION_MINIMUM_COUNT="${SINGLE_WISE_DPO_SELECTION_MINIMUM_COUNT:-0}"

if ! [[ "${TOTAL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS must be a positive integer, got: ${TOTAL_STEPS}" >&2
  exit 1
fi
if ! [[ "${EVAL_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SINGLE_WISE_DPO_CHECKPOINT_EVAL_INTERVAL must be a positive integer, got: ${EVAL_INTERVAL}" >&2
  exit 1
fi
if ! is_truthy "${RUN_EVAL}"; then
  echo "Interleaved training requires SINGLE_WISE_DPO_RUN_CHECKPOINT_EVAL=true." >&2
  exit 1
fi

eval_steps=()
for ((step = EVAL_INTERVAL; step <= TOTAL_STEPS; step += EVAL_INTERVAL)); do
  eval_steps+=("${step}")
done
if (( TOTAL_STEPS % EVAL_INTERVAL != 0 )); then
  eval_steps+=("${TOTAL_STEPS}")
fi

if is_truthy "${DRY_RUN}"; then
  echo "Interleaved train/eval steps: ${eval_steps[*]}"
  echo "Experiment: ${EXPERIMENT_NAME}"
  echo "Generation directory: ${EVAL_GEN_DIR}"
  echo "W&B directory: ${WANDB_DIR}"
  echo "Early stop: patience=${EARLY_STOP_PATIENCE}, min_delta=${EARLY_STOP_MIN_DELTA}, catastrophic_drop=${CATASTROPHIC_DROP}"
  echo "Target: R1>${TARGET_ROUGE_1}, R2>=${MINIMUM_ROUGE_2}, RL>=${MINIMUM_ROUGE_L}"
  echo "Checkpoint selection: mode=${CHECKPOINT_SELECTION_MODE}, baseline=${SELECTION_BASELINE_ROUGE_1}/${SELECTION_BASELINE_ROUGE_2}/${SELECTION_BASELINE_ROUGE_L}, minimum_count=${SELECTION_MINIMUM_COUNT}"
  checkpoint_retention="current only"
  if is_truthy "${SINGLE_WISE_DPO_KEEP_BEST_CHECKPOINT:-false}"; then
    checkpoint_retention="best and current"
  fi
  for step in "${eval_steps[@]}"; do
    echo "DRY RUN: train until step ${step}, save, exit training, evaluate, retain ${checkpoint_retention}, resume"
  done
  exit 0
fi

# Hydra inspection modes should inspect one training invocation only.
for arg in "$@"; do
  case "${arg}" in
    --cfg|--cfg=*|--info|--info=*|--help|-h)
      export SINGLE_WISE_DPO_STOP_AT_STEP="${eval_steps[0]}"
      bash "${TRAIN_SCRIPT}" "$@"
      exit 0
      ;;
  esac
done

for required_file in "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${CHECKPOINT_MANAGER}" "${EVAL_TEST_FILE}" "${EVAL_PROMPT_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing interleaved train/eval input: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing interleaved train/eval Python executable: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}" "${EVAL_GEN_DIR}"

find_current_step() {
  local max_step=0
  local checkpoint_path checkpoint_step
  shopt -s nullglob
  for checkpoint_path in "${CKPT_DIR}"/global_step_*; do
    checkpoint_step="${checkpoint_path##*/global_step_}"
    if [[ -d "${checkpoint_path}" && "${checkpoint_step}" =~ ^[0-9]+$ ]] && (( checkpoint_step > max_step )); then
      max_step="${checkpoint_step}"
    fi
  done
  shopt -u nullglob
  printf '%s' "${max_step}"
}

find_last_evaluated_step() {
  if [[ ! -f "${EVAL_STATE_FILE}" ]]; then
    printf '0'
    return 0
  fi
  "${PYTHON_BIN}" -c \
    'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("last_evaluated_step", 0)))' \
    "${EVAL_STATE_FILE}"
}

export SINGLE_WISE_DPO_RUN_POST_TRAIN_EVAL=false
export SINGLE_WISE_DPO_KEEP_ONLY_LATEST_ROLLING_CKPT=false

echo "Interleaved train/eval steps: ${eval_steps[*]}"
echo "W&B directory: ${WANDB_DIR}"
echo "Checkpoint state: ${EVAL_STATE_FILE}"

for step in "${eval_steps[@]}"; do
  current_step="$(find_current_step)"
  if (( current_step < step )); then
    export SINGLE_WISE_DPO_STOP_AT_STEP="${step}"
    echo "Training from step ${current_step} until step ${step} (total schedule: ${TOTAL_STEPS})"
    bash "${TRAIN_SCRIPT}" "$@"
    current_step="$(find_current_step)"
    if (( current_step != step )); then
      echo "Expected current checkpoint global_step_${step}, found global_step_${current_step}." >&2
      exit 1
    fi
  elif (( current_step > step )); then
    echo "Skipping completed train/eval boundary ${step}; current checkpoint is step ${current_step}."
    continue
  fi

  last_evaluated_step="$(find_last_evaluated_step)"
  if (( last_evaluated_step >= step )); then
    echo "Step ${step} was already evaluated; continuing."
    continue
  fi

  model_path="${CKPT_DIR}/global_step_${step}/${SINGLE_WISE_DPO_CHECKPOINT_EVAL_MODEL_LEAF:-actor/hf_merged}"
  if [[ "${PENS_RUN_KIND:-dpo}" == sft ]]; then
    # Resuming after a failed eval can skip training. Validate/repair that
    # boundary's export before vLLM starts, even when hf_merged already exists.
    "${PYTHON_BIN}" "${REPO_ROOT}/recipe/dpo/scripts/merge_sft_lora_to_hf.py" \
      --src "${CKPT_DIR}/global_step_${step}/huggingface" --dst "${model_path}" \
      --lora-rank "${SFT_LORA_RANK:-64}" --lora-alpha "${SFT_LORA_ALPHA:-128}"
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "Missing merged model for step ${step}: ${model_path}" >&2
    exit 1
  fi

  model_key="${EXPERIMENT_NAME}_step_${step}"
  result_key="${model_key}__${EVAL_THINKING_TAG}__step_${step}"
  raw_file="${EVAL_GEN_DIR}/${model_key}.parquet"
  echo "Training process stopped at step ${step}; starting GPU eval."
  MODEL_PATH="${model_path}" \
    MODEL_KEY="${model_key}" \
    DATE_TAG="step_${step}" \
    TEST_FILE="${EVAL_TEST_FILE}" \
    PROMPT_FILE="${EVAL_PROMPT_FILE}" \
    RAW_FILE="${raw_file}" \
    PENS_EVAL_STATE_FILE="${EVAL_STATE_FILE}" \
    RESULTS_GEN_DIR="${EVAL_GEN_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    WORKDIR="${EVAL_WORKDIR}" \
    NNODES="${EVAL_NNODES}" \
    NGPUS_PER_NODE="${EVAL_NGPUS_PER_NODE}" \
    GEN_TP="${EVAL_GEN_TP}" \
    GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}" \
    VLLM_ENABLE_THINKING="${EVAL_THINKING_MODE}" \
    bash "${EVAL_SCRIPT}"

  manager_extra_args=()
  if is_truthy "${SINGLE_WISE_DPO_KEEP_BEST_CHECKPOINT:-false}"; then
    manager_extra_args+=(--keep-best-checkpoint)
  fi
  manager_output="$("${PYTHON_BIN}" "${CHECKPOINT_MANAGER}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --state-file "${EVAL_STATE_FILE}" \
    --result-key "${result_key}" \
    --step "${step}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
    --catastrophic-drop "${CATASTROPHIC_DROP}" \
    --minimum-output-coverage "${MINIMUM_OUTPUT_COVERAGE}" \
    --target-rouge-1 "${TARGET_ROUGE_1}" \
    --minimum-rouge-2 "${MINIMUM_ROUGE_2}" \
    --minimum-rouge-l "${MINIMUM_ROUGE_L}" \
    --selection-mode "${CHECKPOINT_SELECTION_MODE}" \
    --selection-baseline-rouge-1 "${SELECTION_BASELINE_ROUGE_1}" \
    --selection-baseline-rouge-2 "${SELECTION_BASELINE_ROUGE_2}" \
    --selection-baseline-rouge-l "${SELECTION_BASELINE_ROUGE_L}" \
    --selection-minimum-count "${SELECTION_MINIMUM_COUNT}" "${manager_extra_args[@]}")"
  printf '%s\n' "${manager_output}"

  should_stop_trial="$("${PYTHON_BIN}" -c \
    'import json, sys; print("true" if json.load(open(sys.argv[1], encoding="utf-8")).get("should_stop_trial") else "false")' \
    "${EVAL_STATE_FILE}")"
  if [[ "${should_stop_trial}" == "true" ]]; then
    stop_reason="$("${PYTHON_BIN}" -c \
      'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("stop_reason") or "unspecified")' \
      "${EVAL_STATE_FILE}")"
    echo "Stopping trial at step ${step}: ${stop_reason}"
    break
  fi
done

echo "Interleaved train/eval driver finished. Requested max steps=${TOTAL_STEPS}, interval=${EVAL_INTERVAL}."
