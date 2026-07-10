#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PENS_SCRATCH_ROOT=""

# Resolve the selected profile directly into the variables consumed below.
# Explicit SINGLE_WISE_DPO_* values always take precedence over Run-E defaults.
if [[ -n "${PENS_DPO_DATA_VARIANT:-}" ]]; then
  cd "${VERL_ROOT}"
  export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

  case "${PENS_DPO_DATA_VARIANT}" in
    sampled)
      DATA_ROOT="${SINGLE_WISE_DPO_DATA_ROOT:-${VERL_ROOT}/data/pens_click_hist_runE_23pct_seed42/sampled_313600_seed42}"
      TRAIN_FILE="${SINGLE_WISE_DPO_TRAIN_FILE:-}"
      ;;
    ordered_part)
      TRAIN_FILE="${SINGLE_WISE_DPO_TRAIN_FILE:-/data/data/jiangli/Microsoft-PeNS/runE_step0201_to_step1400_ordered_train.parquet}"
      DATA_ROOT="${SINGLE_WISE_DPO_DATA_ROOT:-$(dirname -- "${TRAIN_FILE}")}"
      ;;
    full)
      DATA_ROOT="${SINGLE_WISE_DPO_DATA_ROOT:-${VERL_ROOT}/data/pens_click_hist_runE_23pct_seed42}"
      TRAIN_FILE="${SINGLE_WISE_DPO_TRAIN_FILE:-}"
      PENS_SCRATCH_ROOT="${SINGLE_WISE_DPO_SCRATCH_ROOT:-/opt/dlami/nvme/lijiang3/pens_dpo}"
      ALL_EVAL_INTERVAL="${SINGLE_WISE_DPO_ALL_EVAL_INTERVAL:-400}"
      export RAY_TMPDIR="${RAY_TMPDIR:-/opt/dlami/nvme/ray}"
      ;;
    *)
      echo "Unsupported PENS_DPO_DATA_VARIANT: ${PENS_DPO_DATA_VARIANT}" >&2
      echo "Expected one of: sampled, ordered_part, full" >&2
      exit 1
      ;;
  esac

  LOSS_TYPE="${POINTWISE_DPO_LOSS_TYPE:-${SINGLE_WISE_DPO_LOSS_TYPE:-prospect_dpo}}"
  MODEL_DIR="${SINGLE_WISE_DPO_MODEL_DIR:-${SINGLE_WISE_DPO_WARMUP_BASE_DIR:-/data/data/jiangli/models/pens/sft_warmup_pos_qwen3-4b-instruct-2507/global_step_50/hf_merged}}"
  REFERENCE_MODEL_DIR="${SINGLE_WISE_DPO_REFERENCE_MODEL_DIR:-${MODEL_DIR}}"
  TRAIN_BATCH_SIZE="${SINGLE_WISE_DPO_TRAIN_BATCH_SIZE:-224}"
  SEED="${SINGLE_WISE_DPO_SEED:-42}"
  APPLY_CHAT_TEMPLATE_KWARGS="${SINGLE_WISE_DPO_APPLY_CHAT_TEMPLATE_KWARGS:-{enable_thinking:false}}"
  ALPHA_TAU="${SINGLE_WISE_DPO_ALPHA_TAU:-0.0}"
  ALPHA_K="${SINGLE_WISE_DPO_ALPHA_K:-4.0}"
  LAMBDA_GAMMA="${SINGLE_WISE_DPO_LAMBDA_GAMMA:-1.5}"
  NNODES="${SINGLE_WISE_DPO_NNODES:-1}"
  N_GPUS_PER_NODE="${SINGLE_WISE_DPO_N_GPUS_PER_NODE:-8}"
  NCCL_TIMEOUT="${SINGLE_WISE_DPO_NCCL_TIMEOUT:-7200}"
  USE_REMOVE_PADDING="${SINGLE_WISE_DPO_USE_REMOVE_PADDING:-true}"
  USE_DYNAMIC_BSZ="${SINGLE_WISE_DPO_USE_DYNAMIC_BSZ:-false}"
  MAX_TOKEN_LEN_PER_GPU="${SINGLE_WISE_DPO_MAX_TOKEN_LEN_PER_GPU:-9216}"
  REFERENCE_LOGPS_NUM_WORKERS="${SINGLE_WISE_DPO_REFERENCE_LOGPS_NUM_WORKERS:-${N_GPUS_PER_NODE}}"
  REFERENCE_LOGPS_ROWS_PER_TASK="${SINGLE_WISE_DPO_REFERENCE_LOGPS_ROWS_PER_TASK:-256}"
  POST_TRAIN_EVAL_NGPUS_PER_NODE="${SINGLE_WISE_DPO_POST_TRAIN_EVAL_NGPUS_PER_NODE:-8}"

  if [[ "${PENS_DPO_DATA_VARIANT}" == "full" ]]; then
    CKPT_ROOT="${SINGLE_WISE_DPO_CKPT_ROOT:-${PENS_SCRATCH_ROOT}/models/pens}"
    RUN_POST_TRAIN_EVAL=false
    SAVE_FREQ="${SINGLE_WISE_DPO_SAVE_FREQ:-${ALL_EVAL_INTERVAL}}"
  else
    CKPT_ROOT="${SINGLE_WISE_DPO_CKPT_ROOT:-/data/data/jiangli/models/pens}"
    RUN_POST_TRAIN_EVAL="${SINGLE_WISE_DPO_RUN_POST_TRAIN_EVAL:-true}"
    SAVE_FREQ="${SINGLE_WISE_DPO_SAVE_FREQ:-200}"
  fi

  export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
  export RESULT_JSON_FILE="${RESULT_JSON_FILE:-${SINGLE_WISE_DPO_POST_TRAIN_EVAL_RESULT_JSON_FILE:-${VERL_ROOT}/results/qwen3-4b-instruct-2507_warmup_base.json}}"
else
  DATA_ROOT="${SINGLE_WISE_DPO_DATA_ROOT:-${VERL_ROOT}/data/pens_click_hist_runE_23pct_seed42/sampled_313600_seed42}"
  TRAIN_FILE="${SINGLE_WISE_DPO_TRAIN_FILE:-}"
  LOSS_TYPE="${POINTWISE_DPO_LOSS_TYPE:-${SINGLE_WISE_DPO_LOSS_TYPE:-single_wise_dpo}}"
  MODEL_DIR="${SINGLE_WISE_DPO_MODEL_DIR:-/data/.huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
  REFERENCE_MODEL_DIR="${SINGLE_WISE_DPO_REFERENCE_MODEL_DIR:-}"
  CKPT_ROOT="${SINGLE_WISE_DPO_CKPT_ROOT:-/data/data/jiangli/models/pens}"
  TRAIN_BATCH_SIZE="${SINGLE_WISE_DPO_TRAIN_BATCH_SIZE:-112}"
  SEED="${SINGLE_WISE_DPO_SEED:-}"
  APPLY_CHAT_TEMPLATE_KWARGS="${SINGLE_WISE_DPO_APPLY_CHAT_TEMPLATE_KWARGS:-}"
  ALPHA_TAU="${SINGLE_WISE_DPO_ALPHA_TAU:-0.2}"
  ALPHA_K="${SINGLE_WISE_DPO_ALPHA_K:-10.0}"
  LAMBDA_GAMMA="${SINGLE_WISE_DPO_LAMBDA_GAMMA:-2.0}"
  NNODES="${SINGLE_WISE_DPO_NNODES:-4}"
  N_GPUS_PER_NODE="${SINGLE_WISE_DPO_N_GPUS_PER_NODE:-4}"
  NCCL_TIMEOUT="${SINGLE_WISE_DPO_NCCL_TIMEOUT:-600}"
  USE_REMOVE_PADDING="${SINGLE_WISE_DPO_USE_REMOVE_PADDING:-false}"
  USE_DYNAMIC_BSZ="${SINGLE_WISE_DPO_USE_DYNAMIC_BSZ:-true}"
  MAX_TOKEN_LEN_PER_GPU="${SINGLE_WISE_DPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
  SAVE_FREQ="${SINGLE_WISE_DPO_SAVE_FREQ:--1}"
  REFERENCE_LOGPS_NUM_WORKERS="${SINGLE_WISE_DPO_REFERENCE_LOGPS_NUM_WORKERS:-0}"
  REFERENCE_LOGPS_ROWS_PER_TASK="${SINGLE_WISE_DPO_REFERENCE_LOGPS_ROWS_PER_TASK:-2048}"
  RUN_POST_TRAIN_EVAL="${SINGLE_WISE_DPO_RUN_POST_TRAIN_EVAL:-false}"
  POST_TRAIN_EVAL_NGPUS_PER_NODE="${SINGLE_WISE_DPO_POST_TRAIN_EVAL_NGPUS_PER_NODE:-1}"
fi

# Runtime environment inherited by the trainer and evaluation subprocesses.
export RAY_TMPDIR="${RAY_TMPDIR:-${SLURM_TMPDIR:-/tmp}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Files and paths shared by generic and Run-E profiles.
INPUT_VARIANT="${SINGLE_WISE_DPO_INPUT_VARIANT:-click_hist}"
SAMPLE_VARIANT="${SINGLE_WISE_DPO_SAMPLE_VARIANT:-all}"
TOKENIZER_PATH="${SINGLE_WISE_DPO_TOKENIZER_PATH:-${MODEL_DIR}}"
PROJECT_NAME="${SINGLE_WISE_DPO_PROJECT_NAME:-PENS}"
USE_LORA="${SINGLE_WISE_DPO_USE_LORA:-true}"
PYTHON_BIN="${SINGLE_WISE_DPO_PYTHON_BIN:-/data/conda/envs/verl/bin/python}"
EXPORT_HF_MERGED="${SINGLE_WISE_DPO_EXPORT_HF_MERGED:-true}"
EXPORT_HF_MERGED_DTYPE="${SINGLE_WISE_DPO_EXPORT_HF_MERGED_DTYPE:-bfloat16}"
EXPORT_HF_MERGED_MAX_SHARD_SIZE="${SINGLE_WISE_DPO_EXPORT_HF_MERGED_MAX_SHARD_SIZE:-5GB}"
EXPORT_TRUST_REMOTE_CODE="${SINGLE_WISE_DPO_EXPORT_TRUST_REMOTE_CODE:-true}"

# Data schema and loading.
PROMPT_KEY="${SINGLE_WISE_DPO_PROMPT_KEY:-prompt}"
RESPONSE_KEY="${SINGLE_WISE_DPO_RESPONSE_KEY:-response}"
LABEL_KEY="${SINGLE_WISE_DPO_LABEL_KEY:-label}"
S_DWELL_KEY="${SINGLE_WISE_DPO_S_DWELL_KEY:-s_dwell}"
P_CTR_KEY="${SINGLE_WISE_DPO_P_CTR_KEY:-p_ctr}"
MAX_PROMPT_LENGTH="${SINGLE_WISE_DPO_MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${SINGLE_WISE_DPO_MAX_RESPONSE_LENGTH:-48}"
PROMPT_TRUNCATION="${SINGLE_WISE_DPO_PROMPT_TRUNCATION:-right}"
DATALOADER_NUM_WORKERS="${SINGLE_WISE_DPO_DATALOADER_NUM_WORKERS:-8}"

# Algorithm hyperparameters.
BETA="${SINGLE_WISE_DPO_BETA:-0.1}"
ALPHA_MAX="${SINGLE_WISE_DPO_ALPHA_MAX:-2.0}"
LAMBDA_MAX="${SINGLE_WISE_DPO_LAMBDA_MAX:-2.0}"
AVERAGE_LOG_PROB="${SINGLE_WISE_DPO_AVERAGE_LOG_PROB:-true}"

# Optimizer / LR schedule.
ACTOR_LR="${SINGLE_WISE_DPO_ACTOR_LR:-5e-6}"
ACTOR_LR_SCHEDULER_TYPE="${SINGLE_WISE_DPO_ACTOR_LR_SCHEDULER_TYPE:-cosine}"
ACTOR_LR_WARMUP_STEPS_RATIO="${SINGLE_WISE_DPO_ACTOR_LR_WARMUP_STEPS_RATIO:-0.03}"
ACTOR_LR_MIN_RATIO="${SINGLE_WISE_DPO_ACTOR_LR_MIN_RATIO:-0.1}"
ACTOR_LR_NUM_CYCLES="${SINGLE_WISE_DPO_ACTOR_LR_NUM_CYCLES:-0.5}"
ACTOR_WEIGHT_DECAY="${SINGLE_WISE_DPO_ACTOR_WEIGHT_DECAY:-0.01}"

# Model and parallelism.
ROLLOUT_TP_SIZE="${SINGLE_WISE_DPO_ROLLOUT_TP_SIZE:-1}"
ATTN_IMPLEMENTATION="${SINGLE_WISE_DPO_ATTN_IMPLEMENTATION:-flash_attention_2}"
MODEL_DTYPE="${SINGLE_WISE_DPO_MODEL_DTYPE:-bf16}"
ALLOW_UNSUPPORTED_REMOVE_PADDING="${SINGLE_WISE_DPO_ALLOW_UNSUPPORTED_REMOVE_PADDING:-true}"
LORA_RANK="${SINGLE_WISE_DPO_LORA_RANK:-64}"
LORA_ALPHA="${SINGLE_WISE_DPO_LORA_ALPHA:-128}"
LORA_TARGET_MODULES="${SINGLE_WISE_DPO_LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
LORA_ADAPTER_PATH="${SINGLE_WISE_DPO_LORA_ADAPTER_PATH:-}"
LORA_TARGET_PARAMETERS="${SINGLE_WISE_DPO_LORA_TARGET_PARAMETERS:-}"
LORA_EXCLUDE_MODULES="${SINGLE_WISE_DPO_LORA_EXCLUDE_MODULES:-}"

# Training and logging.
MICRO_BATCH_SIZE="${SINGLE_WISE_DPO_MICRO_BATCH_SIZE:-8}"
TOTAL_EPOCHS="${SINGLE_WISE_DPO_TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS:-}"
LOG_FREQ="${SINGLE_WISE_DPO_LOG_FREQ:-10}"
LOGGER="${SINGLE_WISE_DPO_LOGGER:-[console]}"
SAVE_FREQ_EPOCHS="${SINGLE_WISE_DPO_SAVE_FREQ_EPOCHS:-1}"
KEEP_ONLY_LATEST_ROLLING_CKPT="${SINGLE_WISE_DPO_KEEP_ONLY_LATEST_ROLLING_CKPT:-false}"
MAX_ACTOR_CKPT_TO_KEEP="${SINGLE_WISE_DPO_MAX_ACTOR_CKPT_TO_KEEP:-}"
RESUME_MODE="${SINGLE_WISE_DPO_RESUME_MODE:-auto}"
LENGTH_BUCKET_SIZE_MULTIPLIER="${SINGLE_WISE_DPO_LENGTH_BUCKET_SIZE_MULTIPLIER:-50}"
LENGTH_ESTIMATION_MODE="${SINGLE_WISE_DPO_LENGTH_ESTIMATION_MODE:-char}"
LENGTH_ESTIMATION_BATCH_SIZE="${SINGLE_WISE_DPO_LENGTH_ESTIMATION_BATCH_SIZE:-2048}"
LENGTH_ESTIMATION_CHARS_PER_TOKEN="${SINGLE_WISE_DPO_LENGTH_ESTIMATION_CHARS_PER_TOKEN:-4.0}"
AUTO_PRECOMPUTE_REFERENCE_LOGPS="${SINGLE_WISE_DPO_AUTO_PRECOMPUTE_REFERENCE_LOGPS:-true}"
REFERENCE_LOGPS_MATERIALIZED_DIR="${SINGLE_WISE_DPO_REFERENCE_LOGPS_MATERIALIZED_DIR:-${CKPT_ROOT}/reference_logps}"
REFERENCE_LOGPS_ALLOW_CROSS_NAMESPACE_REUSE="${SINGLE_WISE_DPO_REFERENCE_LOGPS_ALLOW_CROSS_NAMESPACE_REUSE:-true}"
REFERENCE_LOGPS_ALLOW_SAMPLE_ID_REUSE="${SINGLE_WISE_DPO_REFERENCE_LOGPS_ALLOW_SAMPLE_ID_REUSE:-true}"
REFERENCE_LOGPS_MAX_BATCH_SIZE="${SINGLE_WISE_DPO_REFERENCE_LOGPS_MAX_BATCH_SIZE:-4}"
REFERENCE_LOGPS_MAX_BATCHED_TOKENS="${SINGLE_WISE_DPO_REFERENCE_LOGPS_MAX_BATCHED_TOKENS:-16384}"

# Optional post-training evaluation.
POST_TRAIN_EVAL_SCRIPT="${SINGLE_WISE_DPO_POST_TRAIN_EVAL_SCRIPT:-recipe/dpo/evaluation/run_pens_personalized_eval.sh}"
POST_TRAIN_EVAL_WORKDIR="${SINGLE_WISE_DPO_POST_TRAIN_EVAL_WORKDIR:-$(pwd)}"
POST_TRAIN_EVAL_NNODES="${SINGLE_WISE_DPO_POST_TRAIN_EVAL_NNODES:-1}"
POST_TRAIN_EVAL_GEN_TP="${SINGLE_WISE_DPO_POST_TRAIN_EVAL_GEN_TP:-1}"
POST_TRAIN_EVAL_VLLM_ENABLE_THINKING="${SINGLE_WISE_DPO_POST_TRAIN_EVAL_VLLM_ENABLE_THINKING:-false}"

LOSS_TYPE_NORMALIZED="$(printf '%s' "${LOSS_TYPE}" | tr '[:upper:]' '[:lower:]')"
case "${LOSS_TYPE_NORMALIZED}" in
  single_wise_dpo)
    CONFIG_NAME="dpo_single_wise_dpo"
    USE_LENGTH_BUCKET_SAMPLER="${SINGLE_WISE_DPO_USE_LENGTH_BUCKET_SAMPLER:-true}"
    ;;
  prospect_dpo)
    CONFIG_NAME="dpo_prospect_dpo"
    USE_LENGTH_BUCKET_SAMPLER="${SINGLE_WISE_DPO_USE_LENGTH_BUCKET_SAMPLER:-false}"
    ;;
  *)
    echo "Unsupported point-wise loss type: ${LOSS_TYPE}. Expected single_wise_dpo or prospect_dpo." >&2
    exit 1
    ;;
esac

if [[ "${USE_LORA,,}" == "true" ]]; then
  FINETUNE_VARIANT="lora"
elif [[ "${USE_LORA,,}" == "false" ]]; then
  FINETUNE_VARIANT="fullft"
  LORA_RANK=0
else
  echo "SINGLE_WISE_DPO_USE_LORA must be true or false, got: ${USE_LORA}" >&2
  exit 1
fi

resolve_variant_train_file() {
  local input_variant="$1"
  local sample_variant="$2"

  if [[ "${sample_variant}" == "all" ]]; then
    echo "resolve_variant_train_file does not support sample_variant=all; use resolve_split_train_files instead." >&2
    exit 1
  fi

  case "${input_variant}:${sample_variant}" in
    click_hist:positive_only)
      echo "${DATA_ROOT}/only_positive_click_hist_train.parquet"
      ;;
    click_hist:negative_only)
      echo "${DATA_ROOT}/only_negative_click_hist_train.parquet"
      ;;
    personalization:positive_only)
      echo "${DATA_ROOT}/pens_only_positive_personalization/train.parquet"
      ;;
    personalization:negative_only)
      echo "${DATA_ROOT}/pens_only_negative_personalization/train.parquet"
      ;;
    *)
      echo "Unsupported SINGLE_WISE_DPO variant combination: ${input_variant}:${sample_variant}" >&2
      exit 1
      ;;
  esac
}

resolve_split_train_files() {
  local input_variant="$1"

  case "${input_variant}" in
    click_hist)
      echo "${DATA_ROOT}/only_positive_click_hist_train.parquet|${DATA_ROOT}/only_negative_click_hist_train.parquet"
      ;;
    personalization)
      echo "${DATA_ROOT}/pens_only_positive_personalization/train.parquet|${DATA_ROOT}/pens_only_negative_personalization/train.parquet"
      ;;
    *)
      echo "Unsupported SINGLE_WISE_DPO input variant: ${input_variant}" >&2
      exit 1
      ;;
  esac
}

sanitize_name_component() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^[:alnum:]_-]+/_/g; s/_+/_/g; s/^[_-]+//; s/[_-]+$//')"
  if [[ -z "${value}" ]]; then
    value="unknown"
  fi
  printf '%s' "${value}"
}

# Walk up past generic leaf dirs (hf_merged, actor, global_step_N) until we
# hit a meaningful experiment identifier. Without this, every SFT warmup path
# ending in /hf_merged collapses to the slug "hf_merged", so switching base
# models mid-run (e.g. Qwen3-4B base -> Qwen3-4B-Instruct-2507) lands in the
# same CKPT_DIR and `resume_mode=auto` loads an adapter trained on the wrong
# base. Observed 2026-04-22: that collision is what broke prospect-DPO 5341.
_walk_up_to_meaningful_name() {
  local path="${1%/}"
  local candidate="$(basename "${path}")"
  while [[ -n "${candidate}" ]]; do
    case "${candidate}" in
      hf_merged|hf_merged_final|hf_merged_fixed|huggingface|actor|hf_model|global_step_*)
        path="$(dirname "${path}")"
        if [[ -z "${path}" || "${path}" == "/" ]]; then
          break
        fi
        candidate="$(basename "${path}")"
        ;;
      *)
        break
        ;;
    esac
  done
  printf '%s' "${candidate}"
}

model_slug() {
  local model_path="${1%/}"
  local candidate
  if [[ "${model_path}" == */snapshots/* ]]; then
    candidate="$(basename "$(dirname "$(dirname "${model_path}")")")"
  else
    candidate="$(_walk_up_to_meaningful_name "${model_path}")"
  fi

  if [[ "${candidate}" == models--* ]]; then
    candidate="${candidate#models--}"
    candidate="${candidate##*--}"
  fi

  sanitize_name_component "${candidate}"
}

reference_model_slug() {
  local model_path="${1%/}"
  local candidate
  if [[ "${model_path}" == */snapshots/* ]]; then
    candidate="$(basename "$(dirname "$(dirname "${model_path}")")")"
  else
    candidate="$(_walk_up_to_meaningful_name "${model_path}")"
  fi
  candidate="${candidate#models--}"
  candidate="${candidate//--/_}"
  sanitize_name_component "${candidate}"
}

dataset_tag_for_path() {
  local source_path="$1"
  sanitize_name_component "$(basename "$(dirname "${source_path}")")"
}

split_tag_for_path() {
  local source_path="$1"
  local stem
  stem="$(basename "${source_path}")"
  stem="${stem%.parquet}"
  sanitize_name_component "${stem}"
}

sample_tag_for_path() {
  local source_path="$1"
  local lowered_path
  lowered_path="$(printf '%s' "${source_path}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${lowered_path}" == *positive* ]]; then
    printf 'positive'
    return 0
  fi
  if [[ "${lowered_path}" == *negative* ]]; then
    printf 'negative'
    return 0
  fi

  case "${SAMPLE_VARIANT}" in
    positive_only)
      printf 'positive'
      ;;
    negative_only)
      printf 'negative'
      ;;
    *)
      printf 'positive_negative'
      ;;
  esac
}

build_reference_output_name() {
  local source_path="$1"
  local dataset_tag
  local split_tag
  local sample_tag
  dataset_tag="$(dataset_tag_for_path "${source_path}")"
  split_tag="$(split_tag_for_path "${source_path}")"
  sample_tag="$(sample_tag_for_path "${source_path}")"
  printf 'ref_logps_%s_%s_%s_%s' "${dataset_tag}" "${split_tag}" "${sample_tag}" "${REFERENCE_MODEL_SLUG}"
}

build_reference_output_names_csv() {
  local names=()
  local source_path
  for source_path in "$@"; do
    if [[ -n "${source_path}" ]]; then
      names+=("$(build_reference_output_name "${source_path}")")
    fi
  done
  local IFS=,
  printf '%s' "${names[*]}"
}

hydra_quote_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\'/\\\'}"
  printf "'%s'" "${value}"
}

require_parquet_columns() {
  local parquet_path="$1"
  shift
  if [[ "${parquet_path}" != *.parquet ]]; then
    echo "Column check requires a parquet file, got: ${parquet_path}" >&2
    exit 1
  fi

  local missing_columns
  if ! missing_columns="$("${PYTHON_BIN}" - "${parquet_path}" "$@" <<'PY'
import sys

import pyarrow.parquet as pq

path = sys.argv[1]
required_columns = sys.argv[2:]
schema_columns = set(pq.ParquetFile(path).schema.names)
missing_columns = [column for column in required_columns if column not in schema_columns]
if missing_columns:
    print(",".join(missing_columns))
    raise SystemExit(1)
PY
)"; then
    echo "Missing required columns in ${parquet_path}: ${missing_columns}" >&2
    exit 1
  fi
}

maybe_require_prospect_columns() {
  local parquet_path="$1"
  if [[ "${LOSS_TYPE_NORMALIZED}" != "prospect_dpo" ]]; then
    return 0
  fi
  require_parquet_columns "${parquet_path}" "${S_DWELL_KEY}" "${P_CTR_KEY}"
}

is_truthy() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

find_latest_checkpoint_dir() {
  local tracker_file="${CKPT_DIR}/latest_checkpointed_iteration.txt"
  local latest_step
  if [[ -f "${tracker_file}" ]]; then
    latest_step="$(<"${tracker_file}")"
    if [[ -n "${latest_step}" && -d "${CKPT_DIR}/global_step_${latest_step}" ]]; then
      printf '%s' "${CKPT_DIR}/global_step_${latest_step}"
      return 0
    fi
  fi

  shopt -s nullglob
  local ckpt_dirs=("${CKPT_DIR}"/global_step_*)
  shopt -u nullglob
  if (( ${#ckpt_dirs[@]} == 0 )); then
    return 1
  fi

  local latest_dir=""
  latest_step=-1
  local ckpt_dir step
  for ckpt_dir in "${ckpt_dirs[@]}"; do
    step="${ckpt_dir##*/global_step_}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
      latest_step="${step}"
      latest_dir="${ckpt_dir}"
    fi
  done

  if [[ -z "${latest_dir}" ]]; then
    return 1
  fi
  printf '%s' "${latest_dir}"
}

export_latest_hf_merged() {
  if ! is_truthy "${USE_LORA,,}" || ! is_truthy "${EXPORT_HF_MERGED}"; then
    return 0
  fi

  local latest_ckpt_dir
  if ! latest_ckpt_dir="$(find_latest_checkpoint_dir)"; then
    echo "Unable to find the latest checkpoint under ${CKPT_DIR} for hf_merged export." >&2
    return 1
  fi

  local actor_dir="${latest_ckpt_dir}/actor"
  if [[ ! -d "${actor_dir}/lora_adapter" ]]; then
    echo "Missing LoRA adapter directory for hf_merged export: ${actor_dir}/lora_adapter" >&2
    return 1
  fi

  local output_dir="${actor_dir}/hf_merged"
  if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "hf_merged already exists at ${output_dir}; skipping export."
    return 0
  fi

  local export_cmd=(
    "${PYTHON_BIN}"
    recipe/dpo/export_lora_checkpoint_to_hf.py
    --actor-dir
    "${actor_dir}"
    --output-dir
    "${output_dir}"
    --base-model-dir
    "${MODEL_DIR}"
    --tokenizer-path
    "${TOKENIZER_PATH}"
    --dtype
    "${EXPORT_HF_MERGED_DTYPE}"
    --max-shard-size
    "${EXPORT_HF_MERGED_MAX_SHARD_SIZE}"
  )
  if is_truthy "${EXPORT_TRUST_REMOTE_CODE}"; then
    export_cmd+=(--trust-remote-code)
  fi

  echo "Exporting merged HF model to ${output_dir}"
  "${export_cmd[@]}"
}

run_post_train_eval() {
  if ! is_truthy "${RUN_POST_TRAIN_EVAL}"; then
    return 0
  fi

  local latest_ckpt_dir
  if ! latest_ckpt_dir="$(find_latest_checkpoint_dir)"; then
    echo "Unable to find the latest checkpoint under ${CKPT_DIR} for post-training eval." >&2
    return 1
  fi

  local eval_model_dir="${latest_ckpt_dir}/actor/hf_merged"
  if [[ ! -d "${eval_model_dir}" ]]; then
    echo "Missing merged HF model for post-training eval: ${eval_model_dir}" >&2
    echo "Keep SINGLE_WISE_DPO_EXPORT_HF_MERGED=true or create actor/hf_merged before eval." >&2
    return 1
  fi
  if [[ ! -f "${POST_TRAIN_EVAL_SCRIPT}" ]]; then
    echo "Missing post-training eval script: ${POST_TRAIN_EVAL_SCRIPT}" >&2
    return 1
  fi

  echo "Running post-training PENS eval for ${eval_model_dir}"
  MODEL_PATH="${eval_model_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    WORKDIR="${POST_TRAIN_EVAL_WORKDIR}" \
    NNODES="${POST_TRAIN_EVAL_NNODES}" \
    NGPUS_PER_NODE="${POST_TRAIN_EVAL_NGPUS_PER_NODE}" \
    GEN_TP="${POST_TRAIN_EVAL_GEN_TP}" \
    VLLM_ENABLE_THINKING="${POST_TRAIN_EVAL_VLLM_ENABLE_THINKING}" \
    bash "${POST_TRAIN_EVAL_SCRIPT}"
}

compute_training_shape() {
  local pos_file="$1"
  local neg_file="$2"
  local batch_size="$3"
  "${PYTHON_BIN}" - "${pos_file}" "${neg_file}" "${batch_size}" <<'PY_SHAPE'
import sys
import pyarrow.parquet as pq

pos_file, neg_file, batch_size_s = sys.argv[1:4]
batch_size = int(batch_size_s)
pos_rows = pq.ParquetFile(pos_file).metadata.num_rows
neg_rows = pq.ParquetFile(neg_file).metadata.num_rows
total_rows = pos_rows + neg_rows
total_steps = total_rows // batch_size
if total_steps <= 0:
    raise SystemExit(f"no training steps: rows={total_rows}, batch_size={batch_size}")
print(pos_rows, neg_rows, total_rows, total_steps)
PY_SHAPE
}

build_eval_steps() {
  local total_steps="$1"
  local interval="$2"
  "${PYTHON_BIN}" - "${total_steps}" "${interval}" "${SINGLE_WISE_DPO_ALL_EVAL_STEPS:-}" <<'PY_STEPS'
import sys

total_steps = int(sys.argv[1])
interval = int(sys.argv[2])
explicit = sys.argv[3].strip()
if interval <= 0:
    raise SystemExit(f"eval interval must be positive, got {interval}")
steps = []
if explicit:
    for part in explicit.replace(" ", "").split(","):
        if part:
            step = int(part)
            if 0 < step <= total_steps:
                steps.append(step)
else:
    step = interval
    while step <= total_steps:
        steps.append(step)
        step += interval
if total_steps not in steps:
    steps.append(total_steps)
seen = set()
for step in sorted(steps):
    if step not in seen:
        print(step)
        seen.add(step)
PY_STEPS
}

export_hf_merged_for_step() {
  local step="$1"
  local ckpt_dir="${CKPT_DIR}/global_step_${step}"
  local actor_dir="${ckpt_dir}/actor"
  local output_dir="${actor_dir}/hf_merged"

  if [[ ! -d "${actor_dir}/lora_adapter" ]]; then
    echo "Missing LoRA adapter for step ${step}: ${actor_dir}/lora_adapter" >&2
    return 1
  fi

  if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "hf_merged exists for step ${step}: ${output_dir}"
    return 0
  fi

  echo "Exporting step ${step} -> ${output_dir}"
  local export_cmd=(
    "${PYTHON_BIN}"
    recipe/dpo/export_lora_checkpoint_to_hf.py
    --actor-dir "${actor_dir}"
    --output-dir "${output_dir}"
    --base-model-dir "${MODEL_DIR}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --dtype "${EXPORT_HF_MERGED_DTYPE}"
    --max-shard-size "${EXPORT_HF_MERGED_MAX_SHARD_SIZE}"
  )
  if is_truthy "${EXPORT_TRUST_REMOTE_CODE}"; then
    export_cmd+=(--trust-remote-code)
  fi
  "${export_cmd[@]}"
}

run_eval_for_step() {
  local step="$1"
  local model_dir="${CKPT_DIR}/global_step_${step}/actor/hf_merged"
  local model_key="qwen3_4b_all_step_${step}"
  local raw_file="${ALL_RESULTS_GEN_DIR}/${model_key}.parquet"

  if [[ ! -d "${model_dir}" ]]; then
    echo "Missing model for eval step ${step}: ${model_dir}" >&2
    return 1
  fi

  echo "Evaluating step ${step} -> ${ALL_RAW_EVAL_JSON}"
  MODEL_PATH="${model_dir}" \
    MODEL_KEY="${model_key}" \
    BASE_MODEL_SLUG="qwen3_4b_all" \
    DATE_TAG="step_${step}" \
    RAW_FILE="${raw_file}" \
    RESULT_JSON_FILE="${ALL_RAW_EVAL_JSON}" \
    RESULTS_GEN_DIR="${ALL_RESULTS_GEN_DIR}" \
    RESULTS_DIR="${ALL_RESULTS_DIR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    WORKDIR="${ALL_EVAL_WORKDIR}" \
    NNODES="${ALL_EVAL_NNODES}" \
    NGPUS_PER_NODE="${ALL_EVAL_NGPUS_PER_NODE}" \
    GEN_TP="${ALL_EVAL_GEN_TP}" \
    VLLM_ENABLE_THINKING="${ALL_EVAL_VLLM_ENABLE_THINKING}" \
    bash "${ALL_EVAL_SCRIPT}"
}

aggregate_eval_results() {
  "${PYTHON_BIN}" - \
    "${ALL_RAW_EVAL_JSON}" \
    "${ALL_RESULT_JSON_FILE}" \
    "${CKPT_DIR}" \
    "${SINGLE_WISE_DPO_EXPERIMENT_NAME}" \
    "${DATA_ROOT}" \
    "${POS_ROWS}" \
    "${NEG_ROWS}" \
    "${TOTAL_ROWS}" \
    "${TRAIN_BATCH_SIZE}" \
    "${TOTAL_STEPS}" \
    "${EVAL_COUNT}" \
    "${ALL_EVAL_INTERVAL}" \
    "${MODEL_DIR}" <<'PY_AGG'
import json
import os
import re
import sys
from pathlib import Path

(
    raw_json,
    final_json,
    ckpt_dir,
    experiment_name,
    data_root,
    pos_rows,
    neg_rows,
    total_rows,
    train_batch_size,
    total_steps,
    eval_count,
    eval_interval,
    model_dir,
) = sys.argv[1:14]

raw_path = Path(raw_json)
final_path = Path(final_json)
if not raw_path.exists() or raw_path.stat().st_size == 0:
    raise SystemExit(f"missing raw eval json: {raw_path}")
with raw_path.open("r", encoding="utf-8") as f:
    raw = json.load(f)
if not isinstance(raw, dict):
    raise SystemExit(f"expected object in {raw_path}")

records = []
for key, payload in raw.items():
    if not isinstance(payload, dict):
        continue
    step = None
    model_path = str(payload.get("model_path", ""))
    match = re.search(r"global_step_(\d+)", model_path)
    if match:
        step = int(match.group(1))
    else:
        match = re.search(r"step_(\d+)", key)
        if match:
            step = int(match.group(1))
    if step is None:
        continue
    record = {
        "step": step,
        "result_key": key,
        "model_path": model_path,
        "raw_generation_file": payload.get("raw_generation_file"),
        "rouge_1_f1": payload.get("rouge_1_f1"),
        "rouge_2_f1": payload.get("rouge_2_f1"),
        "rouge_l_f1": payload.get("rouge_l_f1"),
        "count": payload.get("count"),
        "joined_count": payload.get("joined_count"),
        "skipped_count": payload.get("skipped_count"),
        "non_empty_prediction_count": payload.get("non_empty_prediction_count"),
        "parse_status_counts": payload.get("parse_status_counts", {}),
        "thinking_mode": payload.get("thinking_mode"),
        "date": payload.get("date"),
    }
    records.append(record)

records.sort(key=lambda item: item["step"])
if not records:
    raise SystemExit("no evaluation records found")

metric = "rouge_l_f1"
best = max(records, key=lambda item: float(item.get(metric) or float("-inf")))
summary = {
    "experiment_name": experiment_name,
    "checkpoint_dir": ckpt_dir,
    "data_root": data_root,
    "model_init": model_dir,
    "positive_rows": int(pos_rows),
    "negative_rows": int(neg_rows),
    "total_rows": int(total_rows),
    "train_batch_size": int(train_batch_size),
    "total_steps": int(total_steps),
    "eval_count_target": int(eval_count),
    "eval_interval_steps": int(eval_interval),
    "metric_for_best": metric,
    "best_step": best["step"],
    "best_score": best.get(metric),
    "best_model_path": best.get("model_path"),
    "evaluations": records,
}
final_path.parent.mkdir(parents=True, exist_ok=True)
tmp = final_path.with_suffix(final_path.suffix + f".tmp.{os.getpid()}")
with tmp.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
os.replace(tmp, final_path)

ckpt_path = Path(ckpt_dir)
with (ckpt_path / "best_eval_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
with (ckpt_path / "best_checkpoint_step.txt").open("w", encoding="utf-8") as f:
    f.write(str(best["step"]) + "\n")
print(best["step"])
PY_AGG
}

prune_non_best_hf_merged() {
  local best_step="$1"
  local best_dir="${CKPT_DIR}/global_step_${best_step}/actor/hf_merged"
  shopt -s nullglob
  local merged_dirs=("${CKPT_DIR}"/global_step_*/actor/hf_merged)
  shopt -u nullglob
  local dir
  for dir in "${merged_dirs[@]}"; do
    if [[ "${dir}" != "${best_dir}" ]]; then
      echo "Removing non-best hf_merged export: ${dir}"
      rm -rf "${dir}"
    fi
  done
}

prepare_all_checkpoints_workflow() {
  if [[ "${PENS_DPO_WORKFLOW:-train}" != "all_checkpoints" ]]; then
    return 0
  fi
  if [[ "${PENS_DPO_DATA_VARIANT:-}" != "full" ]]; then
    echo "PENS_DPO_WORKFLOW=all_checkpoints requires PENS_DPO_DATA_VARIANT=full." >&2
    exit 1
  fi

  PRUNE_NON_BEST_HF_MERGED="${SINGLE_WISE_DPO_ALL_PRUNE_NON_BEST_HF_MERGED:-true}"

  ALL_EVAL_SCRIPT="${SINGLE_WISE_DPO_ALL_EVAL_SCRIPT:-${VERL_ROOT}/recipe/dpo/evaluation/run_pens_personalized_eval.sh}"
  ALL_EVAL_WORKDIR="${SINGLE_WISE_DPO_ALL_EVAL_WORKDIR:-${VERL_ROOT}}"
  ALL_EVAL_NNODES="${SINGLE_WISE_DPO_ALL_EVAL_NNODES:-1}"
  ALL_EVAL_NGPUS_PER_NODE="${SINGLE_WISE_DPO_ALL_EVAL_NGPUS_PER_NODE:-8}"
  ALL_EVAL_GEN_TP="${SINGLE_WISE_DPO_ALL_EVAL_GEN_TP:-1}"
  ALL_EVAL_VLLM_ENABLE_THINKING="${SINGLE_WISE_DPO_ALL_EVAL_VLLM_ENABLE_THINKING:-false}"
  ALL_RESULTS_DIR="${SINGLE_WISE_DPO_ALL_RESULTS_DIR:-${VERL_ROOT}/results}"
  ALL_RESULTS_GEN_DIR="${SINGLE_WISE_DPO_ALL_RESULTS_GEN_DIR:-${PENS_SCRATCH_ROOT}/gen_results/qwen3_4b_all}"
  ALL_RESULT_JSON_FILE="${SINGLE_WISE_DPO_ALL_RESULT_JSON_FILE:-${ALL_RESULTS_DIR}/qwen3_4b_all.json}"
  ALL_RAW_EVAL_JSON="${SINGLE_WISE_DPO_ALL_RAW_EVAL_JSON:-${ALL_RESULTS_DIR}/qwen3_4b_all.raw_eval.json}"

  local positive_file="${DATA_ROOT}/only_positive_click_hist_train.parquet"
  local negative_file="${DATA_ROOT}/only_negative_click_hist_train.parquet"
  if [[ ! -f "${positive_file}" || ! -f "${negative_file}" ]]; then
    echo "Missing full DPO parquet files under ${DATA_ROOT}" >&2
    echo "Expected: ${positive_file}" >&2
    echo "Expected: ${negative_file}" >&2
    exit 1
  fi

  read -r POS_ROWS NEG_ROWS TOTAL_ROWS TOTAL_STEPS < <(
    compute_training_shape "${positive_file}" "${negative_file}" "${TRAIN_BATCH_SIZE}"
  )
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${TOTAL_STEPS}}"

  mapfile -t EVAL_STEPS < <(
    build_eval_steps "${TOTAL_TRAINING_STEPS}" "${ALL_EVAL_INTERVAL}"
  )
  EVAL_COUNT="${#EVAL_STEPS[@]}"

  echo "Full DPO rows: positive=${POS_ROWS}, negative=${NEG_ROWS}, total=${TOTAL_ROWS}"
  echo "Train batch size: ${TRAIN_BATCH_SIZE}"
  echo "GPUs: nnodes=${NNODES}, n_gpus_per_node=${N_GPUS_PER_NODE}, micro_batch_size=${MICRO_BATCH_SIZE}"
  echo "Reference logps: workers=${REFERENCE_LOGPS_NUM_WORKERS}, rows_per_task=${REFERENCE_LOGPS_ROWS_PER_TASK}"
  echo "Total training steps: ${TOTAL_TRAINING_STEPS}"
  echo "Eval interval: ${ALL_EVAL_INTERVAL}"
  echo "Eval steps: ${EVAL_STEPS[*]}"
  echo "Scratch root: ${PENS_SCRATCH_ROOT}"
  echo "Ray tmp dir: ${RAY_TMPDIR}"
  echo "Checkpoint dir: ${CKPT_DIR}"
  echo "Raw eval generation dir: ${ALL_RESULTS_GEN_DIR}"
  echo "Final eval sequence JSON: ${ALL_RESULT_JSON_FILE}"

  if is_truthy "${SINGLE_WISE_DPO_ALL_DRY_RUN:-false}"; then
    echo "SINGLE_WISE_DPO_ALL_DRY_RUN=true; exiting before training."
    exit 0
  fi

  mkdir -p \
    "${ALL_RESULTS_DIR}" \
    "${ALL_RESULTS_GEN_DIR}" \
    "${CKPT_DIR}" \
    "${RAY_TMPDIR}"
}

run_all_checkpoints_eval() {
  if [[ "${PENS_DPO_WORKFLOW:-train}" != "all_checkpoints" ]]; then
    return 0
  fi

  local step
  for step in "${EVAL_STEPS[@]}"; do
    if [[ ! -d "${CKPT_DIR}/global_step_${step}" ]]; then
      echo "Skipping missing checkpoint step ${step}: ${CKPT_DIR}/global_step_${step}" >&2
      continue
    fi
    export_hf_merged_for_step "${step}"
    run_eval_for_step "${step}"
  done

  local best_step
  best_step="$(aggregate_eval_results)"
  ln -sfn \
    "${CKPT_DIR}/global_step_${best_step}/actor/hf_merged" \
    "${CKPT_DIR}/best_hf_merged"

  echo "Best eval step: ${best_step}"
  echo "Best model symlink: ${CKPT_DIR}/best_hf_merged"
  echo "Eval sequence JSON: ${ALL_RESULT_JSON_FILE}"

  if is_truthy "${PRUNE_NON_BEST_HF_MERGED}"; then
    prune_non_best_hf_merged "${best_step}"
  fi
}

TRAIN_FILES_OVERRIDE=""
REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV=""
POS_TRAIN_FILE=""
NEG_TRAIN_FILE=""
if [[ -z "${MODEL_DIR}" ]]; then
  echo "Missing model path. Set SINGLE_WISE_DPO_MODEL_DIR." >&2
  exit 1
fi

MODEL_SLUG="$(model_slug "${MODEL_DIR}")"
RUN_TIMESTAMP="${SINGLE_WISE_DPO_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${PENS_DPO_WORKFLOW:-train}" == "all_checkpoints" ]]; then
  DEFAULT_EXPERIMENT_NAME="prospect_dpo_click_hist_all_lora_sft_warmup_pos_qwen3-4b-instruct-2507_all_${RUN_TIMESTAMP}"
else
  DEFAULT_EXPERIMENT_NAME="${LOSS_TYPE_NORMALIZED}_${INPUT_VARIANT}_${SAMPLE_VARIANT}_${FINETUNE_VARIANT}_${MODEL_SLUG}_${RUN_TIMESTAMP}"
fi
EXPERIMENT_NAME="${SINGLE_WISE_DPO_EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
CKPT_DIR="${SINGLE_WISE_DPO_CKPT_DIR:-${CKPT_ROOT}/${EXPERIMENT_NAME}}"

prepare_all_checkpoints_workflow
if [[ -z "${TRAIN_FILE}" ]]; then
  if [[ "${SAMPLE_VARIANT}" == "all" ]]; then
    IFS='|' read -r POS_TRAIN_FILE NEG_TRAIN_FILE <<< "$(resolve_split_train_files "${INPUT_VARIANT}")"
    if [[ ! -f "${POS_TRAIN_FILE}" || ! -f "${NEG_TRAIN_FILE}" ]]; then
      echo "SINGLE_WISE_DPO_SAMPLE_VARIANT=all requires both positive and negative train parquets: ${POS_TRAIN_FILE}, ${NEG_TRAIN_FILE}" >&2
      exit 1
    fi
    TRAIN_FILES_OVERRIDE="[${POS_TRAIN_FILE},${NEG_TRAIN_FILE}]"
  else
    TRAIN_FILE="$(resolve_variant_train_file "${INPUT_VARIANT}" "${SAMPLE_VARIANT}")"
  fi
fi

REFERENCE_MODEL_SLUG="$(reference_model_slug "${REFERENCE_MODEL_DIR:-${MODEL_DIR}}")"

if [[ -n "${TRAIN_FILES_OVERRIDE}" ]]; then
  if [[ -n "${POS_TRAIN_FILE}" ]]; then
    maybe_require_prospect_columns "${POS_TRAIN_FILE}"
  fi
  if [[ -n "${NEG_TRAIN_FILE}" ]]; then
    maybe_require_prospect_columns "${NEG_TRAIN_FILE}"
  fi
  REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV="$(build_reference_output_names_csv "${POS_TRAIN_FILE}" "${NEG_TRAIN_FILE}")"
else
  if [[ -z "${TRAIN_FILE}" ]]; then
    echo "Missing single-wise DPO train parquet. Set SINGLE_WISE_DPO_TRAIN_FILE." >&2
    exit 1
  fi
  if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Missing single-wise DPO train parquet: ${TRAIN_FILE}" >&2
    exit 1
  fi
  maybe_require_prospect_columns "${TRAIN_FILE}"
  TRAIN_FILES_OVERRIDE="${TRAIN_FILE}"
  REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV="$(build_reference_output_names_csv "${TRAIN_FILE}")"
fi

mkdir -p "${CKPT_DIR}"
mkdir -p "${REFERENCE_LOGPS_MATERIALIZED_DIR}"

extra_args=()
extra_args+=("data.val_files=null")
if [[ -n "${SEED}" ]]; then
  extra_args+=("data.seed=${SEED}")
fi
if [[ -n "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
  extra_args+=("+data.apply_chat_template_kwargs=${APPLY_CHAT_TEMPLATE_KWARGS}")
fi
if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
  extra_args+=("actor_rollout_ref.model.lora_adapter_path=${LORA_ADAPTER_PATH}")
fi
if [[ -n "${LORA_TARGET_PARAMETERS}" ]]; then
  extra_args+=("actor_rollout_ref.model.target_parameters=${LORA_TARGET_PARAMETERS}")
fi
if [[ -n "${LORA_EXCLUDE_MODULES}" ]]; then
  extra_args+=("actor_rollout_ref.model.exclude_modules=${LORA_EXCLUDE_MODULES}")
fi
if [[ -n "${REFERENCE_MODEL_DIR}" ]]; then
  extra_args+=("+actor_rollout_ref.ref.model.path=${REFERENCE_MODEL_DIR}")
fi
extra_args+=("data.auto_precompute_reference_logps=${AUTO_PRECOMPUTE_REFERENCE_LOGPS}")
extra_args+=("data.reference_logps_materialized_dir=${REFERENCE_LOGPS_MATERIALIZED_DIR}")
extra_args+=("data.reference_logps_allow_cross_namespace_reuse=${REFERENCE_LOGPS_ALLOW_CROSS_NAMESPACE_REUSE}")
extra_args+=("data.reference_logps_allow_sample_id_reuse=${REFERENCE_LOGPS_ALLOW_SAMPLE_ID_REUSE}")
extra_args+=("data.reference_logps_num_workers=${REFERENCE_LOGPS_NUM_WORKERS}")
extra_args+=("data.reference_logps_rows_per_task=${REFERENCE_LOGPS_ROWS_PER_TASK}")
extra_args+=("data.reference_logps_max_batch_size=${REFERENCE_LOGPS_MAX_BATCH_SIZE}")
extra_args+=("data.reference_logps_max_batched_tokens=${REFERENCE_LOGPS_MAX_BATCHED_TOKENS}")
if [[ -n "${REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV}" ]]; then
  if [[ "${LOSS_TYPE_NORMALIZED}" == "prospect_dpo" ]]; then
    extra_args+=(
      "+data.reference_logps_train_output_names_csv=$(hydra_quote_string "${REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV}")"
    )
  else
    extra_args+=(
      "data.reference_logps_train_output_names_csv=$(hydra_quote_string "${REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV}")"
    )
  fi
fi
extra_args+=("trainer.val_before_train=false")
extra_args+=("trainer.test_freq=-1")
extra_args+=("trainer.val_only=false")
extra_args+=("trainer.log_val_generations=0")
if [[ "${LOSS_TYPE_NORMALIZED}" == "prospect_dpo" ]]; then
  extra_args+=("data.s_dwell_key=${S_DWELL_KEY}")
  extra_args+=("data.p_ctr_key=${P_CTR_KEY}")
  extra_args+=("algorithm.prospect_dpo_alpha_tau=${ALPHA_TAU}")
  extra_args+=("algorithm.prospect_dpo_alpha_k=${ALPHA_K}")
  extra_args+=("algorithm.prospect_dpo_alpha_max=${ALPHA_MAX}")
  extra_args+=("algorithm.prospect_dpo_lambda_max=${LAMBDA_MAX}")
  extra_args+=("algorithm.prospect_dpo_lambda_gamma=${LAMBDA_GAMMA}")
fi
extra_args+=("algorithm.average_log_prob=${AVERAGE_LOG_PROB}")

cmd=(
  "${PYTHON_BIN}"
  -m
  recipe.dpo.main_dpo
  --config-name="${CONFIG_NAME}"
  "data.train_files=${TRAIN_FILES_OVERRIDE}"
  "actor_rollout_ref.model.path=${MODEL_DIR}"
  "actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH}"
  "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
  "actor_rollout_ref.model.allow_unsupported_remove_padding=${ALLOW_UNSUPPORTED_REMOVE_PADDING}"
  "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
  "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
  "actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}"
  "+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
  "algorithm.dpo_beta=${BETA}"
  "algorithm.dpo_loss_type=${LOSS_TYPE_NORMALIZED}"
  "algorithm.reference_free=false"
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=${ACTOR_LR_SCHEDULER_TYPE}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${ACTOR_LR_WARMUP_STEPS_RATIO}"
  "actor_rollout_ref.actor.optim.min_lr_ratio=${ACTOR_LR_MIN_RATIO}"
  "actor_rollout_ref.actor.optim.num_cycles=${ACTOR_LR_NUM_CYCLES}"
  "actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY}"
  "data.prompt_key=${PROMPT_KEY}"
  "data.response_key=${RESPONSE_KEY}"
  "data.label_key=${LABEL_KEY}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.prompt_truncation=${PROMPT_TRUNCATION}"
  "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
  "actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE}"
  "actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.nnodes=${NNODES}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.logger=${LOGGER}"
  "+trainer.log_freq=${LOG_FREQ}"
  "+trainer.keep_only_latest_rolling_ckpt=${KEEP_ONLY_LATEST_ROLLING_CKPT}"
)
if [[ -n "${MAX_ACTOR_CKPT_TO_KEEP}" ]]; then
  cmd+=("+trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}")
fi

if [[ "${LOSS_TYPE_NORMALIZED}" == "single_wise_dpo" ]]; then
  cmd+=(
    "data.use_length_bucket_sampler=${USE_LENGTH_BUCKET_SAMPLER}"
    "data.length_bucket_size_multiplier=${LENGTH_BUCKET_SIZE_MULTIPLIER}"
    "data.length_estimation_mode=${LENGTH_ESTIMATION_MODE}"
    "data.length_estimation_batch_size=${LENGTH_ESTIMATION_BATCH_SIZE}"
    "data.length_estimation_chars_per_token=${LENGTH_ESTIMATION_CHARS_PER_TOKEN}"
  )
fi
if [[ "${LOSS_TYPE_NORMALIZED}" == "prospect_dpo" ]]; then
  cmd+=("+trainer.save_freq_epochs=${SAVE_FREQ_EPOCHS}")
else
  cmd+=("trainer.save_freq_epochs=${SAVE_FREQ_EPOCHS}")
fi

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  cmd+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi
cmd+=("${extra_args[@]}")
cmd+=("$@")

"${cmd[@]}"
export_latest_hf_merged
run_post_train_eval
run_all_checkpoints_eval
