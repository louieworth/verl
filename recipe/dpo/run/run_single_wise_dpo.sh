#!/usr/bin/env bash
set -euxo pipefail

# Files and paths
DATA_ROOT="${SINGLE_WISE_DPO_DATA_ROOT:-/data/data/jiangli/data/pens}"
INPUT_VARIANT="${SINGLE_WISE_DPO_INPUT_VARIANT:-click_hist}"
SAMPLE_VARIANT="${SINGLE_WISE_DPO_SAMPLE_VARIANT:-all}"
LOSS_TYPE="${POINTWISE_DPO_LOSS_TYPE:-${SINGLE_WISE_DPO_LOSS_TYPE:-single_wise_dpo}}"
TRAIN_FILE="${SINGLE_WISE_DPO_TRAIN_FILE:-}"
HAS_EXPLICIT_TRAIN_FILE=false
if [[ -n "${TRAIN_FILE}" ]]; then
  HAS_EXPLICIT_TRAIN_FILE=true
fi
MODEL_DIR="${SINGLE_WISE_DPO_MODEL_DIR:-/data/data/jiangli/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
TOKENIZER_PATH="${SINGLE_WISE_DPO_TOKENIZER_PATH:-${MODEL_DIR}}"
REFERENCE_MODEL_DIR="${SINGLE_WISE_DPO_REFERENCE_MODEL_DIR:-}"
PROJECT_NAME="${SINGLE_WISE_DPO_PROJECT_NAME:-PENS}"
USE_LORA="${SINGLE_WISE_DPO_USE_LORA:-true}"
CKPT_ROOT="${SINGLE_WISE_DPO_CKPT_ROOT:-/data/data/jiangli/ckpt/PENS}"
PYTHON_BIN="${SINGLE_WISE_DPO_PYTHON_BIN:-python3}"
EXPORT_HF_MERGED="${SINGLE_WISE_DPO_EXPORT_HF_MERGED:-true}"
EXPORT_HF_MERGED_DTYPE="${SINGLE_WISE_DPO_EXPORT_HF_MERGED_DTYPE:-bfloat16}"
EXPORT_HF_MERGED_MAX_SHARD_SIZE="${SINGLE_WISE_DPO_EXPORT_HF_MERGED_MAX_SHARD_SIZE:-5GB}"
EXPORT_TRUST_REMOTE_CODE="${SINGLE_WISE_DPO_EXPORT_TRUST_REMOTE_CODE:-true}"

# Data schema and loading
PROMPT_KEY="${SINGLE_WISE_DPO_PROMPT_KEY:-prompt}"
RESPONSE_KEY="${SINGLE_WISE_DPO_RESPONSE_KEY:-response}"
LABEL_KEY="${SINGLE_WISE_DPO_LABEL_KEY:-label}"
S_DWELL_KEY="${SINGLE_WISE_DPO_S_DWELL_KEY:-s_dwell}"
P_CTR_KEY="${SINGLE_WISE_DPO_P_CTR_KEY:-p_ctr}"
TRAIN_BATCH_SIZE="${SINGLE_WISE_DPO_TRAIN_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${SINGLE_WISE_DPO_MAX_PROMPT_LENGTH:-3072}"
MAX_RESPONSE_LENGTH="${SINGLE_WISE_DPO_MAX_RESPONSE_LENGTH:-32}"
PROMPT_TRUNCATION="${SINGLE_WISE_DPO_PROMPT_TRUNCATION:-}"
DATALOADER_NUM_WORKERS="${SINGLE_WISE_DPO_DATALOADER_NUM_WORKERS:-8}"

# Algorithm hyperparameters
BETA="${SINGLE_WISE_DPO_BETA:-0.5}"
ALPHA_TAU="${SINGLE_WISE_DPO_ALPHA_TAU:-0.2}"
ALPHA_K="${SINGLE_WISE_DPO_ALPHA_K:-10.0}"
LAMBDA_MAX="${SINGLE_WISE_DPO_LAMBDA_MAX:-2.0}"
LAMBDA_GAMMA="${SINGLE_WISE_DPO_LAMBDA_GAMMA:-2.0}"

# Model and parallelism
NNODES="${SINGLE_WISE_DPO_NNODES:-1}"
N_GPUS_PER_NODE="${SINGLE_WISE_DPO_N_GPUS_PER_NODE:-2}"
ROLLOUT_TP_SIZE="${SINGLE_WISE_DPO_ROLLOUT_TP_SIZE:-1}"
ATTN_IMPLEMENTATION="${SINGLE_WISE_DPO_ATTN_IMPLEMENTATION:-flash_attention_2}"
MODEL_DTYPE="${SINGLE_WISE_DPO_MODEL_DTYPE:-bf16}"
USE_REMOVE_PADDING="${SINGLE_WISE_DPO_USE_REMOVE_PADDING:-false}"
ALLOW_UNSUPPORTED_REMOVE_PADDING="${SINGLE_WISE_DPO_ALLOW_UNSUPPORTED_REMOVE_PADDING:-true}"
USE_DYNAMIC_BSZ="${SINGLE_WISE_DPO_USE_DYNAMIC_BSZ:-false}"
MAX_TOKEN_LEN_PER_GPU="${SINGLE_WISE_DPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
LORA_RANK="${SINGLE_WISE_DPO_LORA_RANK:-64}"
LORA_ALPHA="${SINGLE_WISE_DPO_LORA_ALPHA:-128}"
# LORA_TARGET_MODULES="${SINGLE_WISE_DPO_LORA_TARGET_MODULES:-all-linear}"
LORA_TARGET_MODULES="${SINGLE_WISE_DPO_LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
LORA_ADAPTER_PATH="${SINGLE_WISE_DPO_LORA_ADAPTER_PATH:-}"
LORA_TARGET_PARAMETERS="${SINGLE_WISE_DPO_LORA_TARGET_PARAMETERS:-}"
LORA_EXCLUDE_MODULES="${SINGLE_WISE_DPO_LORA_EXCLUDE_MODULES:-}"

# Training and logging
MICRO_BATCH_SIZE="${SINGLE_WISE_DPO_MICRO_BATCH_SIZE:-16}"
TOTAL_EPOCHS="${SINGLE_WISE_DPO_TOTAL_EPOCHS:-1}"
LOG_FREQ="${SINGLE_WISE_DPO_LOG_FREQ:-10}"
SAVE_FREQ="${SINGLE_WISE_DPO_SAVE_FREQ:--1}"
SAVE_FREQ_EPOCHS="${SINGLE_WISE_DPO_SAVE_FREQ_EPOCHS:-1}"
USE_LENGTH_BUCKET_SAMPLER="${SINGLE_WISE_DPO_USE_LENGTH_BUCKET_SAMPLER:-}"
LENGTH_BUCKET_SIZE_MULTIPLIER="${SINGLE_WISE_DPO_LENGTH_BUCKET_SIZE_MULTIPLIER:-50}"
LENGTH_ESTIMATION_MODE="${SINGLE_WISE_DPO_LENGTH_ESTIMATION_MODE:-char}"
LENGTH_ESTIMATION_BATCH_SIZE="${SINGLE_WISE_DPO_LENGTH_ESTIMATION_BATCH_SIZE:-2048}"
LENGTH_ESTIMATION_CHARS_PER_TOKEN="${SINGLE_WISE_DPO_LENGTH_ESTIMATION_CHARS_PER_TOKEN:-4.0}"
AUTO_PRECOMPUTE_REFERENCE_LOGPS="${SINGLE_WISE_DPO_AUTO_PRECOMPUTE_REFERENCE_LOGPS:-true}"
REFERENCE_LOGPS_MATERIALIZED_DIR="${SINGLE_WISE_DPO_REFERENCE_LOGPS_MATERIALIZED_DIR:-${DATA_ROOT}/pens}"
REFERENCE_LOGPS_ALLOW_CROSS_NAMESPACE_REUSE="${SINGLE_WISE_DPO_REFERENCE_LOGPS_ALLOW_CROSS_NAMESPACE_REUSE:-true}"
REFERENCE_LOGPS_ALLOW_SAMPLE_ID_REUSE="${SINGLE_WISE_DPO_REFERENCE_LOGPS_ALLOW_SAMPLE_ID_REUSE:-true}"
REFERENCE_LOGPS_NUM_WORKERS="${SINGLE_WISE_DPO_REFERENCE_LOGPS_NUM_WORKERS:-0}"
REFERENCE_LOGPS_ROWS_PER_TASK="${SINGLE_WISE_DPO_REFERENCE_LOGPS_ROWS_PER_TASK:-2048}"
REFERENCE_LOGPS_MAX_BATCH_SIZE="${SINGLE_WISE_DPO_REFERENCE_LOGPS_MAX_BATCH_SIZE:-2}"
REFERENCE_LOGPS_MAX_BATCHED_TOKENS="${SINGLE_WISE_DPO_REFERENCE_LOGPS_MAX_BATCHED_TOKENS:-8192}"

LOSS_TYPE_NORMALIZED="$(printf '%s' "${LOSS_TYPE}" | tr '[:upper:]' '[:lower:]')"
case "${LOSS_TYPE_NORMALIZED}" in
  single_wise_dpo)
    CONFIG_NAME="dpo_single_wise_dpo"
    DEFAULT_PROMPT_TRUNCATION="right"
    DEFAULT_USE_LENGTH_BUCKET_SAMPLER="true"
    USE_SINGLE_WISE_LENGTH_BUCKET_CONFIG=true
    REQUIRE_PROSPECT_COLUMNS=false
    ;;
  prospect_dpo)
    CONFIG_NAME="dpo_prospect_dpo"
    DEFAULT_PROMPT_TRUNCATION="left"
    DEFAULT_USE_LENGTH_BUCKET_SAMPLER="false"
    USE_SINGLE_WISE_LENGTH_BUCKET_CONFIG=false
    REQUIRE_PROSPECT_COLUMNS=true
    ;;
  *)
    echo "Unsupported point-wise loss type: ${LOSS_TYPE}. Expected single_wise_dpo or prospect_dpo." >&2
    exit 1
    ;;
esac

if [[ -z "${PROMPT_TRUNCATION}" ]]; then
  PROMPT_TRUNCATION="${DEFAULT_PROMPT_TRUNCATION}"
fi
if [[ -z "${USE_LENGTH_BUCKET_SAMPLER}" ]]; then
  USE_LENGTH_BUCKET_SAMPLER="${DEFAULT_USE_LENGTH_BUCKET_SAMPLER}"
fi

USE_LORA_NORMALIZED="$(echo "${USE_LORA}" | tr '[:upper:]' '[:lower:]')"
if [[ "${USE_LORA_NORMALIZED}" == "true" ]]; then
  FINETUNE_VARIANT="lora"
elif [[ "${USE_LORA_NORMALIZED}" == "false" ]]; then
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

model_slug() {
  local model_path="${1%/}"
  local candidate
  if [[ "${model_path}" == */snapshots/* ]]; then
    candidate="$(basename "$(dirname "$(dirname "${model_path}")")")"
  else
    candidate="$(basename "${model_path}")"
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
    candidate="$(basename "${model_path}")"
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
  if [[ "${REQUIRE_PROSPECT_COLUMNS}" != "true" ]]; then
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
  if ! is_truthy "${USE_LORA_NORMALIZED}" || ! is_truthy "${EXPORT_HF_MERGED}"; then
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

if [[ -z "${MODEL_DIR}" ]]; then
  echo "Missing model path. Set SINGLE_WISE_DPO_MODEL_DIR." >&2
  exit 1
fi

MODEL_SLUG="$(model_slug "${MODEL_DIR}")"
TRAIN_FILES_OVERRIDE=""
REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV=""
POS_TRAIN_FILE=""
NEG_TRAIN_FILE=""
if [[ "${HAS_EXPLICIT_TRAIN_FILE}" == "false" ]]; then
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

REFERENCE_MODEL_EFFECTIVE_DIR="${REFERENCE_MODEL_DIR:-${MODEL_DIR}}"
REFERENCE_MODEL_SLUG="$(reference_model_slug "${REFERENCE_MODEL_EFFECTIVE_DIR}")"
DEFAULT_EXPERIMENT_NAME="${LOSS_TYPE_NORMALIZED}_${INPUT_VARIANT}_${SAMPLE_VARIANT}_${FINETUNE_VARIANT}_${MODEL_SLUG}"
EXPERIMENT_NAME="${SINGLE_WISE_DPO_EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
CKPT_DIR="${SINGLE_WISE_DPO_CKPT_DIR:-${CKPT_ROOT}/${EXPERIMENT_NAME}}"

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
  extra_args+=("algorithm.prospect_dpo_lambda_max=${LAMBDA_MAX}")
  extra_args+=("algorithm.prospect_dpo_lambda_gamma=${LAMBDA_GAMMA}")
fi

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
  "data.prompt_key=${PROMPT_KEY}"
  "data.response_key=${RESPONSE_KEY}"
  "data.label_key=${LABEL_KEY}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.prompt_truncation=${PROMPT_TRUNCATION}"
  "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
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
  "+trainer.log_freq=${LOG_FREQ}"
)

if [[ "${USE_SINGLE_WISE_LENGTH_BUCKET_CONFIG}" == "true" ]]; then
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

cmd+=("${extra_args[@]}")
cmd+=("$@")

"${cmd[@]}"
export_latest_hf_merged
