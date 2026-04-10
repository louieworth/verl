#!/usr/bin/env bash
set -euxo pipefail

# Files and paths
DATA_ROOT="${PROSPECT_DPO_DATA_ROOT:-/data/data/jiangli/data/pens}"
INPUT_VARIANT="${PROSPECT_DPO_INPUT_VARIANT:-click_hist}"
SAMPLE_VARIANT="${PROSPECT_DPO_SAMPLE_VARIANT:-all}"
TRAIN_FILE="${PROSPECT_DPO_TRAIN_FILE:-}"
HAS_EXPLICIT_TRAIN_FILE=false
if [[ -n "${TRAIN_FILE}" ]]; then
  HAS_EXPLICIT_TRAIN_FILE=true
fi
VAL_FILE="${PROSPECT_DPO_VAL_FILE:-}"
MODEL_DIR="${PROSPECT_DPO_MODEL_DIR:-/data/data/jiangli/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
TOKENIZER_PATH="${PROSPECT_DPO_TOKENIZER_PATH:-${MODEL_DIR}}"
REFERENCE_MODEL_DIR="${PROSPECT_DPO_REFERENCE_MODEL_DIR:-}"
PROJECT_NAME="${PROSPECT_DPO_PROJECT_NAME:-PENS}"
EXPERIMENT_NAME="${PROSPECT_DPO_EXPERIMENT_NAME:-prospect_dpo}"
CKPT_ROOT="${PROSPECT_DPO_CKPT_ROOT:-/data/data/jiangli/ckpt/PENS}"
CKPT_DIR="${PROSPECT_DPO_CKPT_DIR:-${CKPT_ROOT}/${EXPERIMENT_NAME}}"
PYTHON_BIN="${PROSPECT_DPO_PYTHON_BIN:-python3}"

# Data schema and loading
PROMPT_KEY="${PROSPECT_DPO_PROMPT_KEY:-prompt}"
RESPONSE_KEY="${PROSPECT_DPO_RESPONSE_KEY:-response}"
LABEL_KEY="${PROSPECT_DPO_LABEL_KEY:-label}"
S_DWELL_KEY="${PROSPECT_DPO_S_DWELL_KEY:-s_dwell}"
P_CTR_KEY="${PROSPECT_DPO_P_CTR_KEY:-p_ctr}"
TRAIN_BATCH_SIZE="${PROSPECT_DPO_TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${PROSPECT_DPO_VAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${PROSPECT_DPO_MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${PROSPECT_DPO_MAX_RESPONSE_LENGTH:-1024}"
DATALOADER_NUM_WORKERS="${PROSPECT_DPO_DATALOADER_NUM_WORKERS:-8}"

# Algorithm hyperparameters
BETA="${PROSPECT_DPO_BETA:-1}"
ALPHA_TAU="${PROSPECT_DPO_ALPHA_TAU:-0.2}"
ALPHA_K="${PROSPECT_DPO_ALPHA_K:-10.0}"
LAMBDA_MAX="${PROSPECT_DPO_LAMBDA_MAX:-2.0}"
LAMBDA_GAMMA="${PROSPECT_DPO_LAMBDA_GAMMA:-2.0}"

# Model and parallelism
NNODES="${PROSPECT_DPO_NNODES:-1}"
N_GPUS_PER_NODE="${PROSPECT_DPO_N_GPUS_PER_NODE:-1}"
ROLLOUT_TP_SIZE="${PROSPECT_DPO_ROLLOUT_TP_SIZE:-1}"
ATTN_IMPLEMENTATION="${PROSPECT_DPO_ATTN_IMPLEMENTATION:-flash_attention_2}"
MODEL_DTYPE="${PROSPECT_DPO_MODEL_DTYPE:-bf16}"
USE_REMOVE_PADDING="${PROSPECT_DPO_USE_REMOVE_PADDING:-true}"
MAX_TOKEN_LEN_PER_GPU="${PROSPECT_DPO_MAX_TOKEN_LEN_PER_GPU:-8096}"

# Training and logging
MICRO_BATCH_SIZE="${PROSPECT_DPO_MICRO_BATCH_SIZE:-4}"
TOTAL_EPOCHS="${PROSPECT_DPO_TOTAL_EPOCHS:-1}"
LOG_FREQ="${PROSPECT_DPO_LOG_FREQ:-10}"
TEST_FREQ="${PROSPECT_DPO_TEST_FREQ:--1}"
SAVE_FREQ="${PROSPECT_DPO_SAVE_FREQ:--1}"
AUTO_PRECOMPUTE_REFERENCE_LOGPS="${PROSPECT_DPO_AUTO_PRECOMPUTE_REFERENCE_LOGPS:-true}"
REFERENCE_LOGPS_MATERIALIZED_DIR="${PROSPECT_DPO_REFERENCE_LOGPS_MATERIALIZED_DIR:-${DATA_ROOT}/pens}"
REFERENCE_LOGPS_REUSE_TRAIN_FILES_CSV="${PROSPECT_DPO_REFERENCE_LOGPS_REUSE_TRAIN_FILES_CSV:-}"
REFERENCE_LOGPS_REUSE_VAL_FILES_CSV="${PROSPECT_DPO_REFERENCE_LOGPS_REUSE_VAL_FILES_CSV:-}"
REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV="${PROSPECT_DPO_REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV:-}"
REFERENCE_LOGPS_VAL_OUTPUT_NAMES_CSV="${PROSPECT_DPO_REFERENCE_LOGPS_VAL_OUTPUT_NAMES_CSV:-}"
REFERENCE_LOGPS_ALLOW_CROSS_NAMESPACE_REUSE="${PROSPECT_DPO_REFERENCE_LOGPS_ALLOW_CROSS_NAMESPACE_REUSE:-true}"
REFERENCE_LOGPS_ALLOW_SAMPLE_ID_REUSE="${PROSPECT_DPO_REFERENCE_LOGPS_ALLOW_SAMPLE_ID_REUSE:-true}"
REFERENCE_LOGPS_NUM_WORKERS="${PROSPECT_DPO_REFERENCE_LOGPS_NUM_WORKERS:-0}"
REFERENCE_LOGPS_ROWS_PER_TASK="${PROSPECT_DPO_REFERENCE_LOGPS_ROWS_PER_TASK:-2048}"
REFERENCE_LOGPS_MAX_BATCH_SIZE="${PROSPECT_DPO_REFERENCE_LOGPS_MAX_BATCH_SIZE:-2}"
REFERENCE_LOGPS_MAX_BATCHED_TOKENS="${PROSPECT_DPO_REFERENCE_LOGPS_MAX_BATCHED_TOKENS:-8192}"

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
      echo "Unsupported PROSPECT_DPO variant combination: ${input_variant}:${sample_variant}" >&2
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
      echo "Unsupported PROSPECT_DPO input variant: ${input_variant}" >&2
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

if [[ -z "${MODEL_DIR}" ]]; then
  echo "Missing model path. Set PROSPECT_DPO_MODEL_DIR." >&2
  exit 1
fi

TRAIN_FILES_OVERRIDE=""
REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_EFFECTIVE="${REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_CSV}"
REFERENCE_LOGPS_VAL_OUTPUT_NAMES_EFFECTIVE="${REFERENCE_LOGPS_VAL_OUTPUT_NAMES_CSV}"
POS_TRAIN_FILE=""
NEG_TRAIN_FILE=""
if [[ "${HAS_EXPLICIT_TRAIN_FILE}" == "false" ]]; then
  if [[ "${SAMPLE_VARIANT}" == "all" ]]; then
    IFS='|' read -r POS_TRAIN_FILE NEG_TRAIN_FILE <<< "$(resolve_split_train_files "${INPUT_VARIANT}")"
    if [[ ! -f "${POS_TRAIN_FILE}" || ! -f "${NEG_TRAIN_FILE}" ]]; then
      echo "PROSPECT_DPO_SAMPLE_VARIANT=all requires both positive and negative train parquets: ${POS_TRAIN_FILE}, ${NEG_TRAIN_FILE}" >&2
      exit 1
    fi
    TRAIN_FILES_OVERRIDE="[${POS_TRAIN_FILE},${NEG_TRAIN_FILE}]"
  else
    TRAIN_FILE="$(resolve_variant_train_file "${INPUT_VARIANT}" "${SAMPLE_VARIANT}")"
  fi
fi

REFERENCE_MODEL_EFFECTIVE_DIR="${REFERENCE_MODEL_DIR:-${MODEL_DIR}}"
REFERENCE_MODEL_SLUG="$(reference_model_slug "${REFERENCE_MODEL_EFFECTIVE_DIR}")"

if [[ -n "${TRAIN_FILES_OVERRIDE}" ]]; then
  if [[ -n "${POS_TRAIN_FILE}" ]]; then
    require_parquet_columns "${POS_TRAIN_FILE}" "${S_DWELL_KEY}" "${P_CTR_KEY}"
  fi
  if [[ -n "${NEG_TRAIN_FILE}" ]]; then
    require_parquet_columns "${NEG_TRAIN_FILE}" "${S_DWELL_KEY}" "${P_CTR_KEY}"
  fi
  if [[ -z "${REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_EFFECTIVE}" ]]; then
    REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_EFFECTIVE="$(build_reference_output_names_csv "${POS_TRAIN_FILE}" "${NEG_TRAIN_FILE}")"
  fi
else
  if [[ -z "${TRAIN_FILE}" ]]; then
    echo "Missing Prospect-DPO train parquet. Set PROSPECT_DPO_TRAIN_FILE or use PROSPECT_DPO_INPUT_VARIANT/PROSPECT_DPO_SAMPLE_VARIANT." >&2
    exit 1
  fi
  if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Missing Prospect-DPO train parquet: ${TRAIN_FILE}" >&2
    exit 1
  fi
  require_parquet_columns "${TRAIN_FILE}" "${S_DWELL_KEY}" "${P_CTR_KEY}"
  TRAIN_FILES_OVERRIDE="${TRAIN_FILE}"
  if [[ -z "${REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_EFFECTIVE}" ]]; then
    REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_EFFECTIVE="$(build_reference_output_names_csv "${TRAIN_FILE}")"
  fi
fi

if [[ -n "${VAL_FILE}" && ! -f "${VAL_FILE}" ]]; then
  echo "Missing Prospect-DPO val parquet: ${VAL_FILE}" >&2
  exit 1
fi
if [[ -n "${VAL_FILE}" ]]; then
  require_parquet_columns "${VAL_FILE}" "${S_DWELL_KEY}" "${P_CTR_KEY}"
  if [[ -z "${REFERENCE_LOGPS_VAL_OUTPUT_NAMES_EFFECTIVE}" ]]; then
    REFERENCE_LOGPS_VAL_OUTPUT_NAMES_EFFECTIVE="$(build_reference_output_names_csv "${VAL_FILE}")"
  fi
fi

mkdir -p "${CKPT_DIR}"
mkdir -p "${REFERENCE_LOGPS_MATERIALIZED_DIR}"

extra_args=()
if [[ -n "${VAL_FILE}" ]]; then
  extra_args+=("data.val_files=${VAL_FILE}")
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
if [[ -n "${REFERENCE_LOGPS_REUSE_TRAIN_FILES_CSV}" ]]; then
  extra_args+=(
    "data.reference_logps_reuse_train_files_csv=$(hydra_quote_string "${REFERENCE_LOGPS_REUSE_TRAIN_FILES_CSV}")"
  )
fi
if [[ -n "${REFERENCE_LOGPS_REUSE_VAL_FILES_CSV}" ]]; then
  extra_args+=(
    "data.reference_logps_reuse_val_files_csv=$(hydra_quote_string "${REFERENCE_LOGPS_REUSE_VAL_FILES_CSV}")"
  )
fi
if [[ -n "${REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_EFFECTIVE}" ]]; then
  extra_args+=(
    "+data.reference_logps_train_output_names_csv=$(hydra_quote_string "${REFERENCE_LOGPS_TRAIN_OUTPUT_NAMES_EFFECTIVE}")"
  )
fi
if [[ -n "${REFERENCE_LOGPS_VAL_OUTPUT_NAMES_EFFECTIVE}" ]]; then
  extra_args+=(
    "+data.reference_logps_val_output_names_csv=$(hydra_quote_string "${REFERENCE_LOGPS_VAL_OUTPUT_NAMES_EFFECTIVE}")"
  )
fi

"${PYTHON_BIN}" -m recipe.dpo.main_dpo \
  --config-name=dpo_prospect_dpo \
  data.train_files="${TRAIN_FILES_OVERRIDE}" \
  actor_rollout_ref.model.path="${MODEL_DIR}" \
  actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  algorithm.dpo_beta="${BETA}" \
  algorithm.dpo_loss_type=prospect_dpo \
  algorithm.reference_free=false \
  algorithm.prospect_dpo_alpha_tau="${ALPHA_TAU}" \
  algorithm.prospect_dpo_alpha_k="${ALPHA_K}" \
  algorithm.prospect_dpo_lambda_max="${LAMBDA_MAX}" \
  algorithm.prospect_dpo_lambda_gamma="${LAMBDA_GAMMA}" \
  data.prompt_key="${PROMPT_KEY}" \
  data.response_key="${RESPONSE_KEY}" \
  data.label_key="${LABEL_KEY}" \
  data.s_dwell_key="${S_DWELL_KEY}" \
  data.p_ctr_key="${P_CTR_KEY}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  +trainer.log_freq="${LOG_FREQ}" \
  "${extra_args[@]}" \
  "$@"
