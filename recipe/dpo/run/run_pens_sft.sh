#!/usr/bin/env bash
# PENS positive-only SFT followed by the same interleaved headline evaluation
# protocol used by run_single_wise_click_hist_all.sh.

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

# Paths and local compiler caches share common.py defaults with DPO.
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export SFT_NNODES="${SFT_NNODES:-1}"
export SFT_N_GPUS_PER_NODE="${SFT_N_GPUS_PER_NODE:-4}"

export SFT_DATA_ROOT="${SFT_DATA_ROOT:-${PENS_SFT_DATA_ROOT}}"
export SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-${SFT_DATA_ROOT}/pens_sft_train.parquet}"
export SFT_VAL_FILE="${SFT_VAL_FILE-}"
export SFT_MODEL_DIR="${SFT_MODEL_DIR:-${PENS_MODEL_DIR}}"
export SFT_TOKENIZER_PATH="${SFT_TOKENIZER_PATH:-${SFT_MODEL_DIR}}"

# Match DPO's optimization recipe: LoRA r=64/alpha=128, global batch 128,
# micro batch 4, bf16, and the shared LR with no warmup or decay.
export SFT_LORA_RANK="${SFT_LORA_RANK:-64}"
export SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-128}"
export SFT_LORA_TARGET_MODULES="${SFT_LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-128}"
export SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-4}"
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-8240}"
export SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-32768}"
export SFT_USE_DYNAMIC_BSZ="${SFT_USE_DYNAMIC_BSZ:-false}"
# The engine-based SFT trainer represents variable-length samples as nested
# tensors. Its FSDP language-model engine only supports this no-padding mode.
export SFT_PAD_MODE="${SFT_PAD_MODE:-no_padding}"
# Keep the same optimization hyperparameters as DPO while avoiding quadratic
# activation waste from padding each four-sample micro-batch to its longest row.
export SFT_USE_REMOVE_PADDING="${SFT_USE_REMOVE_PADDING:-true}"
export SFT_NUM_WORKERS="${SFT_NUM_WORKERS:-8}"
export SFT_MODEL_DTYPE="${SFT_MODEL_DTYPE:-bf16}"
export SFT_ATTN_IMPLEMENTATION="${SFT_ATTN_IMPLEMENTATION:-flash_attention_2}"
export SFT_LR="${SFT_LR:-${PENS_LR}}"
export SFT_LR_SCHEDULER_TYPE="${SFT_LR_SCHEDULER_TYPE:-constant}"
export SFT_LR_WARMUP_STEPS="${SFT_LR_WARMUP_STEPS:-0}"
export SFT_LR_WARMUP_STEPS_RATIO="${SFT_LR_WARMUP_STEPS_RATIO:-0.0}"
export SFT_LR_MIN_RATIO="${SFT_LR_MIN_RATIO:-0.0}"
export SFT_LR_NUM_CYCLES="${SFT_LR_NUM_CYCLES:-0.5}"
export SFT_WEIGHT_DECAY="${SFT_WEIGHT_DECAY:-0.01}"
export SFT_SEED="${SFT_SEED:-42}"

export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
export SFT_TOTAL_TRAINING_STEPS="${SFT_TOTAL_TRAINING_STEPS:-${PENS_TOTAL_TRAINING_STEPS}}"
export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-${PENS_EVAL_INTERVAL}}"
export SFT_CHECKPOINT_EVAL_INTERVAL="${SFT_CHECKPOINT_EVAL_INTERVAL:-${PENS_EVAL_INTERVAL}}"
export SFT_MAX_CKPT_TO_KEEP="${SFT_MAX_CKPT_TO_KEEP:-2}"
export SFT_KEEP_BEST_CHECKPOINT="${SFT_KEEP_BEST_CHECKPOINT:-false}"
export SFT_RESUME_MODE="${SFT_RESUME_MODE:-auto}"
export SFT_LOGGER="${SFT_LOGGER:-${PENS_LOGGER}}"
export SFT_CHECKPOINT_SAVE_CONTENTS="${SFT_CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}"

export SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${PENS_CKPT_ROOT}}"
export SFT_PROJECT_NAME="${SFT_PROJECT_NAME:-${PENS_PROJECT_NAME}}"
export SFT_RUN_TIMESTAMP="${SFT_RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

export SFT_PYTHON_BIN="${SFT_PYTHON_BIN:-${PENS_PYTHON_BIN}}"
export SFT_TORCHRUN_BIN="${SFT_TORCHRUN_BIN:-}"
export SFT_ENTRYPOINT="${SFT_ENTRYPOINT:--m verl.trainer.sft_trainer}"

# External PENS headline evaluation. Defaults intentionally disable early
# stopping so all 10 boundaries through step 500 are evaluated by default.
export SFT_RUN_CHECKPOINT_EVAL="${SFT_RUN_CHECKPOINT_EVAL:-true}"
export SFT_CHECKPOINT_EVAL_PROMPT_FILE="${SFT_CHECKPOINT_EVAL_PROMPT_FILE:-${PENS_EVAL_PROMPT_FILE}}"
export SFT_CHECKPOINT_EVAL_TEST_FILE="${SFT_CHECKPOINT_EVAL_TEST_FILE:-${PENS_EVAL_TEST_FILE}}"
export SFT_CHECKPOINT_EVAL_NGPUS_PER_NODE="${SFT_CHECKPOINT_EVAL_NGPUS_PER_NODE:-${SFT_N_GPUS_PER_NODE}}"
export SFT_EARLY_STOP_PATIENCE="${SFT_EARLY_STOP_PATIENCE:-0}"
export SFT_CATASTROPHIC_DROP="${SFT_CATASTROPHIC_DROP:-1.0}"
export SFT_MINIMUM_OUTPUT_COVERAGE="${SFT_MINIMUM_OUTPUT_COVERAGE:-0.0}"
export SFT_TARGET_ROUGE_1="${SFT_TARGET_ROUGE_1:-1.0}"
export SFT_MINIMUM_ROUGE_2="${SFT_MINIMUM_ROUGE_2:-0.0}"
export SFT_MINIMUM_ROUGE_L="${SFT_MINIMUM_ROUGE_L:-0.0}"

export GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"
export GEN_TOP_P="${GEN_TOP_P:-0.8}"
export GEN_TOP_K="${GEN_TOP_K:-20}"
export GEN_PROMPT_LENGTH="${GEN_PROMPT_LENGTH:-8000}"
export GEN_RESPONSE_LENGTH="${GEN_RESPONSE_LENGTH:-128}"
export GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-8300}"
export GEN_MAX_NUM_SEQS="${GEN_MAX_NUM_SEQS:-32}"
export GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.1}"
export PASS_K="${PASS_K:-1}"
export VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}"
pens_init_run sft "$@"
if is_truthy "${SFT_TRAIN_SEGMENT:-false}"; then
  export SFT_STOP_AT_STEP="${SINGLE_WISE_DPO_STOP_AT_STEP:-${SFT_STOP_AT_STEP:-${SFT_TOTAL_TRAINING_STEPS}}}"
fi

# The outer driver repeatedly invokes this script with SFT_TRAIN_SEGMENT=true.
if ! is_truthy "${SFT_TRAIN_SEGMENT:-false}" && is_truthy "${SFT_RUN_CHECKPOINT_EVAL}"; then
  exec bash "${SCRIPT_DIR}/run_pens_sft_interleaved_eval.sh" "$@"
fi

if [[ "${SFT_NNODES}" != "1" ]]; then
  echo "This local interleaved SFT launcher supports SFT_NNODES=1; got ${SFT_NNODES}." >&2
  exit 1
fi
if [[ "${SFT_PAD_MODE}" != "no_padding" ]]; then
  echo "The FSDP SFT engine only supports SFT_PAD_MODE=no_padding; got ${SFT_PAD_MODE}." >&2
  exit 1
fi
if [[ ! -x "${SFT_PYTHON_BIN}" ]]; then
  echo "Missing SFT Python executable: ${SFT_PYTHON_BIN}" >&2
  exit 1
fi
if [[ -n "${SFT_TORCHRUN_BIN}" && ! -x "${SFT_TORCHRUN_BIN}" ]]; then
  echo "Missing torchrun executable: ${SFT_TORCHRUN_BIN}" >&2
  exit 1
fi
for required_file in "${SFT_TRAIN_FILE}" "${SFT_MODEL_DIR}/config.json" "${SFT_TOKENIZER_PATH}/tokenizer_config.json"; do
  [[ -f "${required_file}" ]] || { echo "Missing SFT input: ${required_file}" >&2; exit 1; }
done

read -r -a entrypoint_args <<< "${SFT_ENTRYPOINT}"
launcher_cmd=("${SFT_PYTHON_BIN}" -m torch.distributed.run)
if [[ -n "${SFT_TORCHRUN_BIN}" ]]; then
  launcher_cmd=("${SFT_TORCHRUN_BIN}")
fi
val_files_override="[]"
if [[ -n "${SFT_VAL_FILE}" ]]; then
  val_files_override="[${SFT_VAL_FILE}]"
fi

cmd=(
  "${launcher_cmd[@]}"
  --standalone
  --nnodes=1
  "--nproc_per_node=${SFT_N_GPUS_PER_NODE}"
  "${entrypoint_args[@]}"
  "data.train_files=[${SFT_TRAIN_FILE}]"
  "data.val_files=${val_files_override}"
  "data.messages_key=messages"
  "data.max_length=${SFT_MAX_LENGTH}"
  "data.max_token_len_per_gpu=${SFT_MAX_TOKEN_LEN_PER_GPU}"
  "data.train_batch_size=${SFT_TRAIN_BATCH_SIZE}"
  "data.micro_batch_size_per_gpu=${SFT_MICRO_BATCH_SIZE_PER_GPU}"
  "data.truncation=right"
  "data.pad_mode=${SFT_PAD_MODE}"
  "data.use_dynamic_bsz=${SFT_USE_DYNAMIC_BSZ}"
  "data.num_workers=${SFT_NUM_WORKERS}"
  "+data.apply_chat_template_kwargs={enable_thinking:false}"
  "data.ignore_input_ids_mismatch=True"
  "model.path=${SFT_MODEL_DIR}"
  "model.tokenizer_path=${SFT_TOKENIZER_PATH}"
  "model.lora_rank=${SFT_LORA_RANK}"
  "model.lora_alpha=${SFT_LORA_ALPHA}"
  "model.target_modules=${SFT_LORA_TARGET_MODULES}"
  "model.enable_gradient_checkpointing=True"
  "model.trust_remote_code=True"
  "model.use_remove_padding=${SFT_USE_REMOVE_PADDING}"
  "model.allow_unsupported_remove_padding=True"
  "+model.override_config.attn_implementation=${SFT_ATTN_IMPLEMENTATION}"
  "engine.model_dtype=${SFT_MODEL_DTYPE}"
  "engine.seed=${SFT_SEED}"
  "optim.lr=${SFT_LR}"
  "optim.lr_scheduler_type=${SFT_LR_SCHEDULER_TYPE}"
  "optim.lr_warmup_steps=${SFT_LR_WARMUP_STEPS}"
  "optim.lr_warmup_steps_ratio=${SFT_LR_WARMUP_STEPS_RATIO}"
  "optim.min_lr_ratio=${SFT_LR_MIN_RATIO}"
  "optim.num_cycles=${SFT_LR_NUM_CYCLES}"
  "optim.weight_decay=${SFT_WEIGHT_DECAY}"
  "optim.clip_grad=1.0"
  "trainer.total_epochs=${SFT_TOTAL_EPOCHS}"
  "trainer.total_training_steps=${SFT_TOTAL_TRAINING_STEPS}"
  "trainer.stop_at_step=${SFT_STOP_AT_STEP:-${SFT_TOTAL_TRAINING_STEPS}}"
  "trainer.seed=${SFT_SEED}"
  "trainer.nnodes=${SFT_NNODES}"
  "trainer.n_gpus_per_node=${SFT_N_GPUS_PER_NODE}"
  "trainer.default_local_dir=${SFT_DEFAULT_LOCAL_DIR}"
  "trainer.save_freq=${SFT_SAVE_FREQ}"
  "trainer.test_freq=-1"
  "trainer.max_ckpt_to_keep=${SFT_MAX_CKPT_TO_KEEP}"
  "trainer.project_name=${SFT_PROJECT_NAME}"
  "trainer.experiment_name=${SFT_EXPERIMENT_NAME}"
  "trainer.logger=${SFT_LOGGER}"
  "+trainer.wandb_run_id=${WANDB_RUN_ID}"
  "+trainer.wandb_dir=${WANDB_DIR}"
  "+trainer.eval_config_json=$(pens_hydra_quote "${PENS_EVAL_CONFIG_JSON}")"
  "trainer.resume_mode=${SFT_RESUME_MODE}"
  "checkpoint.save_contents=${SFT_CHECKPOINT_SAVE_CONTENTS}"
)
cmd+=("$@")

echo "[run_pens_sft] launching: ${cmd[*]}"
if is_truthy "${SFT_DRY_RUN:-false}"; then
  exit 0
fi
"${cmd[@]}"
for arg in "$@"; do
  case "$arg" in --cfg|--cfg=*|--info|--info=*|--help|-h) exit 0 ;; esac
done

# Current FSDP HF weights are already merged; older exports contain PEFT
# wrappers. Validate/reuse flat weights or merge the older format as needed.
latest_step="$(<"${SFT_DEFAULT_LOCAL_DIR}/latest_checkpointed_iteration.txt")"
checkpoint="${SFT_DEFAULT_LOCAL_DIR}/global_step_${latest_step}"
"${SFT_PYTHON_BIN}" "${REPO_ROOT}/recipe/dpo/scripts/merge_sft_lora_to_hf.py" \
  --src "${checkpoint}/huggingface" --dst "${checkpoint}/hf_merged" \
  --lora-rank "${SFT_LORA_RANK}" --lora-alpha "${SFT_LORA_ALPHA}"
