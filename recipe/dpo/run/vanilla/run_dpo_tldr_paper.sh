#!/usr/bin/env bash
set -euxo pipefail

TRAIN_FILE="${DPO_TLDR_TRAIN_FILE:-/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/dpo_tldr/train.parquet}"
VAL_FILE="${DPO_TLDR_VAL_FILE:-/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/dpo_tldr/val.parquet}"
MODEL_DIR="${DPO_TLDR_MODEL_DIR:-Qwen/Qwen3.5-0.8B}"
TOKENIZER_PATH="${DPO_TLDR_TOKENIZER_PATH:-${MODEL_DIR}}"
NNODES="${DPO_TLDR_NNODES:-1}"
N_GPUS_PER_NODE="${DPO_TLDR_N_GPUS_PER_NODE:-2}"
ROLLOUT_TP_SIZE="${DPO_TLDR_ROLLOUT_TP_SIZE:-1}"
ATTN_IMPLEMENTATION="${DPO_TLDR_ATTN_IMPLEMENTATION:-flash_attention_2}"
MODEL_DTYPE="${DPO_TLDR_MODEL_DTYPE:-bf16}"
USE_REMOVE_PADDING="${DPO_TLDR_USE_REMOVE_PADDING:-false}"
DYNAMIC_MAX_TOKEN_LEN_PER_GPU="${DPO_TLDR_MAX_TOKEN_LEN_PER_GPU:-8096}"
PROJECT_NAME="${DPO_TLDR_PROJECT_NAME:-PENS}"
EXPERIMENT_NAME="${DPO_TLDR_EXPERIMENT_NAME:-dpo_tldr_qwen3_5_0_8b}"
CKPT_ROOT="${DPO_TLDR_CKPT_ROOT:-/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/models/ckpt/PENS}"
CKPT_DIR="${DPO_TLDR_CKPT_DIR:-${CKPT_ROOT}/${EXPERIMENT_NAME}}"

mkdir -p "${CKPT_DIR}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing TL;DR train parquet: ${TRAIN_FILE}" >&2
  exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing TL;DR val parquet: ${VAL_FILE}" >&2
  exit 1
fi

python3 -m recipe.dpo.main_dpo \
  --config-name=dpo_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.model.path="${MODEL_DIR}" \
  actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  algorithm.dpo_beta=0.5 \
  algorithm.dpo_loss_type=sigmoid \
  algorithm.dpo_label_smoothing=0.0 \
  algorithm.reference_free=false \
  data.train_batch_size=64 \
  data.val_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=64 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${DYNAMIC_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="bf16" \
  actor_rollout_ref.actor.optim.optimizer=RMSprop \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=150 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=["model","optimizer","extra"]' \
  trainer.total_epochs=1 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  '+trainer.log_freq=10' \
  trainer.max_actor_ckpt_to_keep=1 \
  '+trainer.save_final_checkpoint=true' \
  'trainer.logger=[console,wandb]' \
  "$@"
