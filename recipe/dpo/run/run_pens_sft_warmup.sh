#!/usr/bin/env bash
# SFT warmup for PENS prospect-DPO (Step 2 of PROSPECT_DPO_IMPROVEMENT_PLAN).
#
# Trains a LoRA adapter on the positive-only PENS subset with next-token
# prediction, producing a reference-policy-aligned checkpoint that Step 3
# (fixed prospect-DPO) uses as its base model (MODEL_DIR + REFERENCE_MODEL_DIR).
#
# Launched from inside a SLURM allocation by
# recipe/dpo/run/sbatch/retry_multinode_sft.sh (torchrun-based, no Ray).

set -euxo pipefail

DATA_ROOT="${SFT_DATA_ROOT:-/data/dpo_pens/data/pens_click_hist_runE_23pct_seed42}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-${DATA_ROOT}/pens_sft_train.parquet}"
SFT_VAL_FILE="${SFT_VAL_FILE-}"

MODEL_DIR_DEFAULT="/data/.huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_DIR=${SFT_MODEL_DIR:-${MODEL_DIR_DEFAULT}}
TOKENIZER_PATH=${SFT_TOKENIZER_PATH:-${MODEL_DIR}}

LORA_RANK="${SFT_LORA_RANK:-64}"
LORA_ALPHA="${SFT_LORA_ALPHA:-128}"
LORA_TARGET_MODULES="${SFT_LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"

TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-256}"
MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-8}"
MAX_LENGTH="${SFT_MAX_LENGTH:-8256}"
MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-32768}"
TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${SFT_TOTAL_TRAINING_STEPS:-100}"
LR="${SFT_LR:-1e-5}"
LR_WARMUP_STEPS_RATIO="${SFT_LR_WARMUP_STEPS_RATIO:-0.03}"

NNODES="${SFT_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
N_GPUS_PER_NODE="${SFT_N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"

CKPT_ROOT="${SFT_CKPT_ROOT:-/data/data/jiangli/models/pens}"
PROJECT_NAME="${SFT_PROJECT_NAME:-pens-sft-warmup}"
EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-sft_warmup_pos_qwen3-4b-instruct-2507}"
DEFAULT_LOCAL_DIR="${SFT_DEFAULT_LOCAL_DIR:-${CKPT_ROOT}/${EXPERIMENT_NAME}}"
# Hot-patched 2026-04-18 20:59: loss dropped to 1.6 by step 89 (flash_attn 2
# plus short-headline data makes Qwen3-4B converge 5x faster than planned).
# Save every 50 steps so we capture the early "format-just-learned" ckpt around
# step 100-150 before the model over-fits.
SAVE_FREQ="${SFT_SAVE_FREQ:-50}"
# Hot-patched 2026-04-18 20:40: keep all rolling ckpts so we can fall back to
# earlier checkpoints (e.g., step 267) if a later one fails eval. Overrides
# sbatch default SFT_MAX_CKPT_TO_KEEP=1 regardless of env.
MAX_CKPT_TO_KEEP="${SFT_MAX_CKPT_TO_KEEP:-2}"
RESUME_MODE="${SFT_RESUME_MODE:-disable}"
LOGGER="${SFT_LOGGER:-[console]}"

PYTHON_BIN="${SFT_PYTHON_BIN:-/data/conda/envs/verl/bin/python}"
ENTRYPOINT="${SFT_ENTRYPOINT:--m verl.trainer.sft_trainer}"

cmd=(
  ${ENTRYPOINT}
  "data.train_files=[${SFT_TRAIN_FILE}]"
  "data.val_files=[${SFT_VAL_FILE}]"
  "data.messages_key=messages"
  "data.max_length=${MAX_LENGTH}"
  "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
  "data.truncation=right"
  "data.pad_mode=no_padding"
  "data.use_dynamic_bsz=True"
  # Qwen3 tokenizer adds <think>...</think> tags to the last turn when applying
  # chat template per-turn vs whole conversation, causing an input_ids mismatch
  # assertion. Accept the concatenated ids as authoritative (documented in
  # sft_trainer_engine.yaml:47).
  "data.ignore_input_ids_mismatch=True"
  "model.path=${MODEL_DIR}"
  "model.tokenizer_path=${TOKENIZER_PATH}"
  "model.lora_rank=${LORA_RANK}"
  "model.lora_alpha=${LORA_ALPHA}"
  "model.target_modules=${LORA_TARGET_MODULES}"
  "model.enable_gradient_checkpointing=True"
  "model.trust_remote_code=True"
  # flash_attn 2.8.3 rebuilt against torch 2.9 cxx11abiTRUE at 20:21 — kept as
  # flash_attention_2 for 3-5x speedup over sdpa. Revert to sdpa if symbol
  # mismatch reappears.
  "+model.override_config.attn_implementation=flash_attention_2"
  "optim.lr=${LR}"
  "optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO}"
  "optim.lr_scheduler_type=cosine"
  "optim.clip_grad=1.0"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.nnodes=${NNODES}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.default_local_dir=${DEFAULT_LOCAL_DIR}"
  "trainer.save_freq=${SAVE_FREQ}"
  "+trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.logger=${LOGGER}"
  "trainer.resume_mode=${RESUME_MODE}"
  "checkpoint.save_contents=[hf_model]"
)

cmd+=("$@")

echo "[run_pens_sft_warmup] launching: ${PYTHON_BIN} ${cmd[*]}"
exec ${PYTHON_BIN} "${cmd[@]}"
