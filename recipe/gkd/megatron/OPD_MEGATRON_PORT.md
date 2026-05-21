# OPD Megatron Port

This document describes the OPD compatibility layer added on top of the official
`recipe/gkd/megatron` implementation.

## Scope

The port keeps the current official Megatron actor, async vLLM rollout server
(`AgentLoopManager` + `CheckpointEngineManager`), teacher server, and
vocab-parallel sparse distillation loss path. OPD behavior is added as
configuration and scheduler extensions:

- `opd.y_mode=y_o`: official behavior. Student rolls out `y_o`; teacher returns
  top-k distributions on the generated sequence; actor updates on `x + y_o`.
- `opd.y_mode=y_r`: student first rolls out `y_o`; the teacher server then
  generates a rewrite `y_r` from the OPD/OPSD refine prompt; actor updates on
  `x + y_r`.
- `trainer.optimization_mode=one_step` with `trainer.scheduler=auto`: the
  dataloader uses the full dataset as one batch, and the trainer resolves the
  scheduler to `one_step` so each epoch runs exactly one
  rollout/teacher/update cycle.
- `trainer.scheduler=one_step_off` and `trainer.scheduler=two_step_off` are the
  official async pipeline schedulers. They are not one-step optimization modes.
- `TokenizedPromptDataset` adapts the current repo's `RLHFDataset` raw-prompt
  output back to the tensorized prompt contract expected by the official
  Megatron GKD trainer.
- Rollout generation is online. The runner expects raw prompt data and does not
  read or create cached `y_o`/`y_r` generation files.
- One-step batches are padded to the async rollout worker dispatch
  multiple when needed. Padded duplicate rows are masked out of the
  distillation loss, so they do not contribute gradients.

## Official-First Decisions

When the FSDP OPD implementation and the official GKD/Megatron implementation
disagree, this port follows the official Megatron path.

- Losses use teacher top-k sparse targets from the teacher server, not FSDP
  full-vocabulary teacher logits.
- `opd.kl_type=forward` maps to official `kl` (`KL(P_teacher_topk || Q_student)`).
- `opd.kl_type=reverse` maps to official `rkl` on renormalized teacher top-k
  support.
- `opd.kl_type=jsd` maps to official sparse JSD with analytic student rest term.
- `opd.top_k=0` means "use every top-k row returned by the teacher server";
  `opd.top_k=N` truncates the returned teacher support to `N`.
- `opd.kl_token_clip` clips the per-token Megatron distillation loss before
  reduction.

## y_r Alignment

Megatron actor loss accepts one student sequence and one same-shaped teacher
top-k tensor. For `y_r`, the teacher prompt may differ from the student prompt,
so the port does not feed the teacher prompt into the actor. Instead:

1. Actor rollout produces `y_o` from student prompt `x`.
2. Teacher server generates `y_r` from the refine prompt:
   - OPD: `x + y_o`
   - OPSD: `x + y* + y_o`
3. The trainer rebuilds a student batch as `x + y_r`.
4. Teacher response-position top-k rows are remapped onto the student loss
   positions that predict the `y_r` tokens.

For `opd.teacher_training_prompt=vanilla`, the teacher rewrite is still
generated from the refine prompt, then a second teacher request scores `y_r`
under the vanilla prompt.

## Run Scripts

New scripts live under:

- `recipe/gkd/megatron/run/run_opd_megatron.sh`
- `recipe/gkd/megatron/run/opd/*.sh`
- `recipe/gkd/megatron/run/opsd/*.sh`

Examples:

```bash
# Teacher server must already be running.
CKPT_PATH=/path/to/teacher \
bash recipe/gkd/megatron/teacher/start_server.sh

MODEL_PATH=/path/to/student \
TEACHER_MODEL_PATH=/path/to/teacher \
DATA_PATH=/path/to/deepscaleR_stage1_prompts.parquet \
MAX_SAMPLES=64 \
bash recipe/gkd/megatron/run/opd/forward_y_r.sh
```

The wrapper also accepts the same uppercase `KEY=value` assignments as script
arguments, e.g. `bash recipe/gkd/megatron/run/opd/forward_y_r.sh
MODEL_PATH=/path/to/student DATA_PATH=/path/to/data.parquet`.

`DATA_PATH` or `TRAIN_DATA_PATH` must point to a raw prompt `.parquet`, `.json`,
`.jsonl`, or a HuggingFace dataset directory saved by `save_to_disk`. If
`DATA_PATH` is unset, the wrapper uses `TRAIN_DATA_PATH` directly. It
intentionally no longer consumes or creates offline generated `y_o`/`y_r`
files.

Important environment variables:

- `Y_MODE`: `y_o` or `y_r`
- `DISTILL_MODE`: `opd` or `opsd`
- `KL_TYPE`: `forward`, `reverse`, or `jsd`
- `KL_TOKEN_CLIP`: per-token loss clip. The named `forward_clip_*` scripts set
  the default to `0.06`; other scripts default to `0`.
- `TOP_K`: teacher support truncation. The named `reverse_topk_*` scripts set
  the default to `32`; other scripts default to `0` and use all returned
  teacher top-k rows.
- `KL_METHOD`: accepted for compatibility; Megatron still uses sparse teacher
  top-k targets
- `TEACHER_TRAINING_PROMPT`: `refine` or `vanilla`
- `APPEND_INSTRUCTION_TO_PROMPT`: defaults to `True`. Online rollout prompts
  get the math instruction appended if it is not already present, matching the
  old `y_o_prepare.py` behavior without reading offline generated data.
- `PROMPT_KEY`: defaults to `auto`, which chooses `prompt`, then `problem`,
  then `question`.
- `DATA_SOURCE`: defaults to `math` when the dataset has no `data_source`
  column.
- Training-time OPSD `y*`: uses the dataset `solution` field. If `solution` is
  empty or absent, it falls back to `answer`. Evaluation still uses `answer`
  through `reward_model.ground_truth`.
- `BETA`, `TEMPERATURE`
- `MAX_PROMPT_LENGTH`: defaults to `1024` for both `y_o` and `y_r`.
  `TEACHER_MAX_PROMPT_LENGTH` defaults to `1024 + MAX_RESPONSE_LENGTH` only
  for `y_r` refine prompts; `y_o` and `y_r` vanilla use `1024`.
- `MAX_RESPONSE_LENGTH`: defaults to `8192`
- `MAX_TOKEN_LEN_PER_GPU`: accepted as an alias for Megatron
  `actor_rollout_ref.actor.max_token_len`
- `OPTIMIZATION_MODE=one_step`
- `SCHEDULER=auto` by default. Auto resolves to `one_step` for
  `OPTIMIZATION_MODE=one_step`, otherwise to the official `one_step_off`
  async pipeline.
- `TRAIN_BATCH_SIZE`: defaults to `1024`, following the GKD/Megatron-style
  batch setting in this wrapper. It is ignored by `OPTIMIZATION_MODE=one_step`
  and controls multi-step batch splitting only.
- `CKPT_BASE_DIR`: defaults to `/scratch/l/luli/jiangli/ckpt`.
- `EVAL_DATASETS_DIR`: defaults to `/scratch/l/luli/jiangli/datasets/eval`.
- `SAVE_FREQ`: defaults to `auto`. The wrapper computes total training steps
  before launching Ray and chooses `ceil(total_steps / 5)`, so the trainer saves
  four intermediate checkpoints plus the final checkpoint when `total_steps >=
  5`. Set `SAVE_FREQ` or `SAVE_STEPS` only when you want to override this.
- `TEACHER_SERVER_HOST`, `TEACHER_SERVER_PORT`

Named KL entry points are available for both `opd/` and `opsd/`:

- Forward KL: `forward_y_o.sh`, `forward_y_r.sh`
- Forward KL with clip: `forward_clip_y_o.sh`, `forward_clip_y_r.sh`
- Reverse KL: `reverse_y_o.sh`, `reverse_y_r.sh`
- Reverse KL with top-K: `reverse_topk_y_o.sh`, `reverse_topk_y_r.sh`
- JSD: `jsd_y_o.sh`, `jsd_y_r.sh`

The wrapper also accepts common OPD aliases such as `NGPUS_PER_NODE`,
`GEN_TP`, `NUM_WORKERS`, `TRAIN_EPOCHS_PER_ROUND`, `WANDB_PROJECT`,
`WANDB_RUN_NAME`, `SAVE_STEPS`, `WARMUP_RATIO`, `WEIGHT_DECAY`, and `MIN_LR`.
FSDP/offline-pipeline-only variables such as LoRA flags, FSDP flags, stage2
reward filtering, and difficulty buckets are reported as warnings because the
official Megatron GKD path does not implement those behaviors.

## Dataflow, Checkpoints, and Evaluation

For `OPTIMIZATION_MODE=multi_step`, the trainer uses
`data.train_batch_size` to split the dataset. With `SCHEDULER=auto`, this
resolves to the official `one_step_off` async pipeline. Each batch follows:

1. Policy async rollout with `AgentLoopManager.generate_sequences`.
2. OPD/OPSD teacher request. For `Y_MODE=y_r`, the same online batch generates
   `y_o`, asks the teacher to rewrite `y_r`, rebuilds `x + y_r`, and scores
   the `y_r` tokens.
3. Megatron actor update on the current batch.

Training rollouts are not saved to disk. They live in the in-memory
`DataProto` batch returned by the official async rollout manager and are
discarded after the teacher signal and actor update. The port intentionally
does not write old offline `gen_results` training files.

The default checkpoint root is:

```text
/scratch/l/luli/jiangli/ckpt/${DISTILL_MODE_ID}/${STUDENT_MODEL_ID}/${EXPERIMENT_NAME}
```

`DISTILL_MODE_ID` is `OPD` or `OPSD`.

`EXPERIMENT_NAME` is also the default W&B run name and includes the student
model, `opd`/`opsd`, `y_o`/`y_r`, KL type, optimization mode, computed step
count, teacher model id, teacher training prompt, and date. For multi-step,
the name also includes `spe${steps_per_epoch}` and `bs${TRAIN_BATCH_SIZE}`.

Multi-step step count follows the official dataloader exactly:

```text
steps_per_epoch = floor(effective_train_samples / TRAIN_BATCH_SIZE)
total_steps = steps_per_epoch * TOTAL_EPOCHS
```

because `drop_last=True` for multi-step. One-step uses one full-dataset batch
per epoch.

The actor checkpoint path is:

```text
${CKPT_BASE_DIR}/${DISTILL_MODE_ID}/${STUDENT_MODEL_ID}/${EXPERIMENT_NAME}/global_step_${step}/actor
```

The checkpoint tracker is:

```text
${CKPT_BASE_DIR}/${DISTILL_MODE_ID}/${STUDENT_MODEL_ID}/${EXPERIMENT_NAME}/latest_checkpointed_iteration.txt
```

Run metadata is written before training and updated after training if a
checkpoint is found:

```text
${CKPT_BASE_DIR}/${DISTILL_MODE_ID}/${STUDENT_MODEL_ID}/${EXPERIMENT_NAME}/run_metadata.json
```

The Megatron actor checkpoint contains distributed state under `actor/dist_ckpt`
and HF config/tokenizer under `actor/huggingface`. When
`SAVE_MERGED_MODEL=true` or `RUN_EVAL_AFTER_TRAINING=true`, the wrapper merges
the latest actor checkpoint with:

```bash
python -m verl.model_merger merge \
  --backend megatron \
  --local_dir <...>/global_step_${step}/actor \
  --target_dir <...>/hf_merged \
  --trust-remote-code
```

By default, `verl.model_merger` reads HF config/tokenizer assets from
`<actor_ckpt>/huggingface`, which the Megatron checkpoint manager writes during
save. Set `MERGE_HF_MODEL_CONFIG_PATH=/path/to/hf_assets` only if you need to
override that local source.

Set `MERGE_TIE_WORD_EMBEDDING=True` only for tied-embedding Megatron checkpoints
that require `--tie-word-embedding`.

Post-training async math evaluation is enabled by:

```bash
RUN_EVAL_AFTER_TRAINING=true PASS_K=16 \
bash recipe/gkd/megatron/run/opd/forward_y_r.sh
```

The Slurm wrapper `recipe/gkd/megatron/run/sbatch_opd_megatron.sh` sets
`RUN_EVAL_AFTER_TRAINING=true`, `SAVE_MERGED_MODEL=true`, and `PASS_K=16` by
default, so training, merge, and evaluation stay inside the same `sbatch` job.
The teacher server still needs to be reachable before the job starts.

Default evaluation outputs:

- Generations:
  `${CKPT_BASE_DIR}/${DISTILL_MODE_ID}/${STUDENT_MODEL_ID}/${EXPERIMENT_NAME}/eval/gen_results/{dataset}_pass16_generation.parquet`
- Results JSON:
  `${VERL_ROOT}/results/${DISTILL_MODE_ID}_result.json`
- Default post-training evaluation datasets:
  `aime24,aime25,hmmt24,hmmt25,beyondaime,amobench`.
- Prepared local evaluation dataset cache includes:
  `aime24,aime25,math500,hmmt24,hmmt25,amc23,beyondaime,amobench,gsm8k`.
- `pass@16` key: `{dataset}_pass16_generation_pass_16`
- `avg@16` key: `{dataset}_avg_pass1_generation_pass_16`, the mean
  per-response correctness across all 16 generated responses
- The result entry also contains `_metadata`, copied from `run_metadata.json`,
  so the evaluated model can be traced back to KL type, optimization mode,
  `y_o`/`y_r`, `opd`/`opsd`, computed steps, save frequency, and teacher model.

W&B is only used when `LOGGER` includes `wandb`, for example
`LOGGER="['console','wandb']"`. The W&B project is `PROJECT_NAME` (or
`WANDB_PROJECT`), and the run name is `EXPERIMENT_NAME` (or `WANDB_RUN_NAME`).
If no name is supplied, the wrapper uses:

```text
${STUDENT_MODEL_ID}_${DISTILL_MODE}_${Y_MODE}_${KL_TYPE}_${STEP_TAG}_teacher-${TEACHER_MODEL_ID}_${TEACHER_TRAINING_PROMPT}_${RUN_DATE}
```

## Validation Done

Local checks completed:

```bash
python -m py_compile \
  verl/models/mcore/model_initializer.py \
  verl/workers/rollout/vllm_rollout/utils.py \
  verl/workers/rollout/vllm_rollout/vllm_async_server.py \
  recipe/gkd/megatron/main_gkd.py \
  recipe/gkd/megatron/opd_dataset.py \
  recipe/gkd/megatron/ray_trainer.py \
  recipe/gkd/megatron/megatron_workers.py \
  recipe/gkd/megatron/teacher_utils.py \
  recipe/gkd/megatron/teacher/client.py \
  recipe/gkd/megatron/teacher/vllm_engine.py \
  recipe/gkd/megatron/teacher/worker.py

bash -n \
  recipe/math_evaluation/benchmark_kl_model.sh \
  recipe/gkd/megatron/teacher/start_server.sh \
  recipe/gkd/megatron/run/run_opd_megatron.sh \
  recipe/gkd/megatron/run/sbatch_opd_megatron.sh \
  recipe/gkd/megatron/run/opd/*.sh \
  recipe/gkd/megatron/run/opsd/*.sh
```

Additional checks:

- Hydra `--cfg job` passed for an async `opd.y_mode=y_r`,
  `trainer.optimization_mode=one_step` smoke configuration.
- `recipe/gkd/megatron/run/opd/forward_y_r.sh KEY=value...` preflight parses
  compatibility arguments and fails cleanly when the teacher server is absent.
- `git diff --check` passed after the final documentation update.
- A 4xH100 SLURM allocation was held with `salloc --gres=gpu:h100:4
  --time=01:00:00 --mem=128G`. The successful minimal smoke used GPU 3 for the
  teacher server and GPU 0 for colocated Megatron actor + async vLLM rollout
  (`MAX_SAMPLES=1`, `MAX_RESPONSE_LENGTH=2`, `opd.y_mode=y_r`,
  `trainer.optimization_mode=one_step`).
- The smoke completed an online async rollout, generated `y_r` through the
  teacher, fetched teacher top-k targets, and finished one Megatron actor
  update. The final log included `INFO: update actor done.` with
  `training/global_step:1` and `actor/kl_loss:6.353798866271973`.
