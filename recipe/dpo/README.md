# Offline DPO Recipe

This recipe implements static-pair offline DPO on top of the existing `verl` FSDP worker stack.

It also supports a point-wise Prospect-DPO variant for unpaired `(prompt, response)` samples with precomputed
`label`, `s_dwell`, and `p_ctr` features.
It also supports a point-wise `single_wise_dpo` variant for offline `(prompt, response, label)` training with a
reference model, where only the response tokens contribute to the loss.

## Directory layout

| Location | Responsibility |
| --- | --- |
| `recipe/dpo/*.py` | Core training: entry point, trainer, workers, losses, runtime datasets, batching, and sampling. |
| `prepare_data/` | Download, prepare, sample, and inspect datasets, including evaluation prompts and precomputed reference log probabilities. |
| `evaluation/` | Generate predictions, extract and score headlines, maintain evaluation checkpoint state, and record evaluation metrics in W&B. |
| `run/` | Training launchers, Slurm submission, shared environment loading, paths, and experiment defaults (`common.py` / `common.sh`). |
| `scripts/` | Checkpoint export and model conversion utilities. |
| `config/` | Hydra training configurations. |

Data preparation commands can run either as modules (`python3 -m recipe.dpo.prepare_data.<name>`) or by
their file path. `prepare_data/pens_eval_utils.py` builds evaluation inputs; scoring remains in `evaluation/`.
`run/common.py` is shared by launchers, evaluation, and training tracking and supports `python3 -S`.

The dataset and preprocessing modules follow these boundaries:

- `singlewise_dataset.py` tokenizes and loads single-wise training samples at runtime. It remains part of core
  training and is used by the unweighted single-wise launchers.
- `sampler.py` groups training samples by length and saves/restores iteration state. The PeNS click-history
  launcher enables it by default.
- `prepare_data/reference_logps_materializer.py` generates and reuses Parquet files with reference-model log
  probabilities. Training calls it when a reference model is required; the reference-free PeNS launchers skip it.
- `prepare_data/sample_id.py` generates deterministic sample IDs for data preparation, backfilling, and matching
  cached reference probabilities. Keep it even for reference-free training datasets.

Both checkpoint exporters are required by the current launchers:

- `scripts/export_lora_checkpoint_to_hf.py` merges a DPO actor's `lora_adapter/` into its base model.
- `scripts/merge_sft_lora_to_hf.py` validates and exports SFT Hugging Face checkpoints, including wrapped LoRA
  tensors and already merged weights.

`evaluation/wandb_utils.py` is required by `evaluation/run_pens_personalized_eval.py`. It writes evaluation
metrics and evaluation configuration to the training run. Training configuration is also stored in W&B;
the launchers no longer create a `metrics/` directory or separate `metrics.json`/`config.json` files.
Checkpoint selection and resume still use `interleaved_eval_state.json` under the model directory.

### Optional cleanup suggestions

All files are retained. These are suggestions for a separate, manual cleanup:

- `evaluation/update_pens_results_json.py` has no callers in the repository. The evaluation entry point already
  updates the results JSON through `upsert_result_json()`, using an atomic write. Remove the standalone utility
  only if no external/manual workflow uses it.
- If this checkout is exclusively for PeNS, consider removing `run/vanilla/` and
  `prepare_data/prepare_hf_dpo_datasets.py` together with the corresponding HH/TL;DR/IPO/SimPO configurations.
  They still provide supported comparison workflows; keep them if those experiments are useful.

Expected dataset columns:

```json
{
  "prompt": [{"role": "user", "content": "Tell me a joke."}],
  "chosen": "Why did the model cross the road? To reduce the loss.",
  "rejected": "I don't know."
}
```

Run with:

```bash
python3 -m recipe.dpo.main_dpo --config-name=dpo_trainer \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

For the IPO variant, use:

```bash
python3 -m recipe.dpo.main_dpo --config-name=dpo_ipo \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

`dpo_ipo` keeps the IPO loss, but uses average response log probabilities instead of summed response log probabilities for the chosen/rejected sequence scores. This follows the same length-normalization intuition used by SimPO, without changing the loss into SimPO itself.

For the SimPO variant, use:

```bash
python3 -m recipe.dpo.main_dpo --config-name=dpo_simpo \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

`dpo_simpo` is reference-free, uses average response log probabilities, and adds the SimPO target reward margin `algorithm.simpo_gamma`.

For the Prospect-DPO variant, use:

```bash
python3 -m recipe.dpo.main_dpo --config-name=dpo_prospect_dpo \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

The PeNS click-history launcher exposes the shared single-wise settings as environment variables and
selects weighted Prospect-DPO by default:

```bash
bash recipe/dpo/run/run_single_wise_click_hist_all.sh
```

If `SINGLE_WISE_DPO_TRAIN_FILE` is unset, the launcher resolves the train parquets via
`SINGLE_WISE_DPO_INPUT_VARIANT` and `SINGLE_WISE_DPO_SAMPLE_VARIANT`. In particular,
`SINGLE_WISE_DPO_INPUT_VARIANT=click_hist` with `SINGLE_WISE_DPO_SAMPLE_VARIANT=all` trains on both
`only_positive_click_hist_train.parquet` and `only_negative_click_hist_train.parquet`.

Prospect-DPO expects one response per row with these additional columns:

- `response`
- `label` (`1` for positive, `0` for negative)
- `s_dwell`
- `p_ctr`

For the single-wise offline DPO variant, use:

```bash
python3 -m recipe.dpo.main_dpo --config-name=dpo_single_wise_dpo \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

The shared launcher defaults to `POINTWISE_DPO_LOSS_TYPE=prospect_dpo` and exposes the common single-wise settings as environment variables. Set `POINTWISE_DPO_LOSS_TYPE=single_wise_dpo` explicitly to select the older BCE loss:

```bash
POINTWISE_DPO_LOSS_TYPE=single_wise_dpo bash recipe/dpo/run/run_single_wise.sh
```

The main environment variables are:

- `SINGLE_WISE_DPO_MODEL_DIR`
- `SINGLE_WISE_DPO_TRAIN_FILE`
- `SINGLE_WISE_DPO_VAL_FILE`
- `SINGLE_WISE_DPO_BETA`
- `SINGLE_WISE_DPO_TRAIN_BATCH_SIZE`
- `SINGLE_WISE_DPO_VAL_BATCH_SIZE`
- `SINGLE_WISE_DPO_MAX_PROMPT_LENGTH`
- `SINGLE_WISE_DPO_MAX_RESPONSE_LENGTH`
- `SINGLE_WISE_DPO_MICRO_BATCH_SIZE`
- `SINGLE_WISE_DPO_TOTAL_EPOCHS`
- `SINGLE_WISE_DPO_CKPT_DIR`
- `SINGLE_WISE_DPO_SAVE_FREQ_EPOCHS`

By default, single-wise DPO checkpoints are saved under:

```bash
${SINGLE_WISE_DPO_CKPT_ROOT}/single_wise_dpo_${INPUT_VARIANT}_${SAMPLE_VARIANT}_${FINETUNE_VARIANT}_${MODEL_SLUG}
```

where `MODEL_SLUG` is derived from `SINGLE_WISE_DPO_MODEL_DIR`.

To export a saved LoRA checkpoint into a directly loadable Hugging Face directory, run:

```bash
python3 recipe/dpo/scripts/export_lora_checkpoint_to_hf.py \
  --actor-dir /path/to/global_step_x/actor \
  --trust-remote-code
```

This writes the merged model to `/path/to/global_step_x/actor/hf_merged` by default.

If you want to save once at the end of every epoch, set:

```bash
SINGLE_WISE_DPO_SAVE_FREQ=-1
SINGLE_WISE_DPO_SAVE_FREQ_EPOCHS=1
```

`single_wise_dpo` expects one response per row with these columns:

- `prompt`
- `response`
- `label` (`1` for positive, `0` for negative)

The offline parquet schema is the raw input schema. During training, the dataset layer tokenizes it into
`input_ids / attention_mask / position_ids / responses / response_mask / labels / label`, and only the
`response_mask` region is used for sequence log-prob computation and loss.

For a paper-like smoke test aligned with the default Appendix B DPO hyperparameters, use:

```bash
python3 -m recipe.dpo.main_dpo --config-name=dpo_paper_test \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

The `dpo_paper_test` preset sets:

- `algorithm.dpo_beta=0.1`
- effective batch size `64`
- `RMSprop`
- `lr=1e-6`
- `lr_warmup_steps=150`
- `rollout.temperature=1.0`

For TL;DR summarization, override `algorithm.dpo_beta=0.5`.

For the TL;DR-specific paper preset, use:

```bash
bash recipe/dpo/run/vanilla/run_dpo_tldr_paper.sh \
  actor_rollout_ref.model.path=/path/to/model
```

This preset fixes the paper-specified TL;DR parameters:

- `algorithm.dpo_beta=0.5`
- `data.train_batch_size=64`
- `actor_rollout_ref.actor.optim.optimizer=RMSprop`
- `actor_rollout_ref.actor.optim.lr=1e-6`
- `actor_rollout_ref.actor.optim.lr_warmup_steps=150`

It also sets `max_prompt_length=512` and `max_response_length=512` as implementation choices for a 1024-token total sequence budget, since the paper does not specify these lengths explicitly.

For the Anthropic-HH-specific paper preset, use:

```bash
bash recipe/dpo/run/vanilla/run_dpo_hh_paper.sh \
  actor_rollout_ref.model.path=/path/to/model
```

This preset fixes the paper-style HH parameters:

- `algorithm.dpo_beta=0.1`
- `data.train_batch_size=64`
- `actor_rollout_ref.actor.optim.optimizer=RMSprop`
- `actor_rollout_ref.actor.optim.lr=1e-6`
- `actor_rollout_ref.actor.optim.lr_warmup_steps=150`

It uses `max_prompt_length=256` and `max_response_length=256` as implementation choices for a 512-token total sequence budget, following public HH configs built on the original DPO codebase.

## Data Preparation

The offline DPO recipe expects pairwise parquet files with these columns:

- `prompt`
- `chosen`
- `rejected`

For point-wise offline variants:

- `prospect_dpo`: `prompt / response / label / s_dwell / p_ctr`
- `single_wise_dpo`: `prompt / response / label`

To build a PENS single-wise offline dataset from `train.tsv`, `valid.tsv`, and `news.tsv`, use:

```bash
python3 recipe/dpo/prepare_data/prepare_pens_singlewise_dpo.py \
  --news-file /data/data/jiangli/data/pens/extract/news.tsv \
  --train-file /data/data/jiangli/data/pens/extract/train.tsv \
  --val-file /data/data/jiangli/data/pens/extract/valid.tsv \
  --output-root /data/data/jiangli/parquet/pens_single_wise_dpo
```

This creates split directories:

- `/data/data/jiangli/parquet/pens_single_wise_dpo/train/*.parquet`
- `/data/data/jiangli/parquet/pens_single_wise_dpo/val/*.parquet`

If you do not want shards, use `--single-file`. This writes one parquet per split while still flushing in small
batches internally, so it does not need to hold the full dataset in memory:

```bash
python3 recipe/dpo/prepare_data/prepare_pens_singlewise_dpo.py \
  --news-file /data/data/jiangli/data/pens/extract/news.tsv \
  --train-file /data/data/jiangli/data/pens/extract/train.tsv \
  --val-file /data/data/jiangli/data/pens/extract/valid.tsv \
  --output-root /data/data/jiangli/parquet/pens_single_wise_dpo_single \
  --single-file
```

This writes:

- `/data/data/jiangli/parquet/pens_single_wise_dpo_single/train.parquet`
- `/data/data/jiangli/parquet/pens_single_wise_dpo_single/val.parquet`

If `tqdm` is installed in the environment, the builder shows progress bars for `load news`, `train rows`, and
`val rows`.

Important: the builder does not clean old outputs. If you rerun with a different mode, use a fresh `output-root`
or delete the old output directory first.

Each row contains:

- `prompt`: chat-format user message with clicked history and candidate article body
- `response`: candidate article `Headline`
- `label`: click label (`1/0`)
- `s_dwell`: positive-side Prospect-DPO feature, computed as `min(pos_weight, q90_split) / q90_split`
- `p_ctr`: negative-side Prospect-DPO feature, copied from `neg_weight` and clipped to `[0, 1]`
- metadata columns such as `user_id`, `candidate_news_id`, `candidate_category`, and `candidate_topic`

For PENS click-history exports, `q90_split` is the floor 90th percentile of the split-local raw `pos_weight`
distribution. Positive rows write `p_ctr=0`, and negative rows write `s_dwell=0`, so the same parquet files can
be used for both `single_wise_dpo` and `prospect_dpo`.

To download the original Hugging Face TL;DR and HH datasets and convert them into the required format, use:

```bash
python3 recipe/dpo/prepare_data/prepare_hf_dpo_datasets.py \
  --hf-root /data/data/jiangli/huggingface \
  --parquet-root /data/data/jiangli/parquet \
  --datasets tldr hh
```

This will create:

- `/data/data/jiangli/huggingface/openai__summarize_from_feedback__comparisons`
- `/data/data/jiangli/huggingface/Anthropic__hh-rlhf`
- `/data/data/jiangli/parquet/dpo_tldr/train.parquet`
- `/data/data/jiangli/parquet/dpo_tldr/val.parquet`
- `/data/data/jiangli/parquet/dpo_hh/train.parquet`
- `/data/data/jiangli/parquet/dpo_hh/val.parquet`

The TL;DR directory contains both the raw source files and a local Hugging Face dataset snapshot:

- `/data/data/jiangli/huggingface/openai__summarize_from_feedback__comparisons/raw/*.json`
- `/data/data/jiangli/huggingface/openai__summarize_from_feedback__comparisons/hf_dataset`

The converted parquet files are directly consumable by this recipe. For example:

```bash
bash recipe/dpo/run/vanilla/run_dpo_tldr_paper.sh \
  actor_rollout_ref.model.path=/path/to/model
```

```bash
bash recipe/dpo/run/vanilla/run_dpo_hh_paper.sh \
  actor_rollout_ref.model.path=/path/to/model
```

Both scripts default to these local parquet paths:

- TL;DR: `/data/data/jiangli/parquet/dpo_tldr/train.parquet` and `/data/data/jiangli/parquet/dpo_tldr/val.parquet`
- HH: `/data/data/jiangli/parquet/dpo_hh/train.parquet` and `/data/data/jiangli/parquet/dpo_hh/val.parquet`

You can still override them either with Hydra arguments:

```bash
bash recipe/dpo/run/vanilla/run_dpo_tldr_paper.sh \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/other_train.parquet \
  data.val_files=/path/to/other_val.parquet
```

or with environment variables:

```bash
DPO_HH_TRAIN_FILE=/path/to/hh_train.parquet \
DPO_HH_VAL_FILE=/path/to/hh_val.parquet \
bash recipe/dpo/run/vanilla/run_dpo_hh_paper.sh \
  actor_rollout_ref.model.path=/path/to/model
```
