# Offline DPO Recipe

This recipe implements static-pair offline DPO on top of the existing `verl` FSDP worker stack.

It also supports a point-wise Prospect-DPO variant for unpaired `(prompt, response)` samples with precomputed
`label`, `s_dwell`, and `p_ctr` features.
It also supports a point-wise `single_wise_dpo` variant for offline `(prompt, response, label)` training with a
reference model, where only the response tokens contribute to the loss.

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

Or with the generic launcher:

```bash
DPO_VARIANT=sigmoid bash recipe/dpo/run/run_dpo.sh \
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

The generic launcher also supports IPO directly:

```bash
DPO_VARIANT=ipo bash recipe/dpo/run/run_dpo.sh \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

For the SimPO variant, use:

```bash
python3 -m recipe.dpo.main_dpo --config-name=dpo_simpo \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

`dpo_simpo` is reference-free, uses average response log probabilities, and adds the SimPO target reward margin `algorithm.simpo_gamma`.

The generic launcher also supports SimPO directly:

```bash
DPO_VARIANT=simpo bash recipe/dpo/run/run_dpo.sh \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

For the Prospect-DPO variant, use:

```bash
python3 -m recipe.dpo.main_dpo --config-name=dpo_prospect_dpo \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

The dedicated launcher exposes all Prospect-DPO-specific knobs as environment variables:

```bash
bash recipe/dpo/run/run_prospect_dpo.sh
```

If `PROSPECT_DPO_TRAIN_FILE` is unset, the launcher can also resolve PENS click-history and personalization train
parquets via `PROSPECT_DPO_INPUT_VARIANT` and `PROSPECT_DPO_SAMPLE_VARIANT`. In particular,
`PROSPECT_DPO_INPUT_VARIANT=click_hist` with `PROSPECT_DPO_SAMPLE_VARIANT=all` will train on both
`only_positive_click_hist_train.parquet` and `only_negative_click_hist_train.parquet`.

The generic launcher also supports Prospect-DPO directly:

```bash
DPO_VARIANT=prospect_dpo bash recipe/dpo/run/run_dpo.sh \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

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

The generic launcher also supports it directly:

```bash
DPO_VARIANT=single_wise_dpo bash recipe/dpo/run/run_dpo.sh \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/val.parquet
```

There is also a dedicated launcher that exposes the common single-wise settings as environment variables:

```bash
bash recipe/dpo/run/run_single_wise_dpo.sh
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
python3 recipe/dpo/export_lora_checkpoint_to_hf.py \
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
bash recipe/dpo/run/run_dpo_tldr_paper.sh \
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
bash recipe/dpo/run/run_dpo_hh_paper.sh \
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
python3 recipe/dpo/data/prepare_pens_singlewise_dpo.py \
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
python3 recipe/dpo/data/prepare_pens_singlewise_dpo.py \
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
python3 recipe/dpo/data/prepare_hf_dpo_datasets.py \
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
bash recipe/dpo/run/run_dpo_tldr_paper.sh \
  actor_rollout_ref.model.path=/path/to/model
```

```bash
bash recipe/dpo/run/run_dpo_hh_paper.sh \
  actor_rollout_ref.model.path=/path/to/model
```

Both scripts default to these local parquet paths:

- TL;DR: `/data/data/jiangli/parquet/dpo_tldr/train.parquet` and `/data/data/jiangli/parquet/dpo_tldr/val.parquet`
- HH: `/data/data/jiangli/parquet/dpo_hh/train.parquet` and `/data/data/jiangli/parquet/dpo_hh/val.parquet`

You can still override them either with Hydra arguments:

```bash
bash recipe/dpo/run/run_dpo_tldr_paper.sh \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/other_train.parquet \
  data.val_files=/path/to/other_val.parquet
```

or with environment variables:

```bash
DPO_HH_TRAIN_FILE=/path/to/hh_train.parquet \
DPO_HH_VAL_FILE=/path/to/hh_val.parquet \
bash recipe/dpo/run/run_dpo_hh_paper.sh \
  actor_rollout_ref.model.path=/path/to/model
```
