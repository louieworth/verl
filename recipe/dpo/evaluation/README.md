# PENS Evaluation Notes

This directory adds local evaluation helpers for the public
[`PENS`](https://msnews.github.io/pens.html) dataset:

- official dataset page: <https://msnews.github.io/pens.html>
- official paper: [Ao et al., ACL 2021](https://www.microsoft.com/en-us/research/publication/pens-a-dataset-and-generic-framework-for-personalized-news-headline-generation/)
- official code: <https://github.com/LLluoling/PENS-Personalized-News-Headline-Generation>

## What The Public PENS Baselines Are

The public ACL 2021 paper reports **20 methods** in total:

- 2 non-personalized headline generation baselines:
  - `Pointer-Gen`
  - `PG+RL-ROUGE`
- 18 personalized baselines from:
  - 6 user encoders from personalized news recommendation:
    - `EBNR`
    - `DKN`
    - `NPA`
    - `NRMS`
    - `LSTUR`
    - `NAML`
  - 3 user-preference injection methods into the headline generator:
    - `IM-1`: initialize the decoder hidden state with the user embedding
    - `IM-2`: inject the user embedding into the attention computation
    - `IM-3`: concatenate the user embedding to the decoder output before token prediction

The official paper reports these headline-generation ROUGE numbers on the PENS test set:

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| Pointer-Gen | 19.86 | 7.76 | 18.83 |
| PG+RL-ROUGE | 20.56 | 8.42 | 20.03 |
| EBNR + IM-1 | 25.13 | 9.03 | 20.73 |
| EBNR + IM-2 | 25.49 | 9.14 | 20.82 |
| EBNR + IM-3 | 24.62 | 8.95 | 20.40 |
| DKN + IM-1 | 25.97 | 9.23 | 20.92 |
| DKN + IM-2 | 27.48 | 10.07 | 21.81 |
| DKN + IM-3 | 25.02 | 8.98 | 20.34 |
| NPA + IM-1 | 25.49 | 9.14 | 20.82 |
| NPA + IM-2 | 26.11 | 9.58 | 21.40 |
| NPA + IM-3 | 26.35 | 9.71 | 21.82 |
| NRMS + IM-1 | 24.92 | 9.01 | 20.75 |
| NRMS + IM-2 | 26.15 | 9.37 | 21.03 |
| NRMS + IM-3 | 25.41 | 9.12 | 20.91 |
| LSTUR + IM-1 | 23.71 | 8.73 | 21.13 |
| LSTUR + IM-2 | 24.10 | 8.82 | 20.73 |
| LSTUR + IM-3 | 23.11 | 8.42 | 20.38 |
| NAML + IM-1 | 27.49 | 10.14 | 21.62 |
| NAML + IM-2 | 28.01 | 10.72 | 22.24 |
| NAML + IM-3 | 27.25 | 10.01 | 21.40 |

The strongest official ACL 2021 baseline is `NAML + IM-2`.

## How The Official PENS Evaluation Works

PENS has two evaluation stages in the official framework.

### Headline-generation evaluation

The official offline headline evaluation does **not** require a separate judge model.

It uses the manually-written personalized headlines in the PENS test set as gold references, then reports:

- `ROUGE-1`
- `ROUGE-2`
- `ROUGE-L`

These are overlap-based generation metrics between a predicted title and the gold personalized title.
In the PENS paper, the reported numbers are **ROUGE F1** scores.

Let the predicted title be tokenized as `y_hat = (w_1, ..., w_m)` and the gold title as
`y = (v_1, ..., v_n)`.

- `ROUGE-1`
  - meaning: unigram overlap, i.e. whether the prediction covers the important words in the gold title
  - define the multiset of unigrams as `G_1(y_hat)` and `G_1(y)`
  - overlap count:
    `match_1 = sum_{g in V} min(count(g, y_hat), count(g, y))`
  - precision:
    `P_1 = match_1 / |G_1(y_hat)|`
  - recall:
    `R_1 = match_1 / |G_1(y)|`
  - F1:
    `ROUGE-1 = 2 * P_1 * R_1 / (P_1 + R_1)`

- `ROUGE-2`
  - meaning: bigram overlap, i.e. whether the prediction matches short phrases and local word order in the gold title
  - define the multiset of bigrams as `G_2(y_hat)` and `G_2(y)`
  - overlap count:
    `match_2 = sum_{g in V_2} min(count(g, y_hat), count(g, y))`
  - precision:
    `P_2 = match_2 / |G_2(y_hat)|`
  - recall:
    `R_2 = match_2 / |G_2(y)|`
  - F1:
    `ROUGE-2 = 2 * P_2 * R_2 / (P_2 + R_2)`

- `ROUGE-L`
  - meaning: longest-common-subsequence overlap, i.e. whether the prediction matches the overall word order and global structure of the gold title
  - let `L = LCS(y_hat, y)` be the length of the longest common subsequence between prediction and gold
  - precision:
    `P_L = L / m`
  - recall:
    `R_L = L / n`
  - F1:
    `ROUGE-L = 2 * P_L * R_L / (P_L + R_L)`

In practice:

- higher `ROUGE-1` usually means better content-word coverage
- higher `ROUGE-2` usually means better phrase-level faithfulness
- higher `ROUGE-L` usually means better overall sequence similarity

The ACL 2021 paper states that headline quality is evaluated with **F1 ROUGE** against the manually written headlines, and the official repository also computes ROUGE over `(prediction, gold)` pairs.
The paper also says the reported ROUGE values are averaged over **10 independent runs**, and mentions
the ROUGE configuration `-a -c 95 -m -n 4 -w 1.2`.

The local scorer in this directory uses `user_id` + `news_id` as the primary join key.
The released `personalized_test.tsv` contains `45` duplicated `user_id` + `news_id` pairs with different
human rewrites, so the scorer collapses them into a single example with multiple references:

- `gold_headlines`: all personalized gold titles for that pair
- scoring rule: compute ROUGE against each gold title and keep the maximum score for that pair

This directory provides:

1. download the dataset

```bash
bash recipe/dpo/evaluation/download_pens_data.sh
```

2. score your predictions directly against the official `personalized_test.tsv`

```bash
python3 recipe/dpo/evaluation/score_pens_predictions.py \
  --predictions /path/to/predictions.jsonl \
  --references /path/to/personalized_test.tsv
```

Expected prediction format:

```json
{"user_id": "NT1", "news_id": "N24110", "prediction": "Your generated personalized headline"}
```

The scorer matches rows by `user_id` + `news_id`.
If your prediction file only contains one generated title per line, `.txt` input is also supported and
will be aligned by row order when the prediction count matches the reference count.

3. optionally flatten `personalized_test.tsv` into a row-wise inspection file

```bash
python3 recipe/dpo/evaluation/prepare_pens_eval_examples.py \
  --news-tsv /path/to/PENS/news.tsv \
  --test-tsv /path/to/personalized_test.tsv \
  --output /path/to/pens_eval_examples.jsonl
```

The flattened file is optional. It is only useful if you want a per-row debugging file with:

- `user_id`
- `news_id`
- `candidate_category`
- `candidate_topic`
- `candidate_title`
- `gold_headlines`
- `gold_headline`
- `gold_headline_count`

4. run personalized-title generation end to end with a local HF model

```bash
bash recipe/dpo/evaluation/run_pens_personalized_eval.sh
```

The shell pipeline expects a directly loadable Hugging Face model and defaults to:

```bash
MODEL_PATH=/data/data/jiangli/ckpt/PENS/single_wise_dpo_click_hist_positive_only_lora_qwen3_5-4b/global_step_5341/actor/hf_merged
TEST_FILE=/data/data/jiangli/data/pens/extract/personalized_test.tsv
PROMPT_FILE=/data/data/jiangli/data/pens/eval/prompts.parquet
```

The default output files are:

```bash
RAW_FILE=./gen_results/<model>.parquet
RESULT_JSON_FILE=./results/result.json
```

The pipeline extracts predictions and computes ROUGE in memory. It no longer writes
`predictions.parquet`, `per_example_scores.parquet`, or a standalone metrics file.
The shared `result.json` stores the metrics and metadata under the derived model slug key.

If your machine has multiple busy GPUs, restrict visible GPUs and set the rollout sizes explicitly, for example:

```bash
CUDA_VISIBLE_DEVICES=1 NGPUS_PER_NODE=1 GEN_TP=1 PASS_K=1 \
  bash recipe/dpo/evaluation/run_pens_personalized_eval.sh
```

The shell entrypoint exposes all generation and evaluation parameters, including:

```bash
NNODES=1
NGPUS_PER_NODE=1
GEN_TP=1
PASS_K=1
GEN_TEMPERATURE=0.7
GEN_TOP_P=0.8
GEN_TOP_K=20
GEN_PROMPT_LENGTH=3072
GEN_RESPONSE_LENGTH=64
VLLM_ENABLE_THINKING=false
BACKEND=auto
RESPONSE_INDEX=0
ALIGN_BY_ORDER=false
```

`recipe/dpo/evaluation/run_pens_personalized_eval.sh` now runs generation directly through
`verl.trainer.main_generation_server`, while
`recipe/dpo/evaluation/run_pens_personalized_eval.py` only handles post-generation parsing,
ROUGE scoring, and `results/result.json` upsert.

For Qwen3.5 headline generation, the default evaluation config in this script uses
the model's non-thinking mode. On the current merged checkpoint, this materially
improves extraction quality over longer-response thinking mode runs.

It runs these steps:

```bash
python3 -m verl.trainer.main_generation_server \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.prompt_length=3072 \
  actor_rollout_ref.rollout.response_length=64 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${PASS_K}" \
  data.train_files="['${PROMPT_FILE}']" \
  data.prompt_key=prompt \
  +data.output_path="./gen_results/<model>.parquet"

python3 recipe/dpo/evaluation/run_pens_personalized_eval.py \
  --raw-file "./gen_results/<model>.parquet" \
  --test-file "${TEST_FILE}" \
  --result-json-file "./results/result.json"
```

The pipeline assumes `PROMPT_FILE` already exists. If it is missing, the shell script exits immediately.

The prepared prompt follows the single-wise DPO click-history format:

- user clicked news titles as history
- candidate news body as the source article
- JSON-only output requirement:

```json
{"headline": "<personalized headline>"}
```

The prepared evaluation file should be built ahead of time. The prompt parquet stores `prompt` as a nested
chat-message column so it can be consumed directly by
`verl.trainer.main_generation_server`.

The shared `result.json` uses the model slug as the top-level key:

```json
{
  "<model>": {
    "backend": "python",
    "count": 20554,
    "joined_count": 20554,
    "skipped_count": 0,
    "rouge_1_f1": 0.0,
    "rouge_2_f1": 0.0,
    "rouge_l_f1": 0.0,
    "non_empty_prediction_count": 20554,
    "parse_status_counts": {
      "inline_json": 20554
    },
    "model_path": "/path/to/model",
    "raw_generation_file": "/path/to/gen_results/<model>.parquet"
  }
}
```

## Dummy Sanity Checks For ROUGE

The following dummy prediction files were generated against the released PENS test references:

- `/data/data/jiangli/data/pens/extract/dummy_eval/pred_gold.jsonl`
- `/data/data/jiangli/data/pens/extract/dummy_eval/pred_copy_candidate.jsonl`
- `/data/data/jiangli/data/pens/extract/dummy_eval/pred_constant.jsonl`
- `/data/data/jiangli/data/pens/extract/dummy_eval/pred_random_gold_shuffle.jsonl`
- `/data/data/jiangli/data/pens/extract/dummy_eval/pred_random_tokens.jsonl`

Each dummy prediction row uses:

```json
{"user_id": "NT1", "news_id": "N24110", "prediction": "Your generated personalized headline"}
```

All scores below were computed with:

```bash
python3 recipe/dpo/evaluation/score_pens_predictions.py \
  --predictions /data/data/jiangli/data/pens/extract/dummy_eval/<file>.jsonl \
  --references /data/data/jiangli/data/pens/extract/personalized_test.tsv
```

This evaluates `20554` unique `user_id` + `news_id` pairs after duplicate-pair aggregation.

The local scorer returns values in `[0, 1]`; multiply by `100` if you want the same scale as the ACL 2021 paper table.

| Dummy prediction | Meaning | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | ---: | ---: | ---: |
| `pred_gold.jsonl` | prediction equals one of the pair's gold personalized titles | 0.99995 | 0.99990 | 0.99995 |
| `pred_copy_candidate.jsonl` | copy the original non-personalized `candidate_title` | 0.44783 | 0.25120 | 0.34901 |
| `pred_random_gold_shuffle.jsonl` | assign another sample's gold headline at random | 0.03120 | 0.00056 | 0.02954 |
| `pred_constant.jsonl` | always output `breaking news update` | 0.00334 | 0.00008 | 0.00334 |
| `pred_random_tokens.jsonl` | random token string baseline | 0.00147 | 0.00002 | 0.00147 |

You can use these rows as rough anchors:

- near `1.0`: essentially perfect match to the gold personalized title
- around `0.45 / 0.25 / 0.35`: copying the original article title without personalization
- near `0`: random or degenerate generation

To reproduce the dummy files:

```bash
python3 recipe/dpo/evaluation/make_dummy_pens_predictions.py \
  --references /data/data/jiangli/data/pens/extract/personalized_test.tsv \
  --news-tsv /data/data/jiangli/data/pens/extract/news.tsv \
  --output-dir /data/data/jiangli/data/pens/extract/dummy_eval
```

## Official Training Recipe Behind The Baselines

From the ACL 2021 paper and the official repository:

1. Preprocess `news.tsv`, `train.tsv`, `valid.tsv`, `test.tsv`.
2. Train a user encoder as a news recommendation model.
3. Pretrain the headline generator by maximizing likelihood with a fixed global user embedding.
4. Fine-tune the personalized generator with RL for 2 epochs.

The public codebase exposes:

- `pensmodule/UserEncoder/`: recommendation-side user encoders
- `pensmodule/Generator/`: personalized headline generator

The public repository README also notes:

- the paper uses Monte Carlo search for RL training
- the released code additionally provides an `a2c` training path because Monte Carlo training is slow and unstable

## Reward Function Used In The Official Framework

The official PENS model page says the reward is estimated from:

- personalization
- fluency
- factualness

The released public code implements this training reward using:

- a personalization term from user/news embeddings
- a fluency term from GPT-2 perplexity-like scoring
- a coverage/factualness-related ROUGE term against the news body

This is **training-time reward**, not the official offline test metric.

## Notes

- There is **no extra evaluation model** required for the official offline PENS benchmark.
- If you only want paper-style test metrics, you only need the PENS test set plus ROUGE scoring.
- The lightweight scorer in this directory supports a pure-Python fallback. If you want behavior closer to the public PENS repo, install the `rouge` Python package and run with `--backend rouge`.
