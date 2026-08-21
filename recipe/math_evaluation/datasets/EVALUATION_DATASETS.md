# Math Evaluation Datasets

The canonical OPD/OPSD suite contains only these datasets:

| Name | Hugging Face source | Split | Expected rows | Scorer |
|---|---|---:|---:|---|
| AIME 2025 | `math-ai/aime25` | `test` | 30 | `math_reward` |
| AIME 2026 | `math-ai/aime26` | `test` | 30 | `math_reward` |
| HMMT February 2026 | `MathArena/hmmt_feb_2026` | `train` | 33 | `math_reward` |
| AMO-Bench | `meituan-longcat/AMO-Bench` | `test` | 39 parser-valid rows | `amobench_parser_reward` |

AIME24, HMMT23/24/25, MATH500, BeyondAIME, GSM8K and USAMO remain
available to legacy callers, but are not selected by any canonical default.

## Preparation

From the repository root:

```bash
python recipe/math_evaluation/datasets/prepare_aime.py
python recipe/math_evaluation/datasets/prepare_hmmt.py
python recipe/math_evaluation/datasets/prepare_additional_eval_datasets.py
```

The default output root is `data/eval_dataset/math`, producing:

```text
aime25/aime25_test.parquet
aime26/aime26_test.parquet
hmmt26/hmmt26_test.parquet
amobench/amobench_test.parquet
```

Each parquet has `data_source`, `prompt`, `ability`, `reward_model`, and
`extra_info`. Prompts contain one user message, but canonical generation renders
only its content as a raw Base-model completion; no chat template is applied.

## Canonical metrics

Every problem must have exactly 16 generations. Avg@16 is the correctness mean
over all generations; Pass@16 is the fraction of problems with at least one
correct generation. The reported macro is an unweighted mean over the four
datasets.
