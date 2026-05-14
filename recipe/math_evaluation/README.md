# math_evaluation

Math-reasoning evaluation: benchmark generation, scoring, dataset prep.
Shared by `recipe/opd/` (post-train eval, training-data scoring) and
`recipe/open_math_reasoning/` (SFT pipeline eval).

## Layout

```
recipe/math_evaluation/
├── compute_score.py             — dispatcher: (data_source, response, gt) → routed to verl/utils/reward_score/{gsm8k|math_reward|amobench_parser_reward}.py
├── eval_utils.py                — generation server + main_eval orchestrator
├── run_eval_suite.py            — CLI: evaluate one model across many datasets
├── run_benchmark_batch.py       — CLI: batch-evaluate from a JSON config
├── compute_pass_at_k_from_gen.py — pass@K aggregator from pass@N parquet
├── benchmark_kl_model.sh        — shell entry called by run_kl_training.sh after training
└── datasets/                    — download + parquet conversion for each benchmark
    ├── prepare_aime.py
    ├── prepare_math500.py
    ├── prepare_hmmt.py
    ├── prepare_usamo.py
    ├── prepare_additional_eval_datasets.py
    └── EVALUATION_DATASETS.md
```

## Typical use

After training, `recipe/opd/run/run_kl_training.sh` auto-invokes
`benchmark_kl_model.sh` against `hf_merged`. Standalone:

```bash
bash recipe/math_evaluation/benchmark_kl_model.sh /path/to/model/hf_merged
```

Multiple models in one run:

```bash
bash recipe/math_evaluation/benchmark_kl_model.sh /path/to/m1 /path/to/m2 ...
```

Datasets (override via env):

```bash
DATASETS="aime24 aime25 math500" bash recipe/math_evaluation/benchmark_kl_model.sh ...
```

## Scoring as a library

```python
from recipe.math_evaluation.compute_score import compute_score_data_source
score = compute_score_data_source("aime24", response_text, ground_truth)
```

The function is a pure router; the actual scorers live in `verl/utils/reward_score/`:

| `data_source` | backend |
|---|---|
| `gsm8k`, `openai/gsm8k` | `verl.utils.reward_score.gsm8k` |
| `aime24`/`aime25`/`amc23`/`math500`/`hmmt*`/`beyondaime`/`deepscaleR` | `verl.utils.reward_score.math_reward` |
| `amobench` | `verl.utils.reward_score.amobench_parser_reward` |

Used internally by `recipe/opd/dataset/score_stage1_reward.py`,
`filter_stage2_by_reward.py`, and `backfill_stage2_reward_from_stage1.py` to
score training-time rollouts.

## Preparing eval datasets

```bash
python recipe/math_evaluation/datasets/prepare_aime.py    --local_dataset_path /path/to/datasets
python recipe/math_evaluation/datasets/prepare_math500.py --local_dataset_path /path/to/datasets
python recipe/math_evaluation/datasets/prepare_hmmt.py    --local_dataset_path /path/to/datasets
python recipe/math_evaluation/datasets/prepare_usamo.py   --local_dataset_path /path/to/datasets
```

See `datasets/EVALUATION_DATASETS.md` for the full list (AIME24/25, AMC23,
MATH500, HMMT23/24/25, USAMO24/25, AMO-Bench, BeyondAIME, GSM8K).
