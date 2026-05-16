# opd / generation

Prompt-preparation scripts called from `run_kl_training.sh` before the vllm
rollout step. They take a HuggingFace dataset (or a stage1 parquet) and emit
a parquet of `{prompt, ground_truth, extra_info}` rows ready for
`verl.trainer.main_generation_server`.

| File | Output | Used by |
| --- | --- | --- |
| `y_o_prepare.py` | stage1 prompts (just the math problem + instruction; the model rolls out → y_o = student rollout) | `run_kl_training.sh` |
| `y_r_prepare.py` | stage2 rewrite prompts for **every** stage1 row. `--distill_mode` selects between **opsd** (problem + expert solution + initial response) and **opd** (problem + initial response, no expert reference). | `run_kl_training.sh` |

Both scripts always process the full input — no built-in reward filtering.
If you want to train only on rewrites of failed samples, filter the resulting
parquet downstream via
`recipe/opd/dataset/filter_stage2_by_reward.py` (or set
`FORWARD_FILTER_STAGE2=true` in `run_kl_training.sh`).

Standalone invocation:

```bash
python recipe/opd/generation/y_o_prepare.py \
    --input_path /path/to/DeepScaleR \
    --output_file gen_results/stage1_prompts.parquet

# OPSD y_r prompts (teacher sees expert solution)
python recipe/opd/generation/y_r_prepare.py \
    --stage1_output gen_results/stage1_responses.parquet \
    --use_initial_response true \
    --distill_mode opsd \
    --output_file gen_results/stage2_prompts.parquet

# OPD y_r prompts (no expert reference)
python recipe/opd/generation/y_r_prepare.py \
    --stage1_output gen_results/stage1_responses.parquet \
    --distill_mode opd \
    --output_file gen_results/stage2_prompts.parquet
```

These scripts only build *prompts*. The rollout itself (calling the model
through vllm) is run by `verl.trainer.main_generation_server` from
`run_kl_training.sh`. Reward scoring + filtering of the resulting parquets
lives in `recipe/opd/dataset/`.
