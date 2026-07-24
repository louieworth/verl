# OPSD Best-of-N

These launchers run the `forward_y_o.sh` OPSD training path with one offline
Best-of-4 stage:

1. Generate four responses per prompt.
2. Stop generation at the configured sampling-only wall-clock deadline.
3. Keep only prompts for which all four responses completed.
4. Select the first correct response. If none is correct, select candidate 0.
5. Train one epoch through the existing `opsd/forward_y_o.sh` path.
6. Save only at the end, merge `hf_merged`, then run Avg@16/Pass@16 eval.

Sampling budgets include generation server/model startup, but do not limit
reward scoring, training, checkpoint export, or evaluation.

```bash
bash recipe/opd/run/opsd_best_of_n/qwen3_4b_instruct_math_best_of_n_8h100.sh
bash recipe/opd/run/opsd_best_of_n/qwen3_4b_instruct_code_best_of_n_8h100.sh
bash recipe/opd/run/opsd_best_of_n/qwen3_8b_math_best_of_n_8h100.sh
bash recipe/opd/run/opsd_best_of_n/qwen3_8b_code_best_of_n_8h100.sh
```

Run all four Best-of-N jobs sequentially:

```bash
bash recipe/opd/run/run_sequence_8h100.sh best_of_n
```

Use `all` instead to run all four GRPO jobs followed by all four Best-of-N
jobs. Set `SEQUENCE_DRY_RUN=true` to validate the complete sequence without
allocating GPUs.

The `2.1 hrs` code budget is interpreted as decimal hours: 7560 seconds
(2 hours 6 minutes). Override any budget with
`SAMPLING_BUDGET_SECONDS=<seconds>`.

Use `OPSD_BEST_OF_N_DRY_RUN=true` to print resolved paths and commands without
requiring GPUs or local model weights.
