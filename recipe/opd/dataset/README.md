# kl_training / dataset

Everything about training data — both the **PyTorch Dataset class** loaded
at runtime by the trainer, and the **CLI utilities** that prepare the
parquets it reads.

| File | Type | Role |
| --- | --- | --- |
| `data_utils.py` | runtime library | `KLTrainingDataset` (PyTorch `Dataset`), `create_kl_dataloader`, `build_teacher_prompt`, prompt templates. Imported by `kl_trainer.py`. |
| `score_stage1_reward.py` | CLI | Score stage1 rollouts and backfill `extra_info.reward` in-place. Idempotent. Called by `run_kl_training.sh`. |
| `filter_stage2_by_reward.py` | CLI | Score stage2 rewrites (y_1) and keep rows with `reward >= threshold`. Writes a new parquet. Called by `run_kl_training.sh` when `FORWARD_FILTER_STAGE2=true`. |
| `backfill_stage2_reward_from_stage1.py` | CLI | Join the stage1 `reward` (true pass@1) into stage2 parquet's `extra_info.reward` by `extra_info.index`. Required by the T4 difficulty-bucket diagnostic on stage2 data. Called by `run_kl_training.sh` when `LOG_DIFFICULTY_BUCKETS=true` and `Y_MODE=y_r`. |

CLI invocation example:

```bash
python -m recipe.opd.dataset.score_stage1_reward --parquet path/to/stage1.parquet
```

The scoring CLIs depend on `recipe.math_evaluation.compute_score` to compute
per-row reward (which routes into `verl/utils/reward_score/`).
