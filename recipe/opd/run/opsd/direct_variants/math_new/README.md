# OPSD math direct variants

These launchers run the same five direct variants for
Qwen3-4B-Instruct-2507 and Qwen3-8B:

- `forward_kl.sh`: forward KL on \(y_o\)
- `forward_kl_clip.sh`: clipped forward KL on \(y_o\)
- `forward_kl_y_r.sh`: forward KL on \(y_r\)
- `reverse_kl.sh`: reverse KL on \(y_o\)
- `reverse_kl_topk.sh`: reverse KL on \(y_o\), teacher top-k 32

They are self-contained with respect to repository data. By default they read:

- `data/train_dataset/deepscaler/train_grpo.parquet`
- `data/eval_dataset/math/<dataset>/<dataset>_test.parquet`

The model identifiers remain Hugging Face identifiers, so weights can be
downloaded on a new machine. Persistent caches and outputs are rooted under
`gen_results/`, `outputs/`, `model/`, and `results/` in this repo.

The current hyperparameters are intentionally preserved rather than changed to
match Table 8: learning rate `5e-6`, per-GPU batch size `1`, gradient
accumulation `16`, and temperature `1.0`. The \(y_o\) launchers use max length
`22528`; the \(y_r\) launcher uses `38912`. The clipped default is `0.06`.

## Run

For example:

```bash
bash recipe/opd/run/opsd/direct_variants/math_new/qwen3-4b-instruct/forward_kl.sh
bash recipe/opd/run/opsd/direct_variants/math_new/qwen3-8b/forward_kl.sh
```

Replace `forward_kl.sh` with any of the five filenames above. Environment
variables can still override operational defaults such as `MAX_SAMPLES`,
`RUN_EVAL_AFTER_TRAINING`, and `PIPELINE_RESUME_MODE`. Repo path overrides use
the `MATH_NEW_*` namespace (for example, `MATH_NEW_MODEL_SAVE_DIR`), so stale
machine-wide variables such as `HF_HOME` or `TRAIN_DATA_PATH` cannot silently
redirect a run outside this repository.

The prepared training parquet already contains the OPSD prompt schema and
expert solution metadata. The shared launcher therefore sets
`PRECOMPUTED_STAGE1_PROMPTS_PATH` and uses `MULTI_STEP=1`: this is one complete
generation/training update, and avoids treating the parquet as a raw
`Dataset.save_to_disk` directory.

## CPU-only preflight

Run every wrapper without loading a model or using a GPU:

```bash
bash recipe/opd/run/opsd/direct_variants/math_new/smoke_test.sh
```

For one launcher:

```bash
bash recipe/opd/run/opsd/direct_variants/math_new/qwen3-8b/forward_kl.sh --dry-run
```
