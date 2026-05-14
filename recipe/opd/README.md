# KL Training

Token-level KL distillation for math reasoning. The teacher rewrites or guides
the student's response, and the student learns the teacher's per-token
distribution under a chosen KL objective.

## Layout

```
recipe/opd/
├── run_training.py / kl_trainer.py / kl_utils.py / config.py
│       — core trainer (see kl_utils.py for the KL math)
├── _tokenizer_compat.py / sitecustomize.py — tokenizer compat hooks
├── dataset/                        — training Dataset class + reward-scoring CLI utilities
├── generation/                     — vllm-rollout prompt prep (stage1 + stage2)
├── run/
│   ├── run_kl_training.sh          — shared base: infra defaults, multi-epoch loop;
│   │                                 hands off to recipe/math_evaluation/benchmark_kl_model.sh after training
│   ├── forward_clip_y_o.sh         — canonical 1: forward KL + clip on y_o
│   ├── forward_y_r.sh              — canonical 2: forward KL on y_r (rewrites)
│   ├── reverse_topk_y_o.sh         — canonical 3: reverse KL + top-K on y_o
│   ├── reverse_y_o.sh              — canonical 4: reverse KL on y_o
│   └── ablation/                   — every other run script (MC variants, 4B variants, reward-filtered, OPSD/JSD, queue runners)
├── anlysis/                        — figure scripts
└── improvement/                    — writeup notes
```

Evaluation (post-train benchmark, scoring library, dataset prep) lives in
`recipe/math_evaluation/` — see its README. `recipe/opd/` only owns
the training loop and the training-data prep that feeds it.

## The 4 canonical recipes

| Script                  | KL type | Data | Special knob       | LR    | Temp |
| ----------------------- | ------- | ---- | ------------------ | ----- | ---- |
| `forward_clip_y_o.sh`   | forward | y_o  | `KL_TOKEN_CLIP=0.06` | 5e-6 | 1.0  |
| `forward_y_r.sh`        | forward | y_r  | —                  | 2e-6 | 1.0  |
| `reverse_topk_y_o.sh`   | reverse | y_o  | `TOP_K=32`         | 2e-6 | 1.0  |
| `reverse_y_o.sh`        | reverse | y_o  | —                  | 2e-6 | 1.0  |

Run one:
```bash
bash recipe/opd/run/reverse_topk_y_o.sh
```

## Exposed parameters (env overrides)

| Env var | Applies to | Default | Notes |
| --- | --- | --- | --- |
| `Y_MODE` | all | per-script | `y_o` (stage1 student rollout) or `y_r` (stage2 teacher rewrite). |
| `MAX_PROMPT_LENGTH` | all | `2048` if `y_o`, `24576` if `y_r` | y_r prompts embed the expert solution + initial response. |
| `MAX_RESPONSE_LENGTH` | all | `16384` | |
| `TEMPERATURE` | all | `1.0` | Distillation softmax temperature. |
| `LEARNING_RATE` | all | per-script (2e-6 or 5e-6) | |
| `TOTAL_EPOCHS` | all | `1` | Outer pipeline epochs (multi-epoch resumes from previous merged ckpt). |
| `KL_TOKEN_CLIP` | **forward only** | `0.06` in `forward_clip_y_o.sh`, `0` elsewhere | Per-token KL clamp. |
| `TOP_K` | **reverse only** | `32` in `reverse_topk_y_o.sh`, `0` elsewhere | Teacher top-K local support. |

Everything else (batch size, LoRA rank, FSDP settings, diagnostic metrics)
has a sensible default in `run/run_kl_training.sh` and is intentionally not
surfaced per-recipe — those are infrastructure, not scientific knobs.

`Y_MODE` legacy aliases `y_raw` / `y_cor` are still accepted and emit a
deprecation warning; the canonical names are `y_o` / `y_r`.

## Adding an ablation

Copy one of the four canonical scripts into `run/ablation/`, rename it, and
edit the pinned values. Anything more invasive — new KL types, new data
sources — goes in `kl_trainer.py` / `data_utils.py` and is exposed through
`run_kl_training.sh`.

## Data pipeline

Generation is auto-invoked from `run_kl_training.sh` (stage1 student rollout
→ optional stage2 teacher rewrite → score → backfill rewards → train). See
`generation/README.md` for the file-by-file breakdown.
