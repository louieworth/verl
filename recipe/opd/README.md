# OPD — On-Policy Distillation

Token-level KL distillation for math reasoning. The teacher rewrites or
guides the student's response, and the student learns the teacher's
per-token distribution under a chosen KL objective.

Two distillation modes share the same trainer and loss math:

| Mode | Teacher | y_r prompt sees | When to use |
| --- | --- | --- | --- |
| **OPSD** (default) | = student | problem + initial response + **expert solution y\*** | On-policy *self*-distillation — same model used as both student and teacher reference, leaning on the ground-truth solution as anchor. |
| **OPD** | ≠ student (must set `TEACHER_MODEL_PATH`) | problem + initial response only (no y\*) | Distilling from a larger / stronger teacher model. Teacher rewrites the student's attempt with no expert hint. |

Mode is selected by `DISTILL_MODE=opsd|opd` env var (or `--distill_mode`
CLI flag).

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
│   ├── opsd/                       — 4 canonical wrappers for OPSD (teacher = student, with y*)
│   ├── opd/                        — 4 canonical wrappers for OPD  (teacher ≠ student, no y*)
│   └── ablation/                   — MC / 4B / reward-filtered / OPSD-JSD / queue runners
├── anlysis/                        — figure scripts
└── improvement/                    — writeup notes
```

Evaluation (post-train benchmark, scoring library, dataset prep) lives in
`recipe/math_evaluation/` — see its README. `recipe/opd/` only owns the
training loop and the training-data prep that feeds it.

## The 4 canonical recipes (× 2 modes)

| Script                  | KL type | Data | Special knob       | LR    | Temp |
| ----------------------- | ------- | ---- | ------------------ | ----- | ---- |
| `forward_clip_y_o.sh`   | forward | y_o  | `KL_TOKEN_CLIP=0.06` | 5e-6 | 1.0  |
| `forward_y_r.sh`        | forward | y_r  | —                  | 2e-6 | 1.0  |
| `reverse_topk_y_o.sh`   | reverse | y_o  | `TOP_K=32`         | 2e-6 | 1.0  |
| `reverse_y_o.sh`        | reverse | y_o  | —                  | 2e-6 | 1.0  |

Each one exists in both `run/opsd/` and `run/opd/`. The OPD variants are
thin wrappers that set `DISTILL_MODE=opd`, require `TEACHER_MODEL_PATH`, and
exec the OPSD wrapper for shared config.

```bash
# OPSD — teacher = student
bash recipe/opd/run/opsd/reverse_topk_y_o.sh

# OPD — teacher ≠ student
TEACHER_MODEL_PATH=Qwen/Qwen3-32B bash recipe/opd/run/opd/reverse_topk_y_o.sh
```

## Exposed parameters (env overrides)

| Env var | Applies to | Default | Notes |
| --- | --- | --- | --- |
| `DISTILL_MODE` | all | `opsd` | `opsd` \| `opd`. OPD requires `TEACHER_MODEL_PATH`. |
| `TEACHER_MODEL_PATH` | OPD only | empty (= student) | Path/HF id of teacher model. Must be set for OPD. |
| `Y_MODE` | all | per-script | `y_o` (stage1 student rollout) or `y_r` (stage2 teacher rewrite). |
| `BASE_PROMPT_LENGTH` | all | `1024` | Student problem prompt budget; used for y_o rollout and student-side KL prompt. |
| `MAX_RESPONSE_LENGTH` | all | `16384` | Fixed response budget for y_o, y_r, and KL target responses. |
| `EXPERT_SOLUTION_PROMPT_LENGTH` | OPSD | `3072` | Reserved budget for y\* in OPSD teacher prompts; with the current DeepScaleR-Cleaned data this keeps the full OPSD expert prompt under the derived budget. |
| `MAX_PROMPT_LENGTH` | all | auto | Training-time teacher prompt budget, derived from OPD/OPSD and vanilla/refine. Override only for debugging. |
| `TEMPERATURE` | all | `1.0` | Distillation softmax temperature. |
| `LEARNING_RATE` | all | per-script (2e-6 or 5e-6) | |
| `TOTAL_EPOCHS` | all | `1` | Outer pipeline epochs (multi-epoch resumes from previous merged ckpt). |
| `MULTI_STEP` | all | `0` | `0` keeps the historical one-step path over all samples. `>0` runs exactly this many offline policy updates; chunk size is computed as `floor(num_train_rows / MULTI_STEP)` and tail rows are dropped. Adds an `msN` tag to run/model/result names. |
| `PIPELINE_AUTO_RESUME` | multi-step | `true` | When rerunning the same command, skip steps with a done marker or final `hf_merged/config.json`. Set `false` to fail fast if existing completed output is found. |
| `KL_TOKEN_CLIP` | **forward only** | `0.06` in `forward_clip_y_o.sh`, `0` elsewhere | Per-token KL clamp. |
| `TOP_K` | **reverse only** | `32` in `reverse_topk_y_o.sh`, `0` elsewhere | Teacher top-K local support. |

Everything else (batch size, LoRA rank, FSDP settings, diagnostic metrics)
has a sensible default in `run/run_kl_training.sh` and is intentionally not
surfaced per-recipe — those are infrastructure, not scientific knobs.

`Y_MODE` legacy aliases `y_raw` / `y_cor` are still accepted and emit a
deprecation warning; the canonical names are `y_o` / `y_r`.

Default length budgets use `BASE_PROMPT_LENGTH=1024`, `MAX_RESPONSE_LENGTH=16384`,
and `EXPERT_SOLUTION_PROMPT_LENGTH=3072`. With those defaults, OPD y_o / OPD
y_r vanilla train at `MAX_LENGTH=17408`, OPD y_r refine trains at `33792`, OPSD
y_o / OPSD y_r vanilla train at `20480`, and OPSD y_r refine trains at `36864`.
Stage2 y_r generation always uses refine prompts, so its rollout max model length
is `33792` for OPD and `36864` for OPSD. On the current DeepScaleR-Cleaned data,
the Qwen3-tokenized OPSD `problem + expert + template` prompt has max 3675 tokens,
so the default 4096-token `BASE_PROMPT_LENGTH + EXPERT_SOLUTION_PROMPT_LENGTH`
budget avoids expert-prompt clipping.

## Adding an ablation

Copy a canonical script into `run/ablation/`, rename it, and edit the
pinned values. Anything more invasive — new KL types, new data sources —
goes in `kl_trainer.py` / `dataset/data_utils.py` and is exposed through
`run_kl_training.sh`.

## Data pipeline

Generation is auto-invoked from `run_kl_training.sh` (stage1 student rollout
→ optional stage2 teacher rewrite → score → backfill rewards → train). See
`generation/README.md` for the file-by-file breakdown.

For offline multi-step on-policy optimization, set only `MULTI_STEP`, for
example `MULTI_STEP=39`. The script computes the chunk size from the loaded train
rows; with 40,000 rows and `MULTI_STEP=39`, each update uses 1,025 rows and drops
the final 25 rows. Each step runs `y_o -> optional y_r -> KL train`, then chunk
`N+1` loads chunk `N`'s `hf_merged` policy before generating its own `y_o`.
Temporary per-chunk parquet files live under
`gen_results/<model>/epoch1/ms39/batchXXXXX/` and are deleted after training by
default. The reusable prompt cache is kept at
`gen_results/<model>/deepscaleR_stage1_prompts.parquet`. Sparse model retention
defaults to step 0 plus every `ceil(MULTI_STEP / 5)` updates and the final step,
so `MULTI_STEP=39` keeps 0, 8, 16, 24, 32, and 39. With the default
`PIPELINE_AUTO_RESUME=true`, rerunning the same command skips completed steps and
continues from the latest available policy checkpoint.

Checkpoints default to
`/scratch/l/luli/jiangli/ckpt/<OPD|OPSD>/<student_model>/<run_name>/`, where
OPD run names start with `teacher<teacher_model>_` and all run names include
`y_o`/`y_r`, KL type/method, prompt mode, `msN`, and date. Evaluation writes to
`results/<OPD|OPSD>/<student_model>.json`; `msN` and the final step suffix are
part of the JSON key, not the filename.
