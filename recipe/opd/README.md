# OPD — On-Policy Distillation

Token-level KL distillation for math reasoning. The teacher rewrites or
guides the student's response, and the student learns the teacher's
per-token distribution under a chosen KL objective.

Two distillation modes share the same trainer and loss math:

| Mode | Teacher | y_r prompt sees | When to use |
| --- | --- | --- | --- |
| **OPSD** (default) | The student's frozen step-0 Base checkpoint | problem + initial response + **expert solution y\*** | Privileged self-distillation anchored by the reference solution. |
| **OPD** | External `Qwen/Qwen3-14B` (non-Base) | problem + initial response only (no y\*) | Teacher rewrites the student's attempt with no expert hint. |

Mode is selected by `DISTILL_MODE=opsd|opd` env var (or `--distill_mode`
CLI flag).

The authoritative experiment entrypoints are the matrices in `recipe/opd/scripts_math/` and
`recipe/opd/script_code/`; see `recipe/opd/scripts_math/README_OPD_OPSD.md`. They pin Base checkpoints,
OpenThoughts/TACO data, 2048/16384 prompt-response budgets, micro-batch 1,
top-K 32, four fractional evaluation milestones, and one resumable W&B run.
The canonical OPD matrix has 1.7B/4B/8B Base students with external
`Qwen/Qwen3-14B`; OPSD has 1.7B/4B/8B Base models and self-distills from each
model's frozen step-0 checkpoint.
Older direct wrappers described below remain available for reproducing legacy
experiments and may have different defaults.

The canonical OpenThoughts Math training parquets are stored directly in Git,
so Math launchers work on a fresh clone without a preparation step. Canonical
TACO is stored in Git as <=40 MB byte chunks under
`recipe/opd/train_data_bundle/taco_canonical/`. On a fresh worker, run
`bash recipe/opd/script_code/prepare_code.sh all`; it hash-verifies and
atomically restores all three canonical TACO parquets, then prepares the
machine-local EvalPlus/LiveCodeBench runtime and datasets. Every real code
training launcher also performs this restoration check automatically. The
single fresh-worker bootstrap command is:

```bash
bash recipe/opd/script_code/prepare_code.sh all
```

It restores and verifies GRPO/SFT/distillation train data, installs the pinned
EvalPlus 0.3.1 and vLLM 0.12.0 Code Eval runtime, downloads
HumanEval+/MBPP+/LiveCodeBench v6, writes the local
runtime environment, and runs a CPU-only correct/incorrect code execution test.
Missing Python data/evaluation packages are installed by default; the base
Docker image still supplies Python, pip, CUDA/driver compatibility and the
normal verl training environment.

Every real launcher in `scripts_math/` and `script_code/` resolves W&B auth in
the shared launcher. Export `WANDB_API_KEY` in the calling environment before
an online run; credentials are never stored in Git. Every real
`script_code/` leaf runs `prepare_code.sh all` before dispatching; dry-runs
remain side-effect free.
The code GRPO artifact follows a DeepCoder-style contract: at most 15
longest-input tests per TACO problem and binary all-selected-tests-pass reward.

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
│   ├── opsd/                       — legacy direct wrappers for OPSD (with y*)
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
TEACHER_MODEL_PATH=Qwen/Qwen3-14B bash recipe/opd/run/opd/reverse_topk_y_o.sh
```

## Exposed parameters (env overrides)

| Env var | Applies to | Default | Notes |
| --- | --- | --- | --- |
| `DISTILL_MODE` | all | `opsd` | `opsd` \| `opd`. OPD requires `TEACHER_MODEL_PATH`. |
| `TEACHER_MODEL_PATH` | all | mode-dependent | OPD uses `Qwen/Qwen3-14B`; OPSD must equal the frozen step-0 `MODEL_PATH`. |
| `Y_MODE` | all | per-script | `y_o` (stage1 student rollout) or `y_r` (stage2 teacher rewrite). |
| `BASE_PROMPT_LENGTH` | all | `1024` | Student problem prompt budget; used for y_o rollout and student-side KL prompt. |
| `MAX_RESPONSE_LENGTH` | all | `16384` | Fixed response budget for y_o, y_r, and KL target responses. |
| `EXPERT_SOLUTION_PROMPT_LENGTH` | OPSD | `3072` | Reserved budget for y\* in OPSD teacher prompts; with the current DeepScaleR-Cleaned data this keeps the full OPSD expert prompt under the derived budget. |
| `MAX_PROMPT_LENGTH` | all | auto | Training-time teacher prompt budget, derived from OPD/OPSD and vanilla/refine. Override only for debugging. |
| `TEMPERATURE` | all | `1.0` | Distillation softmax temperature. |
| `LEARNING_RATE` | all | per-script (2e-6 or 5e-6) | |
| `TOTAL_EPOCHS` | all | `1` | Outer pipeline epochs (multi-epoch resumes from previous merged ckpt). |
| `MULTI_STEP` | all | `0` | `0` uses the default 512 prompts per policy optimizer step and derives the step count. `>0` means exactly that many policy optimizer steps; the full dataset is balanced across those steps and gradient accumulation is derived so each partition produces one step. No rows are dropped. Adds an `msN` tag to run/model/result names. |
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
example `MULTI_STEP=39`. A positive value is the exact number of policy
optimizer steps. The script balances every loaded row across those steps and
derives gradient accumulation so each partition produces one optimizer update;
with 40,000 rows and `MULTI_STEP=39`, 25 partitions use 1,026 rows and 14 use
1,025 rows. Each step runs `y_o -> optional y_r -> KL train`, then chunk
`N+1` loads chunk `N`'s `hf_merged` policy before generating its own `y_o`.
Temporary per-chunk parquet files live under
`gen_results/<model>/epoch1/ms39/batchXXXXX/` and are deleted after training by
default. Sparse model retention and evaluation use
`EVAL_FRACTIONS=0.25,0.5,0.75,1.0`; for `MULTI_STEP=39` these are steps
10, 20, 30, and 39. With the default
`PIPELINE_AUTO_RESUME=true`, rerunning the same command skips completed steps and
continues from the latest available policy checkpoint.

Checkpoints default to
`/scratch/l/luli/jiangli/ckpt/<OPD|OPSD>/<student_model>/<run_name>/`, where
OPD run names start with `teacher<teacher_model>_` and all run names include
`y_o`/`y_r`, KL type/method, prompt mode, `msN`, and date. Evaluation writes to
`results/<OPD|OPSD>/<student_model>.json`; `msN` and the final step suffix are
part of the JSON key, not the filename.
