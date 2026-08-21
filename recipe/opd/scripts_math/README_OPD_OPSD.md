# OPD / OPSD experiment launchers

The experiment matrices are split by task:

- `recipe/opd/scripts_math/`: math
- `recipe/opd/script_code/`: code

Each leaf launcher explicitly declares its task, family, variant, model, and
variant-defining algorithm parameters before delegating to
`recipe/opd/run/run_experiment.sh`; the shared launcher validates and forwards
those declarations and never infers the algorithm from a filename. `1B` means
`Qwen3-1.7B-Base`; the other model directories use `Qwen3-4B-Base` and
`Qwen3-8B-Base`. OPD has 1.7B/4B/8B Base students and always uses the external,
non-Base `Qwen/Qwen3-14B` teacher. OPSD has 1.7B/4B/8B Base models and uses
each model's frozen step-0 checkpoint as its own teacher. OPSD teacher prompts
also receive the privileged reference solution, while OPD teacher prompts
receive only the problem (and, for TRD, the student's initial response).

Those Hugging Face Base IDs are the fixed defaults. An offline/local snapshot
can be selected explicitly while keeping the leaf-declared size and Base-model
invariants:

```bash
MODEL_PATH=/models/Qwen3-1.7B-Base \
TEACHER_MODEL_PATH=/models/Qwen3-14B \
  bash recipe/opd/scripts_math/OPD/1B/top_k.sh --dry-run
```

Local overrides must be existing Hugging Face directories with model weights,
tokenizer assets, and a `config.json` matching the launcher's Qwen3
architecture size; paths or config provenance explicitly marked `Instruct`
are rejected. Student snapshots must come from the documented Base repository;
an OPD teacher snapshot must come from `Qwen/Qwen3-14B`. OPSD does not accept a
separate teacher path: its teacher path must equal its Base model path.
Non-default remote repository IDs are rejected.

The canonical OpenThoughts Math parquets and their manifest are checked into
Git, so a fresh clone can run a Math launcher immediately. Regenerate them
only after intentionally changing the pinned source or prompt contract:

```bash
bash recipe/opd/scripts_math/prepare_data.sh
bash recipe/opd/script_code/prepare_data.sh
```

Canonical TACO is prepared once, then transported through Git as hash-verified chunks of at most
40 MB under `data/train_dataset/taco/bundle/`. Every real code
training launcher atomically reconstructs the exact original files under
`data/train_dataset/taco/canonical/`; it does not download, decode, or reprocess
TACO. Code preparation writes three ZSTD parquets: compact
`train_distill.parquet` for OPD/OPSD, `train_sft.parquet`, and the larger
`train_grpo.parquet` whose TACO execution tests are needed only by GRPO. Code
GRPO uses a DeepCoder-style sparse reward contract: preparation selects at
most 15 tests per problem by descending input-string length (stable source
index tie-break), and a rollout receives reward 1 only when every selected
test passes; compile/runtime/timeout/wrong-answer outcomes receive 0. Problems
with fewer than 15 tests retain all their tests so the canonical 18,944-row,
37-step dataset cardinality is unchanged. The canonical Code GRPO launchers
expose the existing verl executor timeout as
`CODE_GRPO_EXEC_TIMEOUT_SECONDS=10`. The Hugging Face preparation cache lives
under `data/download_cache/` and must not
be copied with the training bundle.

After an intentional dataset rebuild, refresh the committed chunks once:

```bash
bash recipe/opd/script_code/prepare_code.sh bundle
```

On each fresh Docker/EC2 worker, restore the Git-bundled train data and prepare
the machine-local code evaluation runtime and datasets:

```bash
bash recipe/opd/script_code/prepare_code.sh all
```

The shared launcher invokes this command automatically before every real
`script_code/` experiment. It is intentionally skipped for `DRY_RUN=1`.

`all` restores and verifies the checksums of the canonical TACO parquets; it
does not download or rebuild TACO. Use `prepare_code.sh train` only on the artifact
preparation machine, and `prepare_code.sh eval` when only evaluation assets
need refreshing. The command finishes only after importing EvalPlus, vLLM and
LiveCodeBench and passing a CPU-only local code execution/reward smoke test.

The math source is pinned to
`siyanzhao/Openthoughts_math_30k_opsd@1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62`;
DeepScaleR is not used by the canonical matrix. Both preparers enforce a
2048-token prompt cap with at least 95% coverage, a 16384-token response cap,
and write a `manifest.json` with measured p95/max lengths. Outputs are padded
deterministically to the global batch multiple so fraction `1.0` consumes the
whole retained dataset. Preparation also removes exact formatting-normalized
overlaps with the canonical math evaluation assets; the current pinned inputs
drop one OpenThoughts row that duplicates AIME26 problem 0. TACO SFT targets
must parse as Python before they are retained.
The public TACO fallback is likewise pinned to
`BAAI/TACO@d593ed0a2becbbc952230bb89be09189bf1056dc`.

All canonical leaf launchers explicitly expose
`MULTI_STEP="${MULTI_STEP:-0}"`. For OPD/OPSD distillation, the default `ms=0`
makes the bottom-level runner use 512 prompts per on-policy optimizer step and
derive Math `ms=58`
(`29696 / 512`) and Code `ms=37` (`18944 / 512`). With an explicit positive
value such as `MULTI_STEP=40`, that value instead means exactly 40 policy
optimizer steps: the entire dataset is balanced into 40 partitions and gradient
accumulation is derived so every partition produces one optimizer step. No rows
are dropped. Positive `MULTI_STEP` values apply to OPD/OPSD distillation;
Base/SFT/GRPO reject them because those baselines use their own step semantics.

The canonical paper profile is prompt=2048, train response=16384, physical
per-device micro-batch=1 with dynamic batching disabled, global prompt
batch=512, LoRA rank/alpha=64/128, and learning rate 1e-6. Evaluation
checkpoints are selected at `ceil(total_steps * f)` for
`EVAL_FRACTIONS=0.25,0.5,0.75,1.0` and logged to the same W&B run. Math maps to
updates 15/29/44/58 and Code maps to updates 10/19/28/37.

Math evaluation is AIME25, AIME26, HMMT February 2026, and the 39-problem
parser-verifiable AMO-Bench subset at Avg@16 and Pass@16 with
prompt/response=2048/16384, temperature=1.0, and top-p=0.7.
Code keeps HumanEval+, MBPP+, and LiveCodeBench v6 with its established
temperature=0.6/top-p=0.95 profile and a 16384-token response cap. All
evaluation prompts are plain Base-model completions rather than chat-template
prompts.

W&B uses four task/family projects: `opd-math`, `opsd-math`, `opd-code`, and
`opsd-code`. The shared Base/SFT/GRPO baselines live in `opd-math` or
`opd-code` and carry `family=baseline`; they are not duplicated into a second
project. Runs are named
`family-task-variant-model-msN-timestamp`, grouped by `variant-model-size`,
and tagged with the task, family, variant, student/teacher, KL/rollout knobs,
batching, sequence lengths, eval profile, LoRA, learning rate, and seed.
Every real math/code leaf reads `WANDB_API_KEY` from the calling environment;
the shared launcher validates it without storing credentials in Git. Each
experiment persists `wandb_run.json` under its
output root so quarter training segments and external evaluations share one
run ID and one custom `global_step` axis.

The stable training dashboard namespace contains `train/loss`,
`train/learning_rate`, `train/grad_norm`, `train/global_step`, epoch, response
tokens, step time, and MFU when the trainer provides them. GRPO additionally
exports reward before/after KL, policy KL, and entropy. Distillation keeps its
full KL diagnostics (student/teacher perplexity, KL p50/p95/p99/max, clip
fraction, and optional gradient/correction/difficulty signals). Native SFT and
GRPO metric namespaces remain available alongside these common aliases.

Evaluation logging accepts partial suites. Every successfully aggregated
dataset is written under its own `avg@16` and `pass@16` keys; an unavailable
benchmark no longer blocks uploading the other datasets. Macro metrics are
computed over the datasets present in that evaluation result. Namespace,
numeric-range, step, and milestone consistency checks remain enabled.

Variant meanings and their actual runtime variables are visible in every leaf
script. `vanilla` is full-vocabulary reverse KL, `top_k` is reverse KL on
teacher support K=32, and `clip` is full-vocabulary forward KL with token clip
0.05. `skd` uses a five-token student draft, teacher accept top-k 25, and
teacher correction temperature 0.2 before unclipped forward-KL training.
`trd` is the teacher-rewrite path: it trains
on a second-stage rewritten student response (`y_r`). It is not TOP-D (Trust
Region Policy Distillation) from arXiv:2607.04751. The dispatcher rejects an
inconsistent explicit contract instead of silently replacing it.

Resolve one experiment without touching data, models, GPUs, or W&B:

```bash
DRY_RUN=1 bash recipe/opd/scripts_math/OPD/1B/top_k.sh
bash recipe/opd/script_code/OPSD/4B/trd.sh --dry-run
```

Run or inspect a filtered matrix:

```bash
DRY_RUN=1 MATRIX_FAMILIES=opd MATRIX_MODELS=1B,4B \
  bash recipe/opd/scripts_math/run_matrix.sh

MATRIX_FAMILIES=baseline MATRIX_MODELS=8B MATRIX_VARIANTS=base \
  bash recipe/opd/script_code/run_matrix.sh
```

`MATRIX_FAMILIES`, `MATRIX_MODELS`, and `MATRIX_VARIANTS` are comma-separated
filters. Matrix execution stops on the first error by default; set
`MATRIX_CONTINUE_ON_ERROR=1` to finish the selected matrix and return failure
afterwards. Extra Hydra overrides are supported by the SFT/GRPO launchers;
configure Base evaluation and OPD/OPSD runs through their documented
environment variables. Unsupported positional arguments are rejected instead
of being silently ignored. When passing Hydra overrides to `run_matrix.sh`,
filter the matrix to only `sft` or only `grpo`. To define a new threshold or
algorithm profile, copy/edit the explicit leaf values; do not rely on renaming
the file. No credentials are stored in these launchers.
