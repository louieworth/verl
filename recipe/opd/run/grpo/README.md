# Portable Qwen3 Base GRPO Runs

The canonical GRPO jobs are exposed through the recipe-local launcher
matrix:

```bash
bash recipe/opd/scripts_math/Baselines/1B/grpo.sh
bash recipe/opd/scripts_math/Baselines/4B/grpo.sh
bash recipe/opd/scripts_math/Baselines/8B/grpo.sh
bash recipe/opd/script_code/Baselines/1B/grpo.sh
```

All three sizes resolve to Qwen3 Base checkpoints. The shared runner is
`_run_qwen3_grpo_8h100.sh`; older model-named wrappers remain for compatibility
but the canonical recipe launchers are authoritative.

Prepare or verify assets with:

```bash
bash recipe/opd/run/grpo/prepare/ready_to_train.sh train-data
bash recipe/opd/run/grpo/prepare/ready_to_train.sh eval-data
bash recipe/opd/run/grpo/prepare/ready_to_train.sh models
bash recipe/opd/run/grpo/prepare/ready_to_train.sh verify
```

For a fresh Docker/EC2 code worker, Git already contains the <=40 MB TACO bundle
chunks. Run `bash recipe/opd/script_code/prepare_code.sh all`; it atomically
restores the byte-identical canonical train parquets, then prepares the
machine-local code evaluator/runtime. Real code launchers also restore the
bundle automatically when needed.

The canonical asset layout is:

```text
data/train_dataset/openthoughts_math_30k_opsd/{train_grpo,train_sft}.parquet
data/train_dataset/taco/canonical/{train_grpo,train_sft,train_distill}.parquet
data/eval_dataset/math/{aime25,aime26,hmmt26,amobench}/
data/eval_dataset/code/{evalplus,livecodebench,LiveCodeBench}/
model/base/{Qwen3-1.7B-Base,Qwen3-4B-Base,Qwen3-8B-Base}/
model/teacher/Qwen3-14B/
```

Math no longer uses DeepScaleR. Its source is the pinned
`siyanzhao/Openthoughts_math_30k_opsd` revision recorded in the preparation
manifest. Both train sets use plain Base-completion prompts, enforce
prompt/response caps 2048/16384, and guarantee at least 95% prompt coverage.
For code GRPO, preparation deterministically retains the 15 longest-input TACO
tests (or all tests when fewer are available). Reward is binary: every selected
test must pass for reward 1; any compile, runtime, timeout, or wrong-answer
failure yields 0. Test pass fraction is not used as policy reward. The canonical
executor timeout is explicitly pinned to 10 seconds in each Code GRPO launcher.

The matched profile uses global prompt batch 512, rollout group size 8, PPO
mini-batch 32, physical per-device micro-batch 1, dynamic batching disabled,
LoRA rank/alpha 64/128, and learning rate 1e-6. One epoch has finite,
precomputed total steps. Training pauses at fractions 0.25, 0.5, 0.75, and
1.0, evaluates the exported checkpoint, then resumes the same optimizer
schedule and W&B run.

The canonical launcher sets `TOTAL_EPOCHS=1` and derives total outer GRPO
steps from `parquet_rows / 512`: Math is 58 steps and Code is 37. Therefore a
normal run consumes every row in the padded canonical parquet exactly once.
`MAX_TRAIN_DURATION_SECONDS=31536000` is only a one-year emergency ceiling,
not the normal stopping rule; if a smaller override interrupts a segment before
its requested milestone, the runner refuses to mark that milestone complete.

Math evaluation runs AIME25, AIME26, HMMT February 2026, and AMO-Bench with
16 samples, Avg@16/Pass@16, prompt/response 2048/16384, temperature 1.0, and top-p 0.7.
Code evaluation keeps HumanEval+, MBPP+, and LiveCodeBench v6 at 16 samples,
temperature 0.6, top-p 0.95, and response length 16384. Every evaluator forces
plain Base prompts.

Set `GRPO_DRY_RUN=true` to validate prepared assets and print the command, or
`GRPO_DRY_RUN=print` to inspect it before assets exist. W&B uses project `trd`;
the canonical shared launcher contains an intentionally empty
`WANDB_API_KEY` export that must be filled directly before an online run.

LiveCodeBench executes generated programs and carries several gigabytes of
private tests. Run code evaluation inside an isolated machine or container.
