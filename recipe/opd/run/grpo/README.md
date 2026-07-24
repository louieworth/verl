# Portable Qwen3 GRPO Runs

All asset preparation utilities are grouped under `prepare/`. The GRPO root
contains only training entrypoints, rewards, and evaluation wrappers.

All runtime assets use paths relative to the repository root:

```text
data/
  train_dataset/
    deepscaler/{raw,train_grpo.parquet}
    taco/{raw,train_grpo.parquet}
  eval_dataset/
    math/{aime24,aime25,hmmt25,beyondaime,amobench}
    code/{evalplus,livecodebench,LiveCodeBench}
model/
  base/{Qwen3-4B-Instruct-2507,Qwen3-8B}
  trained/<project>/<experiment>/checkpoints/
```

Prepare everything after installing the Python environment:

```bash
bash recipe/opd/run/grpo/prepare/ready_to_train.sh
```

Individual preparation targets are also available:

```bash
bash recipe/opd/run/grpo/prepare/ready_to_train.sh models
bash recipe/opd/run/grpo/prepare/ready_to_train.sh train-data
bash recipe/opd/run/grpo/prepare/ready_to_train.sh eval-data
bash recipe/opd/run/grpo/prepare/ready_to_train.sh verify
```

Run one of the four 8xH100 jobs:

```bash
bash recipe/opd/run/grpo/qwen3_4b_instruct_math_grpo_8h100.sh
bash recipe/opd/run/grpo/qwen3_4b_instruct_code_grpo_8h100.sh
bash recipe/opd/run/grpo/qwen3_8b_math_grpo_8h100.sh
bash recipe/opd/run/grpo/qwen3_8b_code_grpo_8h100.sh
```

Run all four GRPO jobs sequentially:

```bash
bash recipe/opd/run/run_sequence_8h100.sh grpo
```

The combined rollout plus optimizer loop is wall-clock limited at complete
optimizer-step boundaries. The timer starts immediately before the first
rollout. Defaults are 5.5 hours for 4B math, 2.5 hours for 4B code, 9.5 hours
for 8B math, and 4 hours for 8B code. A run finishes its current step after
reaching the limit, saves one final checkpoint, and then starts evaluation.
Model/data setup, the final checkpoint save, and evaluation are outside the
budget. Override `MAX_TRAIN_DURATION_SECONDS` to use a different limit.

Set `GRPO_DRY_RUN=true` to validate local model/data files and print the
resolved training command without starting Ray or allocating GPUs.

The math evaluator keeps the reference settings: 4096 prompt tokens, 16384
response tokens, temperature 0.6, top-p 0.95, and avg@16/pass@16. The code
evaluator keeps 2048 prompt tokens, 16384 response tokens, temperature 0.6,
top-p 0.95, and avg@16/pass@16.

LiveCodeBench `release_v6` is not actually small: its encoded private tests
occupy roughly 4.2 GB as portable JSONL. TACO is also large. Ensure the target
server has enough space for raw data, GRPO parquet files, both base models, and
FSDP optimizer checkpoints. TACO and LiveCodeBench execute generated code;
run code training/evaluation inside an isolated machine or container.

`mathruler` is optional. When installed, the math reward uses its stronger
answer-equivalence grader; otherwise it falls back to verl's built-in boxed
answer grader.

The prepared math parquet in this workspace comes from the local
DeepScaleR-Cleaned dataset (40,245 rows). If it is absent, the ready script
downloads the public `agentica-org/DeepScaleR-Preview-Dataset`; override
`MATH_TRAIN_REPO` when an exact cleaned mirror is available.

DeepScaleR train parquet and the small math/EvalPlus evaluation datasets stay
in Git. TACO train parquet, LiveCodeBench JSONL, base models, and checkpoints
are ignored. After cloning on a destination server, run the ready script to
download and rebuild the ignored assets.
