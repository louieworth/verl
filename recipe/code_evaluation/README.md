# Code Evaluation Recipe

Evaluates OPD/OPSD code-task checkpoints on HumanEval+, MBPP+, and LiveCodeBench v6.

The code-suite defaults remain `PASS_K=16`, `temperature=0.6`, `top_p=0.95`,
`max_prompt_length=2048`, and `max_response_length=16384`. HumanEval+/MBPP+
and LiveCodeBench v6 all use raw Base-model completion prompts. Results are
written to the historical JSON, a CSV with dataset/macro Avg@16 and Pass@16,
and `gen_results/.../metrics.json` using the stable
`opd_eval_metrics/v1` schema. Its `wandb` object can be logged directly by the
outer milestone runner at the JSON `step` value.
If `WANDB_RUN_ID`, `WANDB_PROJECT`, `WANDB_GLOBAL_STEP`, and
`EVAL_MILESTONE_FRACTION` are set, the wrapper resumes the existing run and logs
those metrics at the explicit step; credential handling stays outside this
recipe.

On a fresh Docker/EC2 worker, prepare the pinned evaluator environment and
machine-local datasets with:

```bash
bash recipe/opd/script_code/prepare_code.sh all
```

The command expects the separately deployed canonical TACO parquets under
`data/train_dataset/taco/canonical`; it never rebuilds TACO in `all` mode.

Dependencies, in the `verl` conda environment:

```bash
conda run -n verl python -m pip install "evalplus[vllm]"
```

The repository-local EvalPlus files, LiveCodeBench runner, and all six release-v6 JSONL files are expected under:

```text
data/eval_dataset/code/
├── evalplus/
├── LiveCodeBench/
└── livecodebench/code_generation_lite/{test.jsonl,...,test6.jsonl}
```

Run a trained model manually:

```bash
cd /data2/verl
PYTHON_BIN="/data2/conda/envs/verl/bin/python" PASS_K=16 DATASETS="humaneval_plus mbpp_plus livecodebench_v6" bash recipe/code_evaluation/benchmark_code_model.sh /path/to/hf_model
```

Default outputs:

```text
results/<base_model_name>_code.json
gen_results/code_eval/<model_name>/
```

When invoked by `recipe/opd/run/run_kl_training.sh` with `TASK=code`, outputs are task-separated instead:

```text
results/<OPD_or_OPSD>/code/<student_model>_code.json
results/<OPD_or_OPSD>/code/<student_model>_code.csv
gen_results/eval/code/<result_model_key>/
```

For long EvalPlus runs, `run_evalplus_vllm.py` can generate disjoint numeric
task-ID ranges without evaluating partial datasets:

```bash
python recipe/code_evaluation/run_evalplus_vllm.py \
  --dataset mbpp --model /path/to/hf_model --root /path/to/shard-output \
  --n_samples 16 --id_range 400 600 --skip_evaluation
```

After all ranges finish, merge them with strict coverage validation. The merge
fails unless every benchmark task has exactly `--n_samples` sanitized and raw
outputs:

```bash
python recipe/code_evaluation/merge_evalplus_shards.py \
  --dataset mbpp --n_samples 16 \
  --input /path/to/shard-a.jsonl --input /path/to/shard-b.jsonl \
  --raw_input /path/to/shard-a.raw.jsonl \
  --raw_input /path/to/shard-b.raw.jsonl \
  --output /path/to/merged.jsonl --raw_output /path/to/merged.raw.jsonl
```
