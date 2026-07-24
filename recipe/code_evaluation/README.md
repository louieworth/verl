# Code Evaluation Recipe

Evaluates OPD/OPSD code-task checkpoints on HumanEval+, MBPP+, and LiveCodeBench v6.

Defaults follow the code-generation setup used for this extension: `PASS_K=4`, `temperature=1.0`, `top_p=1.0`, `max_prompt_length=2048`, and `max_response_length=8192`. The prompt limit covers the full current code evaluation set by tokenizer scan: HumanEval+ max 446, MBPP+ max 229, and LiveCodeBench v6 max 1938 tokens. Results are written to JSON plus CSV summaries with `Avg@4` and `Pass@4`.

Dependencies, in the `verl` conda environment:

```bash
conda run -n verl python -m pip install "evalplus[vllm]"
git clone https://github.com/LiveCodeBench/LiveCodeBench.git /data/verl/external/LiveCodeBench
conda run -n verl python -m pip install -e /data/verl/external/LiveCodeBench
```

TACO training data is expected at:

```text
/data/data/jiangli/huggingface/datasets/TACO
```

Run a trained model manually:

```bash
cd /data/verl
PYTHON_BIN="/data/conda/envs/verl/bin/python" TASK=code PASS_K=4 DATASETS="humaneval_plus mbpp_plus livecodebench_v6" bash recipe/code_evaluation/benchmark_code_model.sh /path/to/hf_merged
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
