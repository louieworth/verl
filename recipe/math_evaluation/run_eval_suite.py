#!/usr/bin/env python3

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.insert(0, REPO_ROOT)

from recipe.opd._tokenizer_compat import apply_qwen2_tokenizer_vllm_compat

from recipe.math_evaluation.eval_utils import (
    DEFAULT_EVAL_DATASETS,
    DEFAULT_N_SAMPLES,
    DEFAULT_PROMPT_LENGTH,
    DEFAULT_RESPONSE_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    normalize_eval_dataset_name,
    resolve_eval_dataset_paths,
    run_evaluation_suite,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one model across multiple datasets with one server load")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_EVAL_DATASETS))
    parser.add_argument("--datasets_dir", type=str, default="data/eval_dataset/math")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--pass_k", "--n_samples", dest="pass_k", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt_length", type=int, default=DEFAULT_PROMPT_LENGTH)
    parser.add_argument("--response_length", type=int, default=DEFAULT_RESPONSE_LENGTH)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument(
        "--force_base_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render only prompt content, without a chat/instruction template (default: true).",
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n_gpus_per_node", type=int, default=8)
    parser.add_argument("--gen_tp", type=int, default=8)
    return parser.parse_args()


def main():
    pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if SCRIPT_DIR not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([SCRIPT_DIR, *pythonpath_entries])

    apply_qwen2_tokenizer_vllm_compat()

    args = parse_args()
    if args.pass_k <= 0:
        raise ValueError("--pass_k/--n_samples must be positive")
    if args.prompt_length <= 0 or args.response_length <= 0:
        raise ValueError("prompt and response lengths must be positive")
    os.environ["EVAL_PROMPT_LENGTH"] = str(args.prompt_length)
    os.environ["EVAL_RESPONSE_LENGTH"] = str(args.response_length)
    os.environ["EVAL_MAX_MODEL_LEN"] = str(args.max_model_len or args.prompt_length + args.response_length)
    dataset_names = [dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()]
    dataset_paths = resolve_eval_dataset_paths(dataset_names, args.datasets_dir)
    unsupported = [name for name in dataset_names if normalize_eval_dataset_name(name) not in dataset_paths]
    if unsupported:
        raise ValueError(f"Unsupported evaluation datasets: {unsupported}")
    missing = {name: path for name, path in dataset_paths.items() if not os.path.isfile(path)}
    if missing:
        formatted = ", ".join(f"{name}={path}" for name, path in missing.items())
        raise FileNotFoundError(
            f"Prepared evaluation datasets are missing: {formatted}. "
            "Run recipe/math_evaluation/datasets/prepare_aime.py, prepare_hmmt.py, "
            "and prepare_additional_eval_datasets.py first."
        )

    results = run_evaluation_suite(
        args.model_path,
        dataset_paths,
        args.output_dir,
        pass_k=args.pass_k,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        tensor_model_parallel_size=args.gen_tp,
        output_json_path=args.results_file,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path or None,
        force_base_prompt=args.force_base_prompt,
    )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
