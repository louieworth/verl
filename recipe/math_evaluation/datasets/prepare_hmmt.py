#!/usr/bin/env python3
"""Prepare HMMT February datasets for verl math evaluation."""

from __future__ import annotations

import argparse
import os
import re
import sys

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)


INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
HMMT_CONFIGS = {
    "hmmt26": {
        "hf_name": "MathArena/hmmt_feb_2026",
        "revision": "02fba4f74d8e68e73e66a02d540fd979c05c274c",
        "split": "train",
        "expected_rows": 33,
    },
    "hmmt25": {"hf_name": "PraMamba/HMMT-202502", "split": "train"},
    "hmmt24": {"hf_name": "MathArena/hmmt_feb_2024", "split": "train"},
    "hmmt23": {"hf_name": "MathArena/hmmt_feb_2023", "split": "train"},
}


def normalize_answer(answer_raw) -> str:
    answer = str(answer_raw).strip()
    match = re.fullmatch(r"\\boxed\{(.*)\}", answer, flags=re.DOTALL)
    return match.group(1).strip() if match else answer


def make_map_fn(data_source: str):
    def process_fn(example, idx):
        problem = str(example.get("problem", "")).strip()
        answer_raw = example.get("answer")
        if not problem or answer_raw is None or not str(answer_raw).strip():
            raise ValueError(f"{data_source} row {idx} is missing problem or answer")
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": f"{problem}\n{INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": normalize_answer(answer_raw)},
            "extra_info": {
                "index": idx,
                "problem_idx": str(example.get("problem_idx", idx)),
                "problem": problem,
                "answer": str(answer_raw),
            },
        }

    return process_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HMMT datasets for verl evaluation")
    parser.add_argument(
        "--datasets",
        default="hmmt26",
        help="Comma-separated names. Canonical default: hmmt26; also supports hmmt23,hmmt24,hmmt25.",
    )
    parser.add_argument("--raw_dataset_root", default=None, help="Optional root containing raw HF repositories")
    parser.add_argument("--local_save_dir", default="data/eval_dataset/math")
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Deprecated output-root alias retained for existing scripts; prefer --local_save_dir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = os.path.expanduser(args.local_dataset_path or args.local_save_dir)
    selected = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(HMMT_CONFIGS))
    if unknown:
        raise ValueError(f"Unsupported HMMT datasets: {unknown}")

    for dataset_name in selected:
        config = HMMT_CONFIGS[dataset_name]
        output_dir = os.path.join(output_root, dataset_name)
        output_path = os.path.join(output_dir, f"{dataset_name}_test.parquet")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Dataset {dataset_name} already exists at {output_path}, skipping...")
            continue

        dataset_id = config["hf_name"]
        if args.raw_dataset_root:
            dataset_id = os.path.join(os.path.expanduser(args.raw_dataset_root), dataset_id)
        print(f"Loading {dataset_id} ({config['split']})...")
        load_kwargs = {"split": config["split"]}
        if config.get("revision") and not args.raw_dataset_root:
            load_kwargs["revision"] = config["revision"]
        dataset = load_dataset(dataset_id, **load_kwargs)
        if config.get("expected_rows") is not None and len(dataset) != config["expected_rows"]:
            raise ValueError(
                f"Expected {config['expected_rows']} {dataset_name} problems, found {len(dataset)}; "
                "audit the upstream dataset before evaluating."
            )
        dataset = dataset.map(make_map_fn(dataset_name), with_indices=True, remove_columns=dataset.column_names)
        os.makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(output_path)
        print(f"Saved {len(dataset)} {dataset_name} problems to {output_path}")


if __name__ == "__main__":
    main()
