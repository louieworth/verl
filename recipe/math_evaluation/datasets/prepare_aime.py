#!/usr/bin/env python3
"""Prepare AIME evaluation sets in the parquet format used by verl.

The canonical default contains the 30-problem 2025 and 2026 sets. AIME24
remains selectable for backward-compatible experiments.
"""

from __future__ import annotations

import argparse
import os
import sys

import datasets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)

from verl.utils.reward_score.math_reward import remove_boxed


INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
AIME_CONFIGS = {
    "aime26": {
        "hf_name": "math-ai/aime26",
        "revision": "79037aebdb6580008fb960d17cb21fd3099083e3",
        "split": "test",
        "expected_rows": 30,
    },
    "aime25": {
        "hf_name": "math-ai/aime25",
        "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
        "split": "test",
        "expected_rows": 30,
    },
    "aime24": {"hf_name": "math-ai/aime24", "split": "test"},
}


def normalize_answer(answer_raw) -> str:
    answer = str(answer_raw).strip()
    if "\\boxed" in answer:
        try:
            return str(remove_boxed(answer)).strip()
        except Exception:
            pass
    return answer


def make_map_fn(data_source: str):
    def process_fn(example, idx):
        problem = str(example["problem"]).strip()
        answer_raw = example.get("solution", example.get("answer"))
        if answer_raw is None:
            raise ValueError(f"{data_source} row {idx} has no solution/answer field")
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": f"{problem}\n{INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": normalize_answer(answer_raw)},
            "extra_info": {
                "index": idx,
                "id": str(example.get("id", idx)),
                "answer": str(answer_raw),
                "question": problem,
            },
        }

    return process_fn


def resolve_dataset_id(hf_name: str, local_dataset_path: str | None) -> str:
    return hf_name if local_dataset_path is None else os.path.join(os.path.expanduser(local_dataset_path), hf_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AIME datasets for verl evaluation")
    parser.add_argument(
        "--datasets",
        default="aime25,aime26",
        help="Comma-separated names. Canonical default: aime25,aime26; also supports aime24.",
    )
    parser.add_argument("--local_dataset_path", default=None, help="Optional root containing raw HF repositories")
    parser.add_argument("--local_save_dir", default="data/eval_dataset/math")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(AIME_CONFIGS))
    if unknown:
        raise ValueError(f"Unsupported AIME datasets: {unknown}")

    for dataset_name in selected:
        config = AIME_CONFIGS[dataset_name]
        output_dir = os.path.join(os.path.expanduser(args.local_save_dir), dataset_name)
        output_path = os.path.join(output_dir, f"{dataset_name}_test.parquet")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Dataset {dataset_name} already exists at {output_path}, skipping...")
            continue

        dataset_id = resolve_dataset_id(config["hf_name"], args.local_dataset_path)
        print(f"Loading {dataset_id} ({config['split']})...")
        load_kwargs = {"split": config["split"]}
        if config.get("revision") and args.local_dataset_path is None:
            load_kwargs["revision"] = config["revision"]
        dataset = datasets.load_dataset(dataset_id, **load_kwargs)
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
