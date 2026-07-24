#!/usr/bin/env python3
"""Convert DeepScaleR or TACO into verl's GRPO parquet schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import datasets

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from verl.utils.reward_score.math_reward import remove_boxed


R1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>. "
    "Put your final answer within \\boxed{}."
)
CODE_INSTRUCTION = (
    "You will be given a programming problem. Write a correct Python program that solves it. "
    "Return only the code inside a single ```python code block."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["math", "code"])
    parser.add_argument("--input", required=True, help="Local downloaded dataset directory or data file.")
    parser.add_argument("--output", required=True, help="GRPO parquet output path.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_local_dataset(input_path: str, split: str) -> datasets.Dataset:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset input does not exist: {input_path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return datasets.load_dataset("parquet", data_files=str(path), split="train")
        if suffix in {".json", ".jsonl"}:
            return datasets.load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported dataset file: {path}")

    try:
        loaded = datasets.load_from_disk(str(path))
        if isinstance(loaded, datasets.DatasetDict):
            return loaded[split] if split in loaded else loaded["train"]
        return loaded
    except (FileNotFoundError, ValueError):
        pass

    candidates = {
        "parquet": sorted(path.rglob("*.parquet")),
        "json": sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl")),
    }
    for file_type, files in candidates.items():
        if files:
            return datasets.load_dataset(file_type, data_files=[str(item) for item in files], split="train")
    raise FileNotFoundError(f"No parquet/json/jsonl dataset files found under {input_path}")


def normalize_math_answer(raw_answer: Any) -> str:
    answer = "" if raw_answer is None else str(raw_answer)
    try:
        return str(remove_boxed(answer))
    except Exception:
        return answer


def first_solution(raw_solutions: Any) -> str:
    if not raw_solutions:
        return ""
    if isinstance(raw_solutions, list):
        return str(raw_solutions[0]) if raw_solutions else ""
    if isinstance(raw_solutions, str):
        try:
            parsed = json.loads(raw_solutions)
            if isinstance(parsed, list):
                return str(parsed[0]) if parsed else ""
        except json.JSONDecodeError:
            pass
    return str(raw_solutions)


def serialize_test_cases(raw_test_cases: Any) -> str:
    parsed = raw_test_cases
    if isinstance(raw_test_cases, str):
        try:
            parsed = json.loads(raw_test_cases)
        except json.JSONDecodeError as exc:
            raise ValueError("TACO input_output is not valid JSON") from exc
    if not isinstance(parsed, dict) or "inputs" not in parsed or "outputs" not in parsed:
        raise ValueError("TACO input_output must contain inputs and outputs")
    return json.dumps(parsed, ensure_ascii=False)


def map_math(example: dict[str, Any], index: int) -> dict[str, Any]:
    problem = str(example["problem"])
    answer = example["answer"]
    return {
        "data_source": "deepscaler",
        "prompt": [
            {"role": "system", "content": R1_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": normalize_math_answer(answer)},
        "extra_info": {
            "index": index,
            "problem": problem,
            "answer": "" if answer is None else str(answer),
            "expert_cot": str(example.get("solution") or ""),
        },
    }


def map_code(example: dict[str, Any], index: int) -> dict[str, Any]:
    question = str(example["question"]).strip()
    starter_code = str(example.get("starter_code") or "").strip()
    expert_solution = first_solution(example.get("solutions"))
    prompt = question
    if starter_code:
        prompt += f"\n\nStarter code:\n```python\n{starter_code}\n```"
    prompt += f"\n\n{CODE_INSTRUCTION}"
    return {
        "data_source": "BAAI/TACO",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": serialize_test_cases(example["input_output"]),
        },
        "extra_info": {
            "index": index,
            "problem": question,
            "starter_code": starter_code,
            "expert_solution": expert_solution,
            "expert_cot": expert_solution,
            "difficulty": str(example.get("difficulty") or ""),
            "source": str(example.get("source") or ""),
        },
    }


def validate_columns(dataset: datasets.Dataset, task: str) -> None:
    source_columns = set(dataset.column_names)
    required = {"problem", "answer"} if task == "math" else {"question", "input_output"}
    missing = required - source_columns
    if missing:
        raise ValueError(f"{task} dataset is missing columns: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"GRPO parquet already exists, skipping: {output_path}")
        return

    dataset = load_local_dataset(args.input, args.split)
    validate_columns(dataset, args.task)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    mapper = map_math if args.task == "math" else map_code
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_cache = output_path.with_suffix(".map.arrow")
    map_cache.unlink(missing_ok=True)
    converted = dataset.map(
        mapper,
        with_indices=True,
        remove_columns=dataset.column_names,
        cache_file_name=str(map_cache),
        load_from_cache_file=False,
    )
    expected_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    if set(converted.column_names) != expected_columns:
        raise ValueError(f"Unexpected output columns: {converted.column_names}")
    if len(converted) == 0:
        raise ValueError("Converted GRPO dataset is empty")

    try:
        converted.to_parquet(str(output_path))
    finally:
        map_cache.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "task": args.task,
                "input": args.input,
                "output": str(output_path),
                "rows": len(converted),
                "columns": converted.column_names,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
