#!/usr/bin/env python3
# Copyright 2026
#
# Prepare DeepScaleR/DeepScaler math data in verl GRPO parquet format.

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import datasets
import pandas as pd

try:
    from verl.utils.reward_score.math_reward import remove_boxed
except Exception:  # pragma: no cover - keeps the converter usable outside full verl envs.
    remove_boxed = None


R1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>. "
    "Put your final answer within \\boxed{}. "
)

SIMPLE_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DeepScaleR/DeepScaler data as verl GRPO train/test parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        default="/data/data/jiangli/data/DeepScaleR-Cleaned",
        help="Local dataset dir, json/jsonl/parquet file, or Hugging Face dataset id.",
    )
    parser.add_argument("--output_dir", default="gen_results/grpo_data/deepscaler")
    parser.add_argument("--train_file_name", default="train.parquet")
    parser.add_argument("--test_file_name", default="test.parquet")
    parser.add_argument("--write_test", action="store_true", help="Also write a held-out test parquet.")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="Use <1.0 with --write_test to split train/test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default="train", help="Split to load when input_path is a HF dataset id or DatasetDict.")
    parser.add_argument("--question_key", default="problem")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--solution_key", default="solution")
    parser.add_argument("--data_source", default="deepscaler")
    parser.add_argument(
        "--prompt_style",
        choices=["r1", "simple"],
        default="r1",
        help="r1 matches recipe/r1_ascend/deepscaler.py format reward; simple is user-only CoT prompt.",
    )
    parser.add_argument(
        "--keep_boxed_ground_truth",
        action="store_true",
        help="Keep raw answer as ground_truth instead of trying to remove \\boxed{...}.",
    )
    return parser.parse_args()


def load_records(input_path: str, split: str) -> list[dict[str, Any]]:
    path = Path(input_path)
    if path.exists() and path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in (split, "train", "data"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list or a dict containing split/data list: {input_path}")
            return [dict(item) for item in data]
        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        if suffix == ".parquet":
            return pd.read_parquet(path).to_dict("records")
        raise ValueError(f"Unsupported input file type: {path.suffix}")

    if path.exists() and path.is_dir():
        try:
            ds_loaded = datasets.load_from_disk(str(path))
        except (FileNotFoundError, ValueError):
            parquet_files = sorted(str(p) for p in path.rglob("*.parquet"))
            json_files = sorted(str(p) for p in path.rglob("*.json"))
            jsonl_files = sorted(str(p) for p in path.rglob("*.jsonl"))
            if parquet_files:
                ds_loaded = datasets.load_dataset("parquet", data_files=parquet_files, split="train")
            elif json_files:
                ds_loaded = datasets.load_dataset("json", data_files=json_files, split="train")
            elif jsonl_files:
                ds_loaded = datasets.load_dataset("json", data_files=jsonl_files, split="train")
            else:
                raise FileNotFoundError(f"No dataset files found under {input_path}")
        if isinstance(ds_loaded, datasets.DatasetDict):
            ds = ds_loaded[split] if split in ds_loaded else ds_loaded["train"]
        else:
            ds = ds_loaded
        return [dict(item) for item in ds]

    ds = datasets.load_dataset(input_path, split=split)
    return [dict(item) for item in ds]


def normalize_ground_truth(answer: Any, keep_boxed: bool) -> str:
    answer_str = "" if answer is None else str(answer)
    if keep_boxed or remove_boxed is None:
        return answer_str
    try:
        return str(remove_boxed(answer_str))
    except Exception:
        return answer_str


def build_prompt(problem: str, style: str) -> list[dict[str, str]]:
    if style == "r1":
        return [
            {"role": "system", "content": R1_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
    return [{"role": "user", "content": f"{problem} {SIMPLE_SUFFIX}"}]


def convert_record(item: dict[str, Any], idx: int, args: argparse.Namespace) -> dict[str, Any]:
    missing = [key for key in (args.question_key, args.answer_key) if key not in item]
    if missing:
        raise KeyError(f"Input row {idx} is missing required key(s): {missing}")

    problem = str(item[args.question_key])
    answer = item[args.answer_key]
    solution = item.get(args.solution_key, "")
    ground_truth = normalize_ground_truth(answer, args.keep_boxed_ground_truth)

    return {
        "data_source": args.data_source,
        "prompt": build_prompt(problem, args.prompt_style),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": idx,
            "problem": problem,
            "answer": "" if answer is None else str(answer),
            "expert_cot": "" if solution is None else str(solution),
        },
    }


def validate_output(df: pd.DataFrame) -> None:
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Output is missing required GRPO column(s): {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("Output dataset is empty")
    first = df.iloc[0].to_dict()
    if not isinstance(first["prompt"], list):
        raise ValueError("prompt must be a list of chat messages")
    if not isinstance(first["reward_model"], dict) or "ground_truth" not in first["reward_model"]:
        raise ValueError("reward_model must be a dict with ground_truth")


def main() -> None:
    args = parse_args()
    if args.write_test and not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be in (0, 1) when --write_test is used")
    if not args.write_test and args.train_ratio != 1.0:
        raise ValueError("--train_ratio only applies together with --write_test")

    records = load_records(args.input_path, args.split)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    converted = [convert_record(item, idx, args) for idx, item in enumerate(records)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_test:
        rng = random.Random(args.seed)
        rng.shuffle(converted)
        split_idx = int(len(converted) * args.train_ratio)
        train_rows = converted[:split_idx]
        test_rows = converted[split_idx:]
    else:
        train_rows = converted
        test_rows = []

    train_df = pd.DataFrame(train_rows)
    validate_output(train_df)
    train_path = output_dir / args.train_file_name
    train_df.to_parquet(train_path, engine="pyarrow", index=False)

    test_path = None
    if args.write_test:
        test_df = pd.DataFrame(test_rows)
        validate_output(test_df)
        test_path = output_dir / args.test_file_name
        test_df.to_parquet(test_path, engine="pyarrow", index=False)

    summary = {
        "input_path": args.input_path,
        "train_path": str(train_path),
        "test_path": str(test_path) if test_path else None,
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "columns": list(train_df.columns),
        "prompt_style": args.prompt_style,
        "data_source": args.data_source,
    }
    summary_path = output_dir / "prepare_deepscaler_grpo_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
