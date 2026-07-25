#!/usr/bin/env python3
"""Build DeepScaleR SFT data from non-empty raw ``solution`` values only.

The ``answer`` column is metadata. It must never be used as the assistant
training target.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.lib


INSTRUCTION_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def read_parquet_compat(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except pyarrow.lib.ArrowNotImplementedError:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        batches = [batch.to_pandas() for batch in parquet_file.iter_batches(batch_size=1024)]
        return pd.concat(batches, axis=0, ignore_index=True) if batches else pd.DataFrame()


def load_source(path: str, split: str = "train") -> pd.DataFrame:
    """Load either a Hugging Face dataset saved to disk or a tabular file."""
    source_path = Path(path)
    if source_path.is_dir():
        from datasets import DatasetDict, load_from_disk

        dataset = load_from_disk(str(source_path))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(
                    f"Dataset has splits {sorted(dataset.keys())}, but split {split!r} was requested"
                )
            dataset = dataset[split]
        return dataset.to_pandas()

    suffix = source_path.suffix.lower()
    if suffix == ".parquet":
        return read_parquet_compat(str(source_path))
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(source_path, lines=suffix == ".jsonl")
    raise ValueError(
        "Unsupported input. Expected a Hugging Face dataset directory, parquet, json, or jsonl: "
        f"{source_path}"
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def build_sft_dataset(
    source: pd.DataFrame,
    *,
    problem_key: str = "problem",
    solution_key: str = "solution",
    answer_key: str = "answer",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Use only non-empty ``solution_key`` values as assistant targets."""
    required_columns = {problem_key, solution_key}
    missing = required_columns - set(source.columns)
    if missing:
        raise ValueError(
            "Raw DeepScaleR input is missing required columns "
            f"{sorted(missing)}. The SFT target must come from {solution_key!r}; "
            f"it must not fall back to {answer_key!r}."
        )
    if source.empty:
        raise ValueError("Input dataset is empty")

    records = []
    missing_problem_indices = []
    empty_solution_count = 0
    has_answer = answer_key in source.columns

    for source_index, row in source.iterrows():
        solution = clean_text(row.get(solution_key))
        if not solution:
            empty_solution_count += 1
            continue

        problem = clean_text(row.get(problem_key))
        if not problem:
            missing_problem_indices.append(int(source_index))
            continue

        extra_info = {
            "source_index": int(source_index),
            "problem": problem,
            "solution": solution,
            "target_source": solution_key,
        }
        if has_answer:
            # Kept only for auditing/evaluation; never copied into messages.
            extra_info["answer"] = clean_text(row.get(answer_key))

        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{problem} {INSTRUCTION_SUFFIX}",
                    },
                    {
                        "role": "assistant",
                        "content": solution,
                    },
                ],
                "extra_info": extra_info,
            }
        )

    if missing_problem_indices:
        raise ValueError(
            f"Rows with non-empty {solution_key!r} are missing {problem_key!r}; "
            f"indices include {missing_problem_indices[:10]}"
        )
    if not records:
        raise ValueError(f"No rows with non-empty {solution_key!r} were found")

    output = pd.DataFrame.from_records(records)
    stats = {
        "input_rows": len(source),
        "kept_nonempty_solution_rows": len(output),
        "dropped_empty_solution_rows": empty_solution_count,
    }
    return output, stats


def write_atomic(dataset: pd.DataFrame, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".deepscaler_sft_",
        suffix=".parquet",
        dir=output_dir or None,
    )
    os.close(fd)
    try:
        dataset.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Raw DeepScaleR dataset containing top-level problem and solution columns",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--solution-key", default="solution")
    parser.add_argument("--answer-key", default="answer")
    args = parser.parse_args()

    source = load_source(args.input, split=args.split)
    output, stats = build_sft_dataset(
        source,
        problem_key=args.problem_key,
        solution_key=args.solution_key,
        answer_key=args.answer_key,
    )
    write_atomic(output, args.output)
    print(f"Input rows: {stats['input_rows']}")
    print(f"Rows with non-empty raw {args.solution_key}: {stats['kept_nonempty_solution_rows']}")
    print(f"Dropped rows with empty raw {args.solution_key}: {stats['dropped_empty_solution_rows']}")
    print(f"Assistant target source: {args.solution_key} (never {args.answer_key})")
    print(f"Wrote DeepScaleR solution-CoT-only SFT data -> {args.output}")


if __name__ == "__main__":
    main()
