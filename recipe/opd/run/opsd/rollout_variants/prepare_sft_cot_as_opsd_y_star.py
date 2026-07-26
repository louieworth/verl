#!/usr/bin/env python3
"""Convert CoT-only DeepScaleR SFT rows into direct OPSD y* training data."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def as_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"messages must be a list, got {type(value).__name__}")
    return [dict(message) for message in value]


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def convert(source: pd.DataFrame) -> pd.DataFrame:
    required = {"messages", "extra_info"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"SFT parquet is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for row_index, row in source.iterrows():
        messages = as_messages(row["messages"])
        user_messages = [message for message in messages if message.get("role") == "user"]
        assistant_messages = [
            message for message in messages if message.get("role") == "assistant"
        ]
        if len(user_messages) != 1 or len(assistant_messages) != 1:
            raise ValueError(
                f"row {row_index} must contain one user and one assistant message"
            )

        source_info = row["extra_info"]
        if not isinstance(source_info, dict):
            raise ValueError(
                f"row {row_index} extra_info must be a dict, "
                f"got {type(source_info).__name__}"
            )

        problem = clean(source_info.get("problem"))
        answer = clean(source_info.get("answer"))
        solution = clean(source_info.get("solution"))
        assistant_target = clean(assistant_messages[0].get("content"))
        target_source = clean(source_info.get("target_source"))
        if not problem or not solution or not assistant_target:
            raise ValueError(
                f"row {row_index} is not a non-empty solution/CoT training example"
            )
        if solution != assistant_target:
            raise ValueError(
                f"row {row_index} assistant target differs from extra_info.solution"
            )
        if target_source and target_source != "solution":
            raise ValueError(
                f"row {row_index} target_source={target_source!r}, expected 'solution'"
            )

        source_index = source_info.get("source_index", row_index)
        extra_info = {
            "problem": problem,
            "answer": answer,
            "expert_cot": solution,
            "expert_solution_source": "solution",
            "trajectory_source": "expert_solution",
            "index": int(source_index),
        }
        rows.append(
            {
                "data_source": "deepscaleR",
                "prompt": [
                    {
                        "role": "user",
                        "content": clean(user_messages[0].get("content")),
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": extra_info,
                "responses": [solution],
            }
        )

    output = pd.DataFrame(rows)
    if len(output) != len(source):
        raise AssertionError(f"row count changed: {len(source)} -> {len(output)}")
    return output


def write_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp.parquet",
        dir=str(output_path.parent),
    )
    os.close(fd)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = pd.read_parquet(args.input)
    output = convert(source)
    write_atomic(output, args.output)
    print(f"Wrote {len(output)} CoT-only OPSD y* rows -> {args.output}")


if __name__ == "__main__":
    main()
