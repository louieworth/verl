#!/usr/bin/env python3
"""Convert DeepScaleR GRPO data into assistant-only-loss SFT messages."""

from __future__ import annotations

import argparse
import os
import tempfile

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


def extract_problem(row: pd.Series) -> str:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        problem = str(extra_info.get("problem") or "").strip()
        if problem:
            return problem

    prompt = row.get("prompt")
    if isinstance(prompt, (list, tuple)):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "").strip()
    return ""


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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = read_parquet_compat(args.input)
    required_columns = {"prompt", "extra_info"}
    missing = required_columns - set(source.columns)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")
    if source.empty:
        raise ValueError("Input parquet is empty")

    records = []
    missing_problem_indices = []
    empty_cot_count = 0
    for source_index, row in source.iterrows():
        extra_info = row.get("extra_info")
        expert_cot = ""
        if isinstance(extra_info, dict):
            expert_cot = str(extra_info.get("expert_cot") or "").strip()
        if not expert_cot:
            empty_cot_count += 1
            continue

        problem = extract_problem(row)
        if not problem:
            missing_problem_indices.append(int(source_index))
            continue

        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{problem} {INSTRUCTION_SUFFIX}",
                    },
                    {
                        "role": "assistant",
                        "content": expert_cot,
                    },
                ],
                "extra_info": {
                    "source_index": int(source_index),
                    "problem": problem,
                    "expert_cot": expert_cot,
                },
            }
        )

    if missing_problem_indices:
        raise ValueError(
            "Rows with non-empty expert_cot are missing a problem; "
            f"indices include {missing_problem_indices[:10]}"
        )
    if not records:
        raise ValueError("No rows with non-empty expert_cot were found")

    output = pd.DataFrame.from_records(records)
    write_atomic(output, args.output)
    print(f"Input rows: {len(source)}")
    print(f"Rows with non-empty expert_cot: {len(output)}")
    print(f"Dropped rows with empty expert_cot: {empty_cot_count}")
    print(f"Wrote DeepScaleR SFT data -> {args.output}")


if __name__ == "__main__":
    main()
