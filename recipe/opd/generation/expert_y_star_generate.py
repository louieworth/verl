#!/usr/bin/env python3
"""Create Stage-1 trajectory data using each row's expert CoT as y*."""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

import pandas as pd


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def attach_expert_responses(source: pd.DataFrame) -> pd.DataFrame:
    if "extra_info" not in source.columns:
        raise ValueError("Input prompt parquet is missing extra_info")

    responses: list[list[str]] = []
    normalized_extra_info: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    for row_index, extra_info in enumerate(source["extra_info"]):
        if not isinstance(extra_info, dict):
            missing_indices.append(row_index)
            continue
        expert_cot = clean_text(extra_info.get("expert_cot"))
        if not expert_cot:
            missing_indices.append(row_index)
            continue
        normalized = dict(extra_info)
        normalized["trajectory_source"] = "expert_solution"
        normalized_extra_info.append(normalized)
        responses.append([expert_cot])

    if missing_indices:
        raise ValueError(
            "y* rollout requires a non-empty extra_info['expert_cot'] for every row; "
            f"missing indices include {missing_indices[:10]} "
            f"({len(missing_indices)}/{len(source)} rows)"
        )

    output = source.copy()
    output["extra_info"] = normalized_extra_info
    output["responses"] = responses
    return output


def write_atomic(dataset: pd.DataFrame, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".expert_y_star_",
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
    parser.add_argument("--input", required=True, help="Stage-1 prompt parquet")
    parser.add_argument("--output", required=True, help="Stage-1 response parquet")
    args = parser.parse_args()

    source = pd.read_parquet(args.input)
    if source.empty:
        raise ValueError("Input prompt parquet is empty")
    output = attach_expert_responses(source)
    write_atomic(output, args.output)
    print(f"Wrote {len(output)} y* trajectories from expert_cot -> {args.output}")


if __name__ == "__main__":
    main()
