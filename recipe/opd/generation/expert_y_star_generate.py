#!/usr/bin/env python3
"""Create Stage-1 trajectory data using solution, then answer fallback, as y*."""

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
        expert_solution = clean_text(extra_info.get("expert_cot"))
        expert_solution_source = clean_text(extra_info.get("expert_solution_source"))
        if not expert_solution:
            expert_solution = clean_text(extra_info.get("answer"))
            expert_solution_source = "answer"
        if not expert_solution:
            missing_indices.append(row_index)
            continue
        if not expert_solution_source:
            expert_solution_source = "solution"
        normalized = dict(extra_info)
        normalized["expert_cot"] = expert_solution
        normalized["expert_solution_source"] = expert_solution_source
        normalized["trajectory_source"] = f"expert_{expert_solution_source}"
        normalized_extra_info.append(normalized)
        responses.append([expert_solution])

    if missing_indices:
        raise ValueError(
            "y* rollout requires a non-empty solution or answer for every row; "
            f"missing indices include {missing_indices[:10]} "
            f"({len(missing_indices)}/{len(source)} rows)"
        )

    output = source.copy()
    output["extra_info"] = normalized_extra_info
    output["responses"] = responses
    return output


def select_rows(
    source: pd.DataFrame,
    *,
    start_index: int = 0,
    num_samples: int | None = None,
) -> pd.DataFrame:
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    stop = len(source) if num_samples is None else start_index + num_samples
    selected = source.iloc[start_index:stop].reset_index(drop=True)
    expected = len(source) - start_index if num_samples is None else num_samples
    if len(selected) != expected:
        raise ValueError(
            f"Requested {expected} rows at offset {start_index}, got {len(selected)} "
            f"from a {len(source)}-row trajectory parquet"
        )
    return selected


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
    parser.add_argument(
        "--input",
        required=True,
        help="Stage-1 prompt parquet or a precomputed full y* trajectory parquet",
    )
    parser.add_argument("--output", required=True, help="Stage-1 response parquet")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()

    source = pd.read_parquet(args.input)
    if source.empty:
        raise ValueError("Input parquet is empty")
    source = select_rows(
        source,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )
    output = attach_expert_responses(source)
    write_atomic(output, args.output)
    print(
        f"Wrote {len(output)} y* trajectories from solution/answer "
        f"(start={args.start_index}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
