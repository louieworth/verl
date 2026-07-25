#!/usr/bin/env python3
"""Select the first correct response from each Best-of-N candidate group.

Groups without a correct response are excluded from the training dataset.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import pandas as pd

from recipe.opd.dataset.score_stage1_reward import (
    _inject,
    _score_rows_parallel,
    read_parquet_compat,
)


def response_list(value) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return ["" if item is None else str(item) for item in value]
    return ["" if value is None else str(value)]


def write_atomic(df: pd.DataFrame, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".select_best_of_n_",
        suffix=".parquet",
        dir=output_dir or None,
    )
    os.close(fd)
    try:
        df.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SCORE_STAGE1_WORKERS", min(32, os.cpu_count() or 1))),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=int(os.environ.get("SCORE_STAGE1_PROGRESS_INTERVAL", "100")),
    )
    args = parser.parse_args()
    if args.n <= 1:
        raise ValueError("--n must be greater than 1")

    dataset = read_parquet_compat(args.input)
    if dataset.empty:
        raise ValueError("Candidate parquet is empty")
    if "responses" not in dataset:
        raise ValueError("Candidate parquet has no responses column")

    candidate_groups = [response_list(value) for value in dataset["responses"]]
    invalid = [index for index, group in enumerate(candidate_groups) if len(group) != args.n]
    if invalid:
        raise ValueError(
            f"Expected exactly {args.n} responses per row; invalid rows include {invalid[:10]}"
        )

    records = dataset.to_dict("records")
    scores = np.full((len(dataset), args.n), np.nan, dtype=float)
    selected_indices = np.zeros(len(dataset), dtype=int)
    unresolved = np.arange(len(dataset), dtype=int)
    for candidate_index in range(args.n):
        if len(unresolved) == 0:
            break
        print(
            f"Scoring candidate {candidate_index + 1}/{args.n} "
            f"for {len(unresolved)} unresolved rows."
        )
        candidate_records = []
        for row_index in unresolved:
            candidate_record = dict(records[row_index])
            candidate_record["responses"] = [candidate_groups[row_index][candidate_index]]
            candidate_records.append(candidate_record)
        candidate_scores = _score_rows_parallel(
            pd.DataFrame(candidate_records),
            workers=max(1, args.workers),
            progress_interval=max(0, args.progress_interval),
        )
        scores[unresolved, candidate_index] = candidate_scores
        newly_correct = candidate_scores >= args.threshold
        selected_indices[unresolved[newly_correct]] = candidate_index
        unresolved = unresolved[~newly_correct]

    any_correct = np.nan_to_num(scores, nan=-np.inf) >= args.threshold
    keep_indices = np.flatnonzero(any_correct.any(axis=1))

    selected_responses = []
    selected_extra_info = []
    for row_index in keep_indices:
        candidates = candidate_groups[row_index]
        row_scores = scores[row_index]
        correct_mask = any_correct[row_index]
        selected_index = selected_indices[row_index]
        selected_index = int(selected_index)
        selected_responses.append([candidates[selected_index]])

        extra_info = _inject(dataset.iloc[row_index].get("extra_info"), row_scores[selected_index])
        extra_info["best_of_n"] = args.n
        extra_info["best_of_n_selected_index"] = selected_index
        extra_info["best_of_n_any_correct"] = bool(correct_mask.any())
        extra_info["best_of_n_candidate_rewards"] = [
            None if np.isnan(score) else float(score)
            for score in row_scores
        ]
        selected_extra_info.append(extra_info)

    selected = dataset.iloc[keep_indices].copy().reset_index(drop=True)
    selected["responses"] = selected_responses
    selected["extra_info"] = selected_extra_info
    write_atomic(selected, args.output)

    correct_rows = len(keep_indices)
    dropped_rows = len(dataset) - correct_rows
    print(
        f"Selected first correct candidate for {correct_rows}/{len(dataset)} rows; "
        f"dropped {dropped_rows} rows with no correct candidate."
    )
    counts = np.bincount(selected_indices[keep_indices], minlength=args.n)
    print(f"Selected candidate counts: {counts.tolist()}")
    print(f"Wrote {len(selected)} training rows -> {args.output}")


if __name__ == "__main__":
    main()
