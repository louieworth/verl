#!/usr/bin/env python3
"""Filter raw DeepScaleR to examples with a genuine, non-empty solution."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


REQUIRED_COLUMNS = {"problem", "answer", "solution"}


def clean_solution(example: dict) -> bool:
    solution = example.get("solution")
    return solution is not None and bool(str(solution).strip())


def validate(dataset: Dataset) -> None:
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"Raw DeepScaleR dataset is missing columns: {sorted(missing)}")
    empty_indices = [
        index
        for index, value in enumerate(dataset["solution"])
        if value is None or not str(value).strip()
    ]
    if empty_indices:
        raise ValueError(
            "Filtered dataset still contains empty solution values; "
            f"indices include {empty_indices[:10]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw DeepScaleR dataset saved with save_to_disk")
    parser.add_argument("--output", required=True, help="Output save_to_disk directory")
    parser.add_argument("--split", default="train")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    loaded = load_from_disk(args.input)
    if isinstance(loaded, DatasetDict):
        if args.split not in loaded:
            raise ValueError(f"Dataset does not contain split {args.split!r}")
        source = loaded[args.split]
    else:
        source = loaded

    missing = REQUIRED_COLUMNS - set(source.column_names)
    if missing:
        raise ValueError(f"Raw DeepScaleR dataset is missing columns: {sorted(missing)}")

    # The raw source may be mounted read-only. Avoid the datasets default of
    # writing a filter cache next to its Arrow files.
    filtered = source.filter(
        clean_solution,
        desc="Keeping non-empty raw solution",
        keep_in_memory=True,
    )
    filtered = filtered.select_columns(["problem", "answer", "solution"])
    validate(filtered)

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=str(output_path.parent))
    )
    try:
        filtered.save_to_disk(str(temporary_path))
        if output_path.exists():
            shutil.rmtree(output_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)

    print(f"Input rows: {len(source)}")
    print(f"Rows with non-empty raw solution: {len(filtered)}")
    print(f"Dropped rows without CoT: {len(source) - len(filtered)}")
    print(f"Saved OPSD expert-CoT dataset -> {output_path}")


if __name__ == "__main__":
    main()
