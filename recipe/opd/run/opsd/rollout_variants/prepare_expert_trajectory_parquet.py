#!/usr/bin/env python3
"""Build the complete OPSD y* training parquet from a CoT-only dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import DatasetDict, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.opd.generation.expert_y_star_generate import (
    attach_expert_responses,
    write_atomic,
)
from recipe.opd.generation.y_o_prepare import make_map_fn_stage1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CoT-only DeepScaleR save_to_disk directory")
    parser.add_argument("--output", required=True, help="Complete OPSD y* parquet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-source", default="deepscaleR")
    args = parser.parse_args()

    loaded = load_from_disk(args.input)
    if isinstance(loaded, DatasetDict):
        if args.split not in loaded:
            raise ValueError(f"Dataset does not contain split {args.split!r}")
        source = loaded[args.split]
    else:
        source = loaded

    required = {"problem", "answer", "solution"}
    missing = required - set(source.column_names)
    if missing:
        raise ValueError(f"CoT-only dataset is missing columns: {sorted(missing)}")
    empty_indices = [
        index
        for index, value in enumerate(source["solution"])
        if value is None or not str(value).strip()
    ]
    if empty_indices:
        raise ValueError(
            "CoT-only source contains empty solution values; "
            f"indices include {empty_indices[:10]}"
        )

    processed = source.map(
        make_map_fn_stage1(data_source=args.data_source),
        with_indices=True,
        remove_columns=source.column_names,
        keep_in_memory=True,
        desc="Building OPSD y* schema",
    )
    output = attach_expert_responses(processed.to_pandas())
    write_atomic(output, args.output)

    expected_columns = {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
        "responses",
    }
    missing_output = expected_columns - set(output.columns)
    if missing_output:
        raise ValueError(f"Output is missing OPSD columns: {sorted(missing_output)}")
    print(f"Wrote complete OPSD y* trajectory parquet: {len(output)} rows -> {args.output}")


if __name__ == "__main__":
    main()
