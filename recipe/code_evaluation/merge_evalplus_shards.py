#!/usr/bin/env python3
"""Merge non-overlapping EvalPlus JSONL shards with strict sample-count checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
# Avoid shadowing HuggingFace `datasets` with recipe/code_evaluation/datasets.
sys.path = [
    path
    for path in sys.path
    if Path(path or os.getcwd()).resolve() != SCRIPT_DIR
]


def load_expected_task_ids(dataset: str, version: str) -> list[str]:
    # Import after argument parsing so the caller can provide the same local
    # *_OVERRIDE_PATH environment variables used by run_evalplus_vllm.py.
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset == "humaneval":
        return list(get_human_eval_plus(version=version))
    return list(get_mbpp_plus(version=version))


def load_rows(paths: list[Path]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing input shard: {path}")
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                task_id = row.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError(f"missing task_id at {path}:{line_number}")
                if not isinstance(row.get("solution"), str):
                    raise ValueError(f"missing solution at {path}:{line_number}")
                rows[task_id].append(row)
    return rows


def validate(rows: dict[str, list[dict]], expected_ids: list[str], n_samples: int) -> None:
    expected = set(expected_ids)
    unexpected = sorted(set(rows) - expected)
    if unexpected:
        raise ValueError(f"unexpected task IDs: {unexpected[:10]}")
    bad_counts = {
        task_id: len(rows.get(task_id, []))
        for task_id in expected_ids
        if len(rows.get(task_id, [])) != n_samples
    }
    if bad_counts:
        preview = list(bad_counts.items())[:20]
        raise ValueError(
            f"expected {n_samples} samples for each of {len(expected_ids)} tasks; "
            f"bad counts include: {preview}"
        )


def write_rows(path: Path, rows: dict[str, list[dict]], expected_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        for task_id in expected_ids:
            for row in rows[task_id]:
                stream.write(json.dumps(row) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("humaneval", "mbpp"))
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--raw_input", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw_output", type=Path)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--version", default="default")
    args = parser.parse_args()

    if args.n_samples < 1:
        parser.error("--n_samples must be positive")
    if bool(args.raw_input) != bool(args.raw_output):
        parser.error("--raw_input and --raw_output must be provided together")

    expected_ids = load_expected_task_ids(args.dataset, args.version)
    rows = load_rows(args.input)
    validate(rows, expected_ids, args.n_samples)
    write_rows(args.output, rows, expected_ids)

    if args.raw_input:
        raw_rows = load_rows(args.raw_input)
        validate(raw_rows, expected_ids, args.n_samples)
        write_rows(args.raw_output, raw_rows, expected_ids)

    print(
        f"Merged {len(expected_ids)} tasks x {args.n_samples} samples "
        f"into {args.output}"
    )


if __name__ == "__main__":
    main()
