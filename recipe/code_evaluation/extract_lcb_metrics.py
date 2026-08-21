#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_eval_all(root: Path) -> Path:
    candidates = sorted(root.rglob("*_eval_all.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(f"No LiveCodeBench *_eval_all.json found under {root}")


def load_rows(root: Path, aggregate_all: bool) -> tuple[list[dict], list[Path]]:
    if not aggregate_all:
        path = find_eval_all(root)
        with open(path) as f:
            return json.load(f), [path]

    paths = sorted(root.rglob("*_eval_all.json"), key=lambda p: p.stat().st_mtime)
    rows_by_id: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            continue
        with open(path) as f:
            for row in json.load(f):
                question_id = row.get("question_id")
                if question_id is not None:
                    # Newer shard/canonical files replace older duplicates.
                    rows_by_id[str(question_id)] = row
    if not rows_by_id:
        raise SystemExit(f"No LiveCodeBench *_eval_all.json found under {root}")
    return list(rows_by_id.values()), paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--pass_k", type=int, default=16)
    parser.add_argument(
        "--aggregate_all",
        action="store_true",
        help="Aggregate every *_eval_all.json below root and deduplicate by question_id.",
    )
    parser.add_argument(
        "--expected_tasks",
        type=int,
        default=None,
        help="Fail unless this many unique question_id rows are present.",
    )
    args = parser.parse_args()
    if args.pass_k <= 0:
        raise SystemExit("--pass_k must be positive")
    root = Path(args.root)
    rows, paths = load_rows(root, args.aggregate_all)
    if args.expected_tasks is not None and len(rows) != args.expected_tasks:
        raise SystemExit(
            f"Expected {args.expected_tasks} unique LiveCodeBench tasks under {root}, "
            f"found {len(rows)} across {len(paths)} files"
        )
    avg_values = []
    pass_values = []
    for row in rows:
        graded = row.get("graded_list") or []
        if not graded:
            raise SystemExit(f"LiveCodeBench task {row.get('question_id')} has no graded samples")
        if len(graded) < args.pass_k:
            raise SystemExit(
                f"LiveCodeBench task {row.get('question_id')} has {len(graded)} samples; "
                f"Avg@{args.pass_k}/Pass@{args.pass_k} requires at least {args.pass_k}"
            )
        limited = graded[: args.pass_k]
        correct = sum(1 for value in limited if bool(value))
        avg_values.append(correct / len(limited))
        pass_values.append(1.0 if correct > 0 else 0.0)
    if not avg_values:
        raise SystemExit(f"No LiveCodeBench graded_list entries found under {root}")
    print(f"avg={sum(avg_values) / len(avg_values)}")
    print(f"pass={sum(pass_values) / len(pass_values)}")
    print(f"num_problems={len(avg_values)}")


if __name__ == "__main__":
    main()
