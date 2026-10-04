#!/usr/bin/env python3
"""Upsert one model's PENS evaluation metrics into a shared result.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-file", type=Path, required=True, help="Per-model metrics json file")
    parser.add_argument("--result-file", type=Path, required=True, help="Shared result.json path")
    parser.add_argument("--model-key", type=str, required=True, help="Top-level key in result.json")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--raw-file", type=str, default="")
    parser.add_argument("--prediction-file", type=str, default="")
    parser.add_argument("--per-example-file", type=str, default="")
    args = parser.parse_args()

    metrics = load_json(args.metrics_file)
    payload = dict(metrics)
    payload["model_path"] = args.model_path
    if args.raw_file:
        payload["raw_generation_file"] = args.raw_file
    if args.prediction_file:
        payload["prediction_file"] = args.prediction_file
    if args.per_example_file:
        payload["per_example_file"] = args.per_example_file

    result = {}
    if args.result_file.exists():
        existing = load_json(args.result_file)
        if isinstance(existing, dict):
            result = existing
        else:
            raise ValueError(f"Expected {args.result_file} to contain a JSON object.")

    result[args.model_key] = payload
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    with args.result_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        json.dumps(
            {
                "result_file": str(args.result_file),
                "model_key": args.model_key,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
