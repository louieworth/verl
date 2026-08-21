#!/usr/bin/env python3
"""Compute strict Avg@N/Pass@N math metrics and emit milestone JSON.

Avg@N is the mean correctness of all N samples. Pass@N is the fraction of
problems for which at least one of the N samples is correct. Missing samples are
an error: a partial generation must never be reported as Pass@16.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)

from recipe.math_evaluation.compute_score import compute_score_data_source
from recipe.math_evaluation.eval_utils import (
    DEFAULT_EVAL_DATASETS,
    DEFAULT_N_SAMPLES,
    DEFAULT_PROMPT_LENGTH,
    DEFAULT_RESPONSE_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    EXPECTED_CANONICAL_ROWS,
    normalize_eval_dataset_name,
)


SCHEMA_VERSION = "opd_eval_metrics/v1"


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".eval-metrics-", suffix=".json", dir=directory)
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def score_row(response: str, ground_truth, data_source: str) -> float:
    return float(compute_score_data_source(data_source, str(response), ground_truth))


def aggregate_bench(
    parquet_path: str,
    n_samples: int = DEFAULT_N_SAMPLES,
    dataset_name: str | None = None,
) -> dict[str, float | int]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"Generation file is empty: {parquet_path}")
    normalized_name = normalize_eval_dataset_name(dataset_name) if dataset_name else None
    expected_rows = EXPECTED_CANONICAL_ROWS.get(normalized_name)
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"{normalized_name} generation has {len(df)} problems; canonical evaluation requires {expected_rows}"
        )

    per_question_scores = []
    for row_index, row in df.iterrows():
        data_source = str(row.get("data_source", "")).strip()
        if not data_source:
            raise ValueError(f"Row {row_index} has no data_source in {parquet_path}")
        reward_model = row.get("reward_model")
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        if ground_truth is None:
            raise ValueError(f"Row {row_index} has no reward_model.ground_truth in {parquet_path}")
        responses = row.get("responses")
        responses = list(responses) if responses is not None else []
        if len(responses) != n_samples:
            raise ValueError(
                f"Row {row_index} in {parquet_path} has {len(responses)} responses; "
                f"strict Avg@{n_samples}/Pass@{n_samples} requires exactly {n_samples}."
            )
        row_scores = []
        for sample_index, response in enumerate(responses):
            try:
                row_scores.append(score_row(response, ground_truth, data_source))
            except Exception as exc:
                raise RuntimeError(
                    f"Scorer failed for row {row_index}, sample {sample_index}, "
                    f"data_source={data_source!r} in {parquet_path}"
                ) from exc
        per_question_scores.append(row_scores)

    matrix = np.asarray(per_question_scores, dtype=np.float64)
    correct = matrix > 0
    return {
        "avg": float(correct.mean()),
        "pass": float(correct.any(axis=1).mean()),
        "num_problems": int(matrix.shape[0]),
        "n_samples": int(matrix.shape[1]),
    }


def build_metrics_payload(
    *,
    model_name: str,
    model_path: str,
    step: int | None,
    aggregates: dict[str, dict[str, float | int]],
    n_samples: int,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    seed: int,
    milestone_fraction: float | None = None,
) -> dict[str, Any]:
    datasets = {
        dataset: {
            "num_problems": int(metrics["num_problems"]),
            "avg@16": float(metrics["avg"]),
            "pass@16": float(metrics["pass"]),
        }
        for dataset, metrics in aggregates.items()
    }
    macro_avg = float(np.mean([metrics["avg"] for metrics in aggregates.values()]))
    macro_pass = float(np.mean([metrics["pass"] for metrics in aggregates.values()]))
    wandb = {}
    for dataset, metrics in datasets.items():
        wandb[f"eval/math/{dataset}/avg@16"] = metrics["avg@16"]
        wandb[f"eval/math/{dataset}/pass@16"] = metrics["pass@16"]
    wandb["eval/math/macro/avg@16"] = macro_avg
    wandb["eval/math/macro/pass@16"] = macro_pass

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "math",
        "model": {"name": model_name, "path": model_path or None},
        "step": step,
        "milestone_fraction": milestone_fraction,
        "sampling": {
            "prompt_length": prompt_length,
            "response_length": response_length,
            "n": n_samples,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "prompt_format": "base_completion",
        },
        "datasets": datasets,
        "macro": {"avg@16": macro_avg, "pass@16": macro_pass},
        "wandb": wandb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True, help="Directory with {dataset}_pass16_generation.parquet")
    parser.add_argument("--results_file", required=True, help="Backward-compatible model-keyed results JSON")
    parser.add_argument("--metrics_file", default="", help="Stable milestone JSON; defaults to GEN_DIR/metrics.json")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_EVAL_DATASETS),
        help="Comma-separated canonical datasets",
    )
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--milestone_fraction", type=float, default=None)
    parser.add_argument("--prompt_length", type=int, default=DEFAULT_PROMPT_LENGTH)
    parser.add_argument("--response_length", type=int, default=DEFAULT_RESPONSE_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples != 16:
        raise ValueError("Canonical milestone export requires --n_samples=16")
    if args.milestone_fraction is not None and not 0 < args.milestone_fraction <= 1:
        raise ValueError("--milestone_fraction must be in (0, 1]")
    datasets = [normalize_eval_dataset_name(name) for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("At least one dataset is required")

    aggregates = {}
    for dataset in datasets:
        parquet_path = os.path.join(args.gen_dir, f"{dataset}_pass{args.n_samples}_generation.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Missing generation file: {parquet_path}")
        aggregate = aggregate_bench(parquet_path, n_samples=args.n_samples, dataset_name=dataset)
        aggregates[dataset] = aggregate
        print(
            f"  {dataset:12s} n={aggregate['num_problems']} "
            f"avg@16={aggregate['avg']:.4f} pass@16={aggregate['pass']:.4f}"
        )

    results: dict[str, Any] = {}
    if os.path.exists(args.results_file) and os.path.getsize(args.results_file) > 0:
        with open(args.results_file) as f:
            results = json.load(f)
    entry = results.setdefault(args.model_name, {})
    entry["model_path"] = args.model_path or entry.get("model_path")
    for dataset, aggregate in aggregates.items():
        # New compact keys are the canonical flat representation. The verbose
        # legacy keys remain readable by historical analysis scripts.
        entry[f"{dataset}_avg16"] = aggregate["avg"]
        entry[f"{dataset}_pass16"] = aggregate["pass"]
        entry[f"{dataset}_num_problems"] = aggregate["num_problems"]
        entry[f"{dataset}_avg_pass1_generation_pass_16"] = aggregate["avg"]
        entry[f"{dataset}_pass16_generation_pass_16"] = aggregate["pass"]
    entry["macro_avg16"] = float(np.mean([value["avg"] for value in aggregates.values()]))
    entry["macro_pass16"] = float(np.mean([value["pass"] for value in aggregates.values()]))
    atomic_write_json(args.results_file, results)

    payload = build_metrics_payload(
        model_name=args.model_name,
        model_path=args.model_path,
        step=args.step,
        aggregates=aggregates,
        n_samples=args.n_samples,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        milestone_fraction=args.milestone_fraction,
    )
    metrics_file = args.metrics_file or str(Path(args.gen_dir) / "metrics.json")
    atomic_write_json(metrics_file, payload)
    print(f"Wrote model results to {args.results_file}")
    print(f"Wrote stable milestone metrics to {metrics_file}")


if __name__ == "__main__":
    main()
