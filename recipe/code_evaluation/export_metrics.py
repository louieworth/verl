#!/usr/bin/env python3
"""Export code evaluation results using the stable milestone JSON schema."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any


SCHEMA_VERSION = "opd_eval_metrics/v1"
DEFAULT_DATASETS = ("humaneval_plus", "mbpp_plus", "livecodebench_v6")
ALIASES = {
    "humaneval+": "humaneval_plus",
    "mbpp+": "mbpp_plus",
    "lcb_v6": "livecodebench_v6",
    "livecodebench": "livecodebench_v6",
}


def normalize_dataset(name: str) -> str:
    normalized = name.strip().lower()
    return ALIASES.get(normalized, normalized)


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


def build_metrics_payload(
    *,
    entry: dict[str, Any],
    model_name: str,
    model_path: str,
    datasets: list[str],
    step: int | None,
    n_samples: int,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    seed: int,
    milestone_fraction: float | None = None,
) -> dict[str, Any]:
    if n_samples != 16:
        raise ValueError("Canonical milestone export requires n_samples=16")
    dataset_metrics = {}
    for dataset in datasets:
        avg_key = f"{dataset}_avg{n_samples}"
        pass_key = f"{dataset}_pass{n_samples}"
        if avg_key not in entry or pass_key not in entry:
            # Permit partial evaluation suites: upload every dataset that has
            # a complete Avg@16/Pass@16 pair instead of discarding all results
            # because an unrelated benchmark is unavailable.
            continue
        metrics = {
            "avg@16": float(entry[avg_key]),
            "pass@16": float(entry[pass_key]),
        }
        if f"{dataset}_num_problems" in entry:
            metrics["num_problems"] = int(entry[f"{dataset}_num_problems"])
        dataset_metrics[dataset] = metrics

    if not dataset_metrics:
        raise ValueError(f"No complete Avg@16/Pass@16 dataset metrics found for {model_name}")

    macro_avg = sum(metrics["avg@16"] for metrics in dataset_metrics.values()) / len(dataset_metrics)
    macro_pass = sum(metrics["pass@16"] for metrics in dataset_metrics.values()) / len(dataset_metrics)
    wandb = {}
    for dataset, metrics in dataset_metrics.items():
        wandb[f"eval/code/{dataset}/avg@16"] = metrics["avg@16"]
        wandb[f"eval/code/{dataset}/pass@16"] = metrics["pass@16"]
    wandb["eval/code/macro/avg@16"] = macro_avg
    wandb["eval/code/macro/pass@16"] = macro_pass

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "code",
        "model": {"name": model_name, "path": model_path or entry.get("model_path")},
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
        "datasets": dataset_metrics,
        "macro": {"avg@16": macro_avg, "pass@16": macro_pass},
        "wandb": wandb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stable code milestone metrics JSON")
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--milestone_fraction", type=float, default=None)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--prompt_length", type=int, default=2048)
    parser.add_argument("--response_length", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.results_file) as f:
        results = json.load(f)
    if args.model_name not in results or not isinstance(results[args.model_name], dict):
        raise KeyError(f"Model {args.model_name!r} not found in {args.results_file}")
    datasets = [normalize_dataset(name) for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("At least one dataset is required")
    if args.milestone_fraction is not None and not 0 < args.milestone_fraction <= 1:
        raise ValueError("--milestone_fraction must be in (0, 1]")
    payload = build_metrics_payload(
        entry=results[args.model_name],
        model_name=args.model_name,
        model_path=args.model_path,
        datasets=datasets,
        step=args.step,
        n_samples=args.n_samples,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        milestone_fraction=args.milestone_fraction,
    )
    entry = results[args.model_name]
    entry["macro_avg16"] = payload["macro"]["avg@16"]
    entry["macro_pass16"] = payload["macro"]["pass@16"]
    atomic_write_json(args.results_file, results)
    atomic_write_json(args.output_file, payload)
    print(f"Wrote stable milestone metrics to {args.output_file}")


if __name__ == "__main__":
    main()
