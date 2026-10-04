#!/usr/bin/env python3
"""Record one PENS eval, decide whether to stop, and retain current and optionally best."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path


CHECKPOINT_RE = re.compile(r"global_step_(\d+)")

DEFAULT_TARGET_ROUGE_1 = 0.2933888365419484
DEFAULT_MIN_ROUGE_2 = 0.10612834364424847
DEFAULT_MIN_ROUGE_L = 0.24049149824883745
SELECTION_MODES = ("rouge_1", "balanced_relative")


def balanced_relative_rank(
    *,
    step: int,
    rouge_1: float,
    rouge_2: float,
    rouge_l: float,
    baseline_rouge_1: float,
    baseline_rouge_2: float,
    baseline_rouge_l: float,
) -> tuple[float, ...]:
    baselines = (baseline_rouge_1, baseline_rouge_2, baseline_rouge_l)
    if any(not math.isfinite(value) or value <= 0 for value in baselines):
        raise ValueError(f"Selection baselines must be finite and positive, got {baselines}")
    gains = (
        rouge_1 / baseline_rouge_1 - 1.0,
        rouge_2 / baseline_rouge_2 - 1.0,
        rouge_l / baseline_rouge_l - 1.0,
    )
    return (
        min(gains),
        sum(gains) / len(gains),
        rouge_1,
        rouge_2,
        rouge_l,
        -step,
    )


def load_json_object(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def replace_symlink(path: Path, target: str) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise ValueError(f"Refusing to replace non-symlink path: {path}")
    path.symlink_to(target, target_is_directory=True)


def remove_symlink(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise ValueError(f"Refusing to remove non-symlink path: {path}")


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temporary, path)


def record_evaluation_result(state_file: Path, result_key: str, payload: dict) -> None:
    """Hand an evaluation to the manager through its existing checkpoint state."""
    state = load_json_object(state_file)
    state["pending_evaluation"] = {"result_key": result_key, "metrics": payload}
    save_json_atomic(state_file, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    # Read-only compatibility for drivers launched before the W&B-only change.
    parser.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-key", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--catastrophic-drop", type=float, default=0.01)
    parser.add_argument("--minimum-output-coverage", type=float, default=0.99)
    parser.add_argument("--target-rouge-1", type=float, default=DEFAULT_TARGET_ROUGE_1)
    parser.add_argument("--minimum-rouge-2", type=float, default=DEFAULT_MIN_ROUGE_2)
    parser.add_argument("--minimum-rouge-l", type=float, default=DEFAULT_MIN_ROUGE_L)
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default="rouge_1")
    parser.add_argument("--selection-baseline-rouge-1", type=float)
    parser.add_argument("--selection-baseline-rouge-2", type=float)
    parser.add_argument("--selection-baseline-rouge-l", type=float)
    parser.add_argument("--selection-minimum-count", type=int, default=0)
    parser.add_argument("--keep-best-checkpoint", action="store_true", help="Retain the best model as well as the current checkpoint")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    if checkpoint_dir == Path("/") or not checkpoint_dir.is_dir():
        raise ValueError(f"Invalid checkpoint directory: {checkpoint_dir}")
    if args.step <= 0:
        raise ValueError(f"Step must be positive, got {args.step}")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative")
    if args.early_stop_min_delta < 0 or args.catastrophic_drop < 0:
        raise ValueError("decline thresholds must be non-negative")
    if not 0 <= args.minimum_output_coverage <= 1:
        raise ValueError("--minimum-output-coverage must be between zero and one")
    if args.selection_minimum_count < 0:
        raise ValueError("--selection-minimum-count must be non-negative")

    current_dir = checkpoint_dir / f"global_step_{args.step}"
    if not current_dir.is_dir():
        raise FileNotFoundError(f"Missing current checkpoint: {current_dir}")

    state = load_json_object(args.state_file)
    pending = state.get("pending_evaluation")
    if isinstance(pending, dict):
        if pending.get("result_key") != args.result_key or not isinstance(pending.get("metrics"), dict):
            raise KeyError(f"Missing pending eval result {args.result_key!r} in {args.state_file}")
        result = pending["metrics"]
    elif args.result_json is not None:
        results = load_json_object(args.result_json)
        if args.result_key not in results or not isinstance(results[args.result_key], dict):
            raise KeyError(f"Missing eval result key {args.result_key!r} in {args.result_json}")
        result = results[args.result_key]
    else:
        raise KeyError(f"Missing pending eval result {args.result_key!r} in {args.state_file}")
    metric = float(result["rouge_1_f1"])
    rouge_2 = float(result["rouge_2_f1"])
    rouge_l = float(result["rouge_l_f1"])
    metrics_are_finite = all(math.isfinite(value) for value in (metric, rouge_2, rouge_l))

    joined_count = int(result.get("joined_count") or result.get("count") or 0)
    non_empty_count = int(result.get("non_empty_prediction_count") or 0)
    output_coverage = non_empty_count / joined_count if joined_count > 0 else None

    evaluations = state.get("evaluations", {})
    if not isinstance(evaluations, dict):
        evaluations = {}
    previous_best_step = state.get("best_step")
    previous_best_metric = state.get("best_rouge_1_f1")
    previous_best_evaluation = evaluations.get(str(previous_best_step))
    previous_best_recorded = isinstance(previous_best_step, int) and isinstance(
        previous_best_evaluation, dict
    )
    selection_baselines = (
        args.selection_baseline_rouge_1,
        args.selection_baseline_rouge_2,
        args.selection_baseline_rouge_l,
    )
    if args.selection_mode == "balanced_relative" and any(
        value is None for value in selection_baselines
    ):
        raise ValueError(
            "balanced_relative selection requires all three --selection-baseline-* values"
        )

    current_selection_rank: tuple[float, ...] | None = None
    previous_selection_rank: tuple[float, ...] | None = None
    selection_eligible = metrics_are_finite and non_empty_count >= args.selection_minimum_count
    if selection_eligible and args.selection_mode == "balanced_relative":
        current_selection_rank = balanced_relative_rank(
            step=args.step,
            rouge_1=metric,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            baseline_rouge_1=float(args.selection_baseline_rouge_1),
            baseline_rouge_2=float(args.selection_baseline_rouge_2),
            baseline_rouge_l=float(args.selection_baseline_rouge_l),
        )
        if previous_best_recorded:
            previous_count = int(
                previous_best_evaluation.get("non_empty_prediction_count")
                or previous_best_evaluation.get("count")
                or 0
            )
            previous_values = (
                float(previous_best_evaluation.get("rouge_1_f1", float("nan"))),
                float(previous_best_evaluation.get("rouge_2_f1", float("nan"))),
                float(previous_best_evaluation.get("rouge_l_f1", float("nan"))),
            )
            if previous_count >= args.selection_minimum_count and all(
                math.isfinite(value) for value in previous_values
            ):
                previous_selection_rank = balanced_relative_rank(
                    step=int(previous_best_step),
                    rouge_1=previous_values[0],
                    rouge_2=previous_values[1],
                    rouge_l=previous_values[2],
                    baseline_rouge_1=float(args.selection_baseline_rouge_1),
                    baseline_rouge_2=float(args.selection_baseline_rouge_2),
                    baseline_rouge_l=float(args.selection_baseline_rouge_l),
                )
        is_new_best = current_selection_rank is not None and (
            not previous_best_recorded
            or previous_selection_rank is None
            or current_selection_rank > previous_selection_rank
        )
    else:
        is_new_best = selection_eligible and (
            not previous_best_recorded
            or previous_best_metric is None
            or metric > float(previous_best_metric)
        )
    if is_new_best:
        best_step = args.step
        best_metric = metric
        best_selection_rank = current_selection_rank
    else:
        best_step = int(previous_best_step) if previous_best_recorded else args.step
        best_metric = float(previous_best_metric) if previous_best_metric is not None else metric
        best_selection_rank = previous_selection_rank
    prior_steps = sorted(int(key) for key in evaluations if str(key).isdigit() and int(key) < args.step)
    previous_evaluation = evaluations.get(str(prior_steps[-1])) if prior_steps else None
    previous_metric = None
    if isinstance(previous_evaluation, dict):
        candidate = previous_evaluation.get("rouge_1_f1")
        if candidate is not None and math.isfinite(float(candidate)):
            previous_metric = float(candidate)

    significant_decline = (
        metrics_are_finite
        and previous_metric is not None
        and metric <= previous_metric - args.early_stop_min_delta
    )
    previous_decline_count = int(state.get("consecutive_significant_declines", 0))
    consecutive_declines = previous_decline_count + 1 if significant_decline else 0

    target_met = (
        metrics_are_finite
        and metric > args.target_rouge_1
        and rouge_2 >= args.minimum_rouge_2
        and rouge_l >= args.minimum_rouge_l
    )
    stop_reason = None
    if not metrics_are_finite:
        stop_reason = "non_finite_metric"
    elif output_coverage is not None and output_coverage < args.minimum_output_coverage:
        stop_reason = "output_coverage_below_minimum"
    elif target_met:
        stop_reason = "target_met"
    elif metric < best_metric - args.catastrophic_drop:
        stop_reason = "catastrophic_rouge_1_drop"
    elif args.early_stop_patience > 0 and consecutive_declines >= args.early_stop_patience:
        stop_reason = "consecutive_rouge_1_declines"

    evaluations[str(args.step)] = {
        "result_key": args.result_key,
        "rouge_1_f1": metric,
        "rouge_2_f1": rouge_2,
        "rouge_l_f1": rouge_l,
        "count": result.get("count"),
        "joined_count": result.get("joined_count"),
        "non_empty_prediction_count": result.get("non_empty_prediction_count"),
        "output_coverage": output_coverage,
        "raw_generation_file": result.get("raw_generation_file"),
        "model_path": result.get("model_path"),
        "significant_rouge_1_decline": significant_decline,
    }
    state = {
        "best_rouge_1_f1": best_metric,
        "best_step": best_step,
        "best_selection_rank": list(best_selection_rank) if best_selection_rank else None,
        "checkpoint_selection_mode": args.selection_mode,
        "selection_minimum_count": args.selection_minimum_count,
        "consecutive_significant_declines": consecutive_declines,
        "current_step": args.step,
        "evaluations": evaluations,
        "last_evaluated_step": args.step,
        "should_stop_trial": stop_reason is not None,
        "stop_reason": stop_reason,
        "target_met": target_met,
        "thresholds": {
            "catastrophic_drop": args.catastrophic_drop,
            "early_stop_min_delta": args.early_stop_min_delta,
            "early_stop_patience": args.early_stop_patience,
            "minimum_output_coverage": args.minimum_output_coverage,
            "minimum_rouge_2": args.minimum_rouge_2,
            "minimum_rouge_l": args.minimum_rouge_l,
            "target_rouge_1_strictly_greater_than": args.target_rouge_1,
            "selection_baseline_rouge_1": args.selection_baseline_rouge_1,
            "selection_baseline_rouge_2": args.selection_baseline_rouge_2,
            "selection_baseline_rouge_l": args.selection_baseline_rouge_l,
            "selection_minimum_count": args.selection_minimum_count,
        },
    }
    save_json_atomic(args.state_file, state)

    replace_symlink(checkpoint_dir / "current_checkpoint", current_dir.name)
    best_dir = checkpoint_dir / f"global_step_{best_step}"
    if args.keep_best_checkpoint and best_dir.is_dir():
        replace_symlink(checkpoint_dir / "best_checkpoint", best_dir.name)
    else:
        remove_symlink(checkpoint_dir / "best_checkpoint")

    retained_steps = {args.step}
    if args.keep_best_checkpoint:
        retained_steps.add(best_step)
    removed: list[str] = []
    for child in checkpoint_dir.iterdir():
        match = CHECKPOINT_RE.fullmatch(child.name)
        if match is None or not child.is_dir():
            continue
        child_step = int(match.group(1))
        if child_step in retained_steps:
            continue
        shutil.rmtree(child)
        removed.append(child.name)

    print(
        json.dumps(
            {
                "best_rouge_1_f1": best_metric,
                "best_step": best_step,
                "current_rouge_1_f1": metric,
                "current_rouge_2_f1": rouge_2,
                "current_rouge_l_f1": rouge_l,
                "current_step": args.step,
                "consecutive_significant_declines": consecutive_declines,
                "is_new_best": is_new_best,
                "output_coverage": output_coverage,
                "checkpoint_selection_mode": args.selection_mode,
                "selection_eligible": selection_eligible,
                "removed_checkpoints": sorted(removed),
                "retained_checkpoints": sorted(f"global_step_{step}" for step in retained_steps),
                "should_stop_trial": stop_reason is not None,
                "state_file": str(args.state_file),
                "stop_reason": stop_reason,
                "target_met": target_met,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
