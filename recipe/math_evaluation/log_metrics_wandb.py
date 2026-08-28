#!/usr/bin/env python3
"""Resume an existing W&B run and log a stable eval-metrics JSON payload.

Credentials are deliberately outside this interface. Authentication remains a
W&B runtime concern; this script never reads, writes, or prints an API key.
"""

from __future__ import annotations

import argparse
import json
import os

from verl.utils.wandb_metadata import (
    merge_opd_wandb_config,
    wandb_init_metadata_from_env,
)


EXPECTED_SCHEMA = "opd_eval_metrics/v1"


def load_wandb_payload(
    metrics_file: str,
    milestone_fraction: float | None,
    global_step: int | None = None,
) -> tuple[str, dict[str, float]]:
    with open(metrics_file) as f:
        document = json.load(f)
    if document.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(
            f"Unsupported metrics schema {document.get('schema_version')!r}; expected {EXPECTED_SCHEMA!r}"
        )
    task = document.get("task")
    if task not in {"math", "code"}:
        raise ValueError(f"Unsupported evaluation task in {metrics_file}: {task!r}")
    document_step = document.get("step")
    if global_step is not None:
        if document_step is None:
            raise ValueError("Metric file is missing step; refusing to relabel stale evaluation output")
        if int(document_step) != global_step:
            raise ValueError(
                f"Metric file step={document_step} does not match requested global_step={global_step}"
            )
    document_fraction = document.get("milestone_fraction")
    if milestone_fraction is not None:
        if document_fraction is None:
            raise ValueError(
                "Metric file is missing milestone_fraction; refusing to relabel stale evaluation output"
            )
        if abs(float(document_fraction) - milestone_fraction) > 1e-12:
            raise ValueError(
                f"Metric file milestone_fraction={document_fraction} does not match requested {milestone_fraction}"
            )
    elif document_fraction is not None:
        raise ValueError(
            f"Metric file unexpectedly contains milestone_fraction={document_fraction} for a base evaluation"
        )
    raw_payload = document.get("wandb")
    if not isinstance(raw_payload, dict) or not raw_payload:
        raise ValueError(f"No non-empty 'wandb' metric object in {metrics_file}")

    payload: dict[str, float] = {}
    expected_prefix = f"eval/{task}/"
    for key, value in raw_payload.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid W&B metric entry: {key!r}={value!r}")
        if not key.startswith(expected_prefix):
            raise ValueError(f"Metric {key!r} does not belong to task namespace {expected_prefix!r}")
        if not 0 <= float(value) <= 1:
            raise ValueError(f"Evaluation metric {key!r} must be in [0, 1], got {value}")
        payload[key] = float(value)
    if milestone_fraction is not None:
        payload["eval/milestone_fraction"] = milestone_fraction
        payload[f"eval/{task}/milestone_fraction"] = milestone_fraction
    return task, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log stable external evaluation metrics to an existing W&B run")
    parser.add_argument("--metrics_file", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--run_id", default=os.environ.get("WANDB_RUN_ID", ""))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--resume",
        choices=["allow", "must", "never", "auto"],
        default=os.environ.get("WANDB_RESUME", "allow"),
    )
    parser.add_argument("--global_step", type=int, default=os.environ.get("WANDB_GLOBAL_STEP"))
    parser.add_argument(
        "--milestone_fraction",
        type=float,
        default=os.environ.get("EVAL_MILESTONE_FRACTION"),
    )
    parser.add_argument("--eval_kind", choices=["milestone", "base"], default=os.environ.get("EVAL_KIND", "milestone"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.project or not args.run_id:
        raise ValueError("--project/WANDB_PROJECT and --run_id/WANDB_RUN_ID are required")
    if args.global_step is None or args.global_step < 0:
        raise ValueError("--global_step/WANDB_GLOBAL_STEP must be a non-negative integer")
    if args.eval_kind == "milestone" and (
        args.milestone_fraction is None or not 0 < args.milestone_fraction <= 1
    ):
        raise ValueError("Milestone eval requires EVAL_MILESTONE_FRACTION in (0, 1]")
    if args.eval_kind == "base" and args.milestone_fraction is not None and not 0 <= args.milestone_fraction <= 1:
        raise ValueError("Base eval milestone fraction, when supplied, must be in [0, 1]")

    _, payload = load_wandb_payload(args.metrics_file, args.milestone_fraction, args.global_step)
    import wandb

    init_kwargs = dict(
        project=args.project,
        entity=args.entity or None,
        id=args.run_id,
        name=os.environ.get("WANDB_RUN_NAME") or None,
        resume=args.resume,
        mode=os.environ.get("WANDB_MODE", "online"),
        reinit=True,
        config=merge_opd_wandb_config(None),
    )
    init_kwargs.update(wandb_init_metadata_from_env())
    run = wandb.init(**init_kwargs)
    try:
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
        payload["global_step"] = args.global_step
        run.log(payload)
    finally:
        run.finish()
    print(f"Logged {len(payload)} eval metrics at global_step={args.global_step} to W&B run {args.run_id}")


if __name__ == "__main__":
    main()
