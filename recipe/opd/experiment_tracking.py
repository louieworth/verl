#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Experiment cadence and W&B run-continuity helpers for OPD/OPSD recipes.

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

from verl.utils.wandb_metadata import (
    merge_opd_wandb_config,
    wandb_init_metadata_from_env,
    wandb_process_settings,
    wandb_shared_run_enabled,
)


DEFAULT_EVAL_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_WANDB_PROJECT = "trd"
DEFAULT_WANDB_RUN_STATE_FILENAME = "wandb_run.json"

_VALID_WANDB_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_eval_fractions(
    value: str | Iterable[float] | None,
) -> tuple[float, ...]:
    """Parse and normalize fractional evaluation milestones.

    Fractions must be in ``(0, 1]``. The returned tuple is sorted and
    de-duplicated, which gives deterministic behavior for both CLI strings and
    programmatic callers.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_EVAL_FRACTIONS

    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",")]
        if any(not item for item in raw_values):
            raise ValueError(f"eval fractions contain an empty item: {value!r}")
    else:
        raw_values = list(value)
        if not raw_values:
            raise ValueError("eval fractions must not be empty")

    fractions: set[float] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, bool):
            raise ValueError(f"invalid eval fraction: {raw_value!r}")
        try:
            fraction = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid eval fraction: {raw_value!r}") from exc
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"eval fraction must be in (0, 1], got {raw_value!r}")
        fractions.add(fraction)

    return tuple(sorted(fractions))


def compute_eval_milestones(
    total_training_steps: int,
    fractions: str | Iterable[float] | None = None,
) -> tuple[int, ...]:
    """Convert fractional milestones to unique optimizer steps using ceil."""
    if isinstance(total_training_steps, bool) or total_training_steps <= 0:
        raise ValueError(
            f"total_training_steps must be a positive integer, got {total_training_steps!r}"
        )
    if int(total_training_steps) != total_training_steps:
        raise ValueError(
            f"total_training_steps must be a positive integer, got {total_training_steps!r}"
        )

    total_training_steps = int(total_training_steps)
    parsed_fractions = parse_eval_fractions(fractions)
    return tuple(
        sorted(
            {
                min(total_training_steps, math.ceil(total_training_steps * fraction))
                for fraction in parsed_fractions
            }
        )
    )


def compute_eval_milestone_fractions(
    total_training_steps: int,
    fractions: str | Iterable[float] | None = None,
) -> dict[int, float]:
    """Map each unique milestone step to the greatest fraction it reaches."""
    milestones = compute_eval_milestones(total_training_steps, fractions)
    total_training_steps = int(total_training_steps)
    parsed_fractions = parse_eval_fractions(fractions)
    by_step: dict[int, float] = {step: 0.0 for step in milestones}
    for fraction in parsed_fractions:
        step = min(total_training_steps, math.ceil(total_training_steps * fraction))
        by_step[step] = max(by_step[step], fraction)
    return by_step


@dataclass(frozen=True)
class WandbRunState:
    """Stable W&B identity shared by all train/eval processes in one run."""

    run_id: str
    project: str
    run_identity: str
    state_path: Path


def default_wandb_run_state_path(output_dir: str | os.PathLike[str]) -> Path:
    """Return the default persisted run-state path for an experiment."""
    return Path(output_dir) / DEFAULT_WANDB_RUN_STATE_FILENAME


def _validate_wandb_run_id(run_id: str) -> str:
    run_id = run_id.strip()
    if not run_id:
        raise ValueError("W&B run id must not be empty")
    if len(run_id) > 128 or not _VALID_WANDB_RUN_ID.fullmatch(run_id):
        raise ValueError(
            "W&B run id must contain only letters, digits, '_' or '-' and be at most 128 characters"
        )
    return run_id


def deterministic_wandb_run_id(project: str, run_identity: str) -> str:
    """Derive a stable W&B run id without embedding user or model names."""
    project = project.strip()
    run_identity = run_identity.strip()
    if not project:
        raise ValueError("W&B project must not be empty")
    if not run_identity:
        raise ValueError("W&B run identity must not be empty")
    payload = json.dumps(
        {"project": project, "run_identity": run_identity},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _read_wandb_run_state(state_path: Path) -> dict[str, Any]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid W&B run state JSON at {state_path}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"invalid W&B run state at {state_path}: expected an object")
    return state


def resolve_wandb_run_state(
    *,
    project: str = DEFAULT_WANDB_PROJECT,
    run_name: str = "",
    run_identity: str = "",
    state_path: str | os.PathLike[str],
    explicit_run_id: str = "",
) -> WandbRunState:
    """Load or atomically persist a deterministic W&B run identity.

    ``explicit_run_id`` takes precedence over ``WANDB_RUN_ID``. If a state
    file already exists, a conflicting explicit id is rejected instead of
    silently creating a second dashboard run.
    """
    project = project.strip()
    if not project:
        raise ValueError("W&B project must not be empty")

    path = Path(state_path)
    resolved_identity = (run_identity or run_name).strip()
    if not resolved_identity:
        resolved_identity = str(path.parent.resolve())

    requested_run_id = (explicit_run_id or os.environ.get("WANDB_RUN_ID", "")).strip()
    if requested_run_id:
        requested_run_id = _validate_wandb_run_id(requested_run_id)

    if path.exists():
        state = _read_wandb_run_state(path)
        stored_project = str(state.get("project", "")).strip()
        stored_run_id = _validate_wandb_run_id(str(state.get("run_id", "")))
        stored_identity = str(state.get("run_identity", "")).strip()
        if stored_project != project:
            raise ValueError(
                f"W&B state project mismatch at {path}: stored={stored_project!r}, requested={project!r}"
            )
        if requested_run_id and requested_run_id != stored_run_id:
            raise ValueError(
                f"W&B run id mismatch at {path}: stored={stored_run_id!r}, requested={requested_run_id!r}"
            )
        if stored_identity and stored_identity != resolved_identity:
            raise ValueError(
                f"W&B run identity mismatch at {path}: "
                f"stored={stored_identity!r}, requested={resolved_identity!r}"
            )
        return WandbRunState(
            run_id=stored_run_id,
            project=stored_project,
            run_identity=stored_identity or resolved_identity,
            state_path=path,
        )

    run_id = requested_run_id or deterministic_wandb_run_id(project, resolved_identity)
    state = {
        "schema_version": 1,
        "project": project,
        "run_id": run_id,
        "run_identity": resolved_identity,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    try:
        with path.open("x", encoding="utf-8") as state_file:
            state_file.write(serialized)
    except FileExistsError:
        # Another process won the creation race. Re-read and validate it.
        return resolve_wandb_run_state(
            project=project,
            run_name=run_name,
            run_identity=run_identity,
            state_path=path,
            explicit_run_id=run_id,
        )

    return WandbRunState(
        run_id=run_id,
        project=project,
        run_identity=resolved_identity,
        state_path=path,
    )


def initialize_wandb_run(
    wandb_client: Any,
    *,
    project: str = DEFAULT_WANDB_PROJECT,
    run_name: str = "",
    run_identity: str = "",
    state_path: str | os.PathLike[str],
    explicit_run_id: str = "",
    mode: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> WandbRunState:
    """Initialize W&B with one persisted id and resumable semantics."""
    state = resolve_wandb_run_state(
        project=project,
        run_name=run_name,
        run_identity=run_identity,
        state_path=state_path,
        explicit_run_id=explicit_run_id,
    )
    init_kwargs = dict(
        project=state.project,
        id=state.run_id,
        name=run_name or None,
        resume=os.environ.get("WANDB_RESUME", "allow"),
        config=merge_opd_wandb_config(config),
    )
    if wandb_shared_run_enabled():
        init_kwargs["settings"] = wandb_process_settings(
            wandb_client,
            label="kl-trainer",
        )
    else:
        init_kwargs["mode"] = mode or os.environ.get("WANDB_MODE", "online")
    init_kwargs.update(wandb_init_metadata_from_env())
    wandb_client.init(**init_kwargs)
    define_metric = getattr(wandb_client, "define_metric", None)
    if callable(define_metric):
        define_metric("global_step")
        define_metric("*", step_metric="global_step")
    return state


def log_wandb_metrics(
    wandb_client: Any,
    metrics: Mapping[str, Any],
    *,
    global_step: int,
    commit: bool = True,
) -> dict[str, Any]:
    """Log metrics on a caller-supplied, explicit global-step timeline."""
    if isinstance(global_step, bool) or int(global_step) != global_step or global_step < 0:
        raise ValueError(f"global_step must be a non-negative integer, got {global_step!r}")
    global_step = int(global_step)
    payload = dict(metrics)
    payload["global_step"] = global_step
    # Do not pass the SDK's ``step=`` here. Train and eval deliberately log at
    # the same optimizer step; W&B rejects a second committed record at the
    # same SDK step. ``global_step`` is defined as the custom x-axis instead.
    wandb_client.log(payload, commit=commit)
    return payload


def _flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for raw_key, value in metrics.items():
        key = str(raw_key).strip().strip("/")
        if not key:
            raise ValueError("metric names must not be empty")
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_metrics(value, full_key))
        elif isinstance(value, (Number, str)) or value is None:
            flattened[full_key] = value
        else:
            raise TypeError(
                f"evaluation metric {full_key!r} must be scalar or a nested mapping, got {type(value).__name__}"
            )
    return flattened


def build_eval_metrics_payload(
    metrics: Mapping[str, Any],
    *,
    task: str = "",
    milestone_fraction: float | None = None,
) -> dict[str, Any]:
    """Flatten evaluator output under ``eval[/task]/...`` W&B keys."""
    task = task.strip().strip("/")
    prefix = f"eval/{task}" if task else "eval"
    flattened = _flatten_metrics(metrics)
    payload = {
        key if key.startswith("eval/") else f"{prefix}/{key}": value
        for key, value in flattened.items()
    }
    if milestone_fraction is not None:
        parsed = parse_eval_fractions([milestone_fraction])
        payload["eval/milestone_fraction"] = parsed[0]
    return payload


def log_eval_metrics(
    wandb_client: Any,
    metrics: Mapping[str, Any],
    *,
    global_step: int,
    task: str = "",
    milestone_fraction: float | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Log nested evaluator metrics into the shared W&B run."""
    payload = build_eval_metrics_payload(
        metrics,
        task=task,
        milestone_fraction=milestone_fraction,
    )
    payload["eval/global_step"] = int(global_step)
    return log_wandb_metrics(
        wandb_client,
        payload,
        global_step=global_step,
        commit=commit,
    )
