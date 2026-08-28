#!/usr/bin/env python3
"""Stream a math benchmark into an existing W&B run.

The benchmark remains a normal subprocess, so its terminal output is unchanged.
This wrapper keeps W&B active for the full subprocess lifetime, captures that
output in the W&B Logs panel, and turns structured progress lines into metrics.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROGRESS_PREFIX = "[EVAL_PROGRESS] "


def parse_progress_line(line: str) -> dict[str, Any] | None:
    """Parse one structured progress line emitted by ``eval_utils.py``."""
    position = line.find(PROGRESS_PREFIX)
    if position < 0:
        return None
    try:
        event = json.loads(line[position + len(PROGRESS_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


class EvalProgressTracker:
    """Track the current eval location without adding W&B chart metrics."""

    def __init__(self, *, global_step: int, milestone_fraction: float | None):
        self.global_step = global_step
        self.milestone_fraction = milestone_fraction
        self.started_at = time.monotonic()
        self.current_dataset = ""
        self.current_phase = "starting"
        self.terminal_tail = ""

    def observe(self, event: Mapping[str, Any]) -> None:
        phase = str(event.get("phase", "progress"))
        dataset = str(event.get("dataset", self.current_dataset))
        self.current_phase = phase
        self.current_dataset = dataset

    def emit(self, event: Mapping[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("global_step", self.global_step)
        if self.milestone_fraction is not None:
            payload.setdefault("milestone_fraction", self.milestone_fraction)
        payload.setdefault("elapsed_seconds", time.monotonic() - self.started_at)
        self.observe(payload)
        print(f"{PROGRESS_PREFIX}{json.dumps(payload, sort_keys=True, separators=(',', ':'))}", flush=True)

    def fail(self, exit_code: int, error: str = "") -> None:
        event: dict[str, Any] = {
            "phase": "failed",
            "dataset": self.current_dataset,
            "exit_code": exit_code,
        }
        if error:
            event["error"] = error
        self.emit(event)

    def observe_terminal(self, text: str) -> None:
        self.terminal_tail = (self.terminal_tail + text)[-4000:]


def _consume_progress_text(text: str, pending: str, progress: EvalProgressTracker) -> str:
    normalized = (pending + text).replace("\r", "\n")
    lines = normalized.split("\n")
    for line in lines[:-1]:
        if event := parse_progress_line(line):
            progress.observe(event)
    return lines[-1]


def stream_command(
    command: Sequence[str],
    progress: EvalProgressTracker,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run a command while immediately mirroring its combined output."""
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=dict(env) if env is not None else None,
        bufsize=0,
    )
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    while chunk := os.read(process.stdout.fileno(), 4096):
        text = decoder.decode(chunk)
        sys.stdout.write(text)
        sys.stdout.flush()
        progress.observe_terminal(text)
        pending = _consume_progress_text(text, pending, progress)
    tail = decoder.decode(b"", final=True)
    if tail:
        sys.stdout.write(tail)
        sys.stdout.flush()
        progress.observe_terminal(tail)
        pending = _consume_progress_text(tail, pending, progress)
    if pending and (event := parse_progress_line(pending)):
        progress.observe(event)
    return process.wait()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark with live W&B console and progress logging")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a benchmark command is required after --")
    return args


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for live eval W&B logging")
    return value


def _optional_fraction() -> float | None:
    raw = os.environ.get("EVAL_MILESTONE_FRACTION", "").strip()
    return float(raw) if raw else None


def _metrics_file() -> Path | None:
    if explicit := os.environ.get("EVAL_METRICS_FILE", "").strip():
        return Path(explicit)
    if output_dir := os.environ.get("EVAL_OUTPUT_DIR", "").strip():
        return Path(output_dir) / "metrics.json"
    return None


def main() -> None:
    args = _parse_args()
    project = _required_env("WANDB_PROJECT")
    run_id = _required_env("WANDB_RUN_ID")
    global_step = int(_required_env("WANDB_GLOBAL_STEP"))
    milestone_fraction = _optional_fraction()

    import wandb

    from verl.utils.wandb_metadata import merge_opd_wandb_config, wandb_init_metadata_from_env

    init_kwargs = {
        "project": project,
        "entity": os.environ.get("WANDB_ENTITY") or None,
        "id": run_id,
        "name": os.environ.get("WANDB_RUN_NAME") or None,
        "resume": os.environ.get("WANDB_RESUME", "allow"),
        "mode": os.environ.get("WANDB_MODE", "online"),
        "reinit": True,
        "config": merge_opd_wandb_config(None),
        "settings": wandb.Settings(console=os.environ.get("WANDB_CONSOLE", "wrap")),
    }
    init_kwargs.update(wandb_init_metadata_from_env())
    run = wandb.init(**init_kwargs)
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")

    progress = EvalProgressTracker(
        global_step=global_step,
        milestone_fraction=milestone_fraction,
    )
    progress.emit({"phase": "started", "dataset": ""})
    child_env = os.environ.copy()
    child_env["EVAL_WANDB_WRAPPED"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"

    exit_code = 1
    try:
        exit_code = stream_command(args.command, progress, env=child_env)
        if exit_code != 0:
            progress.fail(exit_code, progress.terminal_tail.strip()[-2000:])
            raise subprocess.CalledProcessError(exit_code, args.command)

        metrics_file = _metrics_file()
        if metrics_file is not None and metrics_file.is_file():
            from recipe.math_evaluation.log_metrics_wandb import load_wandb_payload

            _, payload = load_wandb_payload(
                str(metrics_file),
                milestone_fraction,
                global_step,
            )
            payload["global_step"] = global_step
            run.log(payload)
        progress.emit({"phase": "complete", "dataset": "", "exit_code": 0, "overall_fraction": 1.0})
        exit_code = 0
    except BaseException as exc:
        if progress.current_phase != "failed":
            progress.fail(exit_code, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
