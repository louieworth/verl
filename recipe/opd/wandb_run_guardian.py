#!/usr/bin/env python3
"""Own one shared W&B run until the complete train and eval pipeline exits."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from verl.utils.wandb_metadata import (
    merge_opd_wandb_config,
    wandb_init_metadata_from_env,
    wandb_process_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--control-file", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    return parser.parse_args()


def write_ready(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"{os.getpid()}\n", encoding="utf-8")
    temporary.replace(path)


def read_exit_code(path: Path) -> int | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    exit_code = int(document["exit_code"])
    if exit_code < 0:
        raise ValueError(f"exit_code must be non-negative, got {exit_code}")
    return exit_code


def main() -> None:
    args = parse_args()
    project = os.environ["WANDB_PROJECT"]
    run_id = os.environ["WANDB_RUN_ID"]
    run_name = os.environ.get("WANDB_RUN_NAME") or None
    ready_file = Path(args.ready_file)
    control_file = Path(args.control_file)
    ready_file.parent.mkdir(parents=True, exist_ok=True)

    import wandb

    init_kwargs = {
        "project": project,
        "entity": os.environ.get("WANDB_ENTITY") or None,
        "id": run_id,
        "name": run_name,
        "resume": os.environ.get("WANDB_RESUME", "allow"),
        "config": merge_opd_wandb_config(None),
        "settings": wandb_process_settings(
            wandb,
            primary=True,
            label="pipeline",
            console="off",
        ),
    }
    init_kwargs.update(wandb_init_metadata_from_env())
    run = wandb.init(**init_kwargs)
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    write_ready(ready_file)

    interrupted = False

    def request_failure(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, request_failure)
    signal.signal(signal.SIGINT, request_failure)

    exit_code = 1
    try:
        while not interrupted:
            requested_exit_code = read_exit_code(control_file)
            if requested_exit_code is not None:
                exit_code = requested_exit_code
                break
            time.sleep(args.poll_seconds)
    finally:
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
