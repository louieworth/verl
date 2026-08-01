#!/usr/bin/env python3
"""Prepare fixed OPSD teacher-rollout prompts for y_t ~ pi(. | x, y*).

The output keeps the standard stage-1 schema and replaces only ``prompt`` with
the expert-conditioned rewrite prompt used by the OPSD KL teacher.  It can
either build the full prompt parquet from a DeepScaleR save_to_disk dataset or
slice an already prepared parquet for one multi-step update.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DatasetDict, load_from_disk

from recipe.opd.dataset.data_utils import build_teacher_prompt
from recipe.opd.generation.y_o_prepare import make_map_fn_stage1


CONDITIONING_TAG = "opsd_x_y_star_expert_rewrite_v1"


def _as_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    raise ValueError(f"Expected prompt messages, got {type(value).__name__}")


def _extra_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected extra_info dict, got {type(value).__name__}")
    return dict(value)


def is_prepared_teacher_prompt_frame(frame: pd.DataFrame) -> bool:
    if frame.empty or "extra_info" not in frame:
        return False
    info = _extra_info(frame.iloc[0]["extra_info"])
    return info.get("trajectory_conditioning") == CONDITIONING_TAG


def build_teacher_prompt_frame(stage1: pd.DataFrame, *, task: str = "math") -> pd.DataFrame:
    canonical_columns = [
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    ]
    missing = set(canonical_columns) - set(stage1.columns)
    if missing:
        raise ValueError(f"Stage-1 prompt parquet is missing columns: {sorted(missing)}")

    output = stage1.loc[:, canonical_columns].copy(deep=True)
    prompts: list[list[dict[str, str]]] = []
    extra_infos: list[dict[str, Any]] = []

    for row_index, row in stage1.iterrows():
        info = _extra_info(row["extra_info"])
        problem = str(info.get("problem", "")).strip()
        expert_solution = str(info.get("expert_cot", "")).strip()
        if not problem:
            raise ValueError(f"Row {row_index} has no extra_info.problem")
        if not expert_solution:
            raise ValueError(f"Row {row_index} has no non-empty extra_info.expert_cot")

        prompt = build_teacher_prompt(
            problem,
            expert_solution,
            use_initial_response=False,
            distill_mode="opsd",
            task=task,
        )
        prompts.append([{"role": "user", "content": prompt}])
        info["trajectory_conditioning"] = CONDITIONING_TAG
        info["trajectory_prompt_source"] = info.get(
            "expert_solution_source",
            "solution",
        )
        extra_infos.append(info)

    output["prompt"] = prompts
    output["extra_info"] = extra_infos
    return output


def build_from_source(source_path: str, *, task: str, data_source: str) -> pd.DataFrame:
    if task != "math":
        raise ValueError("Building directly from a source dataset currently supports task=math only")

    loaded = load_from_disk(source_path)
    source = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    required = {"problem", "answer", "solution"}
    missing = required - set(source.column_names)
    if missing:
        raise ValueError(f"DeepScaleR source is missing columns: {sorted(missing)}")

    stage1 = source.map(
        make_map_fn_stage1(data_source=data_source),
        with_indices=True,
        remove_columns=source.column_names,
        keep_in_memory=True,
        desc="Building base OPSD stage-1 schema",
    )
    return build_teacher_prompt_frame(stage1.to_pandas(), task=task)


def slice_frame(frame: pd.DataFrame, *, start_index: int, num_samples: int | None) -> pd.DataFrame:
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be > 0 when provided")
    stop = len(frame) if num_samples is None else start_index + num_samples
    output = frame.iloc[start_index:stop].reset_index(drop=True)
    if num_samples is not None and len(output) != num_samples:
        raise ValueError(
            f"Requested {num_samples} rows from offset {start_index}, got {len(output)}"
        )
    return output


def validate_alignment(frame: pd.DataFrame, alignment_path: str) -> None:
    alignment = pd.read_parquet(alignment_path)
    if len(frame) != len(alignment):
        raise ValueError(
            f"Teacher prompt/alignment row mismatch: {len(frame)} != {len(alignment)}"
        )
    for position in range(len(frame)):
        teacher_info = _extra_info(frame.iloc[position]["extra_info"])
        stage1_info = _extra_info(alignment.iloc[position]["extra_info"])
        for key in ("index", "problem"):
            if teacher_info.get(key) != stage1_info.get(key):
                raise ValueError(
                    f"Teacher prompt alignment mismatch at row {position} for {key!r}"
                )


def write_atomic(frame: pd.DataFrame, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp.parquet",
        dir=str(output.parent),
    )
    os.close(fd)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="DeepScaleR save_to_disk directory, stage-1 parquet, or prepared teacher-prompt parquet",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["math", "code"], default="math")
    parser.add_argument("--data-source", default="deepscaleR")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument(
        "--alignment-input",
        default="",
        help="Optional stage-1 prompt parquet whose rows must align with the output slice",
    )
    args = parser.parse_args()

    source_path = Path(args.input)
    if source_path.is_file() and source_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source_path)
        if not is_prepared_teacher_prompt_frame(frame):
            frame = build_teacher_prompt_frame(frame, task=args.task)
    else:
        frame = build_from_source(
            str(source_path),
            task=args.task,
            data_source=args.data_source,
        )

    frame = slice_frame(
        frame,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )
    if args.alignment_input:
        validate_alignment(frame, args.alignment_input)
    write_atomic(frame, args.output)

    first_prompt = _as_messages(frame.iloc[0]["prompt"])[0]["content"] if len(frame) else ""
    expected_expert_marker = (
        "Here is a reference Python solution:"
        if args.task == "code"
        else "**Expert Solution:**"
    )
    if len(frame) and (
        expected_expert_marker not in first_prompt
        or "**Your Initial Solution:**" in first_prompt
    ):
        raise ValueError("Prepared prompt is not the expected pi(. | x, y*) expert rewrite prompt")
    print(
        f"Wrote {len(frame)} OPSD teacher prompts "
        f"({CONDITIONING_TAG}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
