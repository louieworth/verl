#!/usr/bin/env python3
"""Normalize a repo-local GRPO parquet for OPSD stage1 generation."""

from __future__ import annotations

import argparse
import os
import tempfile

import pandas as pd
import pyarrow.lib

from recipe.opd.generation.y_o_prepare import (
    _build_taco_prompt,
    instruction_following,
)


OUTPUT_COLUMNS = (
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
)
REQUIRED_COLUMNS = set(OUTPUT_COLUMNS)


def read_parquet_compat(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except pyarrow.lib.ArrowNotImplementedError:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        batches = [batch.to_pandas() for batch in parquet_file.iter_batches(batch_size=1024)]
        return pd.concat(batches, axis=0, ignore_index=True) if batches else pd.DataFrame()


def normalize_extra_info(value, index: int) -> dict:
    extra_info = dict(value) if isinstance(value, dict) else {}
    extra_info.setdefault("index", index)
    if not extra_info.get("expert_cot"):
        extra_info["expert_cot"] = str(extra_info.get("expert_solution") or "")
    return extra_info


def normalize_prompt(prompt, ability: str, extra_info: dict):
    problem = str(extra_info.get("problem") or "").strip()
    if not problem:
        return prompt
    if ability.lower() == "code":
        return [
            {
                "role": "user",
                "content": _build_taco_prompt(
                    {
                        "question": problem,
                        "starter_code": extra_info.get("starter_code", ""),
                    }
                ),
            }
        ]
    return [{"role": "user", "content": f"{problem} {instruction_following}"}]


def write_atomic(df: pd.DataFrame, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".best_of_n_prompts_",
        suffix=".parquet",
        dir=output_dir or None,
    )
    os.close(fd)
    try:
        df.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    dataset = read_parquet_compat(args.input)
    missing = REQUIRED_COLUMNS - set(dataset.columns)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        dataset = dataset.iloc[: args.max_samples].copy()
    if dataset.empty:
        raise ValueError("Input parquet is empty")

    dataset = dataset[list(OUTPUT_COLUMNS)].copy()
    math_rows = dataset["ability"].astype(str).str.lower().eq("math")
    deepscaler_rows = dataset["data_source"].astype(str).str.lower().eq("deepscaler")
    dataset.loc[math_rows & deepscaler_rows, "data_source"] = "deepscaleR"
    normalized_extra_info = [
        normalize_extra_info(value, index)
        for index, value in enumerate(dataset["extra_info"])
    ]
    dataset["extra_info"] = normalized_extra_info
    dataset["prompt"] = [
        normalize_prompt(prompt, str(ability), extra_info)
        for prompt, ability, extra_info in zip(
            dataset["prompt"],
            dataset["ability"],
            normalized_extra_info,
        )
    ]
    expert_rows = sum(bool(value.get("expert_cot")) for value in normalized_extra_info)
    write_atomic(dataset, args.output)
    print(
        f"Prepared {len(dataset)} OPSD prompts -> {args.output} "
        f"(rows with expert_cot: {expert_rows}/{len(dataset)})"
    )


if __name__ == "__main__":
    main()
