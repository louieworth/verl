#!/usr/bin/env python3
"""
Build a reward-filtered DeepScaleR dataset from Stage-1 evaluation outputs.

This script keeps the source dataset format unchanged by selecting rows from
the original DeepScaleR dataset using `extra_info.index` from the stage-1
responses parquet.

Example:
  python3 recipe/open_math_reasoning/build_reward_filtered_dataset.py \
    --model_name Qwen3-1.7B \
    --stage1_parquet gen_results/Qwen3-1.7B/epoch1/deepscaleR_stage1_responses.parquet \
    --source_dataset /data/data/jiangli/huggingface/datasets/DeepScaleR-Cleaned \
    --output_root /data/data/jiangli/data \
    --target_reward 0 \
    --overwrite
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Tuple

import datasets
import numpy as np
import pandas as pd


def _load_source_dataset(path_or_name: str, split: str):
    """Load a HuggingFace dataset from disk path first, then dataset name."""
    try:
        loaded = datasets.load_from_disk(path_or_name)
        if isinstance(loaded, datasets.DatasetDict):
            if split not in loaded:
                raise ValueError(
                    f"Split '{split}' not found in source DatasetDict. "
                    f"Available: {list(loaded.keys())}"
                )
            return loaded[split]
        return loaded
    except (FileNotFoundError, ValueError):
        return datasets.load_dataset(path_or_name, split=split)


def _to_float_reward(value: Any, reward_index: int) -> float:
    """Convert raw reward value (scalar/list/ndarray) to a float."""
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist()
        if not flat:
            raise ValueError("Empty reward ndarray")
        if reward_index >= len(flat):
            raise ValueError(
                f"reward_index={reward_index} out of range for ndarray "
                f"with length={len(flat)}"
            )
        return float(flat[reward_index])

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty reward list/tuple")
        if reward_index >= len(value):
            raise ValueError(
                f"reward_index={reward_index} out of range for list/tuple "
                f"with length={len(value)}"
            )
        return float(value[reward_index])

    return float(value)


def _collect_indices(
    stage1_df: pd.DataFrame,
    target_reward: float,
    reward_index: int,
    allow_missing: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    """Collect source indices whose reward equals target_reward."""
    matched_indices: List[int] = []
    missing_rows = 0
    bad_reward_rows = 0

    for row_i, row in stage1_df.iterrows():
        extra = row.get("extra_info", {})
        if not isinstance(extra, dict):
            if allow_missing:
                missing_rows += 1
                continue
            raise ValueError(f"Row {row_i}: extra_info is not a dict")

        if "reward" not in extra or "index" not in extra:
            if allow_missing:
                missing_rows += 1
                continue
            raise ValueError(
                f"Row {row_i}: missing 'reward' or 'index' in extra_info"
            )

        try:
            reward_value = _to_float_reward(extra["reward"], reward_index)
        except Exception:
            if allow_missing:
                bad_reward_rows += 1
                continue
            raise

        if reward_value == target_reward:
            matched_indices.append(int(extra["index"]))

    unique_sorted = sorted(set(matched_indices))
    stats = {
        "stage1_rows": int(len(stage1_df)),
        "matched_rows_before_dedup": int(len(matched_indices)),
        "matched_rows_after_dedup": int(len(unique_sorted)),
        "missing_rows": int(missing_rows),
        "bad_reward_rows": int(bad_reward_rows),
    }
    return unique_sorted, stats


def _resolve_output_path(args: argparse.Namespace) -> str:
    if args.output_path:
        return args.output_path
    if not args.model_name:
        raise ValueError("Either --output_path or --model_name must be provided")
    output_name = args.output_name_template.format(model_name=args.model_name)
    return os.path.join(args.output_root, output_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reward-filtered DeepScaleR dataset for reuse."
    )
    parser.add_argument(
        "--stage1_parquet",
        required=True,
        help="Stage-1 evaluated parquet with extra_info.reward and extra_info.index",
    )
    parser.add_argument(
        "--source_dataset",
        default="/data/data/jiangli/huggingface/datasets/DeepScaleR-Cleaned",
        help="Original DeepScaleR dataset path or HF dataset name",
    )
    parser.add_argument(
        "--source_split",
        default="train",
        help="Split to use when source_dataset is a DatasetDict",
    )
    parser.add_argument(
        "--target_reward",
        type=float,
        default=0.0,
        help="Keep rows where reward equals this value (default: 0.0)",
    )
    parser.add_argument(
        "--reward_index",
        type=int,
        default=0,
        help="Reward index for pass@k outputs stored as arrays/lists (default: 0)",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Model name used by --output_name_template when --output_path is not set",
    )
    parser.add_argument(
        "--output_root",
        default="/data/data/jiangli/data",
        help="Root directory for auto-generated output path",
    )
    parser.add_argument(
        "--output_name_template",
        default="{model_name}_fliter_deepsclarR",
        help="Template used with --model_name to generate output directory name",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output dataset path. If set, overrides model_name/output_root/template",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip rows with missing/invalid reward metadata instead of failing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_path if it already exists",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print stats without writing dataset",
    )
    args = parser.parse_args()

    output_path = _resolve_output_path(args)
    print(f"Stage1 parquet: {args.stage1_parquet}")
    print(f"Source dataset: {args.source_dataset}")
    print(f"Target reward: {args.target_reward} (reward_index={args.reward_index})")
    print(f"Output path: {output_path}")
    print(f"Dry run: {args.dry_run}")

    stage1_df = pd.read_parquet(args.stage1_parquet)
    indices, collect_stats = _collect_indices(
        stage1_df=stage1_df,
        target_reward=args.target_reward,
        reward_index=args.reward_index,
        allow_missing=args.allow_missing,
    )
    print("Collected indices:")
    print(
        f"  matched rows: {collect_stats['matched_rows_before_dedup']} "
        f"(unique: {collect_stats['matched_rows_after_dedup']})"
    )
    print(
        f"  missing rows: {collect_stats['missing_rows']}, "
        f"bad reward rows: {collect_stats['bad_reward_rows']}"
    )

    source_ds = _load_source_dataset(args.source_dataset, args.source_split)
    total_source_rows = len(source_ds)
    if not indices:
        raise ValueError("No rows matched target_reward; nothing to save")

    out_of_range = [i for i in indices if i < 0 or i >= total_source_rows]
    if out_of_range:
        raise ValueError(
            f"Found out-of-range source indices. First few: {out_of_range[:5]}"
        )

    filtered_ds = source_ds.select(indices)
    print(f"Source rows: {total_source_rows}")
    print(f"Filtered rows: {len(filtered_ds)}")
    print(f"Columns: {filtered_ds.column_names}")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage1_parquet": args.stage1_parquet,
        "source_dataset": args.source_dataset,
        "source_split": args.source_split,
        "target_reward": args.target_reward,
        "reward_index": args.reward_index,
        "output_path": output_path,
        "source_rows": int(total_source_rows),
        "filtered_rows": int(len(filtered_ds)),
        "filtered_ratio": float(len(filtered_ds) / max(1, len(stage1_df))),
        "columns": filtered_ds.column_names,
        **collect_stats,
    }

    if args.dry_run:
        print("Dry run complete. No files were written.")
        print(json.dumps(metadata, indent=2))
        return

    if os.path.exists(output_path):
        if not args.overwrite:
            raise FileExistsError(
                f"Output path exists: {output_path}. Use --overwrite to replace."
            )
        shutil.rmtree(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_ds.save_to_disk(output_path)

    meta_path = os.path.join(output_path, "FILTER_META.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved filtered dataset to: {output_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
