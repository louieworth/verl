#!/usr/bin/env python3
"""Score stage2 rewrite responses (y_1) and keep rows with reward >= threshold.

Input parquet columns expected:
    data_source, prompt, reward_model(.ground_truth), extra_info, responses
Output parquet: same schema, with extra_info["stage2_reward"] added.
"""
import argparse
import os

import numpy as np
import pandas as pd

from recipe.open_math_reasoning.compute_score import compute_score_data_source


def _first_response(r) -> str:
    if isinstance(r, (list, np.ndarray)):
        return str(r[0]) if len(r) > 0 else ""
    return str(r) if r is not None else ""


def _ground_truth(rm):
    if isinstance(rm, dict):
        return rm.get("ground_truth")
    return None


def _score_row(row) -> float:
    ds = row.get("data_source", "deepscaleR")
    resp = _first_response(row.get("responses"))
    gt = _ground_truth(row.get("reward_model"))
    if gt is None or not resp:
        return 0.0
    try:
        return float(compute_score_data_source(ds, resp, gt))
    except Exception:
        return 0.0


def _stage1_reward(ei) -> float:
    if not isinstance(ei, dict):
        return 0.0
    val = ei.get("reward", 0)
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    try:
        return float(val)
    except Exception:
        return 0.0


def _inject_reward(ei, score: float) -> dict:
    ei = dict(ei) if isinstance(ei, dict) else {}
    ei["stage2_reward"] = float(score)
    return ei


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Stage2 generation parquet")
    ap.add_argument("--output", required=True, help="Filtered parquet path")
    ap.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Keep rows with stage2_reward >= threshold (default 1.0)",
    )
    ap.add_argument(
        "--require_stage1_failed",
        action="store_true",
        help="Also require extra_info.reward == 0 (stage1 failure). Use when "
        "filtering a rewrite_all parquet to emulate reward0_only + stage2 filter.",
    )
    ap.add_argument(
        "--skip_stage2_score",
        action="store_true",
        help="Skip y_1 scoring and filter only by stage1 reward (requires "
        "--require_stage1_failed). Fast path for producing a stage1-fail-only split.",
    )
    args = ap.parse_args()

    if args.skip_stage2_score and not args.require_stage1_failed:
        raise SystemExit("--skip_stage2_score currently requires --require_stage1_failed")

    df = pd.read_parquet(args.input)
    print(f"[filter_stage2] input rows: {len(df)}")

    if args.skip_stage2_score:
        print("[filter_stage2] skipping y_1 scoring (stage1-only filter)")
        keep_mask = pd.Series([True] * len(df), index=df.index)
    else:
        scores = df.apply(_score_row, axis=1).astype(float)
        df["extra_info"] = [_inject_reward(ei, s) for ei, s in zip(df["extra_info"], scores)]
        keep_mask = scores >= args.threshold
    if args.require_stage1_failed:
        stage1_rewards = df["extra_info"].apply(_stage1_reward).astype(float)
        stage1_failed = stage1_rewards == 0.0
        print(
            f"[filter_stage2] stage1_reward==0 rate: "
            f"{stage1_failed.mean():.4f} ({int(stage1_failed.sum())}/{len(df)})"
        )
        keep_mask = keep_mask & stage1_failed
    kept = int(keep_mask.sum())
    print(
        f"[filter_stage2] final keep rate: "
        f"{keep_mask.mean():.4f} ({kept}/{len(df)})"
    )
    if kept == 0:
        raise RuntimeError(
            "No rows pass the filter threshold; aborting so training is not fed empty data."
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df[keep_mask].reset_index(drop=True).to_parquet(args.output)
    print(f"[filter_stage2] wrote {kept} rows -> {args.output}")


if __name__ == "__main__":
    main()
