#!/usr/bin/env python3
"""Score stage1 responses and backfill extra_info.reward in-place.

Idempotent: if every row already has extra_info.reward, exit without rewriting
(unless --force). Designed to be inserted between stage1 generation and stage2
prep in run_kl_training.sh. Required by T4 (LOG_DIFFICULTY_BUCKETS) and by
post-generation reward filtering (FORWARD_FILTER_STAGE2 / y_o stage1_reward_*).
"""
import argparse
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from recipe.math_evaluation.compute_score import compute_score_data_source


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


def _already_scored(df: pd.DataFrame) -> bool:
    def has_reward(ei):
        return isinstance(ei, dict) and "reward" in ei
    return bool(df["extra_info"].apply(has_reward).all())


def _inject(ei, score: float) -> dict:
    ei = dict(ei) if isinstance(ei, dict) else {}
    ei["reward"] = int(score) if float(score).is_integer() else float(score)
    return ei


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Stage1 responses parquet (will be updated)")
    ap.add_argument("--output", default=None, help="Write to this path instead of overwriting --parquet")
    ap.add_argument("--force", action="store_true", help="Re-score even if reward already present")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"[score_stage1] input rows: {len(df)}")

    if not args.force and _already_scored(df):
        print("[score_stage1] extra_info.reward already populated on all rows; skipping.")
        if args.output and os.path.abspath(args.output) != os.path.abspath(args.parquet):
            shutil.copyfile(args.parquet, args.output)
            print(f"[score_stage1] copied original to {args.output}")
        return

    scores = df.apply(_score_row, axis=1).astype(float)
    print(
        f"[score_stage1] pass@1 (reward>=1): {(scores >= 1).mean():.4f} "
        f"({int((scores >= 1).sum())}/{len(df)})"
    )

    df["extra_info"] = [_inject(ei, s) for ei, s in zip(df["extra_info"], scores)]

    out_path = args.output or args.parquet
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Write via tmp file + rename for atomicity when overwriting in-place
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".parquet", prefix=".score_stage1_", dir=out_dir or None
    )
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path)
        os.chmod(tmp_path, 0o664)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print(f"[score_stage1] wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
