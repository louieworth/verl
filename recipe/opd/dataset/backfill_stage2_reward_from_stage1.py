#!/usr/bin/env python3
"""Retroactively backfill stage2 parquet's extra_info.reward from stage1.

The current stage2 parquet was generated before stage1 responses were scored,
so every row has extra_info.reward == 0 (inherited as a placeholder). This
script rebuilds a new stage2 parquet where extra_info.reward reflects the true
stage1 pass@1 outcome, cross-referenced via extra_info.index.

Neither input file is modified. Writes a new parquet at --output.

Stage1 parquet can be pre-scored (has extra_info.reward) or raw. If raw, the
script scores stage1 in memory using the same helpers as score_stage1_reward.py
(nothing written back).
"""
import argparse
import os
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


def _build_stage1_reward_map(stage1_df: pd.DataFrame) -> dict:
    has_reward = stage1_df["extra_info"].apply(
        lambda x: isinstance(x, dict) and "reward" in x
    ).all()
    if has_reward:
        print("[backfill] stage1 already scored; using existing extra_info.reward")
        rewards = stage1_df["extra_info"].apply(lambda x: float(x.get("reward", 0)))
    else:
        print("[backfill] stage1 not scored; scoring in memory (no write-back to stage1)")
        rewards = stage1_df.apply(_score_row, axis=1).astype(float)
        print(
            f"[backfill] stage1 pass@1 = {(rewards >= 1).mean():.4f} "
            f"({int((rewards >= 1).sum())}/{len(stage1_df)})"
        )

    indices = stage1_df["extra_info"].apply(lambda x: x.get("index"))
    reward_map = {}
    for idx, r in zip(indices, rewards):
        if idx is None:
            continue
        reward_map[int(idx)] = int(r) if float(r).is_integer() else float(r)
    return reward_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_parquet", required=True)
    ap.add_argument("--stage2_parquet", required=True)
    ap.add_argument("--output", required=True, help="New stage2 parquet path (must not equal stage2_parquet)")
    args = ap.parse_args()

    if os.path.abspath(args.output) == os.path.abspath(args.stage2_parquet):
        raise SystemExit("--output must differ from --stage2_parquet; refusing to overwrite input.")

    stage1_df = pd.read_parquet(args.stage1_parquet)
    print(f"[backfill] stage1 rows: {len(stage1_df)}")
    reward_map = _build_stage1_reward_map(stage1_df)
    print(f"[backfill] stage1 index->reward map size: {len(reward_map)}")

    stage2_df = pd.read_parquet(args.stage2_parquet)
    print(f"[backfill] stage2 rows: {len(stage2_df)}")

    missing = 0
    changed = 0
    new_extra_info = []
    for ei in stage2_df["extra_info"]:
        ei = dict(ei) if isinstance(ei, dict) else {}
        idx = ei.get("index")
        if idx is None or int(idx) not in reward_map:
            missing += 1
            new_extra_info.append(ei)
            continue
        new_reward = reward_map[int(idx)]
        if ei.get("reward") != new_reward:
            ei["reward"] = new_reward
            changed += 1
        new_extra_info.append(ei)
    stage2_df["extra_info"] = new_extra_info

    # Report the new stage1_reward distribution (as seen from stage2 perspective)
    s1r = stage2_df["extra_info"].apply(lambda x: float(x.get("reward", 0)))
    print(
        f"[backfill] rows with stage1 reward found: {len(stage2_df) - missing}"
        f" (missing: {missing}, changed: {changed})"
    )
    print(
        f"[backfill] stage1 reward==1: {int((s1r == 1).sum())}, "
        f"stage1 reward==0: {int((s1r == 0).sum())}"
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".parquet", prefix=".backfill_", dir=out_dir or None
    )
    os.close(tmp_fd)
    try:
        stage2_df.to_parquet(tmp_path)
        os.chmod(tmp_path, 0o664)
        os.replace(tmp_path, args.output)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print(f"[backfill] wrote {len(stage2_df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
