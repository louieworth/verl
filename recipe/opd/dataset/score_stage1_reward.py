#!/usr/bin/env python3
"""Score stage1 responses and backfill extra_info.reward in-place.

Idempotent: if every row already has extra_info.reward, exit without rewriting
(unless --force). Designed to be inserted between stage1 generation and stage2
prep in run_kl_training.sh. Required by T4 (LOG_DIFFICULTY_BUCKETS) and by
post-generation reward filtering (FORWARD_FILTER_STAGE2 / y_o stage1_reward_*).
"""
import argparse
import concurrent.futures
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow.lib

from recipe.math_evaluation.compute_score import compute_score_data_source
from verl.utils.reward_score import prime_code


def read_parquet_compat(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except pyarrow.lib.ArrowNotImplementedError:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        batches = [
            batch.to_pandas()
            for batch in parquet_file.iter_batches(batch_size=1024)
        ]
        if not batches:
            return pd.DataFrame()
        return pd.concat(batches, axis=0, ignore_index=True)


def _first_response(r) -> str:
    if isinstance(r, (list, np.ndarray)):
        return str(r[0]) if len(r) > 0 else ""
    return str(r) if r is not None else ""


def _ground_truth(rm):
    if isinstance(rm, dict):
        return rm.get("ground_truth")
    return None


def _is_code_row(row) -> bool:
    ability = str(row.get("ability", "")).lower()
    data_source = str(row.get("data_source", "")).lower()
    return ability == "code" or data_source in {"taco", "baai/taco", "apps", "codeforces", "codecontests"}


def _score_code_response(resp: str, gt) -> float:
    try:
        score, _metadata = prime_code.compute_score(resp, gt, continuous=False)
    except Exception:
        return 0.0
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _score_row(row) -> float:
    ds = row.get("data_source", "deepscaleR")
    resp = _first_response(row.get("responses"))
    gt = _ground_truth(row.get("reward_model"))
    if gt is None or not resp:
        return 0.0
    if _is_code_row(row):
        return _score_code_response(resp, gt)
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


def _score_rows_parallel(df: pd.DataFrame, workers: int, progress_interval: int) -> np.ndarray:
    records = df.to_dict("records")
    total = len(records)
    scores = np.zeros(total, dtype=float)
    started = time.monotonic()
    done = 0

    if workers <= 1 or total <= 1:
        for idx, record in enumerate(records):
            scores[idx] = _score_row(record)
            done += 1
            if progress_interval > 0 and (done % progress_interval == 0 or done == total):
                elapsed = time.monotonic() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                print(f"[score_stage1] progress {done}/{total} ({rate:.2f} rows/s)", flush=True)
        return scores

    print(f"[score_stage1] scoring with {workers} worker threads", flush=True)
    max_in_flight = max(workers, workers * 2)
    record_iter = iter(enumerate(records))
    future_to_idx = {}

    def submit_next(executor) -> bool:
        try:
            idx, record = next(record_iter)
        except StopIteration:
            return False
        future_to_idx[executor.submit(_score_row, record)] = idx
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(max_in_flight, total)):
            submit_next(executor)
        while future_to_idx:
            done_futures, _pending = concurrent.futures.wait(
                future_to_idx, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done_futures:
                idx = future_to_idx.pop(future)
                try:
                    scores[idx] = float(future.result())
                except Exception:
                    scores[idx] = 0.0
                done += 1
                submit_next(executor)
                if progress_interval > 0 and (done % progress_interval == 0 or done == total):
                    elapsed = time.monotonic() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    print(f"[score_stage1] progress {done}/{total} ({rate:.2f} rows/s)", flush=True)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Stage1 responses parquet (will be updated)")
    ap.add_argument("--output", default=None, help="Write to this path instead of overwriting --parquet")
    ap.add_argument("--force", action="store_true", help="Re-score even if reward already present")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SCORE_STAGE1_WORKERS", min(32, os.cpu_count() or 1))),
        help="Parallel score workers. Set SCORE_STAGE1_WORKERS to tune for the machine.",
    )
    ap.add_argument(
        "--progress-interval",
        type=int,
        default=int(os.environ.get("SCORE_STAGE1_PROGRESS_INTERVAL", "100")),
        help="Rows between progress logs; 0 disables progress logs.",
    )
    args = ap.parse_args()

    df = read_parquet_compat(args.parquet)
    print(f"[score_stage1] input rows: {len(df)}")

    if not args.force and _already_scored(df):
        print("[score_stage1] extra_info.reward already populated on all rows; skipping.")
        if args.output and os.path.abspath(args.output) != os.path.abspath(args.parquet):
            shutil.copyfile(args.parquet, args.output)
            print(f"[score_stage1] copied original to {args.output}")
        return

    scores = _score_rows_parallel(
        df,
        workers=max(1, int(args.workers)),
        progress_interval=max(0, int(args.progress_interval)),
    )
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
