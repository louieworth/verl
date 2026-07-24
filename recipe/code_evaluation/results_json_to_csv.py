#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

META_COLUMNS = ["model", "task", "is_lora", "teacher", "y_mode", "kl_type", "kl_method", "clip_ratio", "top_K", "multi_turn"]
DATASETS = ["humaneval_plus", "mbpp_plus", "livecodebench_v6"]


def fieldnames(pass_k: int) -> list[str]:
    return META_COLUMNS + [f"{d}_Avg@{pass_k}" for d in DATASETS] + [f"{d}_Pass@{pass_k}" for d in DATASETS]


def parse_meta(key: str) -> dict[str, str]:
    row = {c: "" for c in META_COLUMNS}
    row["model"] = key.split("_")[0] if key else ""
    row["task"] = "code" if "_CODE_" in key.upper() else ""
    u = key.upper()
    row["is_lora"] = "false" if "NO_LORA" in u or key.endswith("_base") else ("true" if "LORA" in u else "")
    m = re.search(r"(?:^|_)teacher([^_]+)", key)
    if m:
        row["teacher"] = m.group(1)
    if "_y_o_" in key:
        row["y_mode"] = "y_o"
    elif "_y_r_" in key:
        row["y_mode"] = "y_r"
    m = re.search(r"_kl_([^_]+)(?:_([^_]+(?:_[^_]+)*))?_clip", key)
    if m:
        row["kl_type"] = m.group(1)
        row["kl_method"] = m.group(2) or ""
    m = re.search(r"_clip([^_]+)", key)
    if m:
        raw = m.group(1)
        row["clip_ratio"] = f"0.{raw[1:].rstrip('0') or '0'}" if raw.isdigit() and raw.startswith("0") and raw != "0" else raw
    m = re.search(r"_topk([0-9]+)", key)
    row["top_K"] = m.group(1) if m else "0"
    m = re.search(r"_ms([0-9]+)(?:_|$)", key)
    if m:
        row["multi_turn"] = m.group(1)
    return row


def fmt(v: Any) -> Any:
    return f"{v:.10g}" if isinstance(v, float) else v


def build_rows(results: dict[str, dict[str, Any]], pass_k: int = 4) -> list[dict[str, Any]]:
    rows = []
    for key, entry in results.items():
        row = parse_meta(key)
        for d in DATASETS:
            row[f"{d}_Avg@{pass_k}"] = fmt(
                entry.get(f"{d}_avg{pass_k}", entry.get(f"{d}_Avg@{pass_k}", ""))
            )
        for d in DATASETS:
            row[f"{d}_Pass@{pass_k}"] = fmt(
                entry.get(f"{d}_pass{pass_k}", entry.get(f"{d}_Pass@{pass_k}", ""))
            )
        rows.append(row)
    return rows



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--pass_k", type=int, default=int(os.environ.get("PASS_K", "4")))
    args = parser.parse_args()
    if args.pass_k <= 0:
        parser.error("--pass_k must be positive")
    output_file = args.output_file or str(Path(args.results_file).with_suffix(".csv"))
    with open(args.results_file) as f:
        data = json.load(f)
    rows = build_rows(data, args.pass_k)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames(args.pass_k))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV comparison: {output_file}")


if __name__ == "__main__":
    main()
