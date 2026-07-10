#!/usr/bin/env python3
"""Sample a fixed number of rows from a parquet file without replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet path.")
    parser.add_argument("--output", required=True, help="Output sampled parquet path.")
    parser.add_argument("--num-rows", type=int, required=True, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--compression", default="snappy", help="Output parquet compression.")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional JSON metadata path. Defaults to <output>.metadata.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(input_path)
    total_rows = parquet.metadata.num_rows
    if args.num_rows < 0:
        raise ValueError("--num-rows must be non-negative")
    if args.num_rows > total_rows:
        raise ValueError(f"requested {args.num_rows} rows, but input only has {total_rows}")

    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(total_rows, size=args.num_rows, replace=False))

    writer: pq.ParquetWriter | None = None
    selected_offset = 0
    row_start = 0
    rows_written = 0
    try:
        for row_group_idx in range(parquet.num_row_groups):
            row_group_rows = parquet.metadata.row_group(row_group_idx).num_rows
            row_end = row_start + row_group_rows

            group_start = np.searchsorted(selected, row_start, side="left", sorter=None)
            group_end = np.searchsorted(selected, row_end, side="left", sorter=None)
            if group_end > group_start:
                if group_start != selected_offset:
                    raise RuntimeError("internal selected-row offset mismatch")
                local_indices = selected[group_start:group_end] - row_start
                table = parquet.read_row_group(row_group_idx)
                sampled = table.take(pa.array(local_indices, type=pa.int64()))
                if writer is None:
                    writer = pq.ParquetWriter(output_path, sampled.schema, compression=args.compression)
                writer.write_table(sampled)
                rows_written += sampled.num_rows
                selected_offset = group_end

            row_start = row_end
    finally:
        if writer is not None:
            writer.close()

    if rows_written != args.num_rows:
        raise RuntimeError(f"sampled row count mismatch: expected {args.num_rows}, wrote {rows_written}")

    metadata_path = Path(args.metadata_output) if args.metadata_output else Path(f"{output_path}.metadata.json")
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "seed": args.seed,
        "input_rows": total_rows,
        "sampled_rows": rows_written,
        "fraction": rows_written / total_rows if total_rows else 0.0,
        "compression": args.compression,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
