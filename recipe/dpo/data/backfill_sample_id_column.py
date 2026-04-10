#!/usr/bin/env python3
"""Backfill deterministic sample_id columns into existing point-wise DPO parquets."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import pyarrow as pa
import pyarrow.parquet as pq

from recipe.dpo.sample_id import DEFAULT_SAMPLE_ID_KEY, compute_sample_id_from_record


def infer_compression(parquet: pq.ParquetFile) -> str:
    if parquet.metadata is None or parquet.metadata.num_row_groups == 0 or parquet.metadata.row_group(0).num_columns == 0:
        return "zstd"
    raw = parquet.metadata.row_group(0).column(0).compression
    if isinstance(raw, str):
        return raw.lower()
    return str(raw).lower()


def append_sample_id_column(table: pa.Table, sample_id_key: str) -> pa.Table:
    if sample_id_key in table.column_names:
        table = table.remove_column(table.column_names.index(sample_id_key))
    sample_ids = [compute_sample_id_from_record(record) for record in table.to_pylist()]
    return table.append_column(sample_id_key, pa.array(sample_ids, type=pa.string()))


def backfill_parquet(path: Path, sample_id_key: str, compression: str | None) -> None:
    parquet = pq.ParquetFile(path)
    codec = compression or infer_compression(parquet)
    temp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    writer: pq.ParquetWriter | None = None
    try:
        for row_group_idx in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group_idx)
            table = append_sample_id_column(table, sample_id_key)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema, compression=codec)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Parquet files to patch in place")
    parser.add_argument("--sample-id-key", default=DEFAULT_SAMPLE_ID_KEY)
    parser.add_argument("--compression", default=None, help="Optional parquet compression override")
    args = parser.parse_args()

    for path in args.paths:
        if not path.exists():
            raise FileNotFoundError(path)
        backfill_parquet(path, sample_id_key=args.sample_id_key, compression=args.compression)
        print(f"Backfilled {args.sample_id_key} for {path}")


if __name__ == "__main__":
    main()
