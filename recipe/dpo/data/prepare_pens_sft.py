#!/usr/bin/env python3
"""Convert the positive-click PENS DPO parquet to an SFT-ready parquet.

Reads rows with columns (prompt, response, sample_id, ...) and writes rows
with a single `messages` column = prompt (list[{role,content}]) appended with
the assistant turn {"role": "assistant", "content": response}.

This output is consumed by verl's MultiTurnSFTDataset (config:
data.multiturn.enable=true, data.multiturn.messages_key=messages).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


MESSAGES_SCHEMA = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
OUTPUT_SCHEMA = pa.schema(
    [
        ("messages", MESSAGES_SCHEMA),
        ("sample_id", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input DPO positive parquet.")
    p.add_argument("--output", required=True, help="Output SFT parquet path.")
    p.add_argument("--prompt-key", default="prompt")
    p.add_argument("--response-key", default="response")
    p.add_argument("--sample-id-key", default="sample_id")
    p.add_argument("--compression", default="zstd")
    p.add_argument("--row-group-size", type=int, default=1024)
    p.add_argument("--label-key", default="label",
                   help="Optional label column to filter on (keeps rows where label > 0.5)")
    p.add_argument("--no-label-filter", action="store_true",
                   help="Do not filter by label (use all rows in input).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = pq.ParquetFile(input_path)
    writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression=args.compression)

    total_rows = reader.metadata.num_rows
    iterator = range(reader.num_row_groups)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="row-groups", unit="rg", total=reader.num_row_groups)

    kept_rows = 0
    for rg_idx in iterator:
        table = reader.read_row_group(rg_idx)

        if args.label_key in table.column_names and not args.no_label_filter:
            labels = table.column(args.label_key).to_pylist()
            keep_mask = [bool(v is not None and float(v) > 0.5) for v in labels]
            if not any(keep_mask):
                continue
            table = table.filter(pa.array(keep_mask))

        prompts = table.column(args.prompt_key).to_pylist()
        responses = table.column(args.response_key).to_pylist()
        sample_ids = (
            table.column(args.sample_id_key).to_pylist()
            if args.sample_id_key in table.column_names
            else [f"row_{rg_idx}_{i}" for i in range(table.num_rows)]
        )

        messages_rows: list[list[dict]] = []
        for prompt, response in zip(prompts, responses, strict=False):
            turns = []
            if prompt is None:
                prompt = []
            for turn in prompt:
                if not isinstance(turn, dict):
                    continue
                turns.append({"role": str(turn.get("role", "")), "content": str(turn.get("content", ""))})
            turns.append({"role": "assistant", "content": str(response or "")})
            messages_rows.append(turns)

        out_table = pa.Table.from_arrays(
            [
                pa.array(messages_rows, type=MESSAGES_SCHEMA),
                pa.array([str(s) for s in sample_ids], type=pa.string()),
            ],
            schema=OUTPUT_SCHEMA,
        )
        writer.write_table(out_table, row_group_size=args.row_group_size)
        kept_rows += out_table.num_rows

    writer.close()
    print(f"[prepare_pens_sft] input_rows={total_rows} kept_rows={kept_rows} -> {output_path}")


if __name__ == "__main__":
    main()
