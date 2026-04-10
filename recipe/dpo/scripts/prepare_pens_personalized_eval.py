#!/usr/bin/env python3
"""Prepare prompt-aligned PENS personalized headline evaluation examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pens_eval_utils import build_personalized_eval_records, dump_jsonl, load_news_table, render_prompt_for_model


SERIALIZED_COLUMNS = {"prompt", "clicked_news_ids", "clicked_titles", "gold_headlines"}
PROMPT_SCHEMA = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
STRING_LIST_SCHEMA = pa.list_(pa.string())
PARQUET_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("news_id", pa.string()),
        ("clicked_news_ids", STRING_LIST_SCHEMA),
        ("clicked_titles", STRING_LIST_SCHEMA),
        ("candidate_category", pa.string()),
        ("candidate_topic", pa.string()),
        ("candidate_title", pa.string()),
        ("candidate_body", pa.string()),
        ("history_count", pa.int64()),
        ("missing_history_count", pa.int64()),
        ("gold_headlines", STRING_LIST_SCHEMA),
        ("prompt", PROMPT_SCHEMA),
        ("gold_headline", pa.string()),
        ("gold_headline_count", pa.int64()),
        ("source_test_file", pa.string()),
        ("source_rows_seen", pa.int64()),
        ("rendered_prompt", pa.string()),
    ]
)


def can_reuse_output(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix.lower() != ".parquet":
        return True

    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception:
        return False

    if "prompt" not in schema.names:
        return False
    prompt_type = schema.field("prompt").type
    return pa.types.is_list(prompt_type) or pa.types.is_large_list(prompt_type)


def save_records(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".jsonl":
        dump_jsonl(output, records)
        return

    if suffix == ".parquet":
        table = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)
        pq.write_table(table, output)
        return

    table_records: list[dict] = []
    for record in records:
        item = dict(record)
        for column in SERIALIZED_COLUMNS:
            if column in item:
                item[column] = json.dumps(item[column], ensure_ascii=False)
        table_records.append(item)

    frame = pd.DataFrame(table_records)
    if suffix in {".tsv", ".txt"}:
        frame.to_csv(output, sep="\t", index=False)
    elif suffix == ".csv":
        frame.to_csv(output, index=False)
    else:
        raise ValueError(f"Unsupported output suffix: {output.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-file", type=Path, required=True, help="PENS news.tsv or news.csv")
    parser.add_argument("--test-file", type=Path, required=True, help="PENS personalized_test.tsv")
    parser.add_argument("--output", type=Path, required=True, help="Output .jsonl/.parquet/.csv/.tsv path")
    parser.add_argument("--model-path", type=str, default=None, help="Optional model path for chat-template rendering")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-examples", type=int, default=-1, help="Optional cap for debugging")
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Reuse an existing prepared evaluation file instead of rebuilding it.",
    )
    args = parser.parse_args()

    if args.skip_if_exists and can_reuse_output(args.output):
        print(
            json.dumps(
                {
                    "reused_existing_output": True,
                    "output": str(args.output),
                    "model_path": args.model_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    news_lookup = load_news_table(args.news_file)
    records = build_personalized_eval_records(
        news_lookup=news_lookup,
        test_file=args.test_file,
        max_examples=args.max_examples,
    )

    if args.model_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        for record in records:
            record["rendered_prompt"] = render_prompt_for_model(
                tokenizer=tokenizer,
                prompt_messages=record["prompt"],
                add_generation_prompt=True,
            )

    save_records(records, args.output)

    print(
        json.dumps(
            {
                "news_count": len(news_lookup),
                "example_count": len(records),
                "output": str(args.output),
                "model_path": args.model_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
