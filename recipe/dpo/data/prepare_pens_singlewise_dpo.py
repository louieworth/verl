#!/usr/bin/env python3
"""Build offline single-wise DPO parquet datasets from public PENS logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recipe.dpo.sample_id import DEFAULT_SAMPLE_ID_KEY, compute_sample_id

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


PROMPT_SCHEMA = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
OUTPUT_SCHEMA = pa.schema(
    [
        ("prompt", PROMPT_SCHEMA),
        ("response", pa.string()),
        ("label", pa.float32()),
        ("s_dwell", pa.float32()),
        ("p_ctr", pa.float32()),
        (DEFAULT_SAMPLE_ID_KEY, pa.string()),
        ("user_id", pa.string()),
        ("candidate_news_id", pa.string()),
        ("candidate_category", pa.string()),
        ("candidate_topic", pa.string()),
        ("history_count", pa.int32()),
        ("split", pa.string()),
        ("data_source", pa.string()),
    ]
)

MAX_HISTORY_ITEMS = 30
PROMPT_INSTRUCTION = (
    "Generate a personalized news headline for this user based on the user's reading interests "
    "and the candidate news article.\n"
    "The click history only contains clicked news titles."
)


def detect_delimiter(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    if suffix == ".csv":
        return ","
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    return "\t" if sample.count("\t") >= sample.count(",") else ","


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def split_id_list(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    if "#TAB#" in text:
        parts = text.split("#TAB#")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = text.split()
    return [part.strip() for part in parts if part and part.strip()]


def split_float_list(value: str) -> list[float]:
    weights: list[float] = []
    for token in split_id_list(value):
        try:
            weights.append(float(token))
        except ValueError:
            weights.append(0.0)
    return weights


def clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def pad_or_trim_float_list(values: list[float], size: int) -> list[float]:
    if len(values) >= size:
        return values[:size]
    return values + [0.0] * (size - len(values))


def compute_floor_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(max(float(value), 0.0) for value in values)
    rank = int((len(ordered) - 1) * quantile)
    return ordered[rank]


def normalize_pos_weight(raw_value: float, positive_weight_q90: float) -> float:
    if positive_weight_q90 <= 0.0:
        return 0.0
    clipped = min(max(raw_value, 0.0), positive_weight_q90)
    return clipped / positive_weight_q90


def count_data_rows(path: Path, max_input_rows: int = -1) -> int | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            total = max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return None
    if max_input_rows > 0:
        total = min(total, max_input_rows)
    return total


def count_total_lines(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def maybe_tqdm(iterable, *, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="row", dynamic_ncols=True)


def load_news_lookup(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row_idx, row in enumerate(maybe_tqdm(reader, desc="load news", total=count_total_lines(path))):
            if not row:
                continue
            if row_idx == 0 and row[0].strip().lower() in {"news id", "news_id"}:
                continue
            if len(row) < 5:
                continue
            news_id = row[0].strip()
            if not news_id:
                continue
            lookup[news_id] = {
                "category": normalize_text(row[1]),
                "topic": normalize_text(row[2]),
                "headline": normalize_text(row[3]),
                "body": normalize_text(row[4]),
            }
    return lookup


def render_history_block(history_ids: list[str], news_lookup: dict[str, dict[str, str]]) -> tuple[str, int, int]:
    history_lines = ["[User Click History Titles]"]
    kept_count = 0
    missing_count = 0
    for news_id in history_ids[-MAX_HISTORY_ITEMS:]:
        news_item = news_lookup.get(news_id)
        if news_item is None:
            missing_count += 1
            continue
        kept_count += 1
        history_lines.append(f"{kept_count}. {news_item['headline']}")
    if kept_count == 0:
        history_lines.append("None")
    return "\n".join(history_lines), kept_count, missing_count


def build_prompt(history_block: str, candidate: dict[str, str]) -> list[dict[str, str]]:
    content = (
        f"{PROMPT_INSTRUCTION}\n\n"
        f"{history_block}\n\n"
        "[Candidate News]\n"
        f"News body: {candidate['body']}"
    )
    return [{"role": "user", "content": content}]


def flush_records(records: list[dict], output_dir: Path, split: str, shard_idx: int, compression: str) -> None:
    if not records:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}-{shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(records, schema=OUTPUT_SCHEMA)
    pq.write_table(table, path, compression=compression)


@dataclass
class SplitOutputWriter:
    split: str
    output_root: Path
    compression: str
    single_file: bool
    output_path: Path | None = None

    def __post_init__(self) -> None:
        self.output_dir = self.output_root / self.split
        self.output_path = self.output_path or (self.output_root / f"{self.split}.parquet")
        self.shard_idx = 0
        self.writer: pq.ParquetWriter | None = None

    def write(self, records: list[dict]) -> None:
        if not records:
            return
        if self.single_file:
            self.output_root.mkdir(parents=True, exist_ok=True)
            if self.writer is None:
                self.writer = pq.ParquetWriter(self.output_path, OUTPUT_SCHEMA, compression=self.compression)
            table = pa.Table.from_pylist(records, schema=OUTPUT_SCHEMA)
            self.writer.write_table(table)
            self.shard_idx = 1
            return
        flush_records(records, self.output_dir, self.split, self.shard_idx, self.compression)
        self.shard_idx += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def output_stats(self) -> dict[str, int | str]:
        if self.single_file:
            return {
                "shards_written": self.shard_idx,
                "output_path": str(self.output_path),
            }
        return {
            "shards_written": self.shard_idx,
            "output_dir": str(self.output_dir),
        }


def extract_row_fields(row: dict[str, str]) -> tuple[str, list[str], list[str], list[str]]:
    user_id = (row.get("UserID") or row.get("user_id") or row.get("userid") or "").strip()
    history_ids = split_id_list(row.get("ClicknewsID") or row.get("clicknewsID") or row.get("clicked_news_ids") or "")
    pos_ids = split_id_list(row.get("pos") or row.get("positive_news_ids") or "")
    neg_ids = split_id_list(row.get("neg") or row.get("negative_news_ids") or "")
    return user_id, history_ids, pos_ids, neg_ids


def default_train_output_path(output_root: Path, sample_mode: str) -> Path:
    filename_map = {
        "positive_only": "only_positive_click_hist_train.parquet",
        "negative_only": "only_negative_click_hist_train.parquet",
        "all": "all_click_hist_train.parquet",
    }
    return output_root / filename_map[sample_mode]


def collect_positive_weight_q90(input_path: Path, max_input_rows: int) -> float:
    delimiter = detect_delimiter(input_path)
    positive_weights: list[float] = []
    rows_seen = 0

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            if max_input_rows > 0 and rows_seen >= max_input_rows:
                break
            rows_seen += 1
            positive_weights.extend(split_float_list(row.get("pos_weight") or ""))

    positive_weight_q90 = compute_floor_quantile(positive_weights, 0.9)
    return positive_weight_q90 if positive_weight_q90 > 0.0 else 1.0


def build_split(
    input_path: Path,
    split: str,
    output_root: Path,
    news_lookup: dict[str, dict[str, str]],
    shard_size: int,
    write_batch_size: int,
    compression: str,
    max_input_rows: int,
    single_file: bool,
    sample_mode: str,
    output_path: Path | None = None,
) -> dict[str, int | str]:
    delimiter = detect_delimiter(input_path)
    flush_size = write_batch_size if single_file else shard_size
    positive_weight_q90 = collect_positive_weight_q90(input_path, max_input_rows=max_input_rows)
    writer = SplitOutputWriter(
        split=split,
        output_root=output_root,
        compression=compression,
        single_file=single_file,
        output_path=output_path,
    )
    buffer: list[dict] = []
    records_written = 0
    positive_records_written = 0
    negative_records_written = 0
    rows_seen = 0
    missing_history = 0
    missing_candidates = 0
    skipped_empty_response = 0
    missing_pos_weight_values = 0
    missing_neg_weight_values = 0

    try:
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            progress = maybe_tqdm(reader, desc=f"{split} rows", total=count_data_rows(input_path, max_input_rows))
            for row in progress:
                if max_input_rows > 0 and rows_seen >= max_input_rows:
                    break
                rows_seen += 1
                user_id, history_ids, pos_ids, neg_ids = extract_row_fields(row)
                raw_pos_weights = split_float_list(row.get("pos_weight") or "")
                raw_neg_weights = split_float_list(row.get("neg_weight") or "")
                pos_weights = pad_or_trim_float_list(raw_pos_weights, len(pos_ids))
                neg_weights = pad_or_trim_float_list(raw_neg_weights, len(neg_ids))
                if len(pos_ids) > len(raw_pos_weights):
                    missing_pos_weight_values += len(pos_ids) - len(raw_pos_weights)
                if len(neg_ids) > len(raw_neg_weights):
                    missing_neg_weight_values += len(neg_ids) - len(raw_neg_weights)
                history_block, history_count, missing_history_count = render_history_block(history_ids, news_lookup)
                missing_history += missing_history_count

                if sample_mode in {"positive_only", "all"}:
                    for pos_idx, candidate_news_id in enumerate(pos_ids):
                        candidate = news_lookup.get(candidate_news_id)
                        if candidate is None:
                            missing_candidates += 1
                            continue
                        response = candidate["headline"]
                        if not response:
                            skipped_empty_response += 1
                            continue
                        prompt = build_prompt(history_block, candidate)
                        data_source = f"pens_{split}"
                        s_dwell = normalize_pos_weight(pos_weights[pos_idx], positive_weight_q90)

                        buffer.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "label": 1.0,
                                "s_dwell": s_dwell,
                                "p_ctr": 0.0,
                                DEFAULT_SAMPLE_ID_KEY: compute_sample_id(
                                    prompt=prompt,
                                    response=response,
                                    label=1.0,
                                    user_id=user_id,
                                    candidate_news_id=candidate_news_id,
                                    split=split,
                                    data_source=data_source,
                                ),
                                "user_id": user_id,
                                "candidate_news_id": candidate_news_id,
                                "candidate_category": candidate["category"],
                                "candidate_topic": candidate["topic"],
                                "history_count": history_count,
                                "split": split,
                                "data_source": data_source,
                            }
                        )
                        positive_records_written += 1
                        if len(buffer) >= flush_size:
                            writer.write(buffer)
                            records_written += len(buffer)
                            buffer.clear()
                            if tqdm is not None and hasattr(progress, "set_postfix"):
                                progress.set_postfix(
                                    records_written=records_written,
                                    positive_records_written=positive_records_written,
                                    negative_records_written=negative_records_written,
                                    refresh=False,
                                )

                if sample_mode in {"negative_only", "all"}:
                    for neg_idx, candidate_news_id in enumerate(neg_ids):
                        candidate = news_lookup.get(candidate_news_id)
                        if candidate is None:
                            missing_candidates += 1
                            continue
                        response = candidate["headline"]
                        if not response:
                            skipped_empty_response += 1
                            continue
                        prompt = build_prompt(history_block, candidate)
                        data_source = f"pens_{split}"
                        p_ctr = clamp_unit_interval(neg_weights[neg_idx])

                        buffer.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "label": 0.0,
                                "s_dwell": 0.0,
                                "p_ctr": p_ctr,
                                DEFAULT_SAMPLE_ID_KEY: compute_sample_id(
                                    prompt=prompt,
                                    response=response,
                                    label=0.0,
                                    user_id=user_id,
                                    candidate_news_id=candidate_news_id,
                                    split=split,
                                    data_source=data_source,
                                ),
                                "user_id": user_id,
                                "candidate_news_id": candidate_news_id,
                                "candidate_category": candidate["category"],
                                "candidate_topic": candidate["topic"],
                                "history_count": history_count,
                                "split": split,
                                "data_source": data_source,
                            }
                        )
                        negative_records_written += 1
                        if len(buffer) >= flush_size:
                            writer.write(buffer)
                            records_written += len(buffer)
                            buffer.clear()
                            if tqdm is not None and hasattr(progress, "set_postfix"):
                                progress.set_postfix(
                                    records_written=records_written,
                                    positive_records_written=positive_records_written,
                                    negative_records_written=negative_records_written,
                                    refresh=False,
                                )

        if buffer:
            writer.write(buffer)
            records_written += len(buffer)
            buffer.clear()
            if tqdm is not None and hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    records_written=records_written,
                    positive_records_written=positive_records_written,
                    negative_records_written=negative_records_written,
                    refresh=False,
                )
    finally:
        writer.close()

    return {
        "input": str(input_path),
        "split": split,
        "mode": "single_file" if single_file else "sharded",
        "sample_mode": sample_mode,
        "rows_seen": rows_seen,
        "records_written": records_written,
        "positive_records_written": positive_records_written,
        "negative_records_written": negative_records_written,
        "missing_history_news": missing_history,
        "missing_candidate_news": missing_candidates,
        "skipped_empty_response": skipped_empty_response,
        "missing_pos_weight_values": missing_pos_weight_values,
        "missing_neg_weight_values": missing_neg_weight_values,
        "positive_weight_q90": positive_weight_q90,
    } | writer.output_stats()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-file", type=Path, required=True, help="PENS news.tsv or news.csv")
    parser.add_argument("--train-file", type=Path, required=True, help="PENS train.tsv or train.csv")
    parser.add_argument("--val-file", type=Path, default=None, help="Optional PENS valid.tsv or valid.csv")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for parquet shards")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional direct train parquet path. When unset and --val-file is omitted, a click-hist train filename is derived under --output-root.",
    )
    parser.add_argument("--shard-size", type=int, default=10_000, help="Rows per parquet shard")
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Write one parquet per split at <output-root>/train.parquet and <output-root>/val.parquet",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=10_000,
        help="Rows buffered before each incremental write in --single-file mode",
    )
    parser.add_argument("--compression", type=str, default="snappy", help="Parquet compression codec")
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="positive_only",
        choices=("all", "positive_only", "negative_only"),
        help="Select which samples to emit from train_unique.tsv.",
    )
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=-1,
        help="Optional limit on source rows per split for smoke tests",
    )
    args = parser.parse_args()

    news_lookup = load_news_lookup(args.news_file)
    train_single_file = args.single_file or args.output_path is not None or args.val_file is None
    train_output_path = args.output_path
    if train_output_path is None and args.val_file is None:
        train_output_path = default_train_output_path(args.output_root, args.sample_mode)
    train_stats = build_split(
        input_path=args.train_file,
        split="train",
        output_root=args.output_root,
        news_lookup=news_lookup,
        shard_size=args.shard_size,
        write_batch_size=args.write_batch_size,
        compression=args.compression,
        max_input_rows=args.max_input_rows,
        single_file=train_single_file,
        sample_mode=args.sample_mode,
        output_path=train_output_path,
    )
    val_stats = None
    if args.val_file is not None:
        val_stats = build_split(
            input_path=args.val_file,
            split="val",
            output_root=args.output_root,
            news_lookup=news_lookup,
            shard_size=args.shard_size,
            write_batch_size=args.write_batch_size,
            compression=args.compression,
            max_input_rows=args.max_input_rows,
            single_file=args.single_file,
            sample_mode=args.sample_mode,
        )

    payload = {
        "news_count": len(news_lookup),
        "train": train_stats,
    }
    if val_stats is not None:
        payload["val"] = val_stats
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
