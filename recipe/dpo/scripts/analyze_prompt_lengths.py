#!/usr/bin/env python3
"""Analyze prompt token lengths for offline DPO parquet datasets."""

from __future__ import annotations

import argparse
import json
import warnings
from array import array
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_THRESHOLDS = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
DEFAULT_PERCENTILES = [50, 90, 95, 99, 99.5, 99.9, 99.99, 100]

_WORKER_PARQUET_PATH: str | None = None
_WORKER_TOKENIZER_PATH: str | None = None
_WORKER_PROMPT_KEY: str | None = None
_WORKER_ADD_GENERATION_PROMPT: bool = True
_WORKER_BATCH_SIZE: int = 1000
_WORKER_TOKENIZER = None


def maybe_tqdm(iterable, *, desc: str, total: int | None = None, unit: str = "it"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True)


def normalize_token_ids(tokenized_output) -> list[int]:
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict) and "input_ids" in tokenized_output:
        token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for token_id in token_ids:
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        normalized_ids.append(int(token_id))
    return normalized_ids


def hf_tokenizer(name_or_path: str, **kwargs):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)
    return tokenizer


def parse_csv_numbers(raw: str, cast):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast(part))
    return values


def prompt_token_length(tokenizer, prompt, add_generation_prompt: bool) -> int:
    if isinstance(prompt, list):
        tokenized = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        return len(normalize_token_ids(tokenized))
    if isinstance(prompt, str):
        return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


def batched_prompt_token_lengths(tokenizer, prompts: list, add_generation_prompt: bool, batch_size: int) -> array:
    lengths = array("I")
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        if not batch:
            continue

        if isinstance(batch[0], list):
            rendered = [tokenizer.apply_chat_template(x, add_generation_prompt=add_generation_prompt, tokenize=False) for x in batch]
            encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            lengths.extend(len(token_ids) for token_ids in encoded)
        elif isinstance(batch[0], str):
            encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
            lengths.extend(len(token_ids) for token_ids in encoded)
        else:
            for prompt in batch:
                lengths.append(prompt_token_length(tokenizer, prompt, add_generation_prompt=add_generation_prompt))
    return lengths


def init_worker(
    parquet_path: str,
    tokenizer_path: str,
    prompt_key: str,
    add_generation_prompt: bool,
    batch_size: int,
) -> None:
    global _WORKER_PARQUET_PATH, _WORKER_TOKENIZER_PATH, _WORKER_PROMPT_KEY
    global _WORKER_ADD_GENERATION_PROMPT, _WORKER_BATCH_SIZE, _WORKER_TOKENIZER
    _WORKER_PARQUET_PATH = parquet_path
    _WORKER_TOKENIZER_PATH = tokenizer_path
    _WORKER_PROMPT_KEY = prompt_key
    _WORKER_ADD_GENERATION_PROMPT = add_generation_prompt
    _WORKER_BATCH_SIZE = batch_size
    _WORKER_TOKENIZER = hf_tokenizer(tokenizer_path, trust_remote_code=True)


def process_row_group_chunk(row_group_indices: list[int]) -> np.ndarray:
    assert _WORKER_PARQUET_PATH is not None
    assert _WORKER_PROMPT_KEY is not None
    assert _WORKER_TOKENIZER is not None

    parquet = pq.ParquetFile(_WORKER_PARQUET_PATH)
    lengths = array("I")
    for row_group_idx in row_group_indices:
        table = parquet.read_row_group(row_group_idx, columns=[_WORKER_PROMPT_KEY])
        prompts = table.column(_WORKER_PROMPT_KEY).to_pylist()
        lengths.extend(
            batched_prompt_token_lengths(
                _WORKER_TOKENIZER,
                prompts,
                add_generation_prompt=_WORKER_ADD_GENERATION_PROMPT,
                batch_size=_WORKER_BATCH_SIZE,
            )
        )
    return np.frombuffer(lengths, dtype=np.uint32).copy()


def split_row_groups(num_row_groups: int, num_workers: int) -> list[list[int]]:
    chunks = [[] for _ in range(num_workers)]
    for idx in range(num_row_groups):
        chunks[idx % num_workers].append(idx)
    return [chunk for chunk in chunks if chunk]


def select_evenly_spaced_items(items: list, take: int) -> list:
    if take <= 0:
        return []
    if take >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, num=take, dtype=int)
    return [items[idx] for idx in indices.tolist()]


def analyze_parquet(
    dataset_path: Path,
    tokenizer_path: str,
    prompt_key: str,
    add_generation_prompt: bool,
    max_samples: int,
    thresholds: list[int],
    percentiles: list[float],
    batch_size: int,
    num_workers: int,
) -> dict:
    parquet = pq.ParquetFile(dataset_path)
    total_rows = parquet.metadata.num_rows
    target_rows = total_rows if max_samples < 0 else min(total_rows, max_samples)
    if max_samples > 0:
        tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=True)
        lengths = array("I")
        row_iter = maybe_tqdm(range(parquet.num_row_groups), desc="row groups", total=parquet.num_row_groups, unit="group")
        rows_seen = 0
        for order_idx, row_group_idx in enumerate(row_iter):
            if rows_seen >= target_rows:
                break
            table = parquet.read_row_group(row_group_idx, columns=[prompt_key])
            prompts = table.column(prompt_key).to_pylist()
            remaining = target_rows - rows_seen
            remaining_groups = parquet.num_row_groups - order_idx
            take = min(len(prompts), max(1, int(np.ceil(remaining / remaining_groups))))
            prompts = select_evenly_spaced_items(prompts, take)
            lengths.extend(
                batched_prompt_token_lengths(
                    tokenizer,
                    prompts,
                    add_generation_prompt=add_generation_prompt,
                    batch_size=batch_size,
                )
            )
            rows_seen += len(prompts)
            if tqdm is not None and hasattr(row_iter, "set_postfix"):
                current_max = max(lengths) if lengths else 0
                row_iter.set_postfix(rows_seen=rows_seen, max_prompt_length=current_max, refresh=False)
        if not lengths:
            raise ValueError(f"No prompts found in {dataset_path}")
        length_array = np.frombuffer(lengths, dtype=np.uint32).copy()
    elif num_workers <= 1:
        tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=True)
        lengths = array("I")
        row_iter = maybe_tqdm(range(parquet.num_row_groups), desc="row groups", total=parquet.num_row_groups, unit="group")
        rows_seen = 0
        for row_group_idx in row_iter:
            table = parquet.read_row_group(row_group_idx, columns=[prompt_key])
            prompts = table.column(prompt_key).to_pylist()
            lengths.extend(
                batched_prompt_token_lengths(
                    tokenizer,
                    prompts,
                    add_generation_prompt=add_generation_prompt,
                    batch_size=batch_size,
                )
            )
            rows_seen += len(prompts)
            if tqdm is not None and hasattr(row_iter, "set_postfix"):
                current_max = max(lengths) if lengths else 0
                row_iter.set_postfix(rows_seen=rows_seen, max_prompt_length=current_max, refresh=False)
        if not lengths:
            raise ValueError(f"No prompts found in {dataset_path}")
        length_array = np.frombuffer(lengths, dtype=np.uint32).copy()
    else:
        row_group_chunks = split_row_groups(parquet.num_row_groups, num_workers)
        ctx = get_context("spawn")
        length_parts: list[np.ndarray] = []
        task_iter = None
        with ctx.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(str(dataset_path), tokenizer_path, prompt_key, add_generation_prompt, batch_size),
        ) as pool:
            task_iter = pool.imap_unordered(process_row_group_chunk, row_group_chunks)
            task_iter = maybe_tqdm(task_iter, desc="worker chunks", total=len(row_group_chunks), unit="chunk")
            for part in task_iter:
                length_parts.append(part)
        if not length_parts:
            raise ValueError(f"No prompts found in {dataset_path}")
        length_array = np.concatenate(length_parts)

    percentile_values = np.percentile(length_array, percentiles, method="linear")
    threshold_coverages = {
        str(threshold): float((length_array <= threshold).sum() / len(length_array)) for threshold in thresholds
    }

    return {
        "dataset_path": str(dataset_path),
        "tokenizer_path": tokenizer_path,
        "prompt_key": prompt_key,
        "add_generation_prompt": add_generation_prompt,
        "samples_analyzed": int(len(length_array)),
        "total_rows_in_dataset": int(total_rows),
        "max_prompt_length_needed_for_full_coverage": int(length_array.max()),
        "summary": {
            "min": int(length_array.min()),
            "mean": float(length_array.mean()),
            "median": float(np.median(length_array)),
            "max": int(length_array.max()),
        },
        "percentiles": {f"p{p}": float(v) for p, v in zip(percentiles, percentile_values, strict=True)},
        "coverage_by_max_prompt_length": threshold_coverages,
        "num_truncated_if_max_prompt_length": {
            str(threshold): int((length_array > threshold).sum()) for threshold in thresholds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Parquet file containing a prompt column")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="HF model/tokenizer path, e.g. your Qwen model")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Prompt column name")
    parser.add_argument(
        "--add-generation-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match verl dataset tokenization for chat prompts. Default: true",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Comma-separated max_prompt_length thresholds to report coverage for",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PERCENTILES),
        help="Comma-separated percentiles to report",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional row limit for a faster estimate. Default: analyze the full dataset",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for tokenizer encoding")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for full-dataset analysis. Ignored when --max-samples is set",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the JSON report")
    args = parser.parse_args()

    thresholds = parse_csv_numbers(args.thresholds, int)
    percentiles = parse_csv_numbers(args.percentiles, float)

    result = analyze_parquet(
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer_path,
        prompt_key=args.prompt_key,
        add_generation_prompt=args.add_generation_prompt,
        max_samples=args.max_samples,
        thresholds=thresholds,
        percentiles=percentiles,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
