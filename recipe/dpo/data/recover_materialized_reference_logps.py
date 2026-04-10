#!/usr/bin/env python3
"""Recover a partially materialized point-wise reference_logps parquet."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from recipe.dpo.data.precompute_point_reference_logps import (
    PointwiseTensorizer,
    compute_reference_logps_for_table,
    maybe_tqdm,
    parse_dtype,
)
from verl.utils import hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True, help="Original parquet without reference_logps.")
    parser.add_argument("--partial-path", required=True, help="Existing partial .tmp parquet path.")
    parser.add_argument("--metadata-path", default=None, help="Namespace metadata.json. Defaults to sibling file.")
    parser.add_argument("--output-path", default=None, help="Final materialized parquet path.")
    parser.add_argument("--device", default=None, help="Device for recovery compute, e.g. cuda:0 or cpu.")
    parser.add_argument("--max-batch-size", type=int, default=1, help="Recovery batch size for missing groups.")
    parser.add_argument(
        "--max-batched-tokens",
        type=int,
        default=4096,
        help="Recovery max batched tokens for missing groups.",
    )
    parser.add_argument("--compression", default="zstd", help="Compression for rewritten parquet.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of GPU workers for missing groups.")
    parser.add_argument("--rows-per-task", type=int, default=1024, help="Rows per GPU task when num-workers > 0.")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Override the model attention implementation used during recovery.",
    )
    return parser.parse_args()


def _default_metadata_path(partial_path: str) -> str:
    output_path = _default_output_path(partial_path)
    candidates = [
        f"{output_path}.metadata.json",
        str(Path(partial_path).with_name("metadata.json")),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _default_output_path(partial_path: str) -> str:
    marker = ".tmp-"
    if marker not in partial_path:
        raise ValueError(f"partial path does not look like a materializer temp file: {partial_path}")
    return partial_path.split(marker, 1)[0]


def _load_metadata(metadata_path: str) -> dict:
    return json.loads(Path(metadata_path).read_text())


def _resolve_attn_implementation(metadata: dict, override: str | None) -> str:
    return override or metadata["attn_implementation"]


def _load_local_runtime(
    metadata: dict,
    device_override: str | None,
    attn_implementation_override: str | None,
) -> tuple[torch.device, torch.dtype, PointwiseTensorizer, object]:
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = parse_dtype(metadata["dtype"], device)
    tokenizer = hf_tokenizer(metadata["tokenizer_path"], trust_remote_code=False)
    tensorizer = PointwiseTensorizer(
        tokenizer=tokenizer,
        max_prompt_length=int(metadata["max_prompt_length"]),
        max_response_length=int(metadata["max_response_length"]),
        prompt_truncation=metadata["prompt_truncation"],
        add_eos=bool(metadata["add_eos"]),
        apply_chat_template_kwargs={},
    )
    model_kwargs = {
        "dtype": model_dtype,
        "trust_remote_code": False,
    }
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = _resolve_attn_implementation(metadata, attn_implementation_override)
    model = AutoModelForCausalLM.from_pretrained(metadata["reference_model_path"], **model_kwargs)
    model.eval().to(device)
    return device, model_dtype, tensorizer, model


def _build_worker_pool(
    metadata: dict,
    *,
    num_workers: int,
    rows_per_task: int,
    max_batch_size: int,
    max_batched_tokens: int,
    attn_implementation_override: str | None,
):
    import ray

    from recipe.dpo.reference_logps_materializer import ReferenceLogpsWorker, ReferenceWorkerPool

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    worker_config = {
        "model_path": metadata["reference_model_path"],
        "tokenizer_path": metadata["tokenizer_path"],
        "prompt_key": metadata["prompt_key"],
        "response_key": metadata["response_key"],
        "max_prompt_length": int(metadata["max_prompt_length"]),
        "max_response_length": int(metadata["max_response_length"]),
        "prompt_truncation": metadata["prompt_truncation"],
        "add_eos": bool(metadata["add_eos"]),
        "apply_chat_template_kwargs": {},
        "dtype": metadata["dtype"],
        "attn_implementation": _resolve_attn_implementation(metadata, attn_implementation_override),
        "trust_remote_code": False,
        "max_batch_size": max_batch_size,
        "max_batched_tokens": max_batched_tokens,
        "device": "cuda",
    }
    workers = [ReferenceLogpsWorker.remote(worker_config) for _ in range(num_workers)]
    return ray, ReferenceWorkerPool(workers=workers, rows_per_task=rows_per_task)


def _validate_paths(source_path: str, partial_path: str, output_path: str) -> None:
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"source parquet not found: {source_path}")
    if not os.path.exists(partial_path):
        raise FileNotFoundError(f"partial parquet not found: {partial_path}")
    output_dir = os.path.dirname(output_path) or "."
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"output directory not found: {output_dir}")


def _copy_completed_row_groups(
    partial_parquet: pq.ParquetFile,
    writer: pq.ParquetWriter | None,
    temp_output_path: str,
    compression: str,
) -> pq.ParquetWriter:
    progress = maybe_tqdm(
        range(partial_parquet.num_row_groups),
        desc="Copying completed row groups",
        total=partial_parquet.num_row_groups,
        unit="group",
    )
    try:
        for row_group_idx in progress:
            table = partial_parquet.read_row_group(row_group_idx)
            if writer is None:
                writer = pq.ParquetWriter(temp_output_path, table.schema, compression=compression)
            writer.write_table(table)
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()
    if writer is None:
        raise ValueError("partial parquet contained no row groups")
    return writer


def _recover_missing_row_groups(
    source_parquet: pq.ParquetFile,
    *,
    start_row_group: int,
    writer: pq.ParquetWriter,
    metadata: dict,
    device: torch.device | None,
    model_dtype: torch.dtype | None,
    tensorizer: PointwiseTensorizer | None,
    model,
    max_batch_size: int,
    max_batched_tokens: int,
    worker_pool=None,
) -> int:
    recovered_rows = 0
    progress = maybe_tqdm(
        range(start_row_group, source_parquet.num_row_groups),
        desc="Recovering missing row groups",
        total=source_parquet.num_row_groups - start_row_group,
        unit="group",
    )
    try:
        for row_group_idx in progress:
            table = source_parquet.read_row_group(row_group_idx)
            if worker_pool is not None:
                prompts = table.column(metadata["prompt_key"]).to_pylist()
                responses = table.column(metadata["response_key"]).to_pylist()
                reference_logps = worker_pool.compute(prompts, responses)
            else:
                if device is None or model_dtype is None or tensorizer is None or model is None:
                    raise ValueError("local runtime is not initialized")
                reference_logps = compute_reference_logps_for_table(
                    table,
                    prompt_key=metadata["prompt_key"],
                    response_key=metadata["response_key"],
                    model=model,
                    tensorizer=tensorizer,
                    device=device,
                    autocast_dtype=model_dtype,
                    max_batch_size=max_batch_size,
                    max_batched_tokens=max_batched_tokens,
                )
            table = table.append_column(
                metadata["reference_logps_key"],
                pa.array(reference_logps, type=pa.float32()),
            )
            writer.write_table(table)
            recovered_rows += table.num_rows
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()
    return recovered_rows


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata_path or _default_metadata_path(args.partial_path)
    output_path = args.output_path or _default_output_path(args.partial_path)
    _validate_paths(args.source_path, args.partial_path, output_path)

    if os.path.exists(output_path):
        final_parquet = pq.ParquetFile(output_path)
        if final_parquet.metadata.num_rows == pq.ParquetFile(args.source_path).metadata.num_rows:
            print(f"final parquet already exists and looks complete: {output_path}")
            return
        raise ValueError(f"final output already exists but is incomplete or mismatched: {output_path}")

    metadata = _load_metadata(metadata_path)
    source_parquet = pq.ParquetFile(args.source_path)
    partial_parquet = pq.ParquetFile(args.partial_path)

    if partial_parquet.num_row_groups > source_parquet.num_row_groups:
        raise ValueError("partial parquet has more row groups than source parquet")

    expected_columns = list(source_parquet.schema.names) + [metadata["reference_logps_key"]]
    if list(partial_parquet.schema.names) != expected_columns:
        raise ValueError(
            f"partial parquet schema mismatch: expected {expected_columns}, got {partial_parquet.schema.names}"
        )

    device = None
    model_dtype = None
    tensorizer = None
    model = None
    ray_module = None
    worker_pool = None
    if args.num_workers > 0:
        ray_module, worker_pool = _build_worker_pool(
            metadata,
            num_workers=args.num_workers,
            rows_per_task=args.rows_per_task,
            max_batch_size=args.max_batch_size,
            max_batched_tokens=args.max_batched_tokens,
            attn_implementation_override=args.attn_implementation,
        )
        runtime_desc = f"{args.num_workers}xgpu rows_per_task={args.rows_per_task}"
    else:
        device, model_dtype, tensorizer, model = _load_local_runtime(
            metadata,
            args.device,
            args.attn_implementation,
        )
        runtime_desc = str(device)

    completed_rows = partial_parquet.metadata.num_rows
    total_rows = source_parquet.metadata.num_rows
    print(
        f"Recovering {args.partial_path} -> {output_path} "
        f"(completed_row_groups={partial_parquet.num_row_groups}/{source_parquet.num_row_groups}, "
        f"completed_rows={completed_rows}/{total_rows}, runtime={runtime_desc}, "
        f"max_batch_size={args.max_batch_size}, max_batched_tokens={args.max_batched_tokens})"
    )

    temp_output_path = f"{output_path}.tmp-recover-{uuid.uuid4().hex}"
    writer: pq.ParquetWriter | None = None
    recovered_rows = 0
    try:
        writer = _copy_completed_row_groups(partial_parquet, writer, temp_output_path, args.compression)
        if partial_parquet.num_row_groups < source_parquet.num_row_groups:
            recovered_rows = _recover_missing_row_groups(
                source_parquet,
                start_row_group=partial_parquet.num_row_groups,
                writer=writer,
                metadata=metadata,
                device=device,
                model_dtype=model_dtype,
                tensorizer=tensorizer,
                model=model,
                max_batch_size=args.max_batch_size,
                max_batched_tokens=args.max_batched_tokens,
                worker_pool=worker_pool,
            )
    finally:
        if writer is not None:
            writer.close()
        if worker_pool is not None:
            worker_pool.close()
        if ray_module is not None and ray_module.is_initialized():
            ray_module.shutdown()

    final_parquet = pq.ParquetFile(temp_output_path)
    if final_parquet.metadata.num_rows != total_rows:
        raise ValueError(
            f"recovered parquet row count mismatch: expected {total_rows}, got {final_parquet.metadata.num_rows}"
        )
    if final_parquet.num_row_groups != source_parquet.num_row_groups:
        raise ValueError(
            "recovered parquet row group mismatch: "
            f"expected {source_parquet.num_row_groups}, got {final_parquet.num_row_groups}"
        )
    if metadata["reference_logps_key"] not in final_parquet.schema.names:
        raise ValueError(f"missing {metadata['reference_logps_key']} in recovered parquet")

    os.replace(temp_output_path, output_path)
    print(f"Recovered materialized parquet: {output_path}")
    print(f"Recovered rows in this run: {recovered_rows}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
