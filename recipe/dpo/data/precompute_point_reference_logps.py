#!/usr/bin/env python3
"""Precompute point-wise offline DPO reference log-probs into parquet."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from recipe.dpo.collate import pointwise_dynamic_prompt_collate_fn
from recipe.dpo.core_algos import get_batch_logps
from verl.utils import hf_tokenizer, normalize_token_ids

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def maybe_tqdm(iterable, *, desc: str, total: int | None = None, unit: str = "it"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True)


def parse_dtype(raw: str, device: torch.device) -> torch.dtype:
    normalized = raw.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16 if device.type == "cuda" else torch.float32
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {raw}")


@dataclass
class PointwiseTensorizer:
    tokenizer: Any
    max_prompt_length: int
    max_response_length: int
    prompt_truncation: str
    add_eos: bool
    apply_chat_template_kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        self.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, prompt, response: str) -> dict[str, torch.Tensor]:
        prompt_ids = self._tokenize_prompt(prompt)
        response_ids = self._tokenize_response(response)
        return self._build_response_tensors(prompt_ids, response_ids)

    def _tokenize_prompt(self, prompt) -> list[int]:
        if isinstance(prompt, list):
            tokenized_prompt = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            token_ids = normalize_token_ids(tokenized_prompt)
        elif isinstance(prompt, str):
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")
        return self._truncate_prompt(token_ids)

    def _tokenize_response(self, response: str) -> list[int]:
        if not isinstance(response, str):
            response = str(response)
        token_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        if self.add_eos and self.eos_token_id is not None:
            token_ids = token_ids + [self.eos_token_id]
        return token_ids[: self.max_response_length]

    def _truncate_prompt(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) <= self.max_prompt_length:
            return token_ids
        if self.prompt_truncation == "left":
            return token_ids[-self.max_prompt_length :]
        if self.prompt_truncation == "right":
            return token_ids[: self.max_prompt_length]
        raise ValueError(f"Unsupported prompt_truncation: {self.prompt_truncation}")

    def _left_pad(self, token_ids: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
        pad_length = length - len(token_ids)
        padded = [self.pad_token_id] * pad_length + token_ids
        mask = [0] * pad_length + [1] * len(token_ids)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def _right_pad(self, token_ids: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
        pad_length = length - len(token_ids)
        padded = token_ids + [self.pad_token_id] * pad_length
        mask = [1] * len(token_ids) + [0] * pad_length
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def _build_response_tensors(self, prompt_ids: list[int], response_ids: list[int]) -> dict[str, torch.Tensor]:
        from verl.utils.model import compute_position_id_with_mask

        prompt_tensor, prompt_mask = self._left_pad(prompt_ids, self.max_prompt_length)
        response_tensor, response_mask = self._right_pad(response_ids, self.max_response_length)
        input_ids = torch.cat([prompt_tensor, response_tensor], dim=0)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=0)
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0].to(torch.long)

        labels = torch.full((self.max_prompt_length + self.max_response_length,), -100, dtype=torch.long)
        valid_response_tokens = int(response_mask.sum().item())
        if valid_response_tokens > 0:
            labels[self.max_prompt_length : self.max_prompt_length + valid_response_tokens] = response_tensor[
                :valid_response_tokens
            ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_tensor,
            "response_mask": response_mask,
            "labels": labels,
        }


def flush_pending_batch(
    pending: list[tuple[int, dict[str, torch.Tensor]]],
    *,
    model,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> tuple[list[int], np.ndarray]:
    batch_indices = [idx for idx, _ in pending]
    batch_samples = [sample for _, sample in pending]
    collated = pointwise_dynamic_prompt_collate_fn(batch_samples)

    batch = {
        "input_ids": collated["input_ids"].to(device),
        "attention_mask": collated["attention_mask"].to(device),
        "position_ids": collated["position_ids"].to(device),
        "labels": collated["labels"].to(device),
    }
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if device.type == "cuda" and autocast_dtype != torch.float32
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        try:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                use_cache=False,
            )
        except TypeError:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        logits = outputs.logits
        if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            logits = logits.float()
        sequence_logps = get_batch_logps(logits, batch["labels"], average_log_prob=False)
    return batch_indices, sequence_logps.detach().cpu().numpy().astype(np.float32, copy=False)


def compute_reference_logps_for_table(
    table: pa.Table,
    *,
    prompt_key: str,
    response_key: str,
    model,
    tensorizer: PointwiseTensorizer,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_batch_size: int,
    max_batched_tokens: int,
) -> np.ndarray:
    prompts = table.column(prompt_key).to_pylist()
    responses = table.column(response_key).to_pylist()
    reference_logps = np.empty(len(prompts), dtype=np.float32)

    pending: list[tuple[int, dict[str, torch.Tensor]]] = []
    current_max_tokens = 0

    def should_flush(next_tokens: int) -> bool:
        if not pending:
            return False
        next_batch_size = len(pending) + 1
        next_max_tokens = max(current_max_tokens, next_tokens)
        if max_batch_size > 0 and next_batch_size > max_batch_size:
            return True
        if max_batched_tokens > 0 and next_batch_size * next_max_tokens > max_batched_tokens:
            return True
        return False

    for row_idx, (prompt, response) in enumerate(zip(prompts, responses, strict=False)):
        sample = tensorizer.encode(prompt, response)
        sample_tokens = int(sample["attention_mask"].sum().item())
        if should_flush(sample_tokens):
            batch_indices, batch_reference_logps = flush_pending_batch(
                pending,
                model=model,
                device=device,
                autocast_dtype=autocast_dtype,
            )
            reference_logps[np.asarray(batch_indices, dtype=np.int64)] = batch_reference_logps
            pending.clear()
            current_max_tokens = 0

        pending.append((row_idx, sample))
        current_max_tokens = max(current_max_tokens, sample_tokens)

    if pending:
        batch_indices, batch_reference_logps = flush_pending_batch(
            pending,
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        reference_logps[np.asarray(batch_indices, dtype=np.int64)] = batch_reference_logps

    return reference_logps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Input parquet path.")
    parser.add_argument("--output-path", required=True, help="Output parquet path with reference_logps appended.")
    parser.add_argument("--model-path", required=True, help="Reference model path or HF model id.")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path. Defaults to model-path.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--response-key", default="response")
    parser.add_argument("--reference-logps-key", default="reference_logps")
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--max-response-length", type=int, default=32)
    parser.add_argument("--prompt-truncation", choices=["left", "right"], default="right")
    parser.add_argument("--dtype", default="bf16", help="Model dtype: bf16, fp16, or fp32.")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batched-tokens", type=int, default=24576)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-row-groups", type=int, default=-1)
    parser.add_argument("--apply-chat-template-kwargs-json", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input-path and output-path must be different")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output parquet already exists: {output_path}. Pass --overwrite to replace it.")
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = parse_dtype(args.dtype, device)
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=args.trust_remote_code)
    apply_chat_template_kwargs = json.loads(args.apply_chat_template_kwargs_json)
    tensorizer = PointwiseTensorizer(
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        prompt_truncation=args.prompt_truncation,
        add_eos=True,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    model_kwargs = {
        "dtype": model_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval().to(device)

    parquet = pq.ParquetFile(input_path)
    total_row_groups = parquet.num_row_groups if args.max_row_groups < 0 else min(parquet.num_row_groups, args.max_row_groups)
    row_group_iter = maybe_tqdm(range(total_row_groups), desc="row groups", total=total_row_groups, unit="group")
    writer: pq.ParquetWriter | None = None

    try:
        for row_group_idx in row_group_iter:
            table = parquet.read_row_group(row_group_idx)
            if args.reference_logps_key in table.column_names:
                if not args.overwrite:
                    raise ValueError(
                        f"Column {args.reference_logps_key!r} already exists in row group {row_group_idx}. "
                        "Pass --overwrite to replace it."
                    )
                column_idx = table.column_names.index(args.reference_logps_key)
                table = table.remove_column(column_idx)

            reference_logps = compute_reference_logps_for_table(
                table,
                prompt_key=args.prompt_key,
                response_key=args.response_key,
                model=model,
                tensorizer=tensorizer,
                device=device,
                autocast_dtype=model_dtype,
                max_batch_size=args.max_batch_size,
                max_batched_tokens=args.max_batched_tokens,
            )
            table = table.append_column(args.reference_logps_key, pa.array(reference_logps, type=pa.float32()))
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=args.compression)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
