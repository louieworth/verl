#!/usr/bin/env python3
"""Generate SKD y_o responses through vLLM speculative decoding.

This is a repo-local entry point for the "vLLM-internal SKD sampler" path. It
installs ``vllm_skd_sampler_patch`` before constructing the vLLM engine. The
installed vLLM still needs to support a plain student draft model; vLLM 0.12.0
does not, so this script fails fast on that build instead of silently falling
back to the slower Python per-token loop.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from recipe.opd.generation.skd_vllm_y_o_generate import (
    _decode_response,
    _gpu_count,
    _prompt_token_ids,
)
from recipe.opd.generation.vllm_skd_sampler_patch import (
    install as install_vllm_skd_sampler,
    installed_vllm_draft_model_is_supported,
    unsupported_draft_model_message,
)


def _bool_arg(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate y_o with vLLM-internal SKD speculative sampler"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--student_model_path", required=True)
    parser.add_argument("--teacher_model_path", default="")
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--distill_mode", choices=["opd", "opsd"], default="opd")
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--prompt_length", type=int, required=True)
    parser.add_argument("--max_model_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--student_temperature", type=float, default=0.6)
    parser.add_argument("--student_top_p", type=float, default=0.95)
    parser.add_argument("--teacher_temperature", type=float, default=0.6)
    parser.add_argument("--teacher_top_p", type=float, default=0.95)
    parser.add_argument(
        "--shared_gpus",
        default=os.environ.get("SKD_SHARED_GPUS", "0,1,2,3,4,5,6,7"),
    )
    parser.add_argument("--student_tp", type=int, default=int(os.environ.get("SKD_STUDENT_TP", "0")))
    parser.add_argument("--teacher_tp", type=int, default=int(os.environ.get("SKD_TEACHER_TP", "0")))
    parser.add_argument("--dtype", default=os.environ.get("SKD_VLLM_DTYPE", "bfloat16"))
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("SKD_VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
    )
    parser.add_argument("--max_num_seqs", type=int, default=int(os.environ.get("SKD_VLLM_MAX_NUM_SEQS", "0")))
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=int(os.environ.get("SKD_VLLM_MAX_NUM_BATCHED_TOKENS", "0")),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SKD_VLLM_SEED", "20")))
    parser.add_argument(
        "--partial_output",
        default=os.environ.get("SKD_PARTIAL_OUTPUT", ""),
        help="JSONL checkpoint for completed row responses; defaults to <output>.partial.jsonl.",
    )
    parser.add_argument(
        "--partial_save_every_batches",
        type=int,
        default=int(os.environ.get("SKD_PARTIAL_SAVE_EVERY_BATCHES", "1")),
        help="Append completed batch responses every N batches; <=0 disables partial writes.",
    )
    parser.add_argument(
        "--fail_if_plain_draft_unsupported",
        type=_bool_arg,
        default=True,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.student_temperature != args.teacher_temperature:
        raise ValueError(
            "vLLM internal speculative decoding uses one SamplingParams object; "
            "student_temperature and teacher_temperature must match to preserve SKD semantics."
        )
    if args.student_top_p != args.teacher_top_p:
        raise ValueError(
            "vLLM internal speculative decoding uses one SamplingParams object; "
            "student_top_p and teacher_top_p must match to preserve SKD semantics."
        )

    teacher_tp = args.teacher_tp or _gpu_count(args.shared_gpus)
    draft_tp = args.student_tp or teacher_tp
    if draft_tp not in {1, teacher_tp}:
        raise ValueError(
            "vLLM only allows draft_tensor_parallel_size to be 1 or equal to "
            f"target tensor_parallel_size; got student_tp={draft_tp}, teacher_tp={teacher_tp}."
        )


def _partial_path(output_path: str, partial_output: str) -> str:
    return partial_output or f"{output_path}.partial.jsonl"


def _load_partial_responses(path: str, expected_rows: int) -> dict[int, str]:
    if not path or not os.path.exists(path):
        return {}

    responses: dict[int, str] = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
                row_idx = int(record["row_idx"])
                response = str(record["response"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                skipped += 1
                continue
            if 0 <= row_idx < expected_rows:
                responses[row_idx] = response

    if skipped:
        print(
            f"[SKD vLLM internal y_o] ignored {skipped} malformed partial rows from {path}",
            flush=True,
        )
    return responses


def _append_partial_responses(path: str, rows: list[tuple[int, str]]) -> None:
    if not rows:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row_idx, response in rows:
            f.write(
                json.dumps(
                    {"row_idx": int(row_idx), "response": response},
                    ensure_ascii=False,
                )
            )
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    args = parse_args()
    _validate_args(args)

    teacher_model_path = args.teacher_model_path or args.student_model_path
    tokenizer_path = args.tokenizer_path or args.student_model_path
    teacher_tp = args.teacher_tp or _gpu_count(args.shared_gpus)
    draft_tp = args.student_tp or teacher_tp

    os.environ["CUDA_VISIBLE_DEVICES"] = args.shared_gpus
    os.environ["VLLM_SKD_SAMPLER"] = "1"
    os.environ["VLLM_SKD_ACCEPT_TOP_K"] = str(args.top_k)
    os.environ["VLLM_SKD_ACCEPT_TOP_P"] = str(args.top_p)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    patched = install_vllm_skd_sampler()
    if args.fail_if_plain_draft_unsupported and not installed_vllm_draft_model_is_supported():
        raise RuntimeError(unsupported_draft_model_message(args.student_model_path))

    from vllm import LLM, SamplingParams  # pylint: disable=import-outside-toplevel

    dataset = pd.read_parquet(args.input)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    if tokenizer.pad_token_id is not None:
        eos_ids.add(int(tokenizer.pad_token_id))

    prompt_ids = [
        _prompt_token_ids(tokenizer, chat, args.prompt_length)
        for chat in dataset[args.prompt_key].tolist()
    ]
    prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    partial_path = _partial_path(args.output, args.partial_output)
    partial_responses = _load_partial_responses(partial_path, len(prompts))
    responses: list[str | None] = [None] * len(prompts)
    for row_idx, response in partial_responses.items():
        responses[row_idx] = response
    if partial_responses:
        print(
            "SKD vLLM internal y_o resume: "
            f"loaded {len(partial_responses)}/{len(prompts)} rows from {partial_path}",
            flush=True,
        )

    llm_kwargs = {
        "model": teacher_model_path,
        "tokenizer": tokenizer_path,
        "tensor_parallel_size": teacher_tp,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "seed": args.seed,
        "enable_prefix_caching": True,
        "disable_log_stats": True,
        "speculative_config": {
            "model": args.student_model_path,
            "num_speculative_tokens": args.gamma,
            "draft_tensor_parallel_size": draft_tp,
        },
    }
    if args.max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    print(
        "SKD vLLM internal y_o rollout: "
        f"rows={len(prompts)} max_tokens={args.max_tokens} prompt_length={args.prompt_length} "
        f"max_model_len={args.max_model_len} batch={args.batch_size} gamma={args.gamma} "
        f"student={args.student_model_path} teacher={teacher_model_path} "
        f"shared_gpus={args.shared_gpus} teacher_tp={teacher_tp} draft_tp={draft_tp} "
        f"sampler_patch={patched}",
        flush=True,
    )

    if any(response is None for response in responses):
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.teacher_temperature,
            top_p=args.teacher_top_p,
            detokenize=False,
            skip_special_tokens=False,
        )

        for batch_idx, start in enumerate(
            tqdm(
                range(0, len(prompts), args.batch_size),
                desc="SKD vLLM internal y_o rollout",
            ),
            start=1,
        ):
            batch_indices = [
                idx
                for idx in range(start, min(start + args.batch_size, len(prompts)))
                if responses[idx] is None
            ]
            if not batch_indices:
                continue
            batch = [prompts[idx] for idx in batch_indices]
            outputs = llm.generate(batch, sampling_params, use_tqdm=False)
            completed_rows: list[tuple[int, str]] = []
            for row_idx, output in zip(batch_indices, outputs, strict=True):
                token_ids = [int(tok) for tok in list(output.outputs[0].token_ids)]
                response = _decode_response(tokenizer, token_ids, eos_ids)
                responses[row_idx] = response
                completed_rows.append((row_idx, response))

            if (
                args.partial_save_every_batches > 0
                and batch_idx % args.partial_save_every_batches == 0
            ):
                _append_partial_responses(partial_path, completed_rows)
    else:
        print(
            "SKD vLLM internal y_o resume: partial checkpoint already covers all rows; "
            "skipping vLLM generation.",
            flush=True,
        )

    missing = [idx for idx, response in enumerate(responses) if response is None]
    if missing:
        raise RuntimeError(f"missing SKD responses for rows: {missing[:10]}")

    dataset["responses"] = [[response] for response in responses]
    dataset.to_parquet(args.output)
    manifest = {
        "mode": "skd_vllm_internal_y_o",
        "distill_mode": args.distill_mode,
        "input": args.input,
        "output": args.output,
        "rows": len(dataset),
        "student_model_path": args.student_model_path,
        "teacher_model_path": teacher_model_path,
        "max_tokens": args.max_tokens,
        "prompt_length": args.prompt_length,
        "max_model_len": args.max_model_len,
        "gamma": args.gamma,
        "accept_top_k": args.top_k,
        "accept_top_p": args.top_p,
        "temperature": args.teacher_temperature,
        "top_p": args.teacher_top_p,
        "shared_gpus": args.shared_gpus,
        "teacher_tp": teacher_tp,
        "draft_tp": draft_tp,
        "vllm_skd_sampler_patch": patched,
        "partial_output": partial_path,
        "partial_rows": sum(response is not None for response in responses),
    }
    with open(f"{args.output}.skd_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved SKD vLLM internal y_o responses to {args.output}", flush=True)


if __name__ == "__main__":
    main()
