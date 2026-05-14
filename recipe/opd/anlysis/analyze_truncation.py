#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Analyze truncation under the exact prompt/sequence construction used by KL training.

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import datasets
from transformers import AutoTokenizer

from recipe.opd.dataset.data_utils import (
    PROMPT_TEMPLATE_REVERSE_KL_STUDENT,
    build_teacher_prompt,
    instruction_following,
)


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


@dataclass
class SampleTruncationStats:
    idx: int
    student_prompt_tokens: int
    teacher_prompt_tokens: int
    response_tokens_raw: int
    student_total_tokens_raw: int
    teacher_total_tokens_raw: int
    student_total_tokens_kept: int
    teacher_total_tokens_kept: int
    student_response_tokens_kept: int
    teacher_response_tokens_kept: int
    aligned_response_tokens_kept: int
    student_truncated: bool
    teacher_truncated: bool
    aligned_response_truncated: bool
    student_prompt_overflow: bool
    teacher_prompt_overflow: bool


def encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def compute_kept_response_tokens(prompt_tokens: int, response_tokens: int, max_length: int) -> tuple[int, int]:
    total_kept = min(prompt_tokens + response_tokens, max_length)
    prompt_kept = min(prompt_tokens, total_kept)
    response_kept = max(total_kept - prompt_kept, 0)
    return total_kept, response_kept


def build_reverse_prompts(item: dict[str, Any], use_initial_response: bool) -> tuple[str, str, str]:
    extra_info = item.get("extra_info", {})
    problem = extra_info.get("problem", "")
    expert_solution = extra_info.get("expert_cot", "")

    responses = item.get("responses", [""])
    response = responses[0] if isinstance(responses, list) else responses
    if not response:
        raise ValueError("Reverse KL requires stage1 responses in the 'responses' field.")

    student_prompt = PROMPT_TEMPLATE_REVERSE_KL_STUDENT.replace("{PROBLEM}", problem).strip()
    student_prompt = student_prompt + " " + instruction_following
    teacher_prompt = build_teacher_prompt(
        problem,
        expert_solution,
        initial_response=response,
        use_initial_response=use_initial_response,
    )
    return student_prompt, teacher_prompt, response


def build_forward_prompts(
    item: dict[str, Any],
    corrected_item: dict[str, Any] | None,
    use_initial_response: bool,
) -> tuple[str, str, str]:
    extra_info = item.get("extra_info", {})
    problem = extra_info.get("problem", "")
    expert_solution = extra_info.get("expert_cot", "")
    initial_response = extra_info.get("initial_response", "")

    if corrected_item is not None:
        if not initial_response:
            responses = item.get("responses", [""])
            initial_response = responses[0] if isinstance(responses, list) else responses
        rewritten_responses = corrected_item.get("responses", [""])
        target_response = rewritten_responses[0] if isinstance(rewritten_responses, list) else rewritten_responses
    else:
        rewritten_responses = item.get("responses", [""])
        target_response = rewritten_responses[0] if isinstance(rewritten_responses, list) else rewritten_responses
        if use_initial_response and not initial_response:
            raise ValueError("Forward KL single-file mode requires extra_info['initial_response'] in the stage2 parquet.")

    student_prompt = problem + " " + instruction_following
    teacher_prompt = build_teacher_prompt(
        problem,
        expert_solution,
        initial_response=initial_response,
        use_initial_response=use_initial_response,
    )
    return student_prompt, teacher_prompt, target_response


def analyze_sample(
    tokenizer,
    item: dict[str, Any],
    *,
    idx: int,
    kl_type: str,
    max_length: int,
    use_initial_response: bool,
    corrected_item: dict[str, Any] | None = None,
) -> SampleTruncationStats:
    if kl_type == "reverse":
        student_prompt, teacher_prompt, response = build_reverse_prompts(item, use_initial_response)
    else:
        student_prompt, teacher_prompt, response = build_forward_prompts(item, corrected_item, use_initial_response)

    student_prompt_tokens = len(encode_text(tokenizer, student_prompt + "\n"))
    teacher_prompt_tokens = len(encode_text(tokenizer, teacher_prompt + "\n"))
    response_tokens_raw = len(encode_text(tokenizer, response))

    student_total_tokens_raw = student_prompt_tokens + response_tokens_raw
    teacher_total_tokens_raw = teacher_prompt_tokens + response_tokens_raw

    student_total_tokens_kept, student_response_tokens_kept = compute_kept_response_tokens(
        student_prompt_tokens, response_tokens_raw, max_length
    )
    teacher_total_tokens_kept, teacher_response_tokens_kept = compute_kept_response_tokens(
        teacher_prompt_tokens, response_tokens_raw, max_length
    )

    aligned_response_tokens_kept = min(student_response_tokens_kept, teacher_response_tokens_kept)

    return SampleTruncationStats(
        idx=idx,
        student_prompt_tokens=student_prompt_tokens,
        teacher_prompt_tokens=teacher_prompt_tokens,
        response_tokens_raw=response_tokens_raw,
        student_total_tokens_raw=student_total_tokens_raw,
        teacher_total_tokens_raw=teacher_total_tokens_raw,
        student_total_tokens_kept=student_total_tokens_kept,
        teacher_total_tokens_kept=teacher_total_tokens_kept,
        student_response_tokens_kept=student_response_tokens_kept,
        teacher_response_tokens_kept=teacher_response_tokens_kept,
        aligned_response_tokens_kept=aligned_response_tokens_kept,
        student_truncated=student_total_tokens_raw > max_length,
        teacher_truncated=teacher_total_tokens_raw > max_length,
        aligned_response_truncated=aligned_response_tokens_kept < response_tokens_raw,
        student_prompt_overflow=student_prompt_tokens >= max_length,
        teacher_prompt_overflow=teacher_prompt_tokens >= max_length,
    )


def summarize(stats: list[SampleTruncationStats], max_length: int) -> dict[str, Any]:
    if not stats:
        return {
            "num_samples": 0,
            "max_length": max_length,
        }

    num_samples = len(stats)
    student_truncated = [s for s in stats if s.student_truncated]
    teacher_truncated = [s for s in stats if s.teacher_truncated]
    any_truncated = [s for s in stats if s.student_truncated or s.teacher_truncated]
    aligned_truncated = [s for s in stats if s.aligned_response_truncated]
    zero_response = [s for s in stats if s.aligned_response_tokens_kept == 0]

    raw_response = [s.response_tokens_raw for s in stats]
    kept_response = [s.aligned_response_tokens_kept for s in stats]
    student_total_raw = sorted(s.student_total_tokens_raw for s in stats)
    teacher_total_raw = sorted(s.teacher_total_tokens_raw for s in stats)
    student_prompt = sorted(s.student_prompt_tokens for s in stats)
    teacher_prompt = sorted(s.teacher_prompt_tokens for s in stats)
    tokens_lost = [s.response_tokens_raw - s.aligned_response_tokens_kept for s in stats]

    worst_samples = sorted(
        stats,
        key=lambda s: (
            s.response_tokens_raw - s.aligned_response_tokens_kept,
            max(s.student_total_tokens_raw, s.teacher_total_tokens_raw),
        ),
        reverse=True,
    )[:10]

    return {
        "num_samples": num_samples,
        "max_length": max_length,
        "student_truncated_samples": len(student_truncated),
        "student_truncated_rate": len(student_truncated) / num_samples,
        "teacher_truncated_samples": len(teacher_truncated),
        "teacher_truncated_rate": len(teacher_truncated) / num_samples,
        "any_truncated_samples": len(any_truncated),
        "any_truncated_rate": len(any_truncated) / num_samples,
        "aligned_response_truncated_samples": len(aligned_truncated),
        "aligned_response_truncated_rate": len(aligned_truncated) / num_samples,
        "zero_response_after_truncation_samples": len(zero_response),
        "zero_response_after_truncation_rate": len(zero_response) / num_samples,
        "avg_response_tokens_raw": mean(raw_response),
        "avg_response_tokens_kept_aligned": mean(kept_response),
        "avg_response_tokens_lost": mean(tokens_lost),
        "max_response_tokens_raw": max(raw_response),
        "max_response_tokens_kept_aligned": max(kept_response),
        "required_max_length_for_zero_truncation": max(
            max(student_total_raw),
            max(teacher_total_raw),
        ),
        "student_total_tokens_raw_p50": percentile(student_total_raw, 0.50),
        "student_total_tokens_raw_p90": percentile(student_total_raw, 0.90),
        "student_total_tokens_raw_p95": percentile(student_total_raw, 0.95),
        "student_total_tokens_raw_p99": percentile(student_total_raw, 0.99),
        "teacher_total_tokens_raw_p50": percentile(teacher_total_raw, 0.50),
        "teacher_total_tokens_raw_p90": percentile(teacher_total_raw, 0.90),
        "teacher_total_tokens_raw_p95": percentile(teacher_total_raw, 0.95),
        "teacher_total_tokens_raw_p99": percentile(teacher_total_raw, 0.99),
        "student_prompt_tokens_p95": percentile(student_prompt, 0.95),
        "teacher_prompt_tokens_p95": percentile(teacher_prompt, 0.95),
        "worst_truncated_samples": [asdict(sample) for sample in worst_samples],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("==========================================")
    print("KL Training Truncation Analysis")
    print("==========================================")
    print(f"Samples:                           {summary['num_samples']}")
    print(f"Configured MAX_LENGTH:             {summary['max_length']}")
    print(f"Required MAX_LENGTH for 0 trunc:   {summary['required_max_length_for_zero_truncation']}")
    print("")
    print(f"Student truncated samples:         {summary['student_truncated_samples']} ({summary['student_truncated_rate']:.2%})")
    print(f"Teacher truncated samples:         {summary['teacher_truncated_samples']} ({summary['teacher_truncated_rate']:.2%})")
    print(f"Any truncated samples:             {summary['any_truncated_samples']} ({summary['any_truncated_rate']:.2%})")
    print(
        f"Aligned response truncated:        {summary['aligned_response_truncated_samples']} "
        f"({summary['aligned_response_truncated_rate']:.2%})"
    )
    print(
        f"Zero response after truncation:    {summary['zero_response_after_truncation_samples']} "
        f"({summary['zero_response_after_truncation_rate']:.2%})"
    )
    print("")
    print(f"Avg raw response tokens:           {summary['avg_response_tokens_raw']:.1f}")
    print(f"Avg kept aligned response tokens:  {summary['avg_response_tokens_kept_aligned']:.1f}")
    print(f"Avg response tokens lost:          {summary['avg_response_tokens_lost']:.1f}")
    print(f"Max raw response tokens:           {summary['max_response_tokens_raw']}")
    print(f"Max kept aligned response tokens:  {summary['max_response_tokens_kept_aligned']}")
    print("")
    print(
        "Teacher total tokens p50/p90/p95/p99: "
        f"{summary['teacher_total_tokens_raw_p50']} / "
        f"{summary['teacher_total_tokens_raw_p90']} / "
        f"{summary['teacher_total_tokens_raw_p95']} / "
        f"{summary['teacher_total_tokens_raw_p99']}"
    )
    print(
        "Student total tokens p50/p90/p95/p99: "
        f"{summary['student_total_tokens_raw_p50']} / "
        f"{summary['student_total_tokens_raw_p90']} / "
        f"{summary['student_total_tokens_raw_p95']} / "
        f"{summary['student_total_tokens_raw_p99']}"
    )
    print("")
    print("Worst Truncated Samples:")
    for sample in summary["worst_truncated_samples"]:
        lost = sample["response_tokens_raw"] - sample["aligned_response_tokens_kept"]
        print(
            f"  idx={sample['idx']} "
            f"raw_response={sample['response_tokens_raw']} "
            f"kept_aligned={sample['aligned_response_tokens_kept']} "
            f"lost={lost} "
            f"student_total_raw={sample['student_total_tokens_raw']} "
            f"teacher_total_raw={sample['teacher_total_tokens_raw']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze truncation under KL training sequence construction.")
    parser.add_argument("--model_path", type=str, required=True, help="Tokenizer/model path.")
    parser.add_argument("--data_path", type=str, required=True, help="Primary parquet used by KL training.")
    parser.add_argument("--kl_type", choices=["reverse", "forward"], required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--corrected_responses_path", type=str, default="")
    parser.add_argument("--use_initial_response", type=parse_bool, default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--strict_zero_truncation", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    data = datasets.load_dataset("parquet", data_files=args.data_path, split="train")
    corrected = None
    if args.kl_type == "forward" and args.corrected_responses_path:
        corrected = datasets.load_dataset("parquet", data_files=args.corrected_responses_path, split="train")
        if len(corrected) != len(data):
            raise ValueError(
                f"corrected_responses_path length mismatch: {len(corrected)} vs primary data {len(data)}"
            )

    if args.max_samples is not None:
        limit = min(args.max_samples, len(data))
        data = data.select(range(limit))
        if corrected is not None:
            corrected = corrected.select(range(limit))

    stats = []
    for idx in range(len(data)):
        corrected_item = corrected[idx] if corrected is not None else None
        stats.append(
            analyze_sample(
                tokenizer,
                data[idx],
                idx=idx,
                kl_type=args.kl_type,
                max_length=args.max_length,
                use_initial_response=args.use_initial_response,
                corrected_item=corrected_item,
            )
        )

    summary = summarize(stats, args.max_length)
    print_summary(summary)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")

    if args.strict_zero_truncation and summary["any_truncated_samples"] > 0:
        print("")
        print("ERROR: truncation detected while --strict_zero_truncation is enabled.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
