#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# y_r prompt preparation — build stage2 rewrite prompts for every stage1 row.
#
# A "y_r" example is the teacher's rewrite of the student's stage1 attempt.
# This script takes the stage1 parquet (student rollouts) and produces the
# prompt that the teacher will rollout against to generate y_r.
#
# The generation prompt is FIXED (always the refine variant — teacher reads
# the initial response). The two templates differ only by whether they show
# the expert solution:
#
#   --distill_mode opsd  →  refine OPSD: π_T(·|x, y*, y_o)
#   --distill_mode opd   →  refine OPD:  π_T(·|x, y_o)
#
# The training-time teacher conditioning is a separate knob (driven by
# TEACHER_TRAINING_PROMPT in run_kl_training.sh / use_initial_response in
# config.py) and may be vanilla or refine independently.
#
# Always processes every stage1 row. If you want to train only on rewrites of
# failed samples, do the reward filtering downstream via
# `recipe/opd/dataset/filter_stage2_by_reward.py`.

import argparse
import os
from typing import Any, Dict

import datasets
from transformers import AutoTokenizer

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
code_instruction_following = (
    "Return only raw corrected Python source code. Do not use Markdown fences or explanations."
)


# refine OPSD: teacher sees problem + expert solution + initial response.
PROMPT_TEMPLATE_OPSD_REFINE = """
Your task is to rewrite your mathematical solution using the reference solution as guidance.

**Problem:**
{PROBLEM}

**Reference Solution:**
{EXPERT_SOLUTION}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Review the reference solution to understand the target reasoning and method
2. Rewrite your solution so it is consistent with the reference solution
3. Keep useful parts of your original structure and style when appropriate
4. Output ONLY the rewritten solution
"""

# refine OPD: teacher sees problem + initial response (no expert reference).
PROMPT_TEMPLATE_OPD_REFINE = """
Your task is to rewrite your mathematical solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Preserve the overall structure and reasoning path of your original solution
2. Identify and fix errors in computation or logic
3. Keep correct intermediate steps and meaningful work
4. Output ONLY the rewritten solution
"""



PROMPT_TEMPLATE_CODE_OPSD_REFINE = """
Your task is to rewrite your Python solution using the reference solution as guidance.

**Problem:**
{PROBLEM}

**Reference Solution:**
```python
{EXPERT_SOLUTION}
```

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Fix correctness issues and edge cases
2. Preserve useful parts of the original approach when appropriate
3. Output ONLY the rewritten Python solution
"""

PROMPT_TEMPLATE_CODE_OPD_REFINE = """
Your task is to rewrite your Python solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Fix correctness issues and edge cases
2. Preserve useful parts of the original approach when appropriate
3. Output ONLY the rewritten Python solution
"""

def _normalize_reward_value(val: Any) -> int:
    if isinstance(val, list):
        val = val[0] if val else 0
    try:
        return int(val)
    except Exception:
        return 0


def _fit_rewrite_prompt(
    render_prompt,
    expert_solution,
    initial_response,
    tokenizer,
    max_prompt_tokens,
    teacher_enable_thinking: bool = False,
    teacher_use_chat_template: bool = False,
):
    """Trim derived fields while preserving the problem and rewrite instructions."""
    fields = {"expert": str(expert_solution), "initial": str(initial_response)}
    prompt = render_prompt(fields["expert"], fields["initial"])
    if tokenizer is None or max_prompt_tokens is None:
        return prompt

    def token_ids(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def prompt_size(text):
        if teacher_use_chat_template:
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=teacher_enable_thinking,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "OPD teacher chat mode requires a tokenizer chat template "
                    "that supports enable_thinking."
                ) from exc
            return len(rendered)
        # Plain completion appends exactly one newline separator.
        return len(token_ids(text + "\n"))

    truncation_marker = "\n[... truncated to fit the native context ...]"

    # y* is authoritative for OPSD, so trim the potentially 16K student draft
    # first. If necessary, trim y* second. Binary-search the retained token
    # prefix because decode/re-encode and the marker itself make a one-shot
    # overflow subtraction unreliable for BPE tokenizers.
    for field_name in ("initial", "expert"):
        if prompt_size(prompt) <= max_prompt_tokens:
            break
        original_ids = token_ids(fields[field_name])
        if not original_ids:
            continue

        def candidate(keep: int) -> tuple[str, str, int]:
            value = tokenizer.decode(original_ids[:keep], skip_special_tokens=True).strip()
            if keep < len(original_ids):
                value += truncation_marker
            candidate_fields = fields.copy()
            candidate_fields[field_name] = value
            candidate_prompt = render_prompt(candidate_fields["expert"], candidate_fields["initial"])
            return value, candidate_prompt, prompt_size(candidate_prompt)

        low, high = 0, len(original_ids)
        best: tuple[str, str] | None = None
        while low <= high:
            mid = (low + high) // 2
            value, candidate_prompt, candidate_size = candidate(mid)
            if candidate_size <= max_prompt_tokens:
                best = (value, candidate_prompt)
                low = mid + 1
            else:
                high = mid - 1

        if best is not None:
            fields[field_name], prompt = best
        else:
            # The other derived field still consumes too much space. Fully
            # trim this lower-priority field, then continue with the next one.
            fields[field_name] = truncation_marker.lstrip("\n")
            prompt = render_prompt(fields["expert"], fields["initial"])

    prompt_tokens = prompt_size(prompt)
    if prompt_tokens > max_prompt_tokens:
        raise ValueError(
            f"Static rewrite prompt has {prompt_tokens} tokens after field truncation; "
            f"cap is {max_prompt_tokens}"
        )
    return prompt


def make_map_fn(
    distill_mode: str,
    task: str = "math",
    *,
    tokenizer=None,
    max_prompt_tokens: int | None = None,
    teacher_enable_thinking: bool = False,
):
    def process_fn(example: Dict[str, Any]):
        extra_info = (example.get("extra_info", {}) or {}).copy()
        extra_info["reward"] = _normalize_reward_value(extra_info.get("reward", 0))

        responses_val = example.get("responses", [""])
        initial_response = responses_val[0] if isinstance(responses_val, list) else responses_val
        extra_info["initial_response"] = initial_response

        problem = extra_info.get("problem", "")
        expert_solution = extra_info.get("expert_cot", "")

        def render_prompt(expert_text: str, initial_text: str) -> str:
            if task == "code":
                if distill_mode == "opd":
                    rendered = (
                        PROMPT_TEMPLATE_CODE_OPD_REFINE
                        .replace("{PROBLEM}", problem)
                        .replace("{INITIAL_RESPONSE}", initial_text)
                    )
                else:
                    rendered = (
                        PROMPT_TEMPLATE_CODE_OPSD_REFINE
                        .replace("{PROBLEM}", problem)
                        .replace("{INITIAL_RESPONSE}", initial_text)
                        .replace("{EXPERT_SOLUTION}", expert_text)
                    )
                return rendered.strip() + "\n\n" + code_instruction_following
            if distill_mode == "opd":
                rendered = (
                    PROMPT_TEMPLATE_OPD_REFINE
                    .replace("{PROBLEM}", problem)
                    .replace("{INITIAL_RESPONSE}", initial_text)
                )
            else:
                rendered = (
                    PROMPT_TEMPLATE_OPSD_REFINE
                    .replace("{PROBLEM}", problem)
                    .replace("{INITIAL_RESPONSE}", initial_text)
                    .replace("{EXPERT_SOLUTION}", expert_text)
                )
            return rendered.strip() + "\n\n" + instruction_following

        prompt_content = _fit_rewrite_prompt(
            render_prompt,
            expert_solution,
            initial_response,
            tokenizer,
            max_prompt_tokens,
            teacher_enable_thinking,
            teacher_use_chat_template=(distill_mode == "opd" and teacher_enable_thinking),
        )
        # KL must condition the teacher on exactly the same (possibly
        # truncated) rewrite prompt used for y_r generation. The prompt itself
        # stays in row["prompt"] to avoid duplicating a potentially 16K field.
        extra_info["teacher_prompt_contract"] = "srd_generation_prompt_v1"

        row = {
            "data_source": example.get("data_source", "deepscaleR"),
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": example.get("ability", "code" if task == "code" else "math"),
            "extra_info": extra_info,
        }
        if task != "code":
            row["reward_model"] = example.get("reward_model")
        return row

    return process_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1_output",
        required=True,
        help="Stage 1 output parquet (with responses and reward).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output parquet for y_r prompts.",
    )
    parser.add_argument(
        "--task",
        choices=["math", "code"],
        default="math",
        help="Task-specific rewrite prompt adapter.",
    )
    parser.add_argument(
        "--distill_mode",
        choices=["opsd", "opd"],
        default="opsd",
        help="opsd: teacher sees expert solution + initial response. "
             "opd:  teacher sees only initial response (no expert reference).",
    )
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument(
        "--teacher_enable_thinking",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        default=False,
    )
    args = parser.parse_args()

    if args.distill_mode == "opsd" and args.teacher_enable_thinking:
        raise ValueError("OPSD uses a Base self-teacher and cannot enable teacher thinking mode")

    if (args.tokenizer_path and args.max_prompt_tokens is None) or (
        args.max_prompt_tokens is not None and not args.tokenizer_path
    ):
        raise ValueError("--tokenizer_path and --max_prompt_tokens must be supplied together")
    tokenizer = None
    if args.tokenizer_path:
        if args.max_prompt_tokens <= 0:
            raise ValueError("--max_prompt_tokens must be positive")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=True)

    print(f"Loading Stage 1 output: {args.stage1_output}")
    ds_raw = datasets.load_dataset("parquet", data_files=args.stage1_output, split="train")

    print(f"Building y_r prompts ({len(ds_raw)} rows, refine {args.distill_mode.upper()})...")
    ds_out = ds_raw.map(
        function=make_map_fn(
            args.distill_mode,
            args.task,
            tokenizer=tokenizer,
            max_prompt_tokens=args.max_prompt_tokens,
            teacher_enable_thinking=args.teacher_enable_thinking,
        ),
        remove_columns=["responses"],
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(ds_out)} y_r prompts to: {args.output_file}")
    ds_out.to_parquet(args.output_file)
    print("y_r prompt preparation completed.")


if __name__ == "__main__":
    main()
