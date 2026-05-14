#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# y_r prompt preparation — build stage2 rewrite prompts for every stage1 row.
#
# A "y_r" example is the teacher's *rewritten* response to a problem, used as
# the training target for forward KL on stage2 data. This script takes the
# stage1 parquet (student rollouts with reward) and produces the prompt that
# the teacher will rollout against to generate y_r.
#
# Always processes every stage1 row. If you want to train only on rewrites of
# failed samples, do the reward filtering downstream via
# `recipe/opd/dataset/filter_stage2_by_reward.py`.

import argparse
import os
from typing import Any, Dict

import datasets

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


PROMPT_TEMPLATE_REWRITE_FROM_EXPERT = """
{PROBLEM}

Here is a reference solution:
{EXPERT_SOLUTION}

After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:
"""

PROMPT_TEMPLATE_REWRITE_WITH_INITIAL_RESPONSE = """
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


def _normalize_reward_value(val: Any) -> int:
    if isinstance(val, list):
        val = val[0] if val else 0
    try:
        return int(val)
    except Exception:
        return 0


def make_map_fn(use_initial_response: bool):
    def process_fn(example: Dict[str, Any]):
        extra_info = (example.get("extra_info", {}) or {}).copy()
        extra_info["reward"] = _normalize_reward_value(extra_info.get("reward", 0))

        responses_val = example.get("responses", [""])
        initial_response = responses_val[0] if isinstance(responses_val, list) else responses_val
        extra_info["initial_response"] = initial_response

        problem = extra_info.get("problem", "")
        expert_solution = extra_info.get("expert_cot", "")

        if use_initial_response:
            prompt_content = (
                PROMPT_TEMPLATE_REWRITE_WITH_INITIAL_RESPONSE
                .replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
                .replace("{EXPERT_SOLUTION}", expert_solution)
            )
        else:
            prompt_content = (
                PROMPT_TEMPLATE_REWRITE_FROM_EXPERT
                .replace("{PROBLEM}", problem)
                .replace("{EXPERT_SOLUTION}", expert_solution)
            )
        prompt_content = prompt_content + " " + instruction_following

        return {
            "data_source": example.get("data_source", "deepscaleR"),
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": example.get("ability", "math"),
            "reward_model": example.get("reward_model"),
            "extra_info": extra_info,
        }

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
        "--use_initial_response",
        type=lambda x: x.lower() == "true",
        default=False,
        help="If true, embed the stage1 response in the prompt; otherwise reference-only.",
    )
    args = parser.parse_args()

    print(f"Loading Stage 1 output: {args.stage1_output}")
    ds_raw = datasets.load_dataset("parquet", data_files=args.stage1_output, split="train")

    prompt_mode = "rewrite_with_initial_response" if args.use_initial_response else "rewrite_from_expert"
    print(f"Building y_r prompts ({len(ds_raw)} rows, prompt={prompt_mode})...")
    ds_out = ds_raw.map(
        function=make_map_fn(args.use_initial_response),
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
