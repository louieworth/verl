#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Stage 2 Data Preparation for KL Forward Training
# Input: Stage 1 output with responses and rewards
# Output: A stage2 prompt dataset over reward==0 samples only

import os
import argparse
import datasets
from typing import Any, Dict

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


PROMPT_TEMPLATE_STAGE_2_REWRITE_REWARD0 = """
{PROBLEM}

Here is a reference solution:
{EXPERT_SOLUTION}

After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:
"""


PROMPT_TEMPLATE_STAGE_2_WITH_INITIAL_RESPONSE_REWARD0 = """
Your task is to rewrite your mathematical solution using the reference solution as guidance.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Reference Solution:**
{EXPERT_SOLUTION}

**Instructions:**
1. Review the reference solution to understand the target reasoning and method
2. Rewrite your solution so it is consistent with the reference solution
3. Keep useful parts of your original structure and style when appropriate
4. Output ONLY the rewritten solution
"""


def _normalize_reward_value(val: Any) -> int:
    """Normalize a raw reward value (possibly list) into an int."""
    if isinstance(val, list):
        val = val[0] if val else 0
    try:
        return int(val)
    except Exception:
        return 0


def _reward_from_example(example: Dict[str, Any]) -> int:
    """Fetch reward from example['extra_info'] robustly."""
    extra_info = example.get("extra_info", {}) or {}
    return _normalize_reward_value(extra_info.get("reward", 0))


def make_map_fn_stage2_rewrite_reward0(use_initial_response: bool):
    """Map reward==0 stage1 samples into stage2 prompts."""

    def process_fn(example: Dict[str, Any]):
        extra_info = (example.get("extra_info", {}) or {}).copy()
        reward = _normalize_reward_value(extra_info.get("reward", 0))
        extra_info["reward"] = reward

        responses_val = example.get("responses", [""])
        initial_response = responses_val[0] if isinstance(responses_val, list) else responses_val
        extra_info["initial_response"] = initial_response

        problem = extra_info.get("problem", "")
        expert_solution = extra_info.get("expert_cot", "")

        if use_initial_response:
            prompt_content = (
                PROMPT_TEMPLATE_STAGE_2_WITH_INITIAL_RESPONSE_REWARD0
                .replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
                .replace("{EXPERT_SOLUTION}", expert_solution)
            )
        else:
            prompt_content = (
                PROMPT_TEMPLATE_STAGE_2_REWRITE_REWARD0
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1_output",
        required=True,
        help="Stage 1 output parquet file (with responses and rewards)",
    )
    parser.add_argument(
        "--output_file",
        default="gen_results/stage2/deepscaleR_stage2_reward0_prompts.parquet",
        help="Output file for reward==0 stage2 prompts",
    )
    parser.add_argument(
        "--use_initial_response",
        type=lambda x: x.lower() == "true",
        default=False,
        help="If true, build rewrite prompts that include the stage1 response; otherwise use reference-only rewrite prompts.",
    )
    args = parser.parse_args()

    print(f"Loading Stage 1 output: {args.stage1_output}")
    ds_raw = datasets.load_dataset("parquet", data_files=args.stage1_output, split="train")

    print("Filtering reward==0 samples...")
    ds_r0 = ds_raw.filter(lambda ex: _reward_from_example(ex) == 0)

    prompt_mode = "rewrite_with_initial_response" if args.use_initial_response else "rewrite_from_expert"
    print(f"Building reward==0 stage2 dataset... mode={prompt_mode}")
    ds_rewrite = ds_r0.map(
        function=make_map_fn_stage2_rewrite_reward0(args.use_initial_response),
        remove_columns=["responses"],
    )

    print("\nDataset stats:")
    print(f"  - Total stage1 samples: {len(ds_raw)}")
    print(f"  - Reward==0 stage2 prompts: {len(ds_rewrite)}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving reward==0 stage2 prompt dataset to: {args.output_file}")
    ds_rewrite.to_parquet(args.output_file)
    print("\nStage 2 reward==0 data preparation completed!")
