#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Stage 2 Data Preparation for KL Forward Training
# Input: Stage 1 output with responses and rewards
# Output: A stage2 prompt dataset over all samples, regardless of reward

import os
import argparse
import datasets
from typing import Any, Dict

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


PROMPT_TEMPLATE_STAGE_2_REWRITE_ALL = """
{PROBLEM}

Here is a reference solution:
{EXPERT_SOLUTION}

After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:
"""


PROMPT_TEMPLATE_STAGE_2_CORRECT_ALL = """
Your task is to correct your wrong mathematical solution using the expert solution as reference.

**Problem:**
{PROBLEM}

**Your Initial Solution (Wrong):**
{INITIAL_RESPONSE}

**Expert Solution (Correct):**
{EXPERT_SOLUTION}

**Correction Strategy:**
1. First, try to MINIMALLY EDIT your initial solution:
   - Keep your original structure, style, and flow
   - Only change specific wrong steps/numbers/equations
   - Preserve your original wording and explanations where correct

2. If minimal editing is NOT feasible (e.g., fundamental approach error):
   - Then rewrite using the expert solution's approach
   - But still try to maintain your original style and format

**Key Principles:**
- Prefer MINIMAL EDITS over complete rewrites
- Stay as close as possible to your original solution style
- Only use the expert solution to identify and fix specific errors
- Output ONLY the corrected solution, no meta-commentary

Please provide your corrected solution:
"""


def _normalize_reward_value(val: Any) -> int:
    """Normalize a raw reward value (possibly list) into an int."""
    if isinstance(val, list):
        val = val[0] if val else 0
    try:
        return int(val)
    except Exception:
        return 0


def make_map_fn_stage2_rewrite_all(use_initial_response: bool):
    """Map every stage1 sample into a stage2 prompt."""

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
                PROMPT_TEMPLATE_STAGE_2_CORRECT_ALL
                .replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
                .replace("{EXPERT_SOLUTION}", expert_solution)
            )
        else:
            prompt_content = (
                PROMPT_TEMPLATE_STAGE_2_REWRITE_ALL
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
        default="gen_results/stage2/deepscaleR_stage2_prompts.parquet",
        help="Output file for all-sample stage2 prompts",
    )
    parser.add_argument(
        "--use_initial_response",
        type=lambda x: x.lower() == "true",
        default=False,
        help="If true, build correction prompts around the stage1 response; otherwise use reverse-style rewrite prompts.",
    )
    args = parser.parse_args()

    print(f"Loading Stage 1 output: {args.stage1_output}")
    ds_raw = datasets.load_dataset("parquet", data_files=args.stage1_output, split="train")

    prompt_mode = "correct_initial_response" if args.use_initial_response else "rewrite_from_expert"
    print(f"Building stage2 prompt dataset over all samples... mode={prompt_mode}")
    ds_rewrite = ds_raw.map(
        function=make_map_fn_stage2_rewrite_all(args.use_initial_response),
        remove_columns=["responses"],
    )

    print("\nDataset stats:")
    print(f"  - Total stage2 prompts: {len(ds_rewrite)}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving stage2 prompt dataset to: {args.output_file}")
    ds_rewrite.to_parquet(args.output_file)
    print("\nStage 2 all-sample data preparation completed!")
