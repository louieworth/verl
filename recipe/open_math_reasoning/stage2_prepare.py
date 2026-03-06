#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Stage 2 Data Preparation: Prepare correction data for incorrect responses
# Input: Stage 1 output with responses and rewards
# Output: Two files - reward=0 (correction) and reward=1 (untouched)

import os
import argparse
import datasets
from typing import Any, Dict

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


PROMPT_TEMPLATE_STEP_2 = """
Your task is to correct a wrong mathematical solution by referring to an expert solution.

**Problem:**
{PROBLEM}

**Initial Solution (Wrong):**
{INITIAL_RESPONSE}

**Expert Solution (Correct):**
{EXPERT_SOLUTION}

**Instructions:**
Review the expert solution to understand the correct approach, then correct your initial solution.
- Reference the expert solution's reasoning and method
- Revise your initial solution accordingly
- Do NOT copy the expert solution word-for-word
- Output ONLY the corrected solution with necessary steps, no explanations or reasoning about the correction process
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


def make_map_fn_stage2_reward0():
    """
    Map function for reward == 0 examples to produce correction inputs.
    """
    def process_fn(example):
        extra_info = example["extra_info"].copy()
        reward = _normalize_reward_value(extra_info.get("reward", 0))
        extra_info["reward"] = reward

        # Extract initial response (string)
        responses_val = example.get("responses", [""])
        initial_response = responses_val[0] if isinstance(responses_val, list) else responses_val

        # Put responses into extra_info
        extra_info["responses"] = initial_response

        # Fields for constructing the correction prompt
        problem = extra_info.get("problem", "")
        expert_solution = extra_info.get("expert_cot", "")

        # Remove expert_cot from extra_info (to avoid duplication/leak)
        if "expert_cot" in extra_info:
            extra_info.pop("expert_cot")

        # Build the simplified correction prompt
        prompt_content = (
            PROMPT_TEMPLATE_STEP_2
            .replace("{PROBLEM}", problem)
            .replace("{INITIAL_RESPONSE}", initial_response)
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
        "--output_reward0",
        default="gen_results/stage2/deepscaleR_stage2_reward0_correction.parquet",
        help="Output file for reward=0 correction dataset",
    )
    parser.add_argument(
        "--output_reward1",
        default="gen_results/stage2/deepscaleR_stage2_reward1.parquet",
        help="Output file for reward=1 untouched dataset",
    )
    args = parser.parse_args()

    # Load Stage 1 output
    print(f"Loading Stage 1 output: {args.stage1_output}")
    ds_raw = datasets.load_dataset("parquet", data_files=args.stage1_output, split="train")

    # Split by reward
    print("Splitting dataset by reward...")
    ds_r0 = ds_raw.filter(lambda ex: _reward_from_example(ex) == 0)
    ds_r1 = ds_raw.filter(lambda ex: _reward_from_example(ex) == 1)

    # Transform reward=0 dataset
    print("Building correction dataset for reward == 0...")
    ds_r0 = ds_r0.map(
        function=make_map_fn_stage2_reward0(),
        remove_columns=["responses"],
    )

    # Keep reward=1 dataset unchanged
    print("Preparing untouched dataset for reward == 1...")
    ds_r1_untouched = ds_r1

    # Stats
    total = len(ds_raw)
    num_r0 = len(ds_r0)
    num_r1 = len(ds_r1_untouched)
    print(f"\nDataset stats:")
    print(f"  - Total: {total}")
    print(f"  - reward == 0 (correction): {num_r0}")
    print(f"  - reward == 1 (untouched): {num_r1}")

    # Create output directories
    for out_path in [args.output_reward0, args.output_reward1]:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # Save outputs
    print(f"\nSaving reward=0 correction dataset to: {args.output_reward0}")
    ds_r0.to_parquet(args.output_reward0)

    print(f"Saving reward=1 untouched dataset to: {args.output_reward1}")
    ds_r1_untouched.to_parquet(args.output_reward1)

    print("\nStage 2 data preparation completed!")
