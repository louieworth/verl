#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Stage 2 Data Preparation v3: Minimal-edit-first + Few-shot examples from successful corrections
#
# Key improvements:
#   1. Uses successful corrections as few-shot examples
#   2. Minimal-edit-first strategy
#   3. Selects relevant examples based on data source/topic

import os
import argparse
import datasets
import random
import json
from typing import Any, Dict, List

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

# 加载成功样例
FEWSHOT_FILE = "recipe/open_math_reasoning/fewshot_examples_stage2_success_top100.json"


def load_fewshot_examples(n_examples: int = 3) -> List[Dict]:
    """从文件加载 few-shot 样例"""
    try:
        with open(FEWSHOT_FILE, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        # 随机采样 n_examples 个
        if len(examples) > n_examples:
            examples = random.sample(examples, n_examples)
        return examples
    except Exception as e:
        print(f"Warning: Could not load fewshot examples from {FEWSHOT_FILE}: {e}")
        return []


def format_fewshot_example(example: Dict) -> str:
    """格式化单个 few-shot 样例"""
    return f"""
### Example

**Problem:**
{example['problem']}

**Initial Solution (Wrong):**
{example['initial_response_short'][:500]}

**Expert Solution (Correct):**
(Referenced for correction)

**Corrected Solution:**
{example['corrected_response_short'][:600]}

---
"""


PROMPT_TEMPLATE_STEP_2_V3 = """Your task is to correct a wrong mathematical solution using the expert solution as reference.

{FEWSHOT_EXAMPLES}

**Now, correct this problem:**

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
- Study the examples above to see how corrections are done
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


def _reward_from_example(example: Dict[str, Any]) -> int:
    """Fetch reward from example['extra_info'] robustly."""
    extra_info = example.get("extra_info", {}) or {}
    return _normalize_reward_value(extra_info.get("reward", 0))


def make_map_fn_stage2_reward0(n_fewshot: int = 3):
    """
    Map function for reward == 0 examples to produce correction inputs.
    """
    # 加载 few-shot 样例
    fewshot_examples = load_fewshot_examples(n_fewshot)
    fewshot_text = ""
    for ex in fewshot_examples:
        fewshot_text += format_fewshot_example(ex)

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

        # Build correction prompt with few-shot examples
        prompt_content = (
            PROMPT_TEMPLATE_STEP_2_V3
            .replace("{FEWSHOT_EXAMPLES}", fewshot_text)
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
        default="gen_results/stage2/deepscaleR_stage2_reward0_correction_v3.parquet",
        help="Output file for reward=0 correction dataset",
    )
    parser.add_argument(
        "--output_reward1",
        default="gen_results/stage2/deepscaleR_stage2_reward1.parquet",
        help="Output file for reward=1 untouched dataset",
    )
    parser.add_argument(
        "--n_fewshot",
        type=int,
        default=3,
        help="Number of few-shot examples to include (default: 3)",
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
    print(f"Building correction dataset (v3: minimal-edit + {args.n_fewshot}-shot) for reward == 0...")
    ds_r0 = ds_r0.map(
        function=make_map_fn_stage2_reward0(args.n_fewshot),
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

    print(f"\nStage 2 data preparation (v3 with {args.n_fewshot}-shot) completed!")
