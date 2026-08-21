#!/usr/bin/env python3
"""DeepCoder-style sparse execution reward for canonical TACO GRPO."""

import json
import os
import sys

from verl.utils.reward_score.prime_code import apps_check_correctness

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

REWARD_CONTRACT = os.environ.get(
    "CODE_GRPO_REWARD_CONTRACT", "deepcoder_binary_15_longest_v1"
)
MAX_SELECTED_TEST_CASES = int(os.environ.get("CODE_GRPO_MAX_TEST_CASES", "15"))
EXEC_TIMEOUT_SECONDS = int(os.environ.get("CODE_GRPO_EXEC_TIMEOUT_SECONDS", "10"))
if (
    REWARD_CONTRACT != "deepcoder_binary_15_longest_v1"
    or MAX_SELECTED_TEST_CASES != 15
    or EXEC_TIMEOUT_SECONDS != 10
):
    raise RuntimeError(
        "canonical code GRPO requires CODE_GRPO_REWARD_CONTRACT="
        "deepcoder_binary_15_longest_v1, CODE_GRPO_MAX_TEST_CASES=15, and "
        "CODE_GRPO_EXEC_TIMEOUT_SECONDS=10"
    )


def extract_python_source(completion: str) -> str:
    """Accept the canonical raw-source response and legacy Python fences."""

    if "```python" in completion:
        return completion.rsplit("```python", 1)[-1].split("```", 1)[0].strip()
    return completion.strip()


def compute_binary_execution_reward(solution_str: str, ground_truth: str | dict) -> float:
    """Return 1 only when every prepared reward test passes.

    The canonical artifact contains at most 15 deterministically selected
    longest-input tests.  The existing verl executor applies a global timeout
    and returns non-True sentinels for compile, runtime, timeout, and wrong
    answer failures; every such outcome maps to zero reward.
    """

    try:
        test_cases = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if not isinstance(test_cases, dict):
            return 0.0
        inputs = test_cases.get("inputs")
        outputs = test_cases.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs:
            return 0.0
        if len(inputs) != len(outputs) or len(inputs) > MAX_SELECTED_TEST_CASES:
            return 0.0
        results, _metadata = apps_check_correctness(
            in_outs=test_cases,
            generation=extract_python_source(solution_str),
            timeout=EXEC_TIMEOUT_SECONDS,
            debug=False,
        )
        return float(len(results) == len(inputs) and all(result is True for result in results))
    except Exception:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return compute_binary_execution_reward(solution_str, ground_truth)
