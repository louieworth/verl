#!/usr/bin/env python3
"""DeepScaleR reward with a dependency-free fallback grader."""

import re

from verl.utils.reward_score import math_reward as builtin_math_reward


FORMAT_PATTERN = re.compile(
    r"<think>.*</think>.*<answer>.*\\boxed\{.*\}.*</answer>",
    re.DOTALL,
)


def answer_is_correct(solution_str, ground_truth):
    try:
        from mathruler.grader import extract_boxed_content, grade_answer

        return bool(grade_answer(extract_boxed_content(solution_str), ground_truth))
    except ImportError:
        return builtin_math_reward.compute_score(solution_str, ground_truth) > 0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    solution_str = solution_str.strip()
    score = 0.33 if re.fullmatch(FORMAT_PATTERN, solution_str) else 0.0
    if answer_is_correct(solution_str, ground_truth):
        score += 0.67
    return score
