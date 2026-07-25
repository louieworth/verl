#!/usr/bin/env python3
"""Continuous TACO execution reward for GRPO."""

import sys

from verl.utils.reward_score import prime_code

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    score, _metadata = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    return float(score)
