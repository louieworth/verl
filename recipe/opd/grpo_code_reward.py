# Copyright 2026
#
# Adapter for running code GRPO with the standard verl naive reward manager.

from verl.utils.reward_score import prime_code


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    score, metadata = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    return {"score": float(score), "metadata": metadata, "data_source": data_source}
