#!/usr/bin/env python3
"""Run LiveCodeBench with a zero-shot raw Base-model completion prompt.

LiveCodeBench's model registry controls prompt style independently of the local
checkpoint. This wrapper keeps the upstream runner/evaluator intact while
replacing only its GenericBase prompt formatter and enforcing the 2K prompt cap.
"""

from __future__ import annotations

import os
import sys

from transformers import AutoTokenizer


def argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} is required by the Base-prompt wrapper") from exc


def build_base_prompt(problem) -> str:
    prompt = f"Problem:\n{problem.question_content.strip()}\n\n"
    if problem.starter_code:
        prompt += f"Starter code:\n```python\n{problem.starter_code.rstrip()}\n```\n\n"
    prompt += (
        "Write a complete, correct Python program that solves the problem and passes all tests. "
        "Read input from stdin and write the answer to stdout when the problem requires it. "
        "Return only raw Python source code. Do not use Markdown fences or explanations.\n"
    )
    return prompt


def main() -> None:
    model_path = argument_value("--local_model_path")
    prompt_limit = int(os.environ.get("LCB_BASE_PROMPT_MAX_TOKENS", "2048"))
    seed = int(os.environ.get("LCB_BASE_SEED", "42"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    from lcb_runner.runner import scenario_router
    from lcb_runner.runner.main import main as lcb_main
    from lcb_runner.runner.vllm_runner import VLLMRunner

    original_runner_init = VLLMRunner.__init__

    def seeded_runner_init(self, args, model):
        original_runner_init(self, args, model)
        self.sampling_params.seed = seed

    VLLMRunner.__init__ = seeded_runner_init

    def format_base_prompt(problem, _model_style):
        prompt = build_base_prompt(problem)
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count > prompt_limit:
            raise ValueError(
                f"LiveCodeBench task {problem.question_id} has a {token_count}-token prompt, "
                f"exceeding the {prompt_limit}-token evaluation cap; refusing to truncate."
            )
        return prompt

    scenario_router.format_prompt_generation = format_base_prompt
    lcb_main()


if __name__ == "__main__":
    main()
