#!/usr/bin/env python3
"""Functional, CPU-only verification of the canonical code reward runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def verify_runtime(evalplus_version: str, vllm_version: str, lcb_repo: Path) -> dict[str, str]:
    installed_evalplus = importlib.metadata.version("evalplus")
    if installed_evalplus != evalplus_version:
        raise RuntimeError(
            f"evalplus version mismatch: installed={installed_evalplus}, expected={evalplus_version}"
        )
    installed_vllm = importlib.metadata.version("vllm")
    if installed_vllm != vllm_version:
        raise RuntimeError(f"vllm version mismatch: installed={installed_vllm}, expected={vllm_version}")

    # LiveCodeBench reads prompt fixtures relative to cwd during import. Match
    # the evaluator's runtime contract instead of merely finding the module.
    previous_cwd = Path.cwd()
    try:
        os.chdir(lcb_repo)
        import lcb_runner.runner.main  # noqa: F401
    finally:
        os.chdir(previous_cwd)
    import vllm  # noqa: F401

    from recipe.opd.run.grpo.code_reward import compute_binary_execution_reward

    tests = json.dumps({"inputs": ["2\n"], "outputs": ["4\n"]})
    correct = compute_binary_execution_reward("value = int(input())\nprint(value * 2)", tests)
    incorrect = compute_binary_execution_reward("print(0)", tests)
    if correct != 1.0 or incorrect != 0.0:
        raise RuntimeError(
            f"local code execution smoke test failed: correct={correct}, incorrect={incorrect}"
        )
    return {
        "evalplus": installed_evalplus,
        "vllm": installed_vllm,
        "code_execution": "pass",
        "deepcoder_reward": "correct=1,incorrect=0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalplus-version", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--lcb-repo", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_runtime(args.evalplus_version, args.vllm_version, args.lcb_repo.resolve()),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
