#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
# Avoid shadowing HuggingFace `datasets` with recipe/code_evaluation/datasets.
sys.path = [path for path in sys.path if Path(path or os.getcwd()).resolve() != SCRIPT_DIR]

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evalplus.evaluate import evaluate
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import extra_eos_for_direct_completion, make_raw_chat_prompt


# EvalPlus's sanitizer recursively walks the generated AST. Pathological model
# outputs can exceed Python's recursion limit and abort the entire resumable
# generation job. Treat a sanitizer failure like any other invalid solution:
# preserve the raw output, emit an empty sanitized solution, and keep going.
_evalplus_codegen = importlib.import_module("evalplus.codegen")
_evalplus_sanitize = _evalplus_codegen.sanitize
_sanitize_timeout_seconds = float(os.environ.get("EVALPLUS_SANITIZE_TIMEOUT_SECONDS", "3"))
_sanitize_max_chars = int(os.environ.get("EVALPLUS_SANITIZE_MAX_CHARS", "50000"))
_sanitize_max_lines = int(os.environ.get("EVALPLUS_SANITIZE_MAX_LINES", "400"))


def _raise_sanitize_timeout(_signum, _frame) -> None:
    raise TimeoutError(
        f"EvalPlus sanitizer exceeded {_sanitize_timeout_seconds:g} seconds"
    )


def _safe_sanitize(*args, **kwargs) -> str:
    code = args[0] if args else kwargs.get("code", "")
    if isinstance(code, str):
        char_count = len(code)
        line_count = code.count("\n") + 1
        if char_count > _sanitize_max_chars or line_count > _sanitize_max_lines:
            print(
                "WARNING: EvalPlus sanitizer input exceeds the safety limit "
                f"(chars={char_count}, lines={line_count}); recording this sample "
                "as an empty solution.",
                file=sys.stderr,
            )
            return ""
    use_timeout = (
        _sanitize_timeout_seconds > 0
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
    )
    previous_handler = None
    previous_timer = (0.0, 0.0)
    started_at = time.monotonic()
    try:
        if use_timeout:
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_sanitize_timeout)
            previous_timer = signal.setitimer(
                signal.ITIMER_REAL, _sanitize_timeout_seconds
            )
        return _evalplus_sanitize(*args, **kwargs)
    except Exception as exc:
        print(
            f"WARNING: EvalPlus sanitizer failed ({type(exc).__name__}: {exc}); "
            "recording this sample as an empty solution.",
            file=sys.stderr,
        )
        return ""
    finally:
        if use_timeout:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            previous_delay, previous_interval = previous_timer
            if previous_delay > 0:
                elapsed = time.monotonic() - started_at
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(previous_delay - elapsed, 1e-6),
                    previous_interval,
                )


_evalplus_codegen.sanitize = _safe_sanitize
codegen = _evalplus_codegen.codegen


class EvalPlusVllmDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        tensor_parallel_size: int,
        max_model_len: int,
        max_num_seqs: int,
        top_p: float,
        force_base_prompt: bool,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self.top_p = top_p
        self.force_base_prompt = force_base_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, trust_remote_code=self.trust_remote_code)
        if self.is_direct_completion():
            self.eos += extra_eos_for_direct_completion(dataset)
        else:
            self.eos += ["\n```\n"]
        self.llm = LLM(
            model=name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            enable_prefix_caching=True,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0"
        batch_size = min(self.batch_size, num_samples)
        raw_prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(prompt, self.instruction_prefix, self.response_prefix, self.tokenizer)
        )
        outputs = self.llm.generate(
            [raw_prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature if do_sample else 0.0,
                max_tokens=self.max_new_tokens,
                top_p=self.top_p if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )
        return [out.outputs[0].text.replace("\t", "    ") for out in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvalPlus generation/evaluation with explicit vLLM sampling parameters.")
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--force_base_prompt", action="store_true")
    parser.add_argument("--version", default="default")
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--id_range",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="Generate only task IDs in the half-open numeric range [LOW, HIGH).",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Stop after generation so disjoint shards can be merged before evaluation.",
    )
    args = parser.parse_args()

    if args.id_range is not None and args.id_range[0] >= args.id_range[1]:
        parser.error("--id_range requires LOW < HIGH")

    os.makedirs(args.root, exist_ok=True)
    batch_size = args.bs if args.bs is not None else min(args.n_samples, 32)
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
    model = EvalPlusVllmDecoder(
        name=args.model,
        dataset=args.dataset,
        batch_size=batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        top_p=args.top_p,
        force_base_prompt=args.force_base_prompt,
    )
    identifier = Path(args.model.strip("./").replace("/", "--")).name + f"_vllm_temp_{args.temperature}_top_p_{args.top_p}_max_tokens_{args.max_tokens}"
    if args.id_range is not None:
        identifier += f"_ids_{args.id_range[0]}_{args.id_range[1]}"
    target_path = os.path.join(args.root, args.dataset, identifier + ".jsonl")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    codegen(
        target_path=target_path,
        model=model,
        dataset=args.dataset,
        n_samples=args.n_samples,
        id_range=tuple(args.id_range) if args.id_range is not None else None,
        version=args.version,
        resume=args.resume,
    )
    del model
    gc.collect()
    if args.skip_evaluation:
        return
    evaluate(
        dataset=args.dataset,
        samples=target_path,
        parallel=args.parallel,
        version=args.version,
        i_just_wanna_run=True,
    )


if __name__ == "__main__":
    main()
