#!/usr/bin/env python3
"""Build a length-safe SFT dataset from the first non-empty TACO solution."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer


CODE_INSTRUCTION = (
    "You will be given a programming problem. Write a correct Python program that solves it. "
    "Return only the code inside a single ```python code block."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local BAAI/TACO data into verl multi-turn SFT parquet."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-length", type=int, default=8192)
    return parser.parse_args()


def load_source(path: str, split: str) -> Dataset:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"TACO source does not exist: {source}")
    if source.is_dir():
        try:
            loaded = load_from_disk(str(source))
            if isinstance(loaded, DatasetDict):
                if split not in loaded:
                    raise ValueError(f"TACO dataset has no split {split!r}: {sorted(loaded)}")
                return loaded[split]
            return loaded
        except (FileNotFoundError, ValueError):
            parquet_files = sorted(source.rglob(f"{split}-*.parquet"))
            if not parquet_files:
                parquet_files = sorted(source.rglob("*.parquet"))
            if parquet_files:
                return load_dataset(
                    "parquet", data_files=[str(item) for item in parquet_files], split="train"
                )
            raise
    if source.suffix.lower() == ".parquet":
        return load_dataset("parquet", data_files=str(source), split="train")
    raise ValueError(f"Unsupported TACO source: {source}")


def first_solution(raw: Any) -> str:
    if raw is None:
        return ""
    parsed = raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"TACO solutions must be a JSON list, got {type(parsed).__name__}")
    return str(parsed[0]).strip() if parsed and str(parsed[0]).strip() else ""


def build_prompt(example: dict[str, Any]) -> str:
    question = str(example.get("question") or "").strip()
    if not question:
        raise ValueError("TACO row has an empty question")
    starter_code = str(example.get("starter_code") or "").strip()
    if starter_code:
        question += f"\n\nStarter code:\n```python\n{starter_code}\n```"
    return f"{question}\n\n{CODE_INSTRUCTION}"


def template_length(tokenizer, messages: list[dict[str, str]]) -> int:
    """Upper-bound verl's per-message assembly and whole-message sanity length."""
    whole = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
    )
    empty_user = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
    )
    two_empty_users = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2,
        add_generation_prompt=False,
        tokenize=True,
    )
    system_prompt_length = max(0, 2 * len(empty_user) - len(two_empty_users))
    per_message_length = 0
    for index, message in enumerate(messages):
        tokens = tokenizer.apply_chat_template(
            [message],
            add_generation_prompt=False,
            tokenize=True,
        )
        if index and message["role"] != "system":
            tokens = tokens[system_prompt_length:]
        per_message_length += len(tokens)
    return max(len(whole), per_message_length)


def write_atomic(dataset: Dataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".taco_sft_", suffix=".parquet", dir=output.parent)
    os.close(fd)
    try:
        dataset.to_parquet(temporary)
        os.replace(temporary, output)
        os.chmod(output, 0o664)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")

    source = load_source(args.input, args.split)
    required = {"question", "solutions"}
    missing = required - set(source.column_names)
    if missing:
        raise ValueError(f"TACO source is missing columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    records: list[dict[str, Any]] = []
    dropped_empty_solution = 0
    dropped_overlength = 0
    overlength_examples: list[dict[str, int]] = []
    maximum_observed_length = 0

    for source_index, example in enumerate(source):
        solution = first_solution(example.get("solutions"))
        if not solution:
            dropped_empty_solution += 1
            continue
        prompt = build_prompt(example)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution},
        ]
        sequence_length = template_length(tokenizer, messages)
        maximum_observed_length = max(maximum_observed_length, sequence_length)
        if sequence_length > args.max_length:
            dropped_overlength += 1
            if len(overlength_examples) < 20:
                overlength_examples.append(
                    {"source_index": source_index, "sequence_length": sequence_length}
                )
            continue
        records.append(
            {
                "messages": messages,
                "extra_info": {
                    "source_index": source_index,
                    "problem": str(example.get("question") or "").strip(),
                    "starter_code": str(example.get("starter_code") or "").strip(),
                    "solution": solution,
                    "target_source": "solutions[0]",
                    "difficulty": str(example.get("difficulty") or ""),
                    "source": str(example.get("source") or ""),
                    "url": str(example.get("url") or ""),
                    "sequence_length": sequence_length,
                },
            }
        )

    if not records:
        raise ValueError("No usable TACO SFT rows remain after filtering")

    output = Path(args.output)
    write_atomic(Dataset.from_list(records), output)
    print(
        json.dumps(
            {
                "dataset": "BAAI/TACO",
                "input": args.input,
                "output": str(output),
                "model_path": args.model_path,
                "max_length": args.max_length,
                "input_rows": len(source),
                "kept_rows": len(records),
                "dropped_empty_solution_rows": dropped_empty_solution,
                "dropped_overlength_rows": dropped_overlength,
                "maximum_observed_length": maximum_observed_length,
                "overlength_examples": overlength_examples,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
