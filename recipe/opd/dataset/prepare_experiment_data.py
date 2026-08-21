#!/usr/bin/env python3
"""Prepare the canonical OpenThoughts/TACO training artifacts.

The output contains verl RL rows, plain-completion SFT rows, and (for code) a
compact distillation parquet without TACO execution tests.  All artifacts use
the same retained examples and deterministic padding.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import datasets
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from recipe.opd.dataset.data_utils import build_student_prompt


MATH_REPO_ID = "siyanzhao/Openthoughts_math_30k_opsd"
MATH_REVISION = "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"
CODE_REPO_ID = "BAAI/TACO"
CODE_REVISION = "d593ed0a2becbbc952230bb89be09189bf1056dc"
PROMPT_CONTRACT_VERSION = "plain_base_completion_v3"
MANIFEST_SCHEMA_VERSION = "opd_canonical_training_data/v2"
CODE_DISTILL_ARTIFACT_VERSION = "taco_tests_externalized_v1"
CODE_GRPO_REWARD_CONTRACT_VERSION = "deepcoder_binary_15_longest_v1"
CODE_GRPO_MAX_TEST_CASES = 15
CODE_GRPO_TEST_SELECTION = "longest_input_chars_desc_then_source_index"
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 9
CONTAMINATION_POLICY = "drop_exact_canonical_math_eval_overlap"
CONTAMINATION_POLICY_VERSION = "canonical_math_eval_decontamination_v1"
CONTAMINATION_HASH_VERSION = "nfkc_casefold_no_whitespace_latex_spacing_v1"
CANONICAL_MATH_EVAL_FILES = (
    ("aime25", Path("aime25/aime25_test.parquet")),
    ("aime26", Path("aime26/aime26_test.parquet")),
    ("hmmt26", Path("hmmt26/hmmt26_test.parquet")),
    ("amobench", Path("amobench/amobench_test.parquet")),
)

# TeX spacing commands do not change the underlying problem.  Removing them,
# together with all Unicode whitespace below, catches formatting-only copies
# without dropping punctuation or mathematical operators.
_LATEX_SPACING_RE = re.compile(
    r"""
    \\(?:quad|qquad|enspace|enskip|thinspace|medspace|thickspace|
          negthinspace|negmedspace|negthickspace|hfill|vfill|
          smallskip|medskip|bigskip)(?![A-Za-z])
    |\\[hv]space\*?\{[^{}]*\}
    |\\(?:kern|mkern|hskip|vskip|mskip)(?![A-Za-z])\s*
       [-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))?\s*
       (?:pt|pc|in|bp|cm|mm|dd|cc|sp|em|ex|mu)?
    |\\[,;:!>/ ]
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("math", "code"))
    parser.add_argument("--input", default="", help="Local dataset path; empty downloads the pinned source.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Writable Hugging Face cache (defaults to <output-dir>/.cache).",
    )
    parser.add_argument("--model-path", required=True, help="Qwen3 Base tokenizer/model path or HF ID.")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=16384)
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        default=512,
        help="Deterministically pad each output to this global batch multiple; 0 disables padding.",
    )
    parser.add_argument("--min-prompt-coverage", type=float, default=0.95)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--eval-root", default="data/eval_dataset/math")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_source(task: str, source: str, split: str, cache_dir: str) -> datasets.Dataset:
    if not source:
        repo_id = MATH_REPO_ID if task == "math" else CODE_REPO_ID
        revision = MATH_REVISION if task == "math" else CODE_REVISION
        return datasets.load_dataset(
            repo_id,
            split=split,
            revision=revision,
            cache_dir=cache_dir,
        )
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset input does not exist: {path}")
    if path.is_file():
        loaders = {".parquet": "parquet", ".json": "json", ".jsonl": "json"}
        loader = loaders.get(path.suffix.lower())
        if loader is None:
            raise ValueError(f"Unsupported dataset input: {path}")
        return datasets.load_dataset(
            loader,
            data_files=str(path),
            split="train",
            cache_dir=cache_dir,
        )
    try:
        loaded = datasets.load_from_disk(str(path))
        if isinstance(loaded, datasets.DatasetDict):
            return loaded[split] if split in loaded else loaded["train"]
        return loaded
    except (FileNotFoundError, ValueError):
        parquet_files = sorted(path.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        return datasets.load_dataset(
            "parquet",
            data_files=[str(file) for file in parquet_files],
            split="train",
            cache_dir=cache_dir,
        )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_solution(value: Any) -> str:
    if value is None:
        return ""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
    if isinstance(parsed, (list, tuple)):
        return clean_text(parsed[0]) if parsed else ""
    return clean_text(parsed)


def math_answer(example: dict[str, Any], solution: str) -> str:
    for key in ("Answer", "answer", "final_answer"):
        answer = clean_text(example.get(key))
        if answer:
            return answer
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", solution)
    return boxed[-1].strip() if boxed else ""


def _stable_test_input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def select_deepcoder_test_cases(
    value: Any, max_test_cases: int = CODE_GRPO_MAX_TEST_CASES
) -> tuple[str, int, int]:
    """Select a deterministic DeepCoder-style reward suite.

    DeepCoder uses the 15 test cases with the longest input strings and gives a
    sparse reward only when every selected case passes.  Selection happens at
    artifact-preparation time so workers never depend on the ordering or size
    of the upstream TACO payload.
    """

    if max_test_cases <= 0:
        raise ValueError("max_test_cases must be positive")
    parsed = value
    if isinstance(value, str):
        parsed = json.loads(value)
    if not isinstance(parsed, dict) or "inputs" not in parsed or "outputs" not in parsed:
        raise ValueError("TACO input_output must contain inputs and outputs")
    inputs = parsed["inputs"]
    outputs = parsed["outputs"]
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs:
        raise ValueError("TACO input_output inputs/outputs must be non-empty lists")
    if len(inputs) != len(outputs):
        raise ValueError("TACO input_output inputs/outputs must have equal lengths")

    ranked_indices = sorted(
        range(len(inputs)),
        key=lambda index: (-len(_stable_test_input_text(inputs[index])), index),
    )
    selected_indices = ranked_indices[:max_test_cases]
    selected = {
        key: copy.deepcopy(item)
        for key, item in parsed.items()
        if key not in {"inputs", "outputs"}
    }
    selected["inputs"] = [copy.deepcopy(inputs[index]) for index in selected_indices]
    selected["outputs"] = [copy.deepcopy(outputs[index]) for index in selected_indices]
    return (
        json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        len(inputs),
        len(selected_indices),
    )


def render_example(task: str, example: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    if task == "math":
        problem = clean_text(example.get("problem") or example.get("Question"))
        solution = clean_text(example.get("solution") or example.get("COT_Reason"))
        answer = math_answer(example, solution)
        prompt = build_student_prompt(problem, task="math")
        reward = answer
        metadata = {"source": clean_text(example.get("source")), "answer": answer}
    else:
        # BAAI/TACO is commonly encountered in two equivalent layouts:
        # the upstream ``question``/``solutions`` schema and the repository's
        # cleaned ``problem``/``solution`` schema.  Accept both so the prompt
        # contract is independent of which pinned local snapshot is used.
        problem = clean_text(example.get("problem") or example.get("question"))
        starter = clean_text(example.get("starter_code"))
        solution = first_solution(example.get("solution") or example.get("solutions"))
        if starter:
            problem = f"{problem}\n\nStarter code:\n```python\n{starter}\n```"
        prompt = build_student_prompt(problem, task="code")
        reward, original_test_count, selected_test_count = select_deepcoder_test_cases(
            example.get("input_output")
        )
        metadata = {
            "starter_code": starter,
            "difficulty": clean_text(example.get("difficulty")),
            "source": clean_text(example.get("source")),
            "original_test_case_count": original_test_count,
            "reward_test_case_count": selected_test_count,
        }
    return problem, prompt, solution, {"reward": reward, **metadata}


def normalize_problem_for_contamination(text: str) -> str:
    """Return the versioned, formatting-insensitive problem representation."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = _LATEX_SPACING_RE.sub("", normalized)
    normalized = normalized.replace("~", "")
    return "".join(character for character in normalized if not character.isspace())


def normalized_problem_hash(text: str) -> str:
    normalized = normalize_problem_for_contamination(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_eval_problem(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    problem = clean_text(
        row.get("problem")
        or row.get("question")
        or extra_info.get("problem")
        or extra_info.get("question")
        or extra_info.get("prompt")
    )
    if problem:
        return problem
    raw_prompt = row.get("prompt")
    if hasattr(raw_prompt, "tolist"):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, (list, tuple)) and raw_prompt:
        first_message = raw_prompt[0]
        if isinstance(first_message, dict):
            return clean_text(first_message.get("content"))
    return ""


def load_canonical_math_eval_hashes(
    eval_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load all required canonical eval problems and fingerprint the hash set.

    Missing, unreadable, empty, or malformed canonical inputs are fatal: an
    apparently clean training set is not meaningful if decontamination ran
    against only a subset of the advertised evaluation suite.
    """
    eval_root = eval_root.resolve()
    hashes: dict[str, list[dict[str, Any]]] = {}
    fingerprint_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []

    for dataset_name, relative_path in CANONICAL_MATH_EVAL_FILES:
        parquet_path = eval_root / relative_path
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Missing canonical math eval data for contamination filtering: {parquet_path}"
            )
        try:
            rows = pq.read_table(parquet_path).to_pylist()
        except Exception as exc:
            raise RuntimeError(f"Could not read canonical math eval data: {parquet_path}") from exc
        if not rows:
            raise ValueError(f"Canonical math eval data is empty: {parquet_path}")

        dataset_summaries.append(
            {
                "dataset": dataset_name,
                "relative_path": relative_path.as_posix(),
                "rows": len(rows),
                "parquet_sha256": _file_sha256(parquet_path),
            }
        )
        for eval_index, row in enumerate(rows):
            problem = _extract_eval_problem(row)
            normalized = normalize_problem_for_contamination(problem)
            if not normalized:
                raise ValueError(
                    f"Canonical math eval row has no usable problem: "
                    f"dataset={dataset_name}, eval_index={eval_index}"
                )
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            match = {
                "dataset": dataset_name,
                "eval_index": eval_index,
                "sha256": digest,
            }
            hashes.setdefault(digest, []).append(match)
            fingerprint_rows.append(match)

    fingerprint_payload = {
        "hash_version": CONTAMINATION_HASH_VERSION,
        "datasets": dataset_summaries,
        "rows": fingerprint_rows,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return hashes, {
        "datasets": dataset_summaries,
        "fingerprint": fingerprint,
        "unique_problem_hashes": len(hashes),
    }


def atomic_to_parquet(dataset: datasets.Dataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".parquet", dir=output.parent)
    os.close(handle)
    try:
        pq.write_table(
            dataset.data.table,
            temporary,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            use_dictionary=True,
            row_group_size=512,
        )
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def pad_rows(rows: list[dict[str, Any]], multiple: int, source_size: int) -> int:
    """Pad with deterministic copies so a one-epoch loader covers every row."""
    if multiple == 0 or not rows:
        return 0
    needed = (-len(rows)) % multiple
    for offset in range(needed):
        row = copy.deepcopy(rows[offset % len(rows)])
        extra_info = row.setdefault("extra_info", {})
        extra_info.setdefault("source_index", extra_info.get("index"))
        extra_info["index"] = source_size + offset
        extra_info["is_batch_padding"] = True
        rows.append(row)
    return needed


def main() -> None:
    args = parse_args()
    if args.max_prompt_length <= 0 or args.max_response_length <= 0:
        raise ValueError("Token limits must be positive")
    if args.pad_to_multiple < 0:
        raise ValueError("--pad-to-multiple must be non-negative")
    if not 0 < args.min_prompt_coverage <= 1:
        raise ValueError("--min-prompt-coverage must be in (0, 1]")

    contamination_hashes: dict[str, list[dict[str, Any]]] = {}
    contamination_eval_summary: dict[str, Any] = {
        "datasets": [],
        "fingerprint": None,
        "unique_problem_hashes": 0,
    }
    contamination_eval_root: str | None = None
    if args.task == "math":
        resolved_eval_root = Path(args.eval_root).resolve()
        contamination_hashes, contamination_eval_summary = load_canonical_math_eval_hashes(resolved_eval_root)
        contamination_eval_root = str(resolved_eval_root)

    output_dir = Path(args.output_dir)
    grpo_path = output_dir / "train_grpo.parquet"
    sft_path = output_dir / "train_sft.parquet"
    distill_path = output_dir / "train_distill.parquet"
    manifest_path = output_dir / "manifest.json"
    required_outputs = [grpo_path, sft_path, manifest_path]
    if args.task == "code":
        required_outputs.append(distill_path)
    source_probe = None
    if not args.overwrite and all(path.is_file() for path in required_outputs):
        expected_source_input = (
            str(Path(args.input).resolve())
            if args.input
            else (MATH_REPO_ID if args.task == "math" else CODE_REPO_ID)
        )
        source_fingerprint = None
        if args.input:
            cache_dir = args.cache_dir or str(output_dir / ".cache")
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            source_probe = load_source(args.task, args.input, args.split, cache_dir)
            source_fingerprint = getattr(source_probe, "_fingerprint", None)
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "task": args.task,
            "repo_id": MATH_REPO_ID if args.task == "math" else CODE_REPO_ID,
            "revision": MATH_REVISION if args.task == "math" else CODE_REVISION,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "prompt_format": "plain_base_completion",
            "completion_separator": "\n",
            "code_solution_validation": "python_ast_v1" if args.task == "code" else "not_applicable",
            "contamination_policy": CONTAMINATION_POLICY if args.task == "math" else "not_applicable",
            "contamination_policy_version": (
                CONTAMINATION_POLICY_VERSION if args.task == "math" else "not_applicable"
            ),
            "contamination_hash_version": (
                CONTAMINATION_HASH_VERSION if args.task == "math" else "not_applicable"
            ),
            "contamination_eval_root": contamination_eval_root,
            "contamination_eval_datasets": contamination_eval_summary["datasets"],
            "contamination_eval_fingerprint": contamination_eval_summary["fingerprint"],
            "model_path": str(Path(args.model_path).resolve()) if Path(args.model_path).exists() else args.model_path,
            "split": args.split,
            "max_samples": args.max_samples,
            "min_prompt_coverage": args.min_prompt_coverage,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "pad_to_multiple": args.pad_to_multiple,
            "source_kind": "local" if args.input else "huggingface",
            "source_input": expected_source_input,
        }
        if source_fingerprint is not None:
            expected["dataset_fingerprint"] = source_fingerprint
        if args.task == "code":
            expected["code_distill_artifact_version"] = CODE_DISTILL_ARTIFACT_VERSION
            expected["code_grpo_reward_contract_version"] = CODE_GRPO_REWARD_CONTRACT_VERSION
            expected["code_grpo_reward_type"] = "binary_all_selected_tests"
            expected["code_grpo_test_selection"] = CODE_GRPO_TEST_SELECTION
            expected["code_grpo_max_test_cases"] = CODE_GRPO_MAX_TEST_CASES
        mismatches = {
            key: {"expected": value, "found": existing.get(key)}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing canonical data at {output_dir} has an incompatible manifest: {mismatches}. "
                "Re-run with --overwrite."
            )
        print(f"Canonical data already prepared and verified: {output_dir}")
        return

    cache_dir = args.cache_dir or str(output_dir / ".cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    source = source_probe if source_probe is not None else load_source(args.task, args.input, args.split, cache_dir)
    source_dataset_fingerprint = getattr(source, "_fingerprint", None)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        source = source.select(range(min(args.max_samples, len(source))))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=cache_dir,
    )

    grpo_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    dropped_empty = 0
    dropped_prompt = 0
    dropped_sft_response = 0
    dropped_invalid_code_solution = 0
    dropped_contamination_details: list[dict[str, Any]] = []
    source_name = MATH_REPO_ID if args.task == "math" else CODE_REPO_ID

    for source_index, example in enumerate(source):
        try:
            problem, prompt, solution, metadata = render_example(args.task, example)
        except (TypeError, ValueError, json.JSONDecodeError):
            dropped_empty += 1
            continue
        if not problem or not solution or not metadata["reward"]:
            dropped_empty += 1
            continue
        if args.task == "code":
            try:
                ast.parse(solution)
            except (SyntaxError, ValueError, TypeError):
                dropped_invalid_code_solution += 1
                continue
        if args.task == "math":
            problem_digest = normalized_problem_hash(problem)
            eval_matches = contamination_hashes.get(problem_digest)
            if eval_matches:
                dropped_contamination_details.append(
                    {
                        "source_index": source_index,
                        "source": clean_text(metadata.get("source")),
                        "sha256": problem_digest,
                        "eval_matches": copy.deepcopy(eval_matches),
                    }
                )
                continue
        # Rollout, KL, and SFT all condition on exactly one newline after the
        # stored prompt. Enforce the cap on that actual model prefix.
        prompt_length = len(tokenizer.encode(prompt + "\n", add_special_tokens=False))
        prompt_lengths.append(prompt_length)
        if prompt_length > args.max_prompt_length:
            dropped_prompt += 1
            continue
        response_length = len(tokenizer.encode(solution, add_special_tokens=False))
        response_lengths.append(response_length)
        extra_info = {
            "index": source_index,
            "problem": problem,
            "expert_cot": solution,
            "prompt_length": prompt_length,
            "expert_length": response_length,
            **{key: value for key, value in metadata.items() if key != "reward"},
        }
        grpo_rows.append(
            {
                "data_source": source_name,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": args.task,
                "reward_model": {"style": "rule", "ground_truth": metadata["reward"]},
                "extra_info": extra_info,
            }
        )
        if args.task == "code":
            # OPD/OPSD only need the prompt and expert solution. TACO's
            # execution tests account for almost all of train_grpo.parquet;
            # keep them exclusively in the GRPO artifact instead of copying
            # gigabytes into every distillation deployment.
            distill_rows.append(
                {
                    "data_source": source_name,
                    "prompt": [{"role": "user", "content": prompt}],
                    "ability": args.task,
                    "reward_model": {"style": "distill_only", "ground_truth": ""},
                    "extra_info": copy.deepcopy(extra_info),
                }
            )
        if response_length <= args.max_response_length:
            sft_rows.append({"prompt": prompt, "response": solution, "extra_info": extra_info})
        else:
            dropped_sft_response += 1

    prompt_coverage = 0.0 if not prompt_lengths else 1.0 - dropped_prompt / len(prompt_lengths)
    if prompt_coverage < args.min_prompt_coverage:
        raise RuntimeError(
            f"Prompt cap {args.max_prompt_length} covers only {prompt_coverage:.2%}; "
            f"required {args.min_prompt_coverage:.2%}"
        )
    if not grpo_rows or not sft_rows:
        raise RuntimeError("No usable training rows remain after canonical filtering")

    retained_grpo_rows = len(grpo_rows)
    retained_sft_rows = len(sft_rows)
    grpo_padding_rows = pad_rows(grpo_rows, args.pad_to_multiple, len(source))
    sft_padding_rows = pad_rows(sft_rows, args.pad_to_multiple, len(source))
    distill_padding_rows = (
        pad_rows(distill_rows, args.pad_to_multiple, len(source)) if args.task == "code" else 0
    )
    grpo_dataset = datasets.Dataset.from_list(grpo_rows)
    sft_dataset = datasets.Dataset.from_list(sft_rows)
    distill_dataset = datasets.Dataset.from_list(distill_rows) if args.task == "code" else None
    output_row_fingerprints = {
        "grpo": grpo_dataset._fingerprint,
        "sft": sft_dataset._fingerprint,
    }
    if distill_dataset is not None:
        output_row_fingerprints["distill"] = distill_dataset._fingerprint
    prepared_dataset_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "output_row_fingerprints": output_row_fingerprints,
                "grpo_rows": len(grpo_rows),
                "sft_rows": len(sft_rows),
                "distill_rows": len(distill_rows),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    atomic_to_parquet(grpo_dataset, grpo_path)
    atomic_to_parquet(sft_dataset, sft_path)
    if distill_dataset is not None:
        atomic_to_parquet(distill_dataset, distill_path)
    prompt_lengths_sorted = sorted(prompt_lengths)
    response_lengths_sorted = sorted(response_lengths)

    def percentile(values: list[int], fraction: float) -> int:
        return values[min(len(values) - 1, int((len(values) - 1) * fraction))] if values else 0

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "task": args.task,
        "repo_id": source_name,
        "revision": MATH_REVISION if args.task == "math" else CODE_REVISION,
        "source_kind": "local" if args.input else "huggingface",
        "source_input": str(Path(args.input).resolve()) if args.input else source_name,
        "dataset_fingerprint": source_dataset_fingerprint,
        "selected_source_dataset_fingerprint": getattr(source, "_fingerprint", None),
        "prepared_dataset_fingerprint": prepared_dataset_fingerprint,
        "output_row_fingerprints": output_row_fingerprints,
        "model_path": str(Path(args.model_path).resolve()) if Path(args.model_path).exists() else args.model_path,
        "split": args.split,
        "max_samples": args.max_samples,
        "min_prompt_coverage": args.min_prompt_coverage,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "prompt_format": "plain_base_completion",
        "completion_separator": "\n",
        "code_solution_validation": "python_ast_v1" if args.task == "code" else "not_applicable",
        "code_distill_artifact_version": (
            CODE_DISTILL_ARTIFACT_VERSION if args.task == "code" else "not_applicable"
        ),
        "code_grpo_reward_contract_version": (
            CODE_GRPO_REWARD_CONTRACT_VERSION if args.task == "code" else "not_applicable"
        ),
        "code_grpo_reward_type": (
            "binary_all_selected_tests" if args.task == "code" else "not_applicable"
        ),
        "code_grpo_test_selection": (
            CODE_GRPO_TEST_SELECTION if args.task == "code" else "not_applicable"
        ),
        "code_grpo_max_test_cases": CODE_GRPO_MAX_TEST_CASES if args.task == "code" else 0,
        "parquet_compression": PARQUET_COMPRESSION,
        "parquet_compression_level": PARQUET_COMPRESSION_LEVEL,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "input_rows": len(source),
        "retained_grpo_rows": retained_grpo_rows,
        "retained_sft_rows": retained_sft_rows,
        "grpo_rows": len(grpo_rows),
        "sft_rows": len(sft_rows),
        "distill_rows": len(distill_rows) if args.task == "code" else 0,
        "pad_to_multiple": args.pad_to_multiple,
        "grpo_padding_rows": grpo_padding_rows,
        "sft_padding_rows": sft_padding_rows,
        "distill_padding_rows": distill_padding_rows,
        "dropped_empty_or_invalid": dropped_empty,
        "dropped_overlong_prompt": dropped_prompt,
        "dropped_overlong_sft_response": dropped_sft_response,
        "dropped_invalid_code_solution": dropped_invalid_code_solution,
        "dropped_contaminated_rows": len(dropped_contamination_details),
        "prompt_coverage": prompt_coverage,
        "prompt_length_p95": percentile(prompt_lengths_sorted, 0.95),
        "prompt_length_max": max(prompt_lengths_sorted, default=0),
        "response_length_p95": percentile(response_lengths_sorted, 0.95),
        "response_length_max": max(response_lengths_sorted, default=0),
        "contamination_policy": CONTAMINATION_POLICY if args.task == "math" else "not_applicable",
        "contamination_policy_version": (
            CONTAMINATION_POLICY_VERSION if args.task == "math" else "not_applicable"
        ),
        "contamination_hash_version": (
            CONTAMINATION_HASH_VERSION if args.task == "math" else "not_applicable"
        ),
        "contamination_eval_root": contamination_eval_root,
        "contamination_eval_datasets": contamination_eval_summary["datasets"],
        "contamination_eval_fingerprint": contamination_eval_summary["fingerprint"],
        "contamination_eval_unique_problem_hashes": contamination_eval_summary["unique_problem_hashes"],
        "dropped_contamination_details": dropped_contamination_details,
        "outputs": {
            "grpo": str(grpo_path),
            "sft": str(sft_path),
            **({"distill": str(distill_path)} if args.task == "code" else {}),
        },
    }
    artifact_paths = {"grpo": grpo_path, "sft": sft_path}
    if args.task == "code":
        artifact_paths["distill"] = distill_path
    manifest["artifacts"] = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
            "rows": pq.ParquetFile(path).metadata.num_rows,
        }
        for name, path in artifact_paths.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
