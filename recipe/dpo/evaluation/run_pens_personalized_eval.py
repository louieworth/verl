#!/usr/bin/env python3
"""Post-process raw PENS generations and update shared metrics JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# On Compute Canada the venv ships a stub pyarrow wheel; the real module lives
# in the CVMFS arrow module site-packages. Inject it before importing pandas so
# `read_parquet` can find an engine without requiring PYTHONPATH to be set.
try:
    import pyarrow  # noqa: F401
except ModuleNotFoundError:
    _CVMFS_ARROW = (
        "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/"
        "Compiler/gcccore/arrow/19.0.1/lib/python3.12/site-packages"
    )
    if _CVMFS_ARROW not in sys.path:
        sys.path.insert(0, _CVMFS_ARROW)
    import pyarrow  # noqa: F401

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_pens_prediction_json import extract_generation_text, extract_prediction, load_rows
from score_pens_predictions import (
    STRUCTURED_PARSE_STATUSES,
    choose_prediction_key,
    join_predictions,
    load_references,
    mean,
    parse_reference_list,
    score_with_rouge_package_multi,
)

DEFAULT_MODEL_PATH = (
    "/data/data/jiangli/ckpt/PENS/"
    "single_wise_dpo_click_hist_positive_only_lora_qwen3_5-4b/global_step_5341/actor/hf_merged"
)
DEFAULT_TEST_FILE = "/data/data/jiangli/data/pens/extract/personalized_test.tsv"


def env_default(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def env_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return Path(value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(env_default("WORKDIR", os.getcwd())))
    parser.add_argument("--model-path", type=str, default=env_default("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--test-file", type=Path, default=env_path("TEST_FILE") or Path(DEFAULT_TEST_FILE))
    parser.add_argument("--results-gen-dir", type=Path, default=env_path("RESULTS_GEN_DIR"))
    parser.add_argument("--results-dir", type=Path, default=env_path("RESULTS_DIR"))
    parser.add_argument("--raw-file", type=Path, default=env_path("RAW_FILE"))
    parser.add_argument("--result-json-file", type=Path, default=env_path("RESULT_JSON_FILE"))
    parser.add_argument("--model-key", type=str, default=env_default("MODEL_KEY"))
    parser.add_argument("--backend", choices=["rouge"], default="rouge")
    parser.add_argument("--response-index", type=int, default=int(env_default("RESPONSE_INDEX", "0")))
    parser.add_argument("--align-by-order", action="store_true", default=env_flag("ALIGN_BY_ORDER"))
    parser.add_argument(
        "--thinking",
        choices=["true", "false"],
        default=env_default("VLLM_ENABLE_THINKING", "false"),
        help="Whether the generation used vLLM thinking mode. Tagged into the result JSON key.",
    )
    parser.add_argument(
        "--date-tag",
        type=str,
        default=env_default("DATE_TAG") or datetime.now().strftime("%Y%m%d"),
        help="Date tag (default: today YYYYMMDD) appended to the result JSON key.",
    )
    return parser.parse_args()


def derive_model_slug(model_path: str) -> str:
    normalized = model_path.rstrip("/")
    path = Path(normalized)
    parts = path.parts
    if "snapshots" in parts:
        candidate = parts[parts.index("snapshots") - 1]
    else:
        candidate = path.name
        if candidate in {"hf_merged", "hf_merged_final", "hf_merged_fixed", "huggingface", "actor"}:
            actor_dir = path.parent
            step_dir = actor_dir.parent.name
            experiment_dir = actor_dir.parent.parent.name
            candidate = f"{experiment_dir}_{step_dir}_{candidate}"
    if candidate.startswith("models--"):
        candidate = candidate.split("--")[-1]
    cleaned = [ch.lower() if ch.isalnum() or ch in {"_", "-"} else "_" for ch in candidate]
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_-")


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.workdir = args.workdir.resolve()
    args.test_file = args.test_file.resolve()
    args.model_key = args.model_key or derive_model_slug(args.model_path)
    args.results_gen_dir = (args.results_gen_dir or (args.workdir / "gen_results")).resolve()
    args.results_dir = (args.results_dir or (args.workdir / "results")).resolve()
    args.raw_file = (args.raw_file or (args.results_gen_dir / f"{args.model_key}.parquet")).resolve()
    args.result_json_file = (args.result_json_file or (args.results_dir / "result.json")).resolve()
    return args


def extract_predictions_frame(raw_file: Path, response_index: int) -> tuple[pd.DataFrame, dict[str, int], int]:
    rows = load_rows(raw_file)
    extracted_rows: list[dict] = []
    status_counts: dict[str, int] = {}
    non_empty_prediction_count = 0

    for row in rows:
        generated_text = extract_generation_text(row, response_index=response_index)
        prediction, parse_status = extract_prediction(generated_text)
        status_counts[parse_status] = status_counts.get(parse_status, 0) + 1
        if prediction:
            non_empty_prediction_count += 1
        extracted_rows.append(
            {
                "user_id": row["user_id"],
                "news_id": row["news_id"],
                "prediction": prediction,
                "generated_text": generated_text,
                "parse_status": parse_status,
            }
        )
    return pd.DataFrame(extracted_rows), status_counts, non_empty_prediction_count


def score_predictions_frame(
    predictions: pd.DataFrame,
    references_file: Path,
    *,
    backend: str,
    align_by_order: bool,
) -> dict:
    references = load_references(references_file)
    prediction_key = choose_prediction_key(predictions, "prediction")
    joined = join_predictions(predictions, references, prediction_key, align_by_order)
    joined_count = int(len(joined))

    if "parse_status" in joined.columns:
        joined = joined[joined["parse_status"].isin(STRUCTURED_PARSE_STATUSES)].copy()
    joined = joined.dropna(subset=[prediction_key]).copy()
    joined = joined[joined[prediction_key].astype(str).str.strip().ne("")].copy()
    if "gold_headlines" in joined.columns:
        joined["gold_headlines"] = joined["gold_headlines"].apply(parse_reference_list)
    else:
        joined["gold_headlines"] = joined["gold_headline"].apply(parse_reference_list)
    joined = joined[joined["gold_headlines"].map(bool)].copy()

    skipped_count = joined_count - int(len(joined))
    hypotheses = joined[prediction_key].astype(str).tolist()
    reference_lists = joined["gold_headlines"].tolist()

    rouge1, rouge2, rougel = score_with_rouge_package_multi(hypotheses, reference_lists)
    backend_used = "rouge"

    return {
        "backend": backend_used,
        "count": int(len(joined)),
        "joined_count": joined_count,
        "skipped_count": skipped_count,
        "rouge_1_f1": mean(rouge1),
        "rouge_2_f1": mean(rouge2),
        "rouge_l_f1": mean(rougel),
    }


def upsert_result_json(result_file: Path, model_key: str, payload: dict) -> None:
    """Insert-or-update a single model_key in result_file."""
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    if result_file.exists() and result_file.stat().st_size > 0:
        with result_file.open("r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            existing = json.loads(content)
            if not isinstance(existing, dict):
                raise ValueError(f"Expected {result_file} to contain a JSON object.")
            result = existing

    result[model_key] = payload

    tmp_path = result_file.with_suffix(result_file.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, result_file)


def main() -> None:
    args = resolve_paths(parse_args())
    if not args.raw_file.exists():
        raise FileNotFoundError(f"Missing raw generation file: {args.raw_file}")
    if not args.test_file.exists():
        raise FileNotFoundError(f"Missing test file: {args.test_file}")

    predictions, parse_status_counts, non_empty_prediction_count = extract_predictions_frame(
        args.raw_file,
        response_index=args.response_index,
    )
    metrics = score_predictions_frame(
        predictions,
        args.test_file,
        backend=args.backend,
        align_by_order=args.align_by_order,
    )
    payload = dict(metrics)
    payload["non_empty_prediction_count"] = non_empty_prediction_count
    payload["parse_status_counts"] = parse_status_counts
    payload["model_path"] = args.model_path
    payload["raw_generation_file"] = str(args.raw_file)
    thinking_on = str(args.thinking).strip().lower() == "true"
    payload["thinking_mode"] = thinking_on
    payload["date"] = args.date_tag
    tagged_key = f"{args.model_key}__think{'ON' if thinking_on else 'OFF'}__{args.date_tag}"
    upsert_result_json(args.result_json_file, tagged_key, payload)
    args.model_key = tagged_key

    print(
        json.dumps(
            {
                "model_key": args.model_key,
                "raw_generation_file": str(args.raw_file),
                "result_json_file": str(args.result_json_file),
                "metrics": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
