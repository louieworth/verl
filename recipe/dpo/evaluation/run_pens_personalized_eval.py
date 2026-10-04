#!/usr/bin/env python3
"""Score raw PENS generations, log to W&B, and hand off checkpoint state."""
# ruff: noqa: E402 -- direct script imports follow the repository path setup.

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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recipe.dpo.evaluation.manage_pens_interleaved_checkpoint import record_evaluation_result
from recipe.dpo.evaluation.score_pens_predictions import (
    STRUCTURED_PARSE_STATUSES,
    choose_prediction_key,
    join_predictions,
    load_references,
    load_table,
    mean,
    parse_reference_list,
    score_with_rouge_package_multi,
)
from recipe.dpo.evaluation.wandb_utils import evaluation_config, log_eval_to_wandb
from recipe.dpo.run.common import default_paths

DEFAULT_MODEL_PATH = default_paths()["PENS_MODEL_DIR"]
DEFAULT_TEST_FILE = default_paths()["PENS_EVAL_TEST_FILE"]


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


def thinking_metadata(value: str) -> tuple[bool | None, str]:
    normalized = value.strip().lower()
    if normalized == "skip":
        return None, "NA"
    return normalized == "true", "ON" if normalized == "true" else "OFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(env_default("WORKDIR", os.getcwd())))
    parser.add_argument("--model-path", type=str, default=env_default("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--test-file", type=Path, default=env_path("TEST_FILE") or Path(DEFAULT_TEST_FILE))
    parser.add_argument("--results-gen-dir", type=Path, default=env_path("RESULTS_GEN_DIR"))
    # Accept old arguments from already launched drivers without writing them.
    parser.add_argument("--results-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--raw-file", type=Path, default=env_path("RAW_FILE"))
    parser.add_argument(
        "--result-json-file", type=Path, default=env_path("RESULT_JSON_FILE"), help=argparse.SUPPRESS
    )
    parser.add_argument("--checkpoint-state-file", type=Path, default=env_path("PENS_EVAL_STATE_FILE"))
    parser.add_argument("--model-key", type=str, default=env_default("MODEL_KEY"))
    parser.add_argument("--backend", choices=["rouge"], default="rouge")
    parser.add_argument("--response-index", type=int, default=int(env_default("RESPONSE_INDEX", "0")))
    parser.add_argument("--align-by-order", action="store_true", default=env_flag("ALIGN_BY_ORDER"))
    parser.add_argument(
        "--thinking",
        choices=["true", "false", "skip"],
        default=env_default("VLLM_ENABLE_THINKING", "false"),
        help=(
            "Whether generation used vLLM thinking mode; skip means the model has no "
            "thinking-mode switch. Tagged into the W&B result key."
        ),
    )
    parser.add_argument(
        "--date-tag",
        type=str,
        default=env_default("DATE_TAG") or datetime.now().strftime("%Y%m%d"),
        help="Date tag (default: today YYYYMMDD) appended to the W&B result key.",
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
    paths = default_paths()
    experiment = env_default("PENS_EXPERIMENT_NAME", args.model_key)
    args.results_gen_dir = (args.results_gen_dir or (Path(paths["PENS_OUTPUT_ROOT"]) / experiment)).resolve()
    args.raw_file = (args.raw_file or (args.results_gen_dir / f"{args.model_key}.parquet")).resolve()
    if args.checkpoint_state_file is None and args.result_json_file is not None:
        # Older running drivers only pass the metrics filename. Use their
        # checkpoint state instead so they can finish without that file.
        kind = env_default("PENS_RUN_KIND", "dpo")
        prefix = "SFT" if kind == "sft" else "SINGLE_WISE_DPO"
        checkpoint_key = f"{prefix}_DEFAULT_LOCAL_DIR" if kind == "sft" else f"{prefix}_CKPT_DIR"
        checkpoint = env_path(checkpoint_key)
        args.checkpoint_state_file = env_path(f"{prefix}_CHECKPOINT_EVAL_STATE_FILE")
        if args.checkpoint_state_file is None and checkpoint is not None:
            args.checkpoint_state_file = checkpoint / "interleaved_eval_state.json"
    if args.checkpoint_state_file is not None:
        args.checkpoint_state_file = args.checkpoint_state_file.resolve()
    return args


def extract_generation_text(row: dict, response_index: int) -> str:
    responses = row.get("responses")
    if hasattr(responses, "tolist"):
        responses = responses.tolist()
    if isinstance(responses, str):
        responses = [responses]
    if isinstance(responses, list | tuple) and responses:
        index = response_index if 0 <= response_index < len(responses) else 0
        value = responses[index]
    else:
        value = row.get("generated_text", "")
    return "" if value is None else str(value).strip()


def extract_prediction(text: str) -> tuple[str, str]:
    """Score the headline verbatim, without interpreting JSON or boxed output."""
    prediction = text.strip()
    return prediction, "plain_text" if prediction else "empty"


def extract_predictions_frame(raw_file: Path, response_index: int) -> tuple[pd.DataFrame, dict[str, int], int]:
    rows = load_table(raw_file).to_dict(orient="records")
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
    thinking_mode, thinking_tag = thinking_metadata(str(args.thinking))
    payload["thinking_mode"] = thinking_mode
    payload["date"] = args.date_tag
    payload["evaluation_config"] = evaluation_config()
    payload["evaluation_config"]["reference_file"] = str(args.test_file)
    tagged_key = f"{args.model_key}__think{thinking_tag}__{args.date_tag}"
    log_eval_to_wandb(payload, result_key=tagged_key, date_tag=args.date_tag, model_path=args.model_path)
    if args.checkpoint_state_file is not None:
        record_evaluation_result(args.checkpoint_state_file, tagged_key, payload)
    args.model_key = tagged_key

    print(
        json.dumps(
            {
                "model_key": args.model_key,
                "raw_generation_file": str(args.raw_file),
                "metrics": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
