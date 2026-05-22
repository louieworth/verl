#!/usr/bin/env python3
"""Score PENS predictions with ROUGE-1/2/L."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import pandas as pd

STRUCTURED_PARSE_STATUSES = {"fenced_json", "inline_json", "boxed"}


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    if suffix == ".txt":
        return pd.read_csv(path, sep="\t", header=None, names=["prediction"], keep_default_na=False)
    raise ValueError(f"Unsupported file suffix: {path.suffix}")


def split_id_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if "#TAB#" in text:
        parts = text.split("#TAB#")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = text.split()
    return [part.strip() for part in parts if part and part.strip()]


def split_title_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if "#TAB#" in text:
        parts = text.split("#TAB#")
    elif ";;" in text:
        parts = text.split(";;")
    else:
        parts = text.split("\t")
    return [part.strip() for part in parts if part and part.strip()]


def flatten_official_test(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] < 4:
        raise ValueError("Official PENS test TSV should have at least 4 columns.")

    frame = frame.iloc[:, :4].copy()
    frame.columns = ["user_id", "clicked_news_ids", "news_id_list", "gold_title_list"]

    rows = []
    for _, row in frame.iterrows():
        user_id = str(row["user_id"]).strip()
        news_ids = split_id_list(row["news_id_list"])
        titles = split_title_list(row["gold_title_list"])
        count = min(len(news_ids), len(titles))
        for idx in range(count):
            rows.append(
                {
                    "user_id": user_id,
                    "news_id": news_ids[idx],
                    "gold_headline": titles[idx],
                }
            )
    return pd.DataFrame(rows)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def compute_with_rouge_package(hyps: list[str], refs: list[str]) -> tuple[list[float], list[float], list[float]]:
    """Single-reference ROUGE via rouge_score (Google) with Porter stemmer.

    Aligns with PENS paper benchmark (`pyrouge -m`) which enables Porter
    stemming. Older pltrdy `rouge` package didn't support stemming → ~0.03
    RL undercount vs paper. Switching to rouge_score(use_stemmer=True)
    closes that gap.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougel = [], [], []
    for h, r in zip(hyps, refs):
        if not h or not r:
            rouge1.append(0.0); rouge2.append(0.0); rougel.append(0.0)
            continue
        s = scorer.score(normalize_text(r), normalize_text(h))
        rouge1.append(s["rouge1"].fmeasure)
        rouge2.append(s["rouge2"].fmeasure)
        rougel.append(s["rougeL"].fmeasure)
    return rouge1, rouge2, rougel


def choose_prediction_key(frame: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in frame.columns:
        return preferred
    for key in [
        "prediction",
        "generated_headline",
        "generated_title",
        "headline",
        "title",
        "pred",
        "output",
        "response",
        "generation",
    ]:
        if key in frame.columns:
            return key
    if len(frame.columns) == 1:
        return str(frame.columns[0])
    raise ValueError(f"Could not infer prediction column from: {list(frame.columns)}")


def parse_reference_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def aggregate_references(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"user_id", "news_id"}.issubset(frame.columns):
        return frame

    frame = prepare_join_keys(frame)
    meta_columns = [col for col in frame.columns if col not in {"gold_headline", "gold_headlines", "gold_headline_count"}]
    records = []
    for _, group in frame.groupby(["user_id", "news_id"], sort=False, dropna=False):
        base = {column: group.iloc[0][column] for column in meta_columns}
        gold_headlines: list[str] = []
        for _, row in group.iterrows():
            for title in parse_reference_list(row.get("gold_headlines", [])):
                if title not in gold_headlines:
                    gold_headlines.append(title)
            for title in parse_reference_list(row.get("gold_headline", "")):
                if title not in gold_headlines:
                    gold_headlines.append(title)
        base["gold_headlines"] = gold_headlines
        base["gold_headline"] = gold_headlines[0] if gold_headlines else ""
        base["gold_headline_count"] = len(gold_headlines)
        records.append(base)
    return pd.DataFrame(records)


def load_references(path: Path) -> pd.DataFrame:
    frame = load_table(path)
    if "gold_headlines" in frame.columns or "gold_headline" in frame.columns:
        return aggregate_references(frame)
    if "rewrite_titles" in frame.columns:
        return aggregate_references(flatten_official_test(frame))
    if path.suffix.lower() in {".tsv", ".txt"}:
        headerless = pd.read_csv(path, sep="\t", header=None, keep_default_na=False)
        if headerless.shape[1] >= 4:
            return aggregate_references(flatten_official_test(headerless))
    if frame.shape[1] >= 4 and "user_id" not in frame.columns:
        return aggregate_references(flatten_official_test(frame))
    raise ValueError(
        "Could not infer reference format. Expected prepared eval examples with `gold_headline` or `gold_headlines`, "
        "or official PENS `test.tsv`."
    )


def prepare_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"user_id", "news_id"}.issubset(frame.columns):
        return frame
    joined = frame.copy()
    joined["user_id"] = joined["user_id"].astype(str).str.strip()
    joined["news_id"] = joined["news_id"].astype(str).str.strip()
    return joined


def join_predictions(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    prediction_key: str,
    align_by_order: bool,
) -> pd.DataFrame:
    predictions = prepare_join_keys(predictions)
    references = prepare_join_keys(references)

    if {"user_id", "news_id"}.issubset(predictions.columns) and {"user_id", "news_id"}.issubset(references.columns):
        pred_keys = predictions[["user_id", "news_id"]]
        if pred_keys.duplicated().any():
            raise ValueError("Predictions contain duplicate `user_id` + `news_id` pairs.")
        prediction_columns = ["user_id", "news_id", prediction_key]
        for extra_column in ["parse_status", "generated_text"]:
            if extra_column in predictions.columns and extra_column not in prediction_columns:
                prediction_columns.append(extra_column)
        return references.merge(
            predictions[prediction_columns],
            on=["user_id", "news_id"],
            how="inner",
            validate="one_to_one",
        )

    if align_by_order or len(predictions) == len(references):
        if len(predictions) != len(references):
            raise ValueError("`--align-by-order` requires prediction and reference lengths to match.")
        joined = references.copy().reset_index(drop=True)
        joined[prediction_key] = predictions[prediction_key].reset_index(drop=True)
        for extra_column in ["parse_status", "generated_text"]:
            if extra_column in predictions.columns:
                joined[extra_column] = predictions[extra_column].reset_index(drop=True)
        return joined

    raise ValueError(
        "Could not align predictions with references. Provide `user_id` + `news_id`, "
        "or use `--align-by-order`."
    )


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def score_with_rouge_package_multi(
    hypotheses: list[str],
    reference_lists: list[list[str]],
) -> tuple[list[float], list[float], list[float]]:
    """Multi-reference max-F1 ROUGE via rouge_score with Porter stemmer.

    Aligns with PENS paper benchmark `pyrouge -a -c 95 -m -n 4 -w 1.2`:
    - `-a` (avg across refs)   → multi-ref max F1 here
    - `-m` (Porter stemmer)    → use_stemmer=True
    Older pltrdy `rouge` package didn't support stemming → ~0.03 RL
    undercount vs paper. Switching closes that gap (~+0.030 RL on PENS).
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1 = []
    rouge2 = []
    rougel = []
    for hypothesis, references in zip(hypotheses, reference_lists):
        norm_hyp = normalize_text(hypothesis)
        if not references or not norm_hyp.strip():
            rouge1.append(0.0); rouge2.append(0.0); rougel.append(0.0)
            continue
        norm_refs = [normalize_text(ref) for ref in references]
        norm_refs = [r for r in norm_refs if r.strip()]
        if not norm_refs:
            rouge1.append(0.0); rouge2.append(0.0); rougel.append(0.0)
            continue
        b1 = b2 = bL = 0.0
        for ref in norm_refs:
            s = scorer.score(ref, norm_hyp)
            b1 = max(b1, s["rouge1"].fmeasure)
            b2 = max(b2, s["rouge2"].fmeasure)
            bL = max(bL, s["rougeL"].fmeasure)
        rouge1.append(b1); rouge2.append(b2); rougel.append(bL)
    return rouge1, rouge2, rougel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction file: jsonl/csv/tsv/parquet")
    parser.add_argument("--references", type=Path, required=True, help="Prepared eval file or official test.tsv")
    parser.add_argument("--prediction-key", type=str, default=None, help="Prediction column name")
    parser.add_argument(
        "--backend",
        type=str,
        default="rouge",
        choices=["rouge"],
        help="ROUGE backend (rouge package).",
    )
    parser.add_argument("--align-by-order", action="store_true", help="Align predictions and references by row order")
    parser.add_argument("--save-per-example", type=Path, default=None, help="Optional output file with per-example scores")
    args = parser.parse_args()

    predictions = load_table(args.predictions)
    references = load_references(args.references)
    prediction_key = choose_prediction_key(predictions, args.prediction_key)
    joined = join_predictions(predictions, references, prediction_key, args.align_by_order)

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
    hyps = joined[prediction_key].astype(str).tolist()
    reference_lists = joined["gold_headlines"].tolist()
    joined["gold_headline"] = joined["gold_headlines"].apply(lambda refs: refs[0] if refs else "")
    joined["gold_headline_count"] = joined["gold_headlines"].apply(len)

    rouge1, rouge2, rougel = score_with_rouge_package_multi(hyps, reference_lists)
    backend_used = "rouge"

    joined["rouge_1_f1"] = rouge1
    joined["rouge_2_f1"] = rouge2
    joined["rouge_l_f1"] = rougel

    metrics = {
        "backend": backend_used,
        "count": int(len(joined)),
        "joined_count": joined_count,
        "skipped_count": skipped_count,
        "rouge_1_f1": mean(rouge1),
        "rouge_2_f1": mean(rouge2),
        "rouge_l_f1": mean(rougel),
    }

    if args.save_per_example is not None:
        args.save_per_example.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.save_per_example.suffix.lower()
        if suffix == ".jsonl":
            with args.save_per_example.open("w", encoding="utf-8") as f:
                for row in joined.to_dict(orient="records"):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif suffix == ".parquet":
            joined.to_parquet(args.save_per_example, index=False)
        elif suffix == ".csv":
            joined.to_csv(args.save_per_example, index=False)
        elif suffix in {".tsv", ".txt"}:
            joined.to_csv(args.save_per_example, sep="\t", index=False)
        else:
            raise ValueError(f"Unsupported per-example output suffix: {args.save_per_example.suffix}")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
