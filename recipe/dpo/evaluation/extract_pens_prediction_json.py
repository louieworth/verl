#!/usr/bin/env python3
"""Extract headline strings from raw PENS model generations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


FENCED_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
HEADLINE_FIELD_RE = re.compile(
    r'"headline"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"', re.IGNORECASE | re.DOTALL
)
HEADLINE_KEYS = [
    "headline",
    "title",
    "prediction",
    "generated_headline",
    "generated_title",
    "response",
    "output",
]
PLACEHOLDER_VALUES = {
    "headline",
    "personalized headline",
    "generated headline",
    "generated title",
    "title",
}
REASONING_MARKERS = (
    "thinking process",
    "analyze the request",
    "analyze user click history",
    "analyze candidate news",
    "the user wants a personalized news headline",
    "the output must be a json object",
    "personalization angle:",
    "observations:",
)


def load_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False).to_dict(orient="records")
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", keep_default_na=False).to_dict(orient="records")
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    frame = pd.DataFrame(rows)
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".tsv", ".txt"}:
        frame.to_csv(path, sep="\t", index=False)
    else:
        raise ValueError(f"Unsupported output suffix: {path.suffix}")


def coerce_prediction(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = coerce_prediction(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in HEADLINE_KEYS:
            if key in value:
                return coerce_prediction(value[key])
        for item in value.values():
            text = coerce_prediction(item)
            if text:
                return text
        return ""
    return str(value).strip()


def is_placeholder_prediction(text: str) -> bool:
    cleaned = str(text).strip()
    if not cleaned:
        return True
    normalized = cleaned.strip().strip('"').strip("'").strip()
    lowered = normalized.lower().strip("<>").strip()
    return lowered in PLACEHOLDER_VALUES


def parse_candidate_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def iter_inline_json_objects(text: str):
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed, idx, idx + end


def classify_unstructured_output(cleaned: str) -> tuple[str, str]:
    stripped = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    lowered = stripped.lower()
    if any(marker in lowered for marker in REASONING_MARKERS):
        return "", "reasoning_no_answer"
    if list(HEADLINE_FIELD_RE.finditer(cleaned)):
        return "", "malformed_json_output"
    return "", "missing_structured_output"


def extract_prediction(text: str) -> tuple[str, str]:
    cleaned = str(text).strip()
    if not cleaned:
        return "", "empty"

    fenced_matches = list(FENCED_JSON_RE.finditer(cleaned))
    for match in reversed(fenced_matches):
        parsed = parse_candidate_json(match.group(1).strip())
        if parsed is not None:
            candidate = coerce_prediction(parsed)
            if candidate and not is_placeholder_prediction(candidate):
                return candidate, "fenced_json"

    inline_matches = list(iter_inline_json_objects(cleaned))
    for parsed, _, _ in reversed(inline_matches):
        candidate = coerce_prediction(parsed)
        if candidate and not is_placeholder_prediction(candidate):
            return candidate, "inline_json"

    return classify_unstructured_output(cleaned)


def normalize_responses(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist") and not isinstance(value, dict):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def extract_generation_text(row: dict, response_index: int) -> str:
    responses = normalize_responses(row.get("responses"))
    if responses:
        if 0 <= response_index < len(responses):
            return responses[response_index]
        return responses[0]
    return str(row.get("generated_text", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw generation jsonl/parquet/csv/tsv")
    parser.add_argument("--output", type=Path, required=True, help="Prediction jsonl/parquet/csv/tsv")
    parser.add_argument("--response-index", type=int, default=0, help="Which sampled response to score when `responses` is a list")
    args = parser.parse_args()

    rows = load_rows(args.input)
    extracted_rows = []
    status_counts: dict[str, int] = {}
    non_empty = 0

    for row in rows:
        generated_text = extract_generation_text(row, response_index=args.response_index)
        prediction, status = extract_prediction(generated_text)
        status_counts[status] = status_counts.get(status, 0) + 1
        if prediction:
            non_empty += 1
        extracted_rows.append(
            {
                "user_id": row["user_id"],
                "news_id": row["news_id"],
                "prediction": prediction,
                "generated_text": generated_text,
                "parse_status": status,
            }
        )

    save_rows(args.output, extracted_rows)

    print(
        json.dumps(
            {
                "count": len(extracted_rows),
                "non_empty_prediction_count": non_empty,
                "parse_status_counts": status_counts,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
