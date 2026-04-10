from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

DEFAULT_SAMPLE_ID_KEY = "sample_id"


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_label(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def build_sample_id_payload(
    *,
    prompt: Any,
    response: Any,
    label: Any,
    user_id: Any = None,
    candidate_news_id: Any = None,
    split: Any = None,
    data_source: Any = None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "response": _normalize_string(response),
        "label": _normalize_label(label),
        "user_id": _normalize_string(user_id),
        "candidate_news_id": _normalize_string(candidate_news_id),
        "split": _normalize_string(split),
        "data_source": _normalize_string(data_source),
    }


def compute_sample_id(
    *,
    prompt: Any,
    response: Any,
    label: Any,
    user_id: Any = None,
    candidate_news_id: Any = None,
    split: Any = None,
    data_source: Any = None,
) -> str:
    payload = build_sample_id_payload(
        prompt=prompt,
        response=response,
        label=label,
        user_id=user_id,
        candidate_news_id=candidate_news_id,
        split=split,
        data_source=data_source,
    )
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def compute_sample_id_from_record(record: Mapping[str, Any]) -> str:
    return compute_sample_id(
        prompt=record.get("prompt"),
        response=record.get("response"),
        label=record.get("label"),
        user_id=record.get("user_id"),
        candidate_news_id=record.get("candidate_news_id"),
        split=record.get("split"),
        data_source=record.get("data_source"),
    )
