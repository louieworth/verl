#!/usr/bin/env python3
"""Shared helpers for PENS personalized headline evaluation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.dpo.data.prepare_pens_singlewise_dpo import (
    detect_delimiter,
    load_news_lookup,
    normalize_text,
    render_history_block,
    split_id_list,
)


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

STRICT_JSON_OUTPUT = '{"headline":"<personalized headline>"}'

OUTPUT_RULES_BLOCK = (
    "Use exactly the following output format:\n"
    "```json\n"
    f"{STRICT_JSON_OUTPUT}\n"
    "```"
)

EVAL_PROMPT_INSTRUCTION = (
    "Generate a personalized news headline for this user based on the user's reading interests "
    "and the candidate news article.\n"
    "The click history only contains clicked news titles.\n"
    f"{OUTPUT_RULES_BLOCK}\n"
    "Do not add any explanation outside the JSON block."
)


def split_title_list(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    if "#TAB#" in text:
        parts = text.split("#TAB#")
    elif ";;" in text:
        parts = text.split(";;")
    else:
        parts = text.split("\t")
    return [normalize_text(part) for part in parts if part and normalize_text(part)]


def load_news_table(news_file: Path) -> dict[str, dict[str, str]]:
    return load_news_lookup(news_file)


def build_eval_prompt(history_block: str, candidate: dict[str, str]) -> list[dict[str, str]]:
    content = (
        f"{EVAL_PROMPT_INSTRUCTION}\n\n"
        f"{history_block}\n\n"
        "[Candidate News]\n"
        f"News body: {candidate.get('body', '')}"
    )
    return [{"role": "user", "content": content}]


def clicked_titles_from_ids(clicked_news_ids: list[str], news_lookup: dict[str, dict[str, str]]) -> list[str]:
    titles: list[str] = []
    for news_id in clicked_news_ids:
        news_item = news_lookup.get(news_id)
        if news_item is None:
            continue
        title = normalize_text(news_item.get("headline", ""))
        if title:
            titles.append(title)
    return titles


def iter_personalized_rows(test_file: Path):
    delimiter = detect_delimiter(test_file)
    with test_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if row_idx == 0 and row[0].strip().lower() in {"userid", "user_id"}:
                continue
            if len(row) < 4:
                raise ValueError(f"Expected at least 4 columns in {test_file}, got {len(row)} at row {row_idx}")
            yield {
                "user_id": row[0].strip(),
                "clicked_news_ids": split_id_list(row[1]),
                "positive_news_ids": split_id_list(row[2]),
                "rewrite_titles": split_title_list(row[3]),
            }


def build_personalized_eval_records(
    *,
    news_lookup: dict[str, dict[str, str]],
    test_file: Path,
    max_examples: int = -1,
) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    rows_seen = 0

    for row in iter_personalized_rows(test_file):
        rows_seen += 1
        if max_examples > 0 and len(grouped) >= max_examples:
            break

        user_id = row["user_id"]
        clicked_news_ids = row["clicked_news_ids"]
        positive_news_ids = row["positive_news_ids"]
        rewrite_titles = row["rewrite_titles"]
        pair_count = min(len(positive_news_ids), len(rewrite_titles))

        for idx in range(pair_count):
            news_id = positive_news_ids[idx]
            gold_headline = rewrite_titles[idx]
            candidate = news_lookup.get(news_id)
            if candidate is None:
                continue
            key = (user_id, news_id)
            if key not in grouped:
                history_block, history_count, missing_history_count = render_history_block(clicked_news_ids, news_lookup)
                grouped[key] = {
                    "user_id": user_id,
                    "news_id": news_id,
                    "clicked_news_ids": clicked_news_ids,
                    "clicked_titles": clicked_titles_from_ids(clicked_news_ids, news_lookup),
                    "candidate_category": candidate.get("category", ""),
                    "candidate_topic": candidate.get("topic", ""),
                    "candidate_title": candidate.get("headline", ""),
                    "candidate_body": candidate.get("body", ""),
                    "history_count": history_count,
                    "missing_history_count": missing_history_count,
                    "gold_headlines": [],
                    "prompt": build_eval_prompt(history_block, candidate),
                }
            if gold_headline and gold_headline not in grouped[key]["gold_headlines"]:
                grouped[key]["gold_headlines"].append(gold_headline)

            if max_examples > 0 and len(grouped) >= max_examples:
                break

    records = list(grouped.values())
    for record in records:
        gold_headlines = record["gold_headlines"]
        record["gold_headline"] = gold_headlines[0] if gold_headlines else ""
        record["gold_headline_count"] = len(gold_headlines)
        record["source_test_file"] = str(test_file)
        record["source_rows_seen"] = rows_seen
    return records


def render_prompt_for_model(
    *,
    tokenizer,
    prompt_messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    return tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
