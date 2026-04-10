#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


TLDR_DATASET_BASE_URL = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset"
TLDR_COMPARISON_BATCH_FILES = [
    "batch3.json",
    "batch4.json",
    "batch5.json",
    "batch10.json",
    "batch11.json",
    "batch12.json",
    "batch13.json",
    "batch14.json",
    "batch15.json",
    "batch16.json",
    "batch17.json",
    "batch18.json",
    "batch19.json",
    "batch20.json",
    "batch22.json",
    "batch6.json",
    "batch7.json",
    "batch8.json",
    "batch9.json",
    "batch0_cnndm.json",
    "cnndm0.json",
    "cnndm2.json",
    "edit_b2_eval_test.json",
]


def choose_split(split_names: Iterable[str], preferred_names: list[str]) -> str:
    split_name_set = set(split_names)
    for name in preferred_names:
        if name in split_name_set:
            return name
    raise ValueError(f"Unable to find a split from {preferred_names}. Available: {sorted(split_name_set)}")


def common_prefix_length(left: str, right: str) -> int:
    idx = 0
    upper = min(len(left), len(right))
    while idx < upper and left[idx] == right[idx]:
        idx += 1
    return idx


def split_hh_prompt_and_responses(chosen: str, rejected: str) -> tuple[str, str, str]:
    prefix_len = common_prefix_length(chosen, rejected)
    prefix = chosen[:prefix_len]

    assistant_markers = ["\n\nAssistant:", "\n\nAssistant", "Assistant:"]
    split_idx = -1
    split_marker = None
    for marker in assistant_markers:
        marker_idx = prefix.rfind(marker)
        if marker_idx > split_idx:
            split_idx = marker_idx
            split_marker = marker

    if split_idx < 0 or split_marker is None:
        raise ValueError("Failed to locate the final assistant marker in hh-rlhf sample")

    prompt_end = split_idx + len(split_marker)
    prompt = chosen[:prompt_end].strip()
    chosen_response = chosen[prompt_end:].strip()
    rejected_response = rejected[prompt_end:].strip()
    return prompt, chosen_response, rejected_response


def convert_hh_split(split: Dataset, split_name: str) -> Dataset:
    rows = []
    for idx, row in enumerate(split):
        prompt, chosen, rejected = split_hh_prompt_and_responses(row["chosen"], row["rejected"])
        if not prompt or not chosen or not rejected:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "data_source": "Anthropic/hh-rlhf",
                "source_split": split_name,
                "source_index": idx,
            }
        )
    return Dataset.from_list(rows)


def build_tldr_prompt(info: dict) -> str:
    subreddit = info.get("subreddit", "")
    title = info.get("title", "")
    post = info.get("post", "")
    parts = []
    if subreddit:
        parts.append(f"SUBREDDIT: r/{subreddit}")
    if title:
        parts.append(f"TITLE: {title}")
    parts.append(f"POST: {post}")
    parts.append("TL;DR:")
    return "\n".join(parts).strip()


def convert_tldr_split(split: Dataset, split_name: str) -> Dataset:
    rows = []
    for idx, row in enumerate(split):
        info = row["info"]
        summaries = row["summaries"]
        choice = row["choice"]
        if choice not in (0, 1):
            continue
        prompt = build_tldr_prompt(info)
        chosen = summaries[choice]["text"].strip()
        rejected = summaries[1 - choice]["text"].strip()
        if not prompt or not chosen or not rejected:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "data_source": "openai/summarize_from_feedback:comparisons",
                "source_split": split_name,
                "source_index": idx,
            }
        )
    return Dataset.from_list(rows)


def extract_preference_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    raise ValueError(f"Unable to extract summary text from value of type {type(value)}")


def convert_columbia_tldr_split(split: Dataset, split_name: str) -> Dataset:
    rows = []
    for idx, row in enumerate(split):
        prompt = row.get("prompt", "").strip()
        chosen = extract_preference_text(row["chosen"])
        rejected = extract_preference_text(row["rejected"])
        if not prompt or not chosen or not rejected:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "data_source": "Columbia-NLP/DPO-tldr-summarisation-preferences",
                "source_split": split_name,
                "source_index": idx,
            }
        )
    return Dataset.from_list(rows)


def download_tldr_batches(raw_dataset_dir: Path) -> list[dict]:
    raw_json_dir = raw_dataset_dir / "raw"
    raw_json_dir.mkdir(parents=True, exist_ok=True)

    raw_examples = []
    for batch_file in TLDR_COMPARISON_BATCH_FILES:
        local_path = raw_json_dir / batch_file
        if not local_path.exists():
            source_url = f"{TLDR_DATASET_BASE_URL}/comparisons/{batch_file}"
            print(f"Downloading {source_url} -> {local_path}")
            urlretrieve(source_url, local_path)

        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_examples.append(json.loads(line))

    return raw_examples


def save_dataset_dict(dataset_dict: DatasetDict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in dataset_dict.items():
        split.to_parquet(str(output_dir / f"{split_name}.parquet"))


def prepare_hh(raw_root: Path, parquet_root: Path) -> None:
    raw_dataset_dir = raw_root / "Anthropic__hh-rlhf"
    if raw_dataset_dir.exists():
        dataset = load_from_disk(str(raw_dataset_dir))
    else:
        dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=str(raw_root / ".cache"))
        dataset.save_to_disk(str(raw_dataset_dir))

    train_split_name = choose_split(dataset.keys(), ["train"])
    val_split_name = choose_split(dataset.keys(), ["test", "validation", "valid"])
    converted = DatasetDict(
        train=convert_hh_split(dataset[train_split_name], train_split_name),
        val=convert_hh_split(dataset[val_split_name], val_split_name),
    )
    save_dataset_dict(converted, parquet_root / "dpo_hh")


def prepare_tldr_openai(raw_root: Path, parquet_root: Path) -> None:
    raw_dataset_dir = raw_root / "openai__summarize_from_feedback__comparisons"
    hf_dataset_dir = raw_dataset_dir / "hf_dataset"
    if hf_dataset_dir.exists():
        dataset = load_from_disk(str(hf_dataset_dir))
    else:
        raw_examples = download_tldr_batches(raw_dataset_dir)

        train_examples = []
        val_examples = []
        for example in raw_examples:
            split_name = example["split"]
            if split_name == "train":
                train_examples.append(example)
            elif split_name in ("valid1", "valid2", "validation", "valid", "test"):
                val_examples.append(example)

        dataset = DatasetDict(
            train=Dataset.from_list(train_examples),
            validation=Dataset.from_list(val_examples),
        )
        dataset.save_to_disk(str(hf_dataset_dir))

    train_split_name = choose_split(dataset.keys(), ["train"])
    val_split_name = choose_split(dataset.keys(), ["validation", "valid", "test", "valid1"])
    converted = DatasetDict(
        train=convert_tldr_split(dataset[train_split_name], train_split_name),
        val=convert_tldr_split(dataset[val_split_name], val_split_name),
    )
    save_dataset_dict(converted, parquet_root / "dpo_tldr")


def maybe_choose_split(split_names: Iterable[str], preferred_names: list[str]) -> str | None:
    split_name_set = set(split_names)
    for name in preferred_names:
        if name in split_name_set:
            return name
    return None


def prepare_tldr_columbia(raw_root: Path, parquet_root: Path) -> None:
    raw_dataset_dir = raw_root / "Columbia-NLP__DPO-tldr-summarisation-preferences"
    if raw_dataset_dir.exists():
        dataset = load_from_disk(str(raw_dataset_dir))
    else:
        dataset = load_dataset(
            "Columbia-NLP/DPO-tldr-summarisation-preferences",
            cache_dir=str(raw_root / ".cache"),
        )
        dataset.save_to_disk(str(raw_dataset_dir))

    train_split_name = choose_split(dataset.keys(), ["train"])
    val_split_name = choose_split(dataset.keys(), ["validation", "valid", "val"])
    test_split_name = maybe_choose_split(dataset.keys(), ["test"])

    converted_splits = {
        "train": convert_columbia_tldr_split(dataset[train_split_name], train_split_name),
        "val": convert_columbia_tldr_split(dataset[val_split_name], val_split_name),
    }
    if test_split_name is not None:
        converted_splits["test"] = convert_columbia_tldr_split(dataset[test_split_name], test_split_name)

    save_dataset_dict(DatasetDict(converted_splits), parquet_root / "dpo_tldr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert DPO paper datasets into verl pairwise parquet")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["tldr", "hh"],
        choices=["tldr", "hh"],
        help="Which datasets to prepare",
    )
    parser.add_argument(
        "--hf-root",
        type=Path,
        default=Path("/data/data/jiangli/huggingface"),
        help="Local root directory for downloaded Hugging Face datasets",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=Path("/data/data/jiangli/parquet"),
        help="Output root directory for converted parquet files",
    )
    parser.add_argument(
        "--tldr-source",
        choices=["columbia", "openai"],
        default="columbia",
        help="Which TL;DR preference source to convert",
    )
    args = parser.parse_args()

    os.makedirs(args.hf_root, exist_ok=True)
    os.makedirs(args.parquet_root, exist_ok=True)

    if "hh" in args.datasets:
        prepare_hh(args.hf_root, args.parquet_root)
    if "tldr" in args.datasets:
        if args.tldr_source == "columbia":
            prepare_tldr_columbia(args.hf_root, args.parquet_root)
        else:
            prepare_tldr_openai(args.hf_root, args.parquet_root)

    print("Prepared datasets:")
    for name in args.datasets:
        print(f"  - {name}: raw -> {args.hf_root}, parquet -> {args.parquet_root}")


if __name__ == "__main__":
    main()
