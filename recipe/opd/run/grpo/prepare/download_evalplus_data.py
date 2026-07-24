#!/usr/bin/env python3
"""Download the exact EvalPlus datasets into the portable GRPO asset tree."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path


DATASETS = {
    "HumanEvalPlus-v0.1.10.jsonl": (
        "https://github.com/evalplus/humanevalplus_release/releases/download/"
        "v0.1.10/HumanEvalPlus.jsonl.gz"
    ),
    "MbppPlus-v0.2.0.jsonl": (
        "https://github.com/evalplus/mbppplus_release/releases/download/"
        "v0.2.0/MbppPlus.jsonl.gz"
    ),
}


def download_and_expand(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".jsonl.gz", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        print(f"Downloading {url}")
        try:
            import certifi

            tls_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            tls_context = ssl.create_default_context()
        request = urllib.request.Request(url, headers={"User-Agent": "verl-grpo-asset-downloader"})
        with urllib.request.urlopen(request, context=tls_context) as source, temp_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        expanded_path = destination.with_suffix(destination.suffix + ".tmp")
        with gzip.open(temp_path, "rb") as source, expanded_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        if expanded_path.stat().st_size == 0:
            raise ValueError(f"Downloaded dataset is empty: {url}")
        os.replace(expanded_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/eval_dataset/code/evalplus")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for filename, url in DATASETS.items():
        destination = output_dir / filename
        if destination.is_file() and destination.stat().st_size > 0 and not args.overwrite:
            print(f"EvalPlus dataset already exists, skipping: {destination}")
            continue
        download_and_expand(url, destination)
        print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
