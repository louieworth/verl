#!/usr/bin/env python3
"""Download the six portable JSONL files that make up LiveCodeBench v6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


FILE_NAMES = [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
]
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/eval_dataset/code/livecodebench/code_generation_lite",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    expected = [output_dir / file_name for file_name in FILE_NAMES]
    if not args.overwrite and all(path.is_file() and path.stat().st_size > 0 for path in expected):
        print(f"LiveCodeBench release_v6 files already exist, skipping: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="livecodebench/code_generation_lite",
        repo_type="dataset",
        revision=LCB_REVISION,
        local_dir=str(output_dir),
        allow_patterns=FILE_NAMES,
    )

    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"LiveCodeBench download is incomplete: {missing}")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": {path.name: path.stat().st_size for path in expected},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
