#!/usr/bin/env python3
"""Generate OPD y_o responses through a resident student vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os

import numpy as np
import pandas as pd

from recipe.opd.rollout_sync import load_resident_rollout_manifest
from verl.trainer.main_generation_server import generate
from verl.utils.hdfs_io import makedirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate y_o with a resident OPD vLLM server")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_resident_rollout_manifest(args.manifest)
    server_addresses = manifest.get("server_addresses") or []
    if not server_addresses:
        raise ValueError(f"resident rollout manifest {args.manifest} has no server_addresses")

    dataset = pd.read_parquet(args.input)
    chat_lst = dataset[args.prompt_key].tolist()
    chat_lst = [chat.tolist() if hasattr(chat, "tolist") else chat for chat in chat_lst]
    chat_numpy = np.array(chat_lst)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    model_path = args.model_path or manifest["model"]["path"]
    gen_results = asyncio.run(
        generate(server_addresses, model_path, args.n_samples, sampling_params, chat_numpy)
    )
    results = list(itertools.chain.from_iterable(gen_results))
    results = np.array([result.choices[0].message.content for result in results])
    results = np.reshape(results, (-1, args.n_samples))
    if results.shape != (len(chat_lst), args.n_samples):
        raise RuntimeError(f"unexpected result shape {results.shape}; expected {(len(chat_lst), args.n_samples)}")

    dataset["responses"] = results.tolist()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        makedirs(output_dir, exist_ok=True)
    print(f"Saving resident y_o results to {args.output}", flush=True)
    dataset.to_parquet(args.output)


if __name__ == "__main__":
    main()
