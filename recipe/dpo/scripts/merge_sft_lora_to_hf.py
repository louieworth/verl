#!/usr/bin/env python3
"""Export SFT checkpoints as validated flat HF weights, without merging twice."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
import uuid
from pathlib import Path

HF_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def inspect_checkpoint(directory: Path, *, flat=False):
    """Read safetensors headers only; reject empty or incomplete exports."""
    directory = Path(directory)
    config = directory / "config.json"
    if not config.is_file() or not isinstance(json.loads(config.read_text()), dict):
        raise ValueError(f"Missing or invalid model config: {config}")
    index_file = directory / "model.safetensors.index.json"
    weight_map = None
    if index_file.is_file():
        weight_map = json.loads(index_file.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Empty or invalid weight_map: {index_file}")
        names = sorted(set(weight_map.values()))
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError(f"Invalid shard filename in {index_file}")
        files = [directory / name for name in names]
    elif (directory / "model.safetensors").is_file():
        files = [directory / "model.safetensors"]
    else:
        files = sorted(directory.glob("model-*-of-*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights found in {directory}")
    tensors = {}
    tensor_files = {}
    for file in files:
        with file.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors file: {file}")
            size = struct.unpack("<Q", prefix)[0]
            payload_size = file.stat().st_size - 8 - size
            if size > 100 * 1024**2 or payload_size < 0:
                raise ValueError(f"Invalid safetensors header length: {file}")
            header = json.loads(handle.read(size))
        weights = {key: value for key, value in header.items() if key != "__metadata__"}
        if not weights:
            raise ValueError(f"Empty safetensors weights: {file}")
        for key, value in weights.items():
            start, end = value["data_offsets"]
            if not 0 <= start <= end <= payload_size:
                raise ValueError(f"Truncated tensor {key} in {file}")
            if key in tensors:
                raise ValueError(f"Duplicate tensor {key} in {directory}")
            if flat and (not key.startswith(("model.", "lm_head.")) or ".lora_" in key or ".base_layer." in key):
                raise ValueError(f"Unmerged PEFT tensor {key} in {directory}")
            tensors[key] = value
            tensor_files[key] = file.name
    if weight_map is not None and weight_map != tensor_files:
        raise ValueError(f"Weight index does not match shard contents in {directory}")
    return files, tensors


def parse_shard_size(value: str) -> int:
    value = value.strip().upper()
    multiplier = 1
    for suffix, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if value.endswith(suffix):
            multiplier, value = size, value[: -len(suffix)]
            break
    result = int(float(value) * multiplier)
    if result <= 0:
        raise ValueError("Shard size must be positive")
    return result


def flat_key(key):
    return key.removeprefix("base_model.model.").replace(".base_layer.weight", ".weight")


def merge_wrapped_weights(files, destination, rank, alpha, shard_size):
    # Older SFT versions saved raw PEFT weights. Only that format needs Torch
    # and LoRA arithmetic; current FSDP HF exports are already merged.
    from safetensors import safe_open
    from safetensors.torch import save_file

    if rank <= 0:
        raise ValueError("Wrapped LoRA weights require a positive rank")
    scale = alpha / rank
    state = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key)
    merged = {}
    for key, value in state.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        if key.endswith(".base_layer.weight"):
            a_key = key.replace(".base_layer.weight", ".lora_A.default.weight")
            b_key = key.replace(".base_layer.weight", ".lora_B.default.weight")
            if a_key not in state or b_key not in state:
                raise ValueError(f"Missing LoRA pair for {key}")
            delta = (state[b_key].float() @ state[a_key].float()) * scale
            value = (value.float() + delta).to(value.dtype)
        merged[flat_key(key)] = value
    shards = [{}]
    current_size = 0
    for key, value in merged.items():
        size = value.element_size() * value.numel()
        if current_size + size > shard_size and shards[-1]:
            shards.append({})
            current_size = 0
        shards[-1][key] = value.contiguous()
        current_size += size
    weight_map = {}
    total_size = 0
    for index, shard in enumerate(shards, 1):
        name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(destination / name), metadata={"format": "pt"})
        for key, value in shard.items():
            weight_map[key] = name
            total_size += value.element_size() * value.numel()
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
    )


def export_checkpoint(source, destination, *, rank=64, alpha=128, shard_size="5GB"):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different directories")
    files, tensors = inspect_checkpoint(source)
    wrapped = any(key.startswith("base_model.model.") or ".lora_" in key or ".base_layer." in key for key in tensors)
    expected = {flat_key(key) for key in tensors if ".lora_A." not in key and ".lora_B." not in key}
    if destination.is_dir():
        try:
            _, existing = inspect_checkpoint(destination, flat=True)
            if set(existing) == expected:
                print(f"[export] Already valid: {destination} ({len(existing)} tensors)")
                return destination
        except (OSError, ValueError, KeyError, TypeError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}_export_", dir=destination.parent) as temporary:
        output = Path(temporary) / "model"
        output.mkdir()
        if wrapped:
            merge_wrapped_weights(files, output, rank, alpha, parse_shard_size(shard_size))
        else:
            # Reuse already-merged files, including the single-file format.
            # Hard links avoid another 8 GB copy within the same checkpoint.
            for file in files:
                try:
                    os.link(file, output / file.name)
                except OSError:
                    shutil.copy2(file, output / file.name)
            index = source / "model.safetensors.index.json"
            if index.is_file():
                shutil.copy2(index, output / index.name)
        for name in HF_FILES:
            file = source / name
            if file.is_file():
                shutil.copy2(file, output / name)
        _, exported = inspect_checkpoint(output, flat=True)
        if set(exported) != expected:
            raise ValueError(f"Incomplete exported weights: expected {len(expected)}, got {len(exported)}")
        backup = None
        if destination.exists() or destination.is_symlink():
            backup = destination.with_name(f"{destination.name}.invalid_{uuid.uuid4().hex[:8]}")
            destination.rename(backup)
        try:
            output.rename(destination)
        except OSError:
            if backup is not None:
                backup.rename(destination)
            raise
        if backup is not None:
            print(f"[export] Preserved invalid previous export: {backup}")
    print(f"[export] Ready: {destination} ({len(exported)} tensors, merged LoRA: {wrapped})")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--shard-size", default="5GB")
    args = parser.parse_args()
    export_checkpoint(args.src, args.dst, rank=args.lora_rank, alpha=args.lora_alpha, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
