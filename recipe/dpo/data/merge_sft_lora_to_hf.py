#!/usr/bin/env python3
"""Merge verl SFT LoRA-wrapped checkpoint into a flat HF model directory.

verl's SFT trainer with save_contents=hf_model saves the PEFT-wrapped state
dict (keys prefixed `base_model.model.model.*` with `base_layer.weight` +
`lora_A.default.weight` + `lora_B.default.weight` per LoRA target). That
format is NOT loadable by AutoModelForCausalLM.from_pretrained directly,
which causes all weights to be "newly initialized".

This script:
1. Reads every shard from the SFT ckpt's `huggingface/` dir
2. For each LoRA-adapted layer (q/k/v/o_proj): merges W' = W + B @ A * (alpha/rank)
3. For non-LoRA layers: strips the `base_model.model.` prefix
4. Writes the merged state_dict + config + tokenizer to OUT_DIR in flat HF format
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="SFT ckpt huggingface/ dir")
    p.add_argument("--dst", required=True, help="Output merged HF dir")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--shard-size", default="5GB", help="max shard size for output")
    return p.parse_args()


def parse_shard_size(s: str) -> int:
    s = s.strip().upper()
    mult = 1
    if s.endswith("GB"):
        mult, s = 1024**3, s[:-2]
    elif s.endswith("MB"):
        mult, s = 1024**2, s[:-2]
    elif s.endswith("KB"):
        mult, s = 1024, s[:-2]
    return int(float(s) * mult)


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    scale = args.lora_alpha / args.lora_rank

    shard_files = sorted(src.glob("model-*-of-*.safetensors"))
    print(f"[merge] {len(shard_files)} shards, scale={scale} (alpha={args.lora_alpha}, rank={args.lora_rank})")

    # 1. Load every tensor into memory
    state: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    print(f"[merge] loaded {len(state)} tensors from src")

    # 2. Merge LoRA
    merged: dict[str, torch.Tensor] = {}
    lora_layers_merged = 0
    for k, v in state.items():
        if ".lora_A.default.weight" in k or ".lora_B.default.weight" in k:
            # will be consumed when we hit the base_layer
            continue
        if k.endswith(".base_layer.weight"):
            lora_a_key = k.replace(".base_layer.weight", ".lora_A.default.weight")
            lora_b_key = k.replace(".base_layer.weight", ".lora_B.default.weight")
            if lora_a_key not in state or lora_b_key not in state:
                print(f"[merge] WARN: base_layer has no lora pair: {k}")
                new_k = k.replace("base_model.model.", "").replace(".base_layer.weight", ".weight")
                merged[new_k] = v
                continue
            # W' = W + B @ A * (alpha/rank)
            base_w = v.to(torch.float32)
            lora_a = state[lora_a_key].to(torch.float32)
            lora_b = state[lora_b_key].to(torch.float32)
            delta = (lora_b @ lora_a) * scale
            merged_w = base_w + delta
            out_dtype = v.dtype
            new_k = k.replace("base_model.model.", "").replace(".base_layer.weight", ".weight")
            merged[new_k] = merged_w.to(out_dtype)
            lora_layers_merged += 1
        else:
            # non-LoRA, strip prefix
            new_k = k.replace("base_model.model.", "")
            merged[new_k] = v
    print(f"[merge] merged {lora_layers_merged} LoRA layers; total output tensors = {len(merged)}")

    # 3. Sanity check: expected keys start with "model." or "lm_head."
    bad = [k for k in merged if not (k.startswith("model.") or k.startswith("lm_head"))]
    if bad:
        print(f"[merge] WARN: {len(bad)} unexpected keys, e.g. {bad[:3]}")

    # 4. Shard output
    max_size = parse_shard_size(args.shard_size)
    print(f"[merge] sharding to max {max_size / 1024**3:.1f} GB each")
    shards: list[dict[str, torch.Tensor]] = [{}]
    cur_size = 0
    for k, v in merged.items():
        tsize = v.element_size() * v.numel()
        if cur_size + tsize > max_size and shards[-1]:
            shards.append({})
            cur_size = 0
        shards[-1][k] = v
        cur_size += tsize
    n_shards = len(shards)
    print(f"[merge] split into {n_shards} shards")

    # 5. Write
    total_size = 0
    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        out_path = dst / fname
        print(f"[merge] writing {fname} ({len(shard)} tensors, {sum(t.element_size()*t.numel() for t in shard.values())/1024**3:.2f} GB)...")
        save_file(shard, str(out_path), metadata={"format": "pt"})
        for k, v in shard.items():
            weight_map[k] = fname
            total_size += v.element_size() * v.numel()

    # 6. Write safetensors index
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"[merge] wrote model.safetensors.index.json ({total_size/1024**3:.2f} GB total)")

    # 7. Copy config/tokenizer/chat_template from src
    for fname in [
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
        "chat_template.jinja",
    ]:
        src_f = src / fname
        if src_f.exists():
            shutil.copy2(src_f, dst / fname)
            print(f"[merge] copied {fname}")

    print(f"[merge] DONE → {dst}")


if __name__ == "__main__":
    main()
