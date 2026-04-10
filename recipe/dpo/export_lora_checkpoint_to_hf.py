#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a saved LoRA adapter checkpoint into a Hugging Face model.")
    parser.add_argument("--actor-dir", required=True, help="Path to the checkpoint actor directory.")
    parser.add_argument("--output-dir", default=None, help="Directory to save the merged HF model. Defaults to <actor-dir>/hf_merged.")
    parser.add_argument("--base-model-dir", default=None, help="Override base model path. Defaults to adapter_config.json value.")
    parser.add_argument("--tokenizer-path", default=None, help="Override tokenizer path. Defaults to the resolved base model path.")
    parser.add_argument("--processor-path", default=None, help="Override processor path. Defaults to tokenizer path.")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"

    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def collect_target_layers(adapter_state_dict: dict[str, object]) -> list[int]:
    target_layers = sorted(
        {
            int(part)
            for key in adapter_state_dict
            for parts in [key.split(".")]
            if "layers" in parts
            for part in [parts[parts.index("layers") + 1]]
        }
    )
    return target_layers


def normalize_adapter(actor_dir: Path) -> tuple[Path, str]:
    adapter_dir = actor_dir / "lora_adapter"
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_model_path = adapter_dir / "adapter_model.safetensors"

    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {adapter_config_path}")
    if not adapter_model_path.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {adapter_model_path}")

    adapter_config = json.loads(adapter_config_path.read_text())
    adapter_state_dict = load_file(str(adapter_model_path))

    target_layers = collect_target_layers(adapter_state_dict)
    if target_layers and adapter_config.get("layers_to_transform") is None:
        adapter_config["layers_to_transform"] = target_layers
    adapter_config["inference_mode"] = True
    adapter_config.pop("runtime_config", None)

    fixed_adapter_dir = Path(tempfile.mkdtemp(prefix="single_wise_dpo_adapter_"))
    (fixed_adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

    remapped_state_dict = {}
    for key, value in adapter_state_dict.items():
        remapped_state_dict[key.replace(".model.language_model.", ".model.")] = value
    save_file(remapped_state_dict, str(fixed_adapter_dir / "adapter_model.safetensors"))

    return fixed_adapter_dir, adapter_config["base_model_name_or_path"]


def main() -> None:
    args = parse_args()

    actor_dir = Path(args.actor_dir).resolve()
    if not actor_dir.is_dir():
        raise FileNotFoundError(f"Missing actor directory: {actor_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else actor_dir / "hf_merged"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_adapter_dir, base_model_from_config = normalize_adapter(actor_dir)
    base_model_dir = Path(args.base_model_dir or base_model_from_config).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve() if args.tokenizer_path else base_model_dir
    processor_path = Path(args.processor_path).resolve() if args.processor_path else tokenizer_path

    torch_dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, str(fixed_adapter_dir), is_trainable=False)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size=args.max_shard_size)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(str(output_dir))

    try:
        processor = AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=args.trust_remote_code)
    except Exception:
        processor = None
    if processor is not None:
        processor.save_pretrained(str(output_dir))

    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
