#!/usr/bin/env python3
"""Diagnostics for OPD checkpoints that score zero on AIME evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from recipe.math_evaluation.compute_score import compute_score_data_source


DEFAULT_BAD_MODEL = (
    "/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/"
    "teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/"
    "epoch1/ms1/batch00001/hf_merged"
)
DEFAULT_REF_MODEL = (
    "/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/"
    "teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms40_20260520-234705/"
    "epoch1/ms40/batch00040/hf_merged"
)
DEFAULT_BASE_MODEL = (
    "/scratch/l/luli/hf/hub/models--Qwen--Qwen3-1.7B/"
    "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
DEFAULT_AIME24 = "/scratch/l/luli/jiangli/datasets/eval/aime24/aime24_test.parquet"

WEIGHT_KEYS = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.13.self_attn.q_proj.weight",
    "model.layers.13.self_attn.o_proj.weight",
    "model.layers.13.mlp.down_proj.weight",
    "model.layers.27.self_attn.q_proj.weight",
    "model.layers.27.self_attn.o_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_dump(obj: Any, path: Path) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def short_text(text: str, limit: int = 500) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def model_files(model_dir: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(model_dir).glob("*")):
        if path.is_file() or path.is_symlink():
            try:
                stat = path.stat()
                size = stat.st_size
            except OSError:
                size = None
            rows.append({"name": path.name, "size": size, "is_symlink": path.is_symlink()})
    return rows


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def tensor_file_for(model_dir: str, key: str) -> Path | None:
    model_path = Path(model_dir)
    single = model_path / "model.safetensors"
    if single.exists():
        return single
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json_if_exists(index_path)
        rel = index.get("weight_map", {}).get(key)
        if rel:
            return model_path / rel
    return None


def load_tensor_part(model_dir: str, key: str, row_limit: int = 4096) -> tuple[torch.Tensor | None, dict[str, Any]]:
    file_path = tensor_file_for(model_dir, key)
    if file_path is None:
        return None, {"missing": True}

    with safe_open(str(file_path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            return None, {"missing": True, "file": str(file_path)}
        try:
            view = handle.get_slice(key)
            shape = list(view.get_shape())
            used_rows = None
            if len(shape) >= 2 and shape[0] > row_limit:
                tensor = view[:row_limit]
                used_rows = row_limit
            else:
                tensor = view[:]
            return tensor.contiguous(), {"file": str(file_path), "shape": shape, "used_rows": used_rows}
        except Exception:
            tensor = handle.get_tensor(key)
            shape = list(tensor.shape)
            used_rows = None
            if len(shape) >= 2 and shape[0] > row_limit:
                tensor = tensor[:row_limit].contiguous()
                used_rows = row_limit
            return tensor, {"file": str(file_path), "shape": shape, "used_rows": used_rows}


def tensor_stats(tensor: torch.Tensor | None, meta: dict[str, Any]) -> dict[str, Any]:
    if tensor is None:
        return dict(meta)
    x = tensor.detach().float()
    finite = torch.isfinite(x)
    all_finite = bool(finite.all().item())
    x_finite = x[finite]
    if x_finite.numel() == 0:
        values = {"all_finite": all_finite, "finite_fraction": 0.0}
    else:
        values = {
            "all_finite": all_finite,
            "finite_fraction": float(finite.float().mean().item()),
            "mean": float(x_finite.mean().item()),
            "std": float(x_finite.std(unbiased=False).item()),
            "min": float(x_finite.min().item()),
            "max": float(x_finite.max().item()),
            "abs_max": float(x_finite.abs().max().item()),
            "l2_norm": float(torch.linalg.vector_norm(x_finite).item()),
        }
    values.update(meta)
    values["dtype"] = str(tensor.dtype)
    return values


def tensor_delta(model_a: str, model_b: str, key: str) -> dict[str, Any]:
    a, meta_a = load_tensor_part(model_a, key)
    b, meta_b = load_tensor_part(model_b, key)
    if a is None or b is None:
        return {"missing": True, "a": meta_a, "b": meta_b}
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_mismatch": True, "a": list(a.shape), "b": list(b.shape), "meta_a": meta_a, "meta_b": meta_b}
    af = a.float()
    bf = b.float()
    diff = af - bf
    b_norm = torch.linalg.vector_norm(bf)
    diff_norm = torch.linalg.vector_norm(diff)
    return {
        "shape": list(a.shape),
        "meta_a": meta_a,
        "meta_b": meta_b,
        "all_finite": bool(torch.isfinite(diff).all().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "l2": float(diff_norm.item()),
        "relative_l2": float((diff_norm / b_norm.clamp_min(1e-12)).item()),
    }


def tokenizer_summary(model_dir: str) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    prompt = [{"role": "user", "content": "Find 2+2. Put the final answer in \\boxed{}."}]
    chat = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return {
        "class": tok.__class__.__name__,
        "len": len(tok),
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "bos_token": tok.bos_token,
        "bos_token_id": tok.bos_token_id,
        "special_tokens_map": tok.special_tokens_map,
        "chat_template_tail": chat[-300:],
    }


def config_summary(model_dir: str) -> dict[str, Any]:
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    keys = [
        "architectures",
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "tie_word_embeddings",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "dtype",
        "torch_dtype",
        "transformers_version",
    ]
    return {key: getattr(cfg, key, None) for key in keys}


def dataset_summary(dataset_path: str) -> dict[str, Any]:
    df = pd.read_parquet(dataset_path)
    row = df.iloc[0]
    prompt = row["prompt"]
    return {
        "path": dataset_path,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "first_data_source": row.get("data_source"),
        "first_ground_truth": row.get("reward_model", {}).get("ground_truth"),
        "first_prompt": short_text(prompt, 1200),
    }


def run_metadata(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)
    models = {"bad": args.bad_model, "ref": args.ref_model, "base": args.base_model}
    result: dict[str, Any] = {
        "models": {},
        "dataset": dataset_summary(args.dataset),
        "weight_stats": {},
        "weight_deltas": {},
    }

    for name, path in models.items():
        result["models"][name] = {
            "path": path,
            "files": model_files(path),
            "config": config_summary(path),
            "generation_config": load_json_if_exists(Path(path) / "generation_config.json"),
            "tokenizer": tokenizer_summary(path),
        }
        stats = {}
        for key in WEIGHT_KEYS:
            tensor, meta = load_tensor_part(path, key)
            stats[key] = tensor_stats(tensor, meta)
            del tensor
        result["weight_stats"][name] = stats

    comparisons = {
        "bad_vs_base": (args.bad_model, args.base_model),
        "ref_vs_base": (args.ref_model, args.base_model),
        "bad_vs_ref": (args.bad_model, args.ref_model),
    }
    for label, (left, right) in comparisons.items():
        result["weight_deltas"][label] = {key: tensor_delta(left, right, key) for key in WEIGHT_KEYS}

    json_dump(result, out / "metadata_and_weight_stats.json")
    print(f"Wrote {out / 'metadata_and_weight_stats.json'}")


def get_prompt_rows(dataset_path: str) -> list[dict[str, Any]]:
    df = pd.read_parquet(dataset_path)
    rows = []
    for idx in [0, 1]:
        row = df.iloc[idx]
        rows.append(
            {
                "name": f"aime24_{idx}",
                "data_source": row["data_source"],
                "prompt": row["prompt"],
                "ground_truth": row["reward_model"]["ground_truth"],
            }
        )
    rows.append(
        {
            "name": "simple_2_plus_2",
            "data_source": "aime24",
            "prompt": [{"role": "user", "content": "Compute 2+2. Put your final answer within \\boxed{}."}],
            "ground_truth": "4",
        }
    )
    return rows


def load_model_for_smoke(model_dir: str):
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **kwargs)
    model.to("cuda:0")
    model.eval()
    return model


def run_hf_smoke(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)
    torch.manual_seed(args.seed)
    models = [("bad", args.bad_model), ("ref", args.ref_model), ("base", args.base_model)]
    prompt_rows = get_prompt_rows(args.dataset)
    all_results: dict[str, Any] = {"max_new_tokens": args.smoke_max_new_tokens, "num_return": args.smoke_num_return, "models": {}}

    for label, model_dir in models:
        print(f"[hf-smoke] loading {label}: {model_dir}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = load_model_for_smoke(model_dir)
        eos_token_id = model.generation_config.eos_token_id
        pad_token_id = model.generation_config.pad_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id
        model_results = []
        for row in prompt_rows:
            chat_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(chat_text, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    num_return_sequences=args.smoke_num_return,
                    max_new_tokens=args.smoke_max_new_tokens,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
            input_len = int(encoded["input_ids"].shape[-1])
            generations = []
            for seq in outputs:
                gen_ids = seq[input_len:].detach().cpu().tolist()
                raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                try:
                    score = float(compute_score_data_source(row["data_source"], text, row["ground_truth"]))
                except Exception:
                    score = 0.0
                generations.append(
                    {
                        "generated_tokens": len(gen_ids),
                        "first_token_id": gen_ids[0] if gen_ids else None,
                        "score": score,
                        "contains_boxed": "\\boxed" in text,
                        "raw_prefix": short_text(raw, 900),
                        "text_prefix": short_text(text, 900),
                    }
                )
            model_results.append(
                {
                    "prompt_name": row["name"],
                    "ground_truth": row["ground_truth"],
                    "input_tokens": input_len,
                    "generations": generations,
                }
            )
        all_results["models"][label] = {"path": model_dir, "prompts": model_results}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    json_path = out / "hf_smoke_generations.json"
    json_dump(all_results, json_path)
    text_path = out / "hf_smoke_generations.txt"
    with text_path.open("w") as f:
        for label, model_info in all_results["models"].items():
            f.write(f"\n===== {label}: {model_info['path']} =====\n")
            for prompt in model_info["prompts"]:
                f.write(f"\n--- {prompt['prompt_name']} gt={prompt['ground_truth']} input_tokens={prompt['input_tokens']} ---\n")
                for idx, generation in enumerate(prompt["generations"]):
                    f.write(
                        f"[{idx}] score={generation['score']} tokens={generation['generated_tokens']} "
                        f"first_token={generation['first_token_id']} boxed={generation['contains_boxed']}\n"
                    )
                    f.write(generation["text_prefix"] + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), p))


def run_analyze_parquet(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)
    df = pd.read_parquet(args.generation_parquet)
    tokenizer = AutoTokenizer.from_pretrained(args.bad_model, trust_remote_code=True)
    per_question = []
    all_scores = []
    all_char_lens = []
    all_token_lens = []
    empty = 0
    boxed = 0

    for q_idx, row in df.iterrows():
        data_source = row.get("data_source", "aime24")
        reward_model = row.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        responses = list(row.get("responses") or [])
        scores = []
        response_summaries = []
        for resp_idx, resp in enumerate(responses):
            text = "" if resp is None else str(resp)
            try:
                score = float(compute_score_data_source(data_source, text, ground_truth))
            except Exception:
                score = 0.0
            token_len = len(tokenizer.encode(text, add_special_tokens=False))
            char_len = len(text)
            all_scores.append(score)
            all_char_lens.append(char_len)
            all_token_lens.append(token_len)
            if not text.strip():
                empty += 1
            if "\\boxed" in text:
                boxed += 1
            scores.append(score)
            if resp_idx < args.example_responses:
                response_summaries.append(
                    {
                        "idx": resp_idx,
                        "score": score,
                        "char_len": char_len,
                        "token_len": token_len,
                        "contains_boxed": "\\boxed" in text,
                        "prefix": short_text(text, 1000),
                    }
                )
        per_question.append(
            {
                "question_idx": int(q_idx),
                "data_source": data_source,
                "ground_truth": ground_truth,
                "scores": scores,
                "pass16": float(any(s > 0 for s in scores[:16])),
                "pass8": float(any(s > 0 for s in scores[:8])),
                "avg_pass1": float(mean(scores)) if scores else 0.0,
                "examples": response_summaries,
            }
        )

    total_responses = len(all_scores)
    summary = {
        "generation_parquet": args.generation_parquet,
        "n_questions": len(per_question),
        "total_responses": total_responses,
        "avg_pass1": float(mean(all_scores)) if all_scores else 0.0,
        "pass8": float(mean(q["pass8"] for q in per_question)) if per_question else 0.0,
        "pass16": float(mean(q["pass16"] for q in per_question)) if per_question else 0.0,
        "empty_responses": empty,
        "boxed_responses": boxed,
        "char_len": {
            "min": min(all_char_lens) if all_char_lens else 0,
            "p50": percentile(all_char_lens, 50),
            "p90": percentile(all_char_lens, 90),
            "max": max(all_char_lens) if all_char_lens else 0,
        },
        "token_len": {
            "min": min(all_token_lens) if all_token_lens else 0,
            "p50": percentile(all_token_lens, 50),
            "p90": percentile(all_token_lens, 90),
            "max": max(all_token_lens) if all_token_lens else 0,
        },
        "questions": per_question,
    }
    json_path = out / "aime24_parquet_analysis.json"
    json_dump(summary, json_path)
    text_path = out / "aime24_parquet_examples.txt"
    with text_path.open("w") as f:
        f.write(
            f"pass16={summary['pass16']} pass8={summary['pass8']} avg_pass1={summary['avg_pass1']}\n"
            f"empty={empty}/{total_responses} boxed={boxed}/{total_responses}\n"
            f"char_len={summary['char_len']} token_len={summary['token_len']}\n"
        )
        for question in per_question[: args.example_questions]:
            f.write(
                f"\n===== question {question['question_idx']} gt={question['ground_truth']} "
                f"pass16={question['pass16']} avg={question['avg_pass1']} =====\n"
            )
            for example in question["examples"]:
                f.write(
                    f"\n--- response {example['idx']} score={example['score']} "
                    f"chars={example['char_len']} tokens={example['token_len']} boxed={example['contains_boxed']} ---\n"
                )
                f.write(example["prefix"] + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["metadata", "hf-smoke", "analyze-parquet"], required=True)
    parser.add_argument("--bad-model", default=os.environ.get("BAD_MODEL", DEFAULT_BAD_MODEL))
    parser.add_argument("--ref-model", default=os.environ.get("REF_MODEL", DEFAULT_REF_MODEL))
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--dataset", default=os.environ.get("AIME24_PATH", DEFAULT_AIME24))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generation-parquet", default="")
    parser.add_argument("--smoke-max-new-tokens", type=int, default=2048)
    parser.add_argument("--smoke-num-return", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--example-questions", type=int, default=5)
    parser.add_argument("--example-responses", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "metadata":
        run_metadata(args)
    elif args.stage == "hf-smoke":
        run_hf_smoke(args)
    elif args.stage == "analyze-parquet":
        if not args.generation_parquet:
            raise ValueError("--generation-parquet is required for analyze-parquet")
        run_analyze_parquet(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
