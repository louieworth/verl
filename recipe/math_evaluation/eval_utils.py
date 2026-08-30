#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Evaluation utilities for KL Training

import asyncio
import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf

from verl.trainer.generation_server_env import (
    build_generation_server_runtime_env,
    temporarily_clear_torch_launch_env,
)
from verl.trainer.main_generation_server import generate, shutdown_rollout_servers, start_server

logger = logging.getLogger(__name__)

DEFAULT_EVAL_DATASETS = ("aime25", "aime26", "hmmt26", "amobench")
DEFAULT_PROMPT_LENGTH = 2048
DEFAULT_RESPONSE_LENGTH = 16384
DEFAULT_N_SAMPLES = 16
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.7
DEFAULT_SEED = 42
EXPECTED_CANONICAL_ROWS = {"aime25": 30, "aime26": 30, "hmmt26": 33, "amobench": 39}
GENERATION_CACHE_SCHEMA_VERSION = "opd_math_generation_cache/v1"
BASE_EVAL_PROMPT_CONTRACT = "plain_base_completion_trailing_newline_v1"
CHAT_EVAL_PROMPT_CONTRACT = "tokenizer_chat_template_generation_prompt_v1"
_SMALL_CHECKPOINT_FILE_HASH_LIMIT = 8 * 1024 * 1024
EVAL_PROGRESS_PREFIX = "[EVAL_PROGRESS] "


def emit_eval_progress(phase: str, **details: Any) -> None:
    """Emit one machine-readable line while preserving normal terminal logs."""
    event = {"phase": phase, **details}
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":"))
    print(f"{EVAL_PROGRESS_PREFIX}{encoded}", flush=True)


# The standalone generation server currently exposes the OpenAI chat endpoint.
# Supplying this template makes that endpoint behave as a raw completion API:
# no system/user/assistant markers are added around a Base-model prompt.
BASE_COMPLETION_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}"
    "{% if not loop.last %}{{ '\n' }}{% endif %}{% endfor %}{{ '\n' }}"
)


def completion_text(result) -> str:
    """Extract text from either OpenAI chat or plain completion responses."""
    choice = result.choices[0]
    if hasattr(choice, "text"):
        return choice.text or ""
    return choice.message.content or ""


DATASET_ALIASES = {
    "math-ai/aime25": "aime25",
    "math-ai/aime26": "aime26",
    "matharena/aime_2026": "aime26",
    "aime_2026": "aime26",
    "matharena/hmmt_feb_2026": "hmmt26",
    "hmmt_feb_2026": "hmmt26",
    "bytedance-seed/beyondaime": "beyondaime",
    "beyondaime": "beyondaime",
    "meituan-longcat/amo-bench": "amobench",
    "amo-bench": "amobench",
    "amobench": "amobench",
    "openai/gsm8k": "gsm8k",
    "gsm8k": "gsm8k",
}

DATASET_RESULT_NAMES = {
    "gsm8k": "openai/gsm8k",
}

EVAL_DATASET_ROOTS = {
    "aime26": "aime26/aime26_test.parquet",
    "aime24": "aime24/aime24_test.parquet",
    "aime25": "aime25/aime25_test.parquet",
    "math500": "math500/math500_test.parquet",
    "hmmt23": "hmmt23/hmmt23_test.parquet",
    "hmmt24": "hmmt24/hmmt24_test.parquet",
    "hmmt25": "hmmt25/hmmt25_test.parquet",
    "hmmt26": "hmmt26/hmmt26_test.parquet",
    "amc23": "amc23/amc23_test.parquet",
    "beyondaime": "beyondaime/beyondaime_test.parquet",
    "amobench": "amobench/amobench_test.parquet",
    "gsm8k": "gsm8k/gsm8k_test.parquet",
}


def normalize_eval_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    return DATASET_ALIASES.get(normalized, normalized)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_file_marker(path: Path, size: int) -> str:
    """Hash small files fully and sample both ends of large weight shards."""
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        if size <= _SMALL_CHECKPOINT_FILE_HASH_LIMIT:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        else:
            sample_size = _SMALL_CHECKPOINT_FILE_HASH_LIMIT // 2
            digest.update(handle.read(sample_size))
            handle.seek(max(0, size - sample_size))
            digest.update(handle.read(sample_size))
    return digest.hexdigest()


def _model_or_tokenizer_identity(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.exists():
        marker = hashlib.sha256(f"hf_repo:{path_value}".encode("utf-8")).hexdigest()
        return {"kind": "huggingface_repo", "path": path_value, "marker": marker}

    resolved = path.resolve()
    files = [resolved] if resolved.is_file() else sorted(item for item in resolved.rglob("*") if item.is_file())
    inventory = []
    for file_path in files:
        stat = file_path.stat()
        relative_path = file_path.name if resolved.is_file() else file_path.relative_to(resolved).as_posix()
        inventory.append(
            {
                "path": relative_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "content_marker": _checkpoint_file_marker(file_path, stat.st_size),
            }
        )
    marker = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "local_checkpoint",
        "path": str(resolved),
        "files": len(inventory),
        "marker": marker,
    }


def build_generation_cache_provenance(
    *,
    model_path: str,
    tokenizer_path: Optional[str],
    dataset_path: str,
    dataset_name: str,
    prompt_key: str,
    pass_k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    prompt_length: int,
    seed: int,
    force_base_prompt: bool,
) -> dict[str, Any]:
    """Build the exact provenance contract required to reuse generations."""
    resolved_dataset = Path(dataset_path).resolve()
    if not resolved_dataset.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {resolved_dataset}")
    prompt_contract = BASE_EVAL_PROMPT_CONTRACT if force_base_prompt else CHAT_EVAL_PROMPT_CONTRACT
    return {
        "schema_version": GENERATION_CACHE_SCHEMA_VERSION,
        "model": _model_or_tokenizer_identity(model_path),
        "tokenizer": _model_or_tokenizer_identity(tokenizer_path or model_path),
        "dataset": {
            "name": normalize_eval_dataset_name(dataset_name),
            "path": str(resolved_dataset),
            "sha256": _sha256_file(resolved_dataset),
            "size": resolved_dataset.stat().st_size,
        },
        "sampling": {
            "pass_k": int(pass_k),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": -1,
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            "prompt_length": int(prompt_length),
            "max_model_len": int(
                os.environ.get("EVAL_MAX_MODEL_LEN", str(int(prompt_length) + int(max_tokens)))
            ),
        },
        "prompt": {
            "key": prompt_key,
            "force_base_prompt": bool(force_base_prompt),
            "contract": prompt_contract,
            "base_completion_chat_template_sha256": (
                hashlib.sha256(BASE_COMPLETION_CHAT_TEMPLATE.encode("utf-8")).hexdigest()
                if force_base_prompt
                else None
            ),
        },
    }


def generation_cache_manifest_path(generation_path: str) -> str:
    return f"{generation_path}.manifest.json"


def write_generation_cache_manifest(generation_path: str, provenance: dict[str, Any]) -> None:
    manifest_path = Path(generation_cache_manifest_path(generation_path))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": provenance,
        "generation_sha256": _sha256_file(generation_path),
    }
    handle, temporary = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        dir=manifest_path.parent,
    )
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def generation_cache_manifest_matches(generation_path: str, expected_provenance: dict[str, Any]) -> bool:
    manifest_path = generation_cache_manifest_path(generation_path)
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        logger.warning("Cannot validate generation provenance %s: %s", manifest_path, exc)
        return False
    if manifest.get("provenance") != expected_provenance:
        logger.warning("Generation provenance does not match current request: %s", generation_path)
        return False
    actual_sha256 = _sha256_file(generation_path)
    if manifest.get("generation_sha256") != actual_sha256:
        logger.warning("Generation parquet hash does not match its sidecar: %s", generation_path)
        return False
    return True


def generation_parquet_is_complete(
    generation_path: str,
    source_path: str,
    dataset_name: str,
    pass_k: int,
    *,
    expected_provenance: Optional[dict[str, Any]] = None,
) -> bool:
    """Reject partial/stale generation artifacts before reusing them."""
    try:
        source = pd.read_parquet(source_path, columns=["data_source"])
        generated = pd.read_parquet(generation_path, columns=["responses"])
    except Exception as exc:
        logger.warning("Cannot validate generation cache %s: %s", generation_path, exc)
        return False
    normalized = normalize_eval_dataset_name(dataset_name)
    expected = EXPECTED_CANONICAL_ROWS.get(normalized, len(source))
    if len(source) != expected or len(generated) != expected:
        logger.warning(
            "Incomplete %s cache: source=%s generated=%s expected=%s",
            normalized,
            len(source),
            len(generated),
            expected,
        )
        return False
    responses_complete = all(
        responses is not None and len(responses) == pass_k for responses in generated["responses"]
    )
    if not responses_complete:
        return False
    if expected_provenance is not None:
        return generation_cache_manifest_matches(generation_path, expected_provenance)
    return True


def get_result_data_source_name(dataset_name: str) -> str:
    normalized_name = normalize_eval_dataset_name(dataset_name)
    return DATASET_RESULT_NAMES.get(normalized_name, normalized_name)


def build_result_key(dataset_name: str, pass_k: int) -> str:
    result_name = get_result_data_source_name(dataset_name)
    return f"{result_name}_pass{pass_k}_generation_pass_{pass_k}"


def build_legacy_result_key(dataset_name: str, pass_k: int) -> str | None:
    if pass_k != 1:
        return None
    result_name = get_result_data_source_name(dataset_name)
    return f"{result_name}_generation_pass_1"


def find_existing_result_value(model_results: dict, dataset_name: str, pass_k: int):
    key_candidates = [build_result_key(dataset_name, pass_k)]
    legacy_key = build_legacy_result_key(dataset_name, pass_k)
    if legacy_key is not None:
        key_candidates.append(legacy_key)

    for key in key_candidates:
        if key in model_results:
            return model_results[key]
    return None


def resolve_eval_dataset_paths(dataset_names: list[str], datasets_dir: str) -> dict[str, str]:
    resolved_paths: dict[str, str] = {}
    for dataset_name in dataset_names:
        normalized_name = normalize_eval_dataset_name(dataset_name)
        if normalized_name in EVAL_DATASET_ROOTS:
            resolved_paths[normalized_name] = os.path.join(datasets_dir, EVAL_DATASET_ROOTS[normalized_name])
    return resolved_paths


def _build_generation_config(
    model_path: str,
    tokenizer_path: Optional[str],
    *,
    temperature: float,
    top_p: float,
    pass_k: int,
    prompt_length: int,
    response_length: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_model_parallel_size: int,
):
    model_config = {
        "path": model_path,
        "trust_remote_code": True,
    }
    if tokenizer_path:
        model_config["tokenizer_path"] = tokenizer_path

    return OmegaConf.create(
        {
            "trainer": {
                "nnodes": nnodes,
                "n_gpus_per_node": n_gpus_per_node,
            },
            "actor_rollout_ref": {
                "model": model_config,
                "rollout": {
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": "vllm",
                    "mode": "async",
                    "nnodes": nnodes,
                    "n_gpus_per_node": n_gpus_per_node,
                    "temperature": temperature,
                    "top_k": -1,
                    "top_p": top_p,
                    "do_sample": True,
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "tensor_model_parallel_size": tensor_model_parallel_size,
                    "pipeline_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "n": pass_k,
                    "dtype": "bfloat16",
                    "enforce_eager": False,
                },
            },
        }
    )


def _generate_responses_with_server(
    server_addresses: list[str],
    model_path: str,
    dataset_path: str,
    output_path: str,
    *,
    prompt_key: str,
    pass_k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    prompt_length: int,
    seed: int,
    force_base_prompt: bool,
    tokenizer_path: Optional[str],
    dataset_name: str = "",
    dataset_index: int = 0,
    dataset_total: int = 0,
):
    dataset = pd.read_parquet(dataset_path)
    chat_lst = dataset[prompt_key].tolist()
    chat_lst = [chat.tolist() if hasattr(chat, "tolist") else chat for chat in chat_lst]
    chat_numpy = np.array(chat_lst, dtype=object)
    emit_eval_progress(
        "dataset_loaded",
        dataset=dataset_name,
        dataset_index=dataset_index,
        dataset_total=dataset_total,
        problem_count=len(chat_lst),
        sample_total=pass_k,
        overall_fraction=(dataset_index - 1) / dataset_total if dataset_total else 0.0,
    )

    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if force_base_prompt:
        sampling_params.update(
            {
                "chat_template": BASE_COMPLETION_CHAT_TEMPLATE,
                "add_generation_prompt": False,
            }
        )

    # Do not silently truncate benchmark problems. The server's prompt_length is
    # an allocation hint, while this check enforces the public evaluation cap.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True, use_fast=False)
    over_limit = []
    for index, messages in enumerate(chat_lst):
        rendered = render_base_prompt(messages) if force_base_prompt else tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if token_count > prompt_length:
            over_limit.append((index, token_count))
    if over_limit:
        preview = ", ".join(f"row {idx}: {length}" for idx, length in over_limit[:5])
        raise ValueError(
            f"{len(over_limit)} evaluation prompts exceed the {prompt_length}-token cap ({preview}). "
            "Fix the prepared dataset instead of truncating benchmark problems."
        )

    # The OpenAI seed is request-scoped. Reusing one seed for N duplicated
    # requests can collapse pass@N into N identical samples, so generate one
    # deterministic column at a time with seed, seed+1, ..., seed+N-1.
    responses = [[] for _ in range(len(chat_lst))]
    for sample_index in range(pass_k):
        emit_eval_progress(
            "sample_started",
            dataset=dataset_name,
            dataset_index=dataset_index,
            dataset_total=dataset_total,
            problem_count=len(chat_lst),
            sample_index=sample_index + 1,
            sample_total=pass_k,
            overall_fraction=(
                ((dataset_index - 1) + sample_index / pass_k) / dataset_total if dataset_total else 0.0
            ),
        )
        sample_params = {**sampling_params, "seed": seed + sample_index}
        gen_results = asyncio.run(
            generate(
                server_addresses,
                model_path,
                1,
                sample_params,
                chat_numpy,
            )
        )
        results = list(itertools.chain.from_iterable(gen_results))
        if len(results) != len(chat_lst):
            raise RuntimeError(
                f"Generation seed {seed + sample_index} returned {len(results)} results "
                f"for {len(chat_lst)} prompts"
            )
        for prompt_index, result in enumerate(results):
            responses[prompt_index].append(completion_text(result))
        emit_eval_progress(
            "sample_complete",
            dataset=dataset_name,
            dataset_index=dataset_index,
            dataset_total=dataset_total,
            problem_count=len(chat_lst),
            sample_index=sample_index + 1,
            sample_total=pass_k,
            overall_fraction=(
                ((dataset_index - 1) + (sample_index + 1) / pass_k) / dataset_total
                if dataset_total
                else 0.0
            ),
        )

    dataset["responses"] = responses
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_parquet(output_path)


def generate_responses_with_server(
    server_addresses: list[str],
    model_path: str,
    dataset_path: str,
    output_path: str,
    *,
    prompt_key: str,
    pass_k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    prompt_length: int = DEFAULT_PROMPT_LENGTH,
    seed: int = DEFAULT_SEED,
    force_base_prompt: bool = True,
    tokenizer_path: Optional[str] = None,
    cache_provenance: Optional[dict[str, Any]] = None,
    dataset_name: str = "",
    dataset_index: int = 0,
    dataset_total: int = 0,
):
    _generate_responses_with_server(
        server_addresses,
        model_path,
        dataset_path,
        output_path,
        prompt_key=prompt_key,
        pass_k=pass_k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt_length=prompt_length,
        seed=seed,
        force_base_prompt=force_base_prompt,
        tokenizer_path=tokenizer_path,
        dataset_name=dataset_name,
        dataset_index=dataset_index,
        dataset_total=dataset_total,
    )
    if cache_provenance is not None:
        write_generation_cache_manifest(output_path, cache_provenance)


def render_base_prompt(messages) -> str:
    """Render a chat-shaped parquet prompt as a raw Base-model completion."""

    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, str):
        prompt = messages
        if not prompt.strip():
            raise ValueError("Base evaluation prompt is empty")
        return prompt.strip() + "\n"
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError(f"Expected a non-empty prompt message list, got {type(messages).__name__}")

    contents = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Each prompt message must be an object with a string 'content' field")
        contents.append(message["content"])
    prompt = "\n".join(contents).strip()
    if not prompt:
        raise ValueError("Base evaluation prompt is empty")
    return prompt + "\n"


def _evaluate_generated_output(
    dataset_name: str,
    gen_output: str,
    *,
    prompt_key: str,
    pass_k: int,
    output_json_path: Optional[str],
    model_name: Optional[str],
    model_path: Optional[str],
) -> float:
    eval_cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_eval",
        f"data.path={gen_output}",
        f"data.prompt_key={prompt_key}",
        "custom_reward_function.path=recipe/math_evaluation/compute_score.py",
        "custom_reward_function.name=compute_score_data_source",
        "sample_aggregation=pass_at_k",
    ]

    if output_json_path and model_name:
        eval_cmd.extend(
            [
                f"+output_json_path={output_json_path}",
                f"+model_name={model_name}",
                f"+pass_k={pass_k}",
                f"+model_path={model_path}",
            ]
        )

    subprocess.run(eval_cmd, check=True, capture_output=False)

    if output_json_path and model_name and os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            results = json.load(f)
        if model_name in results:
            value = find_existing_result_value(results[model_name], dataset_name, pass_k)
            if value is not None:
                return value

    return 0.0


def evaluate_generated_output(
    dataset_name: str,
    gen_output: str,
    *,
    prompt_key: str,
    pass_k: int,
    output_json_path: Optional[str],
    model_name: Optional[str],
    model_path: Optional[str] = None,
) -> float:
    return _evaluate_generated_output(
        dataset_name,
        gen_output,
        prompt_key=prompt_key,
        pass_k=pass_k,
        output_json_path=output_json_path,
        model_name=model_name,
        model_path=model_path,
    )


def slice_responses_in_generation_file(source_path: str, target_path: str, pass_k: int):
    dataset = pd.read_parquet(source_path)
    dataset["responses"] = dataset["responses"].apply(lambda responses: list(responses[:pass_k]))
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    dataset.to_parquet(target_path)


def launch_generation_server(
    model_path: str,
    tokenizer_path: Optional[str],
    *,
    temperature: float,
    top_p: float,
    pass_k: int,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_model_parallel_size: int,
):
    prompt_length = int(os.environ.get("EVAL_PROMPT_LENGTH", str(DEFAULT_PROMPT_LENGTH)))
    response_length = int(os.environ.get("EVAL_RESPONSE_LENGTH", str(DEFAULT_RESPONSE_LENGTH)))
    max_model_len = int(os.environ.get("EVAL_MAX_MODEL_LEN", str(prompt_length + response_length)))
    max_num_seqs = int(os.environ.get("EVAL_MAX_NUM_SEQS", "1024"))
    gpu_memory_utilization = float(os.environ.get("EVAL_GPU_MEMORY_UTILIZATION", "0.95"))

    config = _build_generation_config(
        model_path,
        tokenizer_path,
        temperature=temperature,
        top_p=top_p,
        pass_k=pass_k,
        prompt_length=prompt_length,
        response_length=response_length,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
    )

    if ray.is_initialized():
        ray.shutdown()

    with temporarily_clear_torch_launch_env():
        ray.init(runtime_env=build_generation_server_runtime_env())
        rollout_servers, server_addresses = asyncio.run(start_server(config, return_replicas=True))
        return rollout_servers, server_addresses, response_length


def shutdown_generation_server(rollout_servers: list):
    shutdown_rollout_servers(rollout_servers)
    if ray.is_initialized():
        ray.shutdown()


def run_evaluation_suite(
    model_path: str,
    dataset_paths: dict[str, str],
    output_dir: str,
    *,
    pass_k: int = DEFAULT_N_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    nnodes: int = 1,
    n_gpus_per_node: int = 8,
    tensor_model_parallel_size: int = 8,
    output_json_path: Optional[str] = None,
    model_name: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    prompt_key: str = "prompt",
    force_base_prompt: bool = True,
) -> dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    rollout_servers = []
    eval_results: dict[str, float] = {}

    try:
        emit_eval_progress(
            "server_starting",
            dataset="",
            dataset_index=0,
            dataset_total=len(dataset_paths),
            overall_fraction=0.0,
        )
        rollout_servers, server_addresses, response_length = launch_generation_server(
            model_path,
            tokenizer_path,
            temperature=temperature,
            top_p=top_p,
            pass_k=pass_k,
            nnodes=nnodes,
            n_gpus_per_node=n_gpus_per_node,
            tensor_model_parallel_size=tensor_model_parallel_size,
        )
        emit_eval_progress(
            "server_ready",
            dataset="",
            dataset_index=0,
            dataset_total=len(dataset_paths),
            overall_fraction=0.0,
        )

        dataset_total = len(dataset_paths)
        for dataset_index, (dataset_name, dataset_path) in enumerate(dataset_paths.items(), start=1):
            if not dataset_path or not os.path.exists(dataset_path):
                logger.warning("Skipping evaluation dataset %s because %s does not exist", dataset_name, dataset_path)
                continue
            emit_eval_progress(
                "dataset_started",
                dataset=dataset_name,
                dataset_index=dataset_index,
                dataset_total=dataset_total,
                sample_total=pass_k,
                overall_fraction=(dataset_index - 1) / dataset_total,
            )

            gen_output = os.path.join(output_dir, f"{dataset_name}_pass{pass_k}_generation.parquet")
            prompt_length = int(os.environ.get("EVAL_PROMPT_LENGTH", str(DEFAULT_PROMPT_LENGTH)))
            cache_provenance = build_generation_cache_provenance(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                prompt_key=prompt_key,
                pass_k=pass_k,
                temperature=temperature,
                top_p=top_p,
                max_tokens=response_length,
                prompt_length=prompt_length,
                seed=seed,
                force_base_prompt=force_base_prompt,
            )

            if (
                os.path.exists(gen_output)
                and os.path.getsize(gen_output) > 0
                and generation_parquet_is_complete(
                    gen_output,
                    dataset_path,
                    dataset_name,
                    pass_k,
                    expected_provenance=cache_provenance,
                )
            ):
                logger.info("  Reusing existing generated responses for %s: %s", dataset_name, gen_output)
                emit_eval_progress(
                    "generation_cache_reused",
                    dataset=dataset_name,
                    dataset_index=dataset_index,
                    dataset_total=dataset_total,
                    sample_index=pass_k,
                    sample_total=pass_k,
                    overall_fraction=dataset_index / dataset_total,
                )
            else:
                logger.info("  Generating responses for %s with pass@%s...", dataset_name, pass_k)
                generate_responses_with_server(
                    server_addresses,
                    model_path,
                    dataset_path,
                    gen_output,
                    prompt_key=prompt_key,
                    pass_k=pass_k,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=response_length,
                    prompt_length=prompt_length,
                    seed=seed,
                    force_base_prompt=force_base_prompt,
                    tokenizer_path=tokenizer_path,
                    cache_provenance=cache_provenance,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    dataset_total=dataset_total,
                )

            logger.info("  Computing scores for %s...", dataset_name)
            emit_eval_progress(
                "scoring_started",
                dataset=dataset_name,
                dataset_index=dataset_index,
                dataset_total=dataset_total,
                sample_index=pass_k,
                sample_total=pass_k,
                overall_fraction=dataset_index / dataset_total,
            )
            accuracy = evaluate_generated_output(
                dataset_name,
                gen_output,
                prompt_key=prompt_key,
                pass_k=pass_k,
                output_json_path=output_json_path,
                model_name=model_name,
                model_path=model_path,
            )
            eval_results[dataset_name] = accuracy
            emit_eval_progress(
                "dataset_complete",
                dataset=dataset_name,
                dataset_index=dataset_index,
                dataset_total=dataset_total,
                sample_index=pass_k,
                sample_total=pass_k,
                overall_fraction=dataset_index / dataset_total,
            )

        emit_eval_progress(
            "suite_complete",
            dataset="",
            dataset_index=dataset_total,
            dataset_total=dataset_total,
            overall_fraction=1.0,
        )
        return eval_results
    finally:
        shutdown_generation_server(rollout_servers)


def run_evaluation_on_dataset(
    model_path: str,
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    pass_k: int = DEFAULT_N_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    ngpus: int = 8,
    output_json_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> float:
    try:
        results = run_evaluation_suite(
            model_path,
            {dataset_name: dataset_path},
            output_dir,
            pass_k=pass_k,
            temperature=temperature,
            top_p=top_p,
            n_gpus_per_node=ngpus,
            output_json_path=output_json_path,
            model_name=model_name,
        )
        return results.get(dataset_name, 0.0)
    except subprocess.CalledProcessError as e:
        logger.error("  Evaluation failed for %s: %s", dataset_name, e)
        return 0.0


def parse_accuracy_from_output(output: str) -> float:
    import re

    match = re.search(r"accuracy[:\\s]+([0-9.]+)", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def save_eval_results(
    results: Dict[str, float],
    output_path: str,
    model_name: str,
    config: Dict = None,
    model_path: Optional[str] = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    formatted_results = {}
    for dataset, accuracy in results.items():
        key = f"{dataset}_pass1_generation_pass_1"
        formatted_results[key] = accuracy

    existing_results = dict(all_results.get(model_name, {}))
    existing_results["model_path"] = model_path
    existing_results.update(formatted_results)
    all_results[model_name] = existing_results

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    logger.info("Evaluation results saved to: %s", output_path)

    if config:
        detailed_path = output_path.replace(".json", "_detailed.json")
        detailed_output = {
            "model_name": model_name,
            "model_path": model_path,
            "results": formatted_results,
            "config": config,
        }
        with open(detailed_path, "w") as f:
            json.dump(detailed_output, f, indent=4)
        logger.info("Detailed results saved to: %s", detailed_path)
