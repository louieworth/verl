from __future__ import annotations

import glob
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from recipe.dpo.data.precompute_point_reference_logps import PointwiseTensorizer, compute_reference_logps_for_table, parse_dtype
from recipe.dpo.sample_id import DEFAULT_SAMPLE_ID_KEY

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def maybe_tqdm(iterable, *, desc: str, total: int | None = None, unit: str = "it"):
    if tqdm is None:
        return iterable
    if iterable is None:
        return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True)


def _normalize_files(data_files: str | list[str] | Any) -> list[str]:
    if data_files is None:
        return []
    if isinstance(data_files, str):
        return [data_files]
    return [str(path) for path in data_files]


def _path_fingerprint(path: str) -> str:
    stat = os.stat(path)
    payload = {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)),
    }
    return _stable_fingerprint(payload)


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _sanitize_output_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_-")
    return sanitized or "reference_logps"


def _reference_namespace(config: DictConfig) -> str:
    ref_model_path = config.actor_rollout_ref.ref.get("model", {}).get("path", config.actor_rollout_ref.model.path)
    tokenizer_path = config.actor_rollout_ref.model.get("tokenizer_path", None) or config.actor_rollout_ref.model.path
    payload = {
        "ref_model_path": ref_model_path,
        "tokenizer_path": tokenizer_path,
        "reference_logps_key": config.data.get("reference_logps_key", "reference_logps"),
        "prompt_key": config.data.get("prompt_key", "prompt"),
        "response_key": config.data.get("response_key", "response"),
        "max_prompt_length": int(config.data.get("max_prompt_length", 1024)),
        "max_response_length": int(config.data.get("max_response_length", 1024)),
        "prompt_truncation": config.data.get("prompt_truncation", "left"),
        "add_eos": bool(config.data.get("add_eos", True)),
        "apply_chat_template_kwargs": dict(config.data.get("apply_chat_template_kwargs", {})),
        "dtype": config.actor_rollout_ref.ref.fsdp_config.get(
            "model_dtype",
            config.actor_rollout_ref.actor.fsdp_config.get("model_dtype", "bf16"),
        ),
        "attn_implementation": config.actor_rollout_ref.model.get("override_config", {}).get(
            "attn_implementation",
            "flash_attention_2",
        ),
        "trust_remote_code": bool(config.actor_rollout_ref.model.get("trust_remote_code", False)),
        "average_log_prob": bool(config.algorithm.get("average_log_prob", False)),
    }
    return _stable_fingerprint(payload)


def _output_root(config: DictConfig) -> str:
    configured = config.data.get("reference_logps_materialized_dir", None)
    if configured:
        return os.path.expanduser(str(configured))
    return os.path.expanduser("~/.cache/verl/dpo/reference_logps_materialized")


def _allow_cross_namespace_reuse(config: DictConfig) -> bool:
    return bool(config.data.get("reference_logps_allow_cross_namespace_reuse", False))


def _allow_sample_id_reuse(config: DictConfig) -> bool:
    return bool(config.data.get("reference_logps_allow_sample_id_reuse", False))


def _sample_id_key(config: DictConfig) -> str:
    return str(config.data.get("sample_id_key", DEFAULT_SAMPLE_ID_KEY))


def _output_name_components(
    path: str,
    explicit_output_name: str | None = None,
) -> tuple[str, str, str, str]:
    source_fingerprint = _path_fingerprint(path)
    short_source_fingerprint = source_fingerprint[:10]
    base_name = Path(path).stem
    suffix = Path(path).suffix or ".parquet"
    if explicit_output_name is None:
        return base_name, suffix, source_fingerprint, short_source_fingerprint

    explicit_name = Path(str(explicit_output_name)).name
    explicit_suffix = suffix
    explicit_base = explicit_name
    if explicit_name.endswith(".parquet"):
        explicit_base = explicit_name[: -len(".parquet")]
        explicit_suffix = ".parquet"
    explicit_base = _sanitize_output_name(explicit_base)
    return explicit_base, explicit_suffix, source_fingerprint, short_source_fingerprint


def _output_path_for_file(
    path: str,
    namespace: str,
    output_root: str,
    explicit_output_name: str | None = None,
) -> str:
    output_base, output_suffix, source_fingerprint, short_source_fingerprint = _output_name_components(
        path, explicit_output_name
    )
    short_namespace = namespace[:8]
    if explicit_output_name:
        return os.path.join(
            output_root,
            f"{output_base}_ns{short_namespace}_src{short_source_fingerprint}{output_suffix}",
        )
    return os.path.join(output_root, namespace, f"{output_base}_{source_fingerprint}{output_suffix}")


def _cross_namespace_output_candidates(
    path: str,
    output_root: str,
    *,
    explicit_output_name: str | None = None,
) -> list[str]:
    output_base, output_suffix, source_fingerprint, short_source_fingerprint = _output_name_components(
        path, explicit_output_name
    )
    if explicit_output_name:
        pattern = os.path.join(output_root, f"{output_base}_ns*_src{short_source_fingerprint}{output_suffix}")
    else:
        pattern = os.path.join(output_root, "*", f"{output_base}_{source_fingerprint}{output_suffix}")
    return sorted(candidate for candidate in glob.glob(pattern) if os.path.isfile(candidate))


def _sample_id_reuse_candidates(
    path: str,
    output_root: str,
    *,
    explicit_output_name: str | None = None,
) -> list[str]:
    output_base, output_suffix, _, _ = _output_name_components(path, explicit_output_name)
    if explicit_output_name:
        pattern = os.path.join(output_root, f"{output_base}_ns*_src*{output_suffix}")
    else:
        pattern = os.path.join(output_root, "*", f"{output_base}_*{output_suffix}")
    return sorted(candidate for candidate in glob.glob(pattern) if os.path.isfile(candidate))


def _parse_csv_files(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _choose_num_workers(config: DictConfig) -> int:
    configured = int(config.data.get("reference_logps_num_workers", 0))
    if configured > 0:
        return configured

    trainer_world_size = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    if cluster_gpus > 0:
        return min(trainer_world_size, cluster_gpus)
    return 0


def _reference_model_path(config: DictConfig) -> str:
    return config.actor_rollout_ref.ref.get("model", {}).get("path", config.actor_rollout_ref.model.path)


def _reference_tokenizer_path(config: DictConfig) -> str:
    return config.actor_rollout_ref.model.get("tokenizer_path", None) or config.actor_rollout_ref.model.path


def _parquet_has_column(path: str, column_name: str) -> bool:
    parquet = pq.ParquetFile(path)
    return column_name in parquet.schema.names


def _parquet_has_reference_logps(path: str, column_name: str) -> bool:
    return _parquet_has_column(path, column_name)


def _metadata_path_for_output(output_path: str) -> Path:
    return Path(f"{output_path}.metadata.json")


def _write_output_metadata(config: DictConfig, namespace: str, output_path: str) -> None:
    metadata_path = _metadata_path_for_output(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        return

    payload = {
        "namespace": namespace,
        "output_path": output_path,
        "allow_cross_namespace_reuse": _allow_cross_namespace_reuse(config),
        "allow_sample_id_reuse": _allow_sample_id_reuse(config),
        "sample_id_key": _sample_id_key(config),
        "reference_model_path": _reference_model_path(config),
        "tokenizer_path": _reference_tokenizer_path(config),
        "reference_logps_key": config.data.get("reference_logps_key", "reference_logps"),
        "prompt_key": config.data.get("prompt_key", "prompt"),
        "response_key": config.data.get("response_key", "response"),
        "max_prompt_length": int(config.data.get("max_prompt_length", 1024)),
        "max_response_length": int(config.data.get("max_response_length", 1024)),
        "prompt_truncation": config.data.get("prompt_truncation", "left"),
        "add_eos": bool(config.data.get("add_eos", True)),
        "attn_implementation": config.actor_rollout_ref.model.get("override_config", {}).get(
            "attn_implementation",
            "flash_attention_2",
        ),
        "dtype": config.actor_rollout_ref.ref.fsdp_config.get(
            "model_dtype",
            config.actor_rollout_ref.actor.fsdp_config.get("model_dtype", "bf16"),
        ),
        "generated_at_epoch_seconds": time.time(),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


@ray.remote(num_gpus=1)
class ReferenceLogpsWorker:
    def __init__(self, worker_config: dict[str, Any]):
        from verl.utils import hf_tokenizer

        self.prompt_key = worker_config["prompt_key"]
        self.response_key = worker_config["response_key"]
        self.device = torch.device(worker_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dtype = parse_dtype(worker_config["dtype"], self.device)
        tokenizer = hf_tokenizer(worker_config["tokenizer_path"], trust_remote_code=worker_config["trust_remote_code"])
        self.tensorizer = PointwiseTensorizer(
            tokenizer=tokenizer,
            max_prompt_length=worker_config["max_prompt_length"],
            max_response_length=worker_config["max_response_length"],
            prompt_truncation=worker_config["prompt_truncation"],
            add_eos=worker_config["add_eos"],
            apply_chat_template_kwargs=worker_config["apply_chat_template_kwargs"],
        )

        model_kwargs = {
            "dtype": self.model_dtype,
            "trust_remote_code": worker_config["trust_remote_code"],
        }
        if self.device.type == "cuda":
            model_kwargs["attn_implementation"] = worker_config["attn_implementation"]
        self.model = AutoModelForCausalLM.from_pretrained(worker_config["model_path"], **model_kwargs)
        self.model.eval().to(self.device)
        self.max_batch_size = worker_config["max_batch_size"]
        self.max_batched_tokens = worker_config["max_batched_tokens"]
        self.average_log_prob = bool(worker_config.get("average_log_prob", False))

    def compute(self, prompts: list[Any], responses: list[Any]) -> list[float]:
        table = pa.table(
            {
                self.prompt_key: pa.array(prompts),
                self.response_key: pa.array(responses),
            }
        )
        logps = compute_reference_logps_for_table(
            table,
            prompt_key=self.prompt_key,
            response_key=self.response_key,
            model=self.model,
            tensorizer=self.tensorizer,
            device=self.device,
            autocast_dtype=self.model_dtype,
            max_batch_size=self.max_batch_size,
            max_batched_tokens=self.max_batched_tokens,
            average_log_prob=self.average_log_prob,
        )
        return logps.tolist()


@dataclass
class ReferenceWorkerPool:
    workers: list[Any]
    rows_per_task: int

    def close(self) -> None:
        for worker in self.workers:
            ray.kill(worker, no_restart=True)

    def compute(self, prompts: list[Any], responses: list[Any]) -> np.ndarray:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        if not prompts:
            return np.empty((0,), dtype=np.float32)

        num_workers = len(self.workers)
        if num_workers == 0:
            raise ValueError("ReferenceWorkerPool requires at least one worker")

        outputs = np.empty((len(prompts),), dtype=np.float32)
        pending: dict[Any, tuple[int, int]] = {}
        next_worker_idx = 0

        def submit(start: int, end: int) -> None:
            nonlocal next_worker_idx
            worker = self.workers[next_worker_idx]
            next_worker_idx = (next_worker_idx + 1) % num_workers
            object_ref = worker.compute.remote(prompts[start:end], responses[start:end])
            pending[object_ref] = (start, end)

        for start in range(0, len(prompts), self.rows_per_task):
            end = min(start + self.rows_per_task, len(prompts))
            submit(start, end)
            if len(pending) >= max(1, num_workers * 2):
                done, _ = ray.wait(list(pending.keys()), num_returns=1)
                object_ref = done[0]
                chunk_start, chunk_end = pending.pop(object_ref)
                outputs[chunk_start:chunk_end] = np.asarray(ray.get(object_ref), dtype=np.float32)

        while pending:
            done, _ = ray.wait(list(pending.keys()), num_returns=1)
            object_ref = done[0]
            chunk_start, chunk_end = pending.pop(object_ref)
            outputs[chunk_start:chunk_end] = np.asarray(ray.get(object_ref), dtype=np.float32)

        return outputs


class LocalReferenceLogpsRunner:
    def __init__(self, config: DictConfig):
        from verl.utils import hf_tokenizer

        self.prompt_key = config.data.get("prompt_key", "prompt")
        self.response_key = config.data.get("response_key", "response")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = config.actor_rollout_ref.ref.fsdp_config.get(
            "model_dtype",
            config.actor_rollout_ref.actor.fsdp_config.get("model_dtype", "bf16"),
        )
        self.model_dtype = parse_dtype(dtype, self.device)
        tokenizer = hf_tokenizer(
            _reference_tokenizer_path(config),
            trust_remote_code=bool(config.actor_rollout_ref.model.get("trust_remote_code", False)),
        )
        self.tensorizer = PointwiseTensorizer(
            tokenizer=tokenizer,
            max_prompt_length=int(config.data.get("max_prompt_length", 1024)),
            max_response_length=int(config.data.get("max_response_length", 1024)),
            prompt_truncation=config.data.get("prompt_truncation", "left"),
            add_eos=bool(config.data.get("add_eos", True)),
            apply_chat_template_kwargs=dict(config.data.get("apply_chat_template_kwargs", {})),
        )
        model_kwargs = {
            "dtype": self.model_dtype,
            "trust_remote_code": bool(config.actor_rollout_ref.model.get("trust_remote_code", False)),
        }
        if self.device.type == "cuda":
            model_kwargs["attn_implementation"] = config.actor_rollout_ref.model.get("override_config", {}).get(
                "attn_implementation",
                "flash_attention_2",
            )
        self.model = AutoModelForCausalLM.from_pretrained(_reference_model_path(config), **model_kwargs)
        self.model.eval().to(self.device)
        self.max_batch_size = int(config.data.get("reference_logps_max_batch_size", 8))
        self.max_batched_tokens = int(config.data.get("reference_logps_max_batched_tokens", 24576))
        self.average_log_prob = bool(config.algorithm.get("average_log_prob", False))

    def compute(self, prompts: list[Any], responses: list[Any]) -> np.ndarray:
        table = pa.table(
            {
                self.prompt_key: pa.array(prompts),
                self.response_key: pa.array(responses),
            }
        )
        return compute_reference_logps_for_table(
            table,
            prompt_key=self.prompt_key,
            response_key=self.response_key,
            model=self.model,
            tensorizer=self.tensorizer,
            device=self.device,
            autocast_dtype=self.model_dtype,
            max_batch_size=self.max_batch_size,
            max_batched_tokens=self.max_batched_tokens,
            average_log_prob=self.average_log_prob,
        )


def _build_worker_pool(config: DictConfig) -> ReferenceWorkerPool | None:
    num_workers = _choose_num_workers(config)
    rows_per_task = int(config.data.get("reference_logps_rows_per_task", 2048))
    if num_workers <= 0:
        return None

    worker_config = {
        "model_path": _reference_model_path(config),
        "tokenizer_path": _reference_tokenizer_path(config),
        "prompt_key": config.data.get("prompt_key", "prompt"),
        "response_key": config.data.get("response_key", "response"),
        "max_prompt_length": int(config.data.get("max_prompt_length", 1024)),
        "max_response_length": int(config.data.get("max_response_length", 1024)),
        "prompt_truncation": config.data.get("prompt_truncation", "left"),
        "add_eos": bool(config.data.get("add_eos", True)),
        "apply_chat_template_kwargs": dict(config.data.get("apply_chat_template_kwargs", {})),
        "dtype": config.actor_rollout_ref.ref.fsdp_config.get(
            "model_dtype",
            config.actor_rollout_ref.actor.fsdp_config.get("model_dtype", "bf16"),
        ),
        "attn_implementation": config.actor_rollout_ref.model.get("override_config", {}).get(
            "attn_implementation",
            "flash_attention_2",
        ),
        "trust_remote_code": bool(config.actor_rollout_ref.model.get("trust_remote_code", False)),
        "max_batch_size": int(config.data.get("reference_logps_max_batch_size", 8)),
        "max_batched_tokens": int(config.data.get("reference_logps_max_batched_tokens", 24576)),
        "average_log_prob": bool(config.algorithm.get("average_log_prob", False)),
        "device": "cuda",
    }
    workers = [ReferenceLogpsWorker.remote(worker_config) for _ in range(num_workers)]
    return ReferenceWorkerPool(workers=workers, rows_per_task=rows_per_task)


def _compute_reference_logps(
    prompts: list[Any],
    responses: list[Any],
    worker_pool: ReferenceWorkerPool | None,
    local_runner: LocalReferenceLogpsRunner | None,
) -> np.ndarray:
    if worker_pool is not None:
        return worker_pool.compute(prompts, responses)
    if local_runner is None:
        raise ValueError("Either worker_pool or local_runner must be provided")
    return local_runner.compute(prompts, responses)


def _load_sample_id_reference_logps_lookup(
    path: str,
    *,
    sample_id_key: str,
    reference_logps_key: str,
) -> dict[str, float]:
    parquet = pq.ParquetFile(path)
    lookup: dict[str, float] = {}
    for row_group_idx in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group_idx, columns=[sample_id_key, reference_logps_key])
        sample_ids = table.column(sample_id_key).to_pylist()
        reference_logps = table.column(reference_logps_key).to_pylist()
        for sample_id, reference_logp in zip(sample_ids, reference_logps, strict=False):
            if sample_id is None or reference_logp is None:
                continue
            if isinstance(reference_logp, (float, np.floating)) and np.isnan(reference_logp):
                continue
            lookup[str(sample_id)] = float(reference_logp)
    return lookup


def _load_sample_id_set(path: str, *, sample_id_key: str) -> set[str]:
    parquet = pq.ParquetFile(path)
    sample_ids: set[str] = set()
    for row_group_idx in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group_idx, columns=[sample_id_key])
        sample_ids.update(str(sample_id) for sample_id in table.column(sample_id_key).to_pylist() if sample_id is not None)
    return sample_ids


def _select_sample_id_reuse_lookup(
    path: str,
    output_root: str,
    *,
    explicit_output_name: str | None,
    sample_id_key: str,
    reference_logps_key: str,
    current_output_path: str,
) -> tuple[str, dict[str, float], int, int] | None:
    candidates = [
        candidate
        for candidate in _sample_id_reuse_candidates(
            path,
            output_root,
            explicit_output_name=explicit_output_name,
        )
        if candidate != current_output_path
        and _parquet_has_column(candidate, sample_id_key)
        and _parquet_has_reference_logps(candidate, reference_logps_key)
    ]
    if not candidates:
        return None

    target_sample_ids = _load_sample_id_set(path, sample_id_key=sample_id_key)
    if not target_sample_ids:
        return None

    best_path: str | None = None
    best_lookup: dict[str, float] | None = None
    best_coverage = 0
    for candidate in sorted(candidates, key=lambda item: os.path.getmtime(item), reverse=True):
        lookup = _load_sample_id_reference_logps_lookup(
            candidate,
            sample_id_key=sample_id_key,
            reference_logps_key=reference_logps_key,
        )
        coverage = sum(1 for sample_id in target_sample_ids if sample_id in lookup)
        if coverage > best_coverage:
            best_path = candidate
            best_lookup = lookup
            best_coverage = coverage
        if coverage == len(target_sample_ids):
            break

    if best_path is None or best_lookup is None or best_coverage <= 0:
        return None
    return best_path, best_lookup, best_coverage, len(target_sample_ids)


def _materialize_output(
    path: str,
    output_path: str,
    config: DictConfig,
    worker_pool: ReferenceWorkerPool | None,
    local_runner: LocalReferenceLogpsRunner | None,
    *,
    namespace: str,
    sample_id_lookup: dict[str, float] | None = None,
    sample_id_key: str | None = None,
    reuse_path: str | None = None,
) -> str:
    reference_logps_key = config.data.get("reference_logps_key", "reference_logps")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _write_output_metadata(config, namespace, output_path)
    temp_output_path = f"{output_path}.tmp-{uuid.uuid4().hex}"

    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    worker_count = len(worker_pool.workers) if worker_pool is not None else 1
    if sample_id_lookup is None:
        print(
            "[ref-logps] Cache miss. Materializing "
            f"{reference_logps_key} for {path} -> {output_path} "
            f"(row_groups={parquet.num_row_groups}, rows={total_rows}, workers={worker_count})"
        )
    else:
        print(
            "[ref-logps] Materializing with sample_id reuse "
            f"for {path} -> {output_path} from {reuse_path} "
            f"(row_groups={parquet.num_row_groups}, rows={total_rows}, workers={worker_count})"
        )

    writer: pq.ParquetWriter | None = None
    progress = maybe_tqdm(
        None,
        desc=f"Precomputing ref logps {os.path.basename(path)}",
        total=total_rows,
        unit="row",
    )
    rows_done = 0
    rows_reused = 0
    rows_computed = 0

    try:
        for row_group_idx in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group_idx)
            if reference_logps_key in table.column_names:
                column_idx = table.column_names.index(reference_logps_key)
                table = table.remove_column(column_idx)

            reference_logps: np.ndarray | None = None
            if sample_id_lookup is not None and sample_id_key is not None and sample_id_key in table.column_names:
                sample_ids = table.column(sample_id_key).to_pylist()
                reference_logps = np.empty(len(sample_ids), dtype=np.float32)
                missing_indices: list[int] = []
                for sample_idx, sample_id in enumerate(sample_ids):
                    cached_value = sample_id_lookup.get(str(sample_id)) if sample_id is not None else None
                    if cached_value is None:
                        missing_indices.append(sample_idx)
                        continue
                    reference_logps[sample_idx] = cached_value
                rows_reused += len(sample_ids) - len(missing_indices)
                if missing_indices:
                    prompts = table.column(config.data.get("prompt_key", "prompt")).to_pylist()
                    responses = table.column(config.data.get("response_key", "response")).to_pylist()
                    missing_reference_logps = _compute_reference_logps(
                        [prompts[idx] for idx in missing_indices],
                        [responses[idx] for idx in missing_indices],
                        worker_pool,
                        local_runner,
                    )
                    reference_logps[np.asarray(missing_indices, dtype=np.int64)] = missing_reference_logps
                    rows_computed += len(missing_indices)

            if reference_logps is None:
                prompts = table.column(config.data.get("prompt_key", "prompt")).to_pylist()
                responses = table.column(config.data.get("response_key", "response")).to_pylist()
                reference_logps = _compute_reference_logps(prompts, responses, worker_pool, local_runner)
                rows_computed += len(prompts)

            table = table.append_column(reference_logps_key, pa.array(reference_logps, type=pa.float32()))

            if writer is None:
                writer = pq.ParquetWriter(
                    temp_output_path,
                    table.schema,
                    compression=config.data.get("reference_logps_compression", "zstd"),
                )
            writer.write_table(table)
            rows_done += table.num_rows
            if progress is not None:
                progress.update(table.num_rows)
                postfix = {"groups": f"{row_group_idx + 1}/{parquet.num_row_groups}", "workers": worker_count}
                if sample_id_lookup is not None:
                    postfix["reused"] = rows_reused
                    postfix["computed"] = rows_computed
                progress.set_postfix(**postfix)
    finally:
        if hasattr(progress, "close"):
            progress.close()
        if writer is not None:
            writer.close()

    os.replace(temp_output_path, output_path)
    if sample_id_lookup is not None:
        print(
            f"[ref-logps] Saved materialized parquet with sample_id reuse: {output_path} "
            f"(reused={rows_reused}, computed={rows_computed})"
        )
    else:
        print(f"[ref-logps] Saved materialized parquet: {output_path}")
    return output_path


def _materialize_file(
    path: str,
    config: DictConfig,
    namespace: str,
    worker_pool: ReferenceWorkerPool | None,
    local_runner: LocalReferenceLogpsRunner | None,
    *,
    explicit_output_name: str | None = None,
) -> str:
    reference_logps_key = config.data.get("reference_logps_key", "reference_logps")
    output_root = _output_root(config)
    output_path = _output_path_for_file(path, namespace, output_root, explicit_output_name=explicit_output_name)
    sample_id_key = _sample_id_key(config)

    if _parquet_has_reference_logps(path, reference_logps_key):
        print(f"[ref-logps] Reusing source parquet with existing {reference_logps_key}: {path}")
        return path
    if os.path.exists(output_path) and _parquet_has_reference_logps(output_path, reference_logps_key):
        _write_output_metadata(config, namespace, output_path)
        print(f"[ref-logps] Reusing cached parquet: {output_path}")
        return output_path
    if _allow_cross_namespace_reuse(config):
        fallback_candidates = [
            candidate
            for candidate in _cross_namespace_output_candidates(
                path,
                output_root,
                explicit_output_name=explicit_output_name,
            )
            if candidate != output_path and _parquet_has_reference_logps(candidate, reference_logps_key)
        ]
        if len(fallback_candidates) == 1:
            print(f"[ref-logps] Reusing cached parquet via namespace fallback: {fallback_candidates[0]}")
            return fallback_candidates[0]
        if len(fallback_candidates) > 1:
            print(
                "[ref-logps] Namespace fallback skipped because multiple cached parquets matched: "
                + ", ".join(fallback_candidates)
            )
    if _allow_sample_id_reuse(config) and _parquet_has_column(path, sample_id_key):
        selected_lookup = _select_sample_id_reuse_lookup(
            path,
            output_root,
            explicit_output_name=explicit_output_name,
            sample_id_key=sample_id_key,
            reference_logps_key=reference_logps_key,
            current_output_path=output_path,
        )
        if selected_lookup is not None:
            reuse_path, sample_id_lookup, covered_rows, total_rows = selected_lookup
            print(
                "[ref-logps] Reusing cached parquet via sample_id lookup: "
                f"{reuse_path} (covered_rows={covered_rows}/{total_rows})"
            )
            return _materialize_output(
                path,
                output_path,
                config,
                worker_pool,
                local_runner,
                namespace=namespace,
                sample_id_lookup=sample_id_lookup,
                sample_id_key=sample_id_key,
                reuse_path=reuse_path,
            )

    return _materialize_output(
        path,
        output_path,
        config,
        worker_pool,
        local_runner,
        namespace=namespace,
    )


def _prepare_group_files(
    files: str | list[str] | Any,
    config: DictConfig,
    namespace: str,
    worker_pool: ReferenceWorkerPool | None,
    local_runner: LocalReferenceLogpsRunner | None,
    *,
    reuse_csv: str | None = None,
    output_names_csv: str | None = None,
) -> list[str]:
    source_files = _parse_csv_files(reuse_csv) or _normalize_files(files)
    explicit_output_names = _parse_csv_files(output_names_csv)
    if explicit_output_names and len(explicit_output_names) != len(source_files):
        raise ValueError(
            "reference_logps output name count mismatch: "
            f"expected {len(source_files)}, got {len(explicit_output_names)}"
        )
    prepared_files = []
    for idx, path in enumerate(source_files):
        explicit_output_name = explicit_output_names[idx] if explicit_output_names else None
        if not path.endswith(".parquet"):
            prepared_files.append(path)
            continue
        prepared_files.append(
            _materialize_file(
                path,
                config,
                namespace,
                worker_pool,
                local_runner,
                explicit_output_name=explicit_output_name,
            )
        )
    return prepared_files


def maybe_materialize_pointwise_reference_logps(config: DictConfig) -> tuple[list[str], list[str] | None]:
    if not bool(config.data.get("auto_precompute_reference_logps", False)):
        return _normalize_files(config.data.train_files), _normalize_files(config.data.get("val_files", None)) or None

    namespace = _reference_namespace(config)
    worker_pool = _build_worker_pool(config)
    local_runner = None if worker_pool is not None else LocalReferenceLogpsRunner(config)
    worker_count = len(worker_pool.workers) if worker_pool is not None else 1
    print(
        "[ref-logps] Auto precompute enabled. "
        f"namespace={namespace}, workers={worker_count}, output_root={_output_root(config)}, "
        f"allow_cross_namespace_reuse={_allow_cross_namespace_reuse(config)}, "
        f"allow_sample_id_reuse={_allow_sample_id_reuse(config)}"
    )
    try:
        train_files = _prepare_group_files(
            config.data.train_files,
            config,
            namespace,
            worker_pool,
            local_runner,
            reuse_csv=config.data.get("reference_logps_reuse_train_files_csv", None),
            output_names_csv=config.data.get("reference_logps_train_output_names_csv", None),
        )
        val_files_raw = config.data.get("val_files", None)
        val_files = None
        if val_files_raw:
            val_files = _prepare_group_files(
                val_files_raw,
                config,
                namespace,
                worker_pool,
                local_runner,
                reuse_csv=config.data.get("reference_logps_reuse_val_files_csv", None),
                output_names_csv=config.data.get("reference_logps_val_output_names_csv", None),
            )
        return train_files, val_files
    finally:
        if worker_pool is not None:
            worker_pool.close()
