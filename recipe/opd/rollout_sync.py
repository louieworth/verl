#!/usr/bin/env python3
"""Utilities for syncing an OPD student FSDP engine into a resident vLLM rollout."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import ray
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter


def load_resident_rollout_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    required = {"model", "rollout", "num_replicas"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"resident rollout manifest {path} is missing keys: {missing}")
    return manifest


def ensure_ray_connected(manifest: dict[str, Any]) -> None:
    if ray.is_initialized():
        return
    address = os.environ.get("RAY_ADDRESS") or manifest.get("ray_address") or "auto"
    if address == "":
        address = "auto"
    namespace = manifest.get("ray_namespace") or os.environ.get("RESIDENT_YO_RAY_NAMESPACE") or os.environ.get("RAY_NAMESPACE") or None
    ray.init(address=address, namespace=namespace, ignore_reinit_error=True)


def _build_rollout_adapter(manifest: dict[str, Any]) -> ServerAdapter:
    rollout_config = omega_conf_to_dataclass(OmegaConf.create(manifest["rollout"]), dataclass_type=RolloutConfig)
    model_config = omega_conf_to_dataclass(OmegaConf.create(manifest["model"]), dataclass_type=HFModelConfig)

    infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
    infer_pp = rollout_config.pipeline_model_parallel_size
    infer_world_size = infer_tp * infer_pp
    world_size = dist.get_world_size()
    if world_size % infer_world_size != 0:
        raise ValueError(
            f"trainer world_size={world_size} is not divisible by rollout infer_world_size={infer_world_size}"
        )

    rollout_device_mesh = init_device_mesh(
        get_device_name(),
        mesh_shape=(world_size // infer_world_size, infer_tp, infer_pp),
        mesh_dim_names=["dp", "infer_tp", "infer_pp"],
    )

    if "RAY_LOCAL_WORLD_SIZE" not in os.environ:
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE") or manifest.get("n_gpus_per_node") or world_size
        os.environ["RAY_LOCAL_WORLD_SIZE"] = str(local_world_size)

    return ServerAdapter(config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh)


async def _sync_student_engine_to_rollout_async(
    *,
    student_engine,
    manifest: dict[str, Any],
    global_step: int | None,
) -> None:
    adapter = _build_rollout_adapter(manifest)
    rollout_config = omega_conf_to_dataclass(OmegaConf.create(manifest["rollout"]), dataclass_type=RolloutConfig)

    if rollout_config.free_cache_engine:
        await adapter.resume(tags=["weights"])

    per_tensor_param, peft_config = student_engine.get_per_tensor_param(
        layered_summon=rollout_config.layered_summon,
        base_sync_done=True,
    )
    await adapter.update_weights(
        per_tensor_param,
        peft_config=peft_config,
        base_sync_done=True,
        global_steps=global_step,
    )

    if rollout_config.free_cache_engine:
        await adapter.resume(tags=["kv_cache"])


def sync_student_engine_to_resident_rollout(
    *,
    student_engine,
    manifest_path: str | os.PathLike[str],
    global_step: int | None,
) -> None:
    manifest = load_resident_rollout_manifest(manifest_path)
    ensure_ray_connected(manifest)
    asyncio.run(
        _sync_student_engine_to_rollout_async(
            student_engine=student_engine,
            manifest=manifest,
            global_step=global_step,
        )
    )


def write_manifest(path: str | os.PathLike[str], manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, target)
