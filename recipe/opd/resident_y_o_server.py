#!/usr/bin/env python3
"""Start a resident student vLLM server for OPD y_o rollout."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import time

import ray
from omegaconf import OmegaConf

from recipe.opd.rollout_sync import write_manifest
from verl.trainer.generation_server_env import (
    build_generation_server_runtime_env,
    temporarily_clear_torch_launch_env,
)
from verl.trainer.main_generation_server import start_server


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resident OPD y_o vLLM server")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--n_gpus_per_node", type=int, required=True)
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--max_num_seqs", type=int, default=1024)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--prompt_length", type=int, required=True)
    parser.add_argument("--response_length", type=int, required=True)
    parser.add_argument("--max_model_len", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--load_format", default="auto")
    parser.add_argument("--free_cache_engine", type=str_to_bool, default=True)
    parser.add_argument("--enable_sleep_mode", type=str_to_bool, default=True)
    parser.add_argument("--enable_standalone_sleep", type=str_to_bool, default=True)
    parser.add_argument("--ray_address", default=os.environ.get("RAY_ADDRESS", ""))
    parser.add_argument(
        "--ray_namespace",
        default=os.environ.get("RESIDENT_YO_RAY_NAMESPACE") or os.environ.get("RAY_NAMESPACE") or "opd_resident_y_o",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    rollout = {
        "_target_": "verl.workers.config.RolloutConfig",
        "name": "vllm",
        "mode": "async",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "prompt_length": args.prompt_length,
        "response_length": args.response_length,
        "max_model_len": args.max_model_len,
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "data_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "dtype": "bfloat16",
        "load_format": args.load_format,
        "free_cache_engine": args.free_cache_engine,
        "enable_sleep_mode": args.enable_sleep_mode,
        "enforce_eager": True,
        "n": 1,
        "checkpoint_engine": {
            "_target_": "verl.workers.config.CheckpointEngineConfig",
            "backend": "naive",
            "update_weights_bucket_megabytes": 2048,
            "engine_kwargs": {},
        },
    }
    model = {
        "path": args.model_path,
        "tokenizer_path": args.model_path,
        "trust_remote_code": True,
        "load_tokenizer": True,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }
    return OmegaConf.create(
        {
            "trainer": {
                "nnodes": args.nnodes,
                "n_gpus_per_node": args.n_gpus_per_node,
            },
            "actor_rollout_ref": {
                "model": model,
                "rollout": rollout,
            },
        }
    )


def main() -> None:
    args = parse_args()
    if args.enable_standalone_sleep:
        os.environ["VERL_ENABLE_STANDALONE_VLLM_SLEEP"] = "1"

    config = build_config(args)
    with temporarily_clear_torch_launch_env():
        ray_address = args.ray_address or None
        ray.init(
            address=ray_address,
            namespace=args.ray_namespace or None,
            runtime_env=build_generation_server_runtime_env(),
        )
        _, server_addresses = asyncio.run(start_server(config))

    num_replicas = (args.nnodes * args.n_gpus_per_node) // args.tensor_model_parallel_size
    manifest = {
        "kind": "opd_resident_y_o_vllm",
        "ray_address": args.ray_address or os.environ.get("RAY_ADDRESS") or "auto",
        "ray_namespace": args.ray_namespace or None,
        "model": OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True),
        "rollout": OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True),
        "server_addresses": server_addresses,
        "server_actor_prefix": "vllm_server",
        "num_replicas": num_replicas,
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "nnodes": args.nnodes,
        "n_gpus_per_node": args.n_gpus_per_node,
    }
    write_manifest(args.manifest, manifest)
    print(f"[resident-y_o] ready: {args.manifest}", flush=True)

    stop = False

    def _request_stop(signum, _frame):
        nonlocal stop
        print(f"[resident-y_o] received signal {signum}, shutting down", flush=True)
        stop = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    while not stop:
        time.sleep(5)
    ray.shutdown()


if __name__ == "__main__":
    main()
