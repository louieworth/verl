#!/usr/bin/env python3
"""Control a resident OPD y_o vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import os

import ray

from recipe.opd.rollout_sync import load_resident_rollout_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control resident OPD y_o vLLM actors")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--action", required=True, choices=["sleep", "wake", "clear_kv_cache"])
    parser.add_argument("--ray_address", default=os.environ.get("RAY_ADDRESS", ""))
    parser.add_argument("--ray_namespace", default=os.environ.get("RESIDENT_YO_RAY_NAMESPACE") or os.environ.get("RAY_NAMESPACE") or "")
    return parser.parse_args()


async def _run_action(manifest: dict, action: str) -> None:
    tasks = []
    for replica_rank in range(int(manifest["num_replicas"])):
        namespace = manifest.get("ray_namespace") or None
        handle = ray.get_actor(
            f"{manifest.get('server_actor_prefix', 'vllm_server')}_{replica_rank}_0",
            namespace=namespace,
        )
        if action == "sleep":
            tasks.append(handle.sleep.remote())
        elif action == "wake":
            tasks.append(handle.wake_up.remote())
        elif action == "clear_kv_cache":
            tasks.append(handle.clear_kv_cache.remote())
        else:
            raise ValueError(action)
    await asyncio.gather(*tasks)


def main() -> None:
    args = parse_args()
    manifest = load_resident_rollout_manifest(args.manifest)
    address = args.ray_address or manifest.get("ray_address") or "auto"
    namespace = args.ray_namespace or manifest.get("ray_namespace") or None
    ray.init(address=address, namespace=namespace, ignore_reinit_error=True)
    asyncio.run(_run_action(manifest, args.action))


if __name__ == "__main__":
    main()
