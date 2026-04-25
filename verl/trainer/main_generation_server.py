# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os
import sys as _sys

try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
except ModuleNotFoundError:
    _extra = "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/19.0.1/lib/python3.12/site-packages"
    if _extra not in _sys.path:
        _sys.path.insert(0, _extra)
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401

import aiohttp
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio
from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion
from tqdm import tqdm

from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class

PROGRESS_UPDATE_INTERVAL = 500


async def start_server(config):
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(session: aiohttp.ClientSession, server_address, **chat_complete_request):
    extra_headers = chat_complete_request.pop("extra_headers", {})
    async with session.post(
        url=f"http://{server_address}/v1/chat/completions",
        headers={"Authorization": "Bearer token-abc123", **extra_headers},
        json=chat_complete_request,
    ) as resp:
        data = await resp.json()
        # Handle request-level errors (e.g. context length exceeded) without
        # crashing the whole batch. Returning None lets the caller record an
        # empty response for this row and continue.
        if "error" in data or "choices" not in data:
            err_msg = ""
            if isinstance(data.get("error"), dict):
                err_msg = data["error"].get("message", "")
            else:
                err_msg = str(data.get("error", data))
            print(f"[submit_request] skipping failed request: {err_msg[:200]}")
            return None
        return ChatCompletion(**data)


async def generate_per_replica(
    server_address,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_lst: list,
    request_concurrency: int | None,
    progress_bar=None,
):
    # here we should sample n_samples for each chat_lst.
    # we use aiohttp to avoid hang in AsyncOpenAI when the number of requests is large.

    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,
        }
        for messages in chat_lst
        for _ in range(n_samples)
    ]

    total_requests = len(chat_complete_request)
    if total_requests == 0:
        return []

    if request_concurrency is None:
        request_concurrency = total_requests
    else:
        request_concurrency = max(1, min(request_concurrency, total_requests))

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=request_concurrency if request_concurrency < total_requests else 0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        results = [None] * total_requests
        in_flight = {}
        next_request_idx = 0
        pending_progress_updates = 0

        for _ in range(request_concurrency):
            task = asyncio.create_task(
                submit_request(session, server_address, **chat_complete_request[next_request_idx])
            )
            in_flight[task] = next_request_idx
            next_request_idx += 1

        try:
            while in_flight:
                done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    request_idx = in_flight.pop(task)
                    results[request_idx] = await task
                    if progress_bar is not None:
                        pending_progress_updates += 1
                        if pending_progress_updates >= PROGRESS_UPDATE_INTERVAL:
                            progress_bar.update(pending_progress_updates)
                            pending_progress_updates = 0
                    if next_request_idx < total_requests:
                        next_task = asyncio.create_task(
                            submit_request(session, server_address, **chat_complete_request[next_request_idx])
                        )
                        in_flight[next_task] = next_request_idx
                        next_request_idx += 1
        finally:
            if progress_bar is not None and pending_progress_updates > 0:
                progress_bar.update(pending_progress_updates)
            for task in in_flight:
                task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)

    return results


async def generate(
    server_addresses: list,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_numpy: np.ndarray,
    request_concurrency: int | None,
):
    num_replicas = len(server_addresses)
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]
    assert len(server_addresses) == len(chat_sub_array)
    total_requests = len(chat_numpy) * n_samples
    progress_unit = "row" if n_samples == 1 else "sample"
    with tqdm(total=total_requests, desc="Generating", unit=progress_unit) as progress_bar:
        results = await asyncio.gather(
            *[
                generate_per_replica(
                    server_addresses[i],
                    model_path,
                    n_samples,
                    sampling_params,
                    chat_sub_array[i],
                    request_concurrency,
                    progress_bar,
                )
                for i in range(num_replicas)
            ]
        )
    return results


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    default_runtime_env = {
        "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}
    }
    ray_init_kwargs = OmegaConf.select(config, "ray_kwargs.ray_init", default={}) or {}
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    n_samples = config.actor_rollout_ref.rollout.n

    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."
    assert n_samples >= 1, "n_samples should always >= 1"

    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        # "top_k": config.actor_rollout_ref.rollout.top_k,
        "max_tokens": config.actor_rollout_ref.rollout.response_length,
    }

    # Optional anti-repetition knobs. Stored under data.* (free OmegaConf dict)
    # because RolloutConfig is a strict dataclass. vllm_async_server hardcodes
    # repetition_penalty=1.0 at server init; passing it per-request overrides
    # that default. See sample payload in /v1/chat/completions docs.
    _rp = OmegaConf.select(config, "data.repetition_penalty", default=None)
    if _rp is not None and float(_rp) != 1.0:
        sampling_params["repetition_penalty"] = float(_rp)
    _fp = OmegaConf.select(config, "data.frequency_penalty", default=None)
    if _fp is not None and float(_fp) != 0.0:
        sampling_params["frequency_penalty"] = float(_fp)
    _pp = OmegaConf.select(config, "data.presence_penalty", default=None)
    if _pp is not None and float(_pp) != 0.0:
        sampling_params["presence_penalty"] = float(_pp)
    if any(k in sampling_params for k in ("repetition_penalty", "frequency_penalty", "presence_penalty")):
        print(
            "Per-request anti-repetition: "
            f"repetition_penalty={sampling_params.get('repetition_penalty', 1.0)}, "
            f"frequency_penalty={sampling_params.get('frequency_penalty', 0.0)}, "
            f"presence_penalty={sampling_params.get('presence_penalty', 0.0)}"
        )

    # Per-request chat_template_kwargs (e.g. {"enable_thinking": false} for Qwen3).
    # This bypasses vLLM server-init `default_chat_template_kwargs`, which doesn't
    # exist on vLLM <0.13. The kwarg is forwarded as a top-level field in the
    # /v1/chat/completions request body — supported on vLLM 0.12+.
    # Stored under data.* (free OmegaConf dict) because RolloutConfig is a strict
    # dataclass and refuses unknown keys.
    _ctk = OmegaConf.select(config, "data.chat_template_kwargs", default=None)
    if _ctk is not None:
        _ctk = OmegaConf.to_container(_ctk, resolve=True) if not isinstance(_ctk, dict) else _ctk
        if _ctk:
            sampling_params["chat_template_kwargs"] = _ctk
            print(f"Per-request chat_template_kwargs: {_ctk}")

    from omegaconf import ListConfig

    train_files = config.data.train_files
    if not isinstance(train_files, list | ListConfig):
        train_files = [train_files]

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)

    datasets = []
    for train_file in train_files:
        import pyarrow.parquet as _pq
        dataset = _pq.read_table(train_file).to_pandas()
        datasets.append(dataset)

    # concat dataset
    dataset = pd.concat(datasets, axis=0, ignore_index=True)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]
    chat_numpy = np.array(chat_lst)
    request_concurrency = OmegaConf.select(config, "data.request_concurrency", default=None)
    request_concurrency = int(request_concurrency) if request_concurrency is not None else None
    print(f"Per-replica request concurrency: {request_concurrency or 'unbounded'}")

    # start native server
    server_handles, server_addresses = asyncio.run(start_server(config))

    # run generate
    gen_results = asyncio.run(
        generate(
            server_addresses,
            config.actor_rollout_ref.model.path,
            n_samples,
            sampling_params,
            chat_numpy,
            request_concurrency,
        )
    )

    # reshape results into a numpy array
    import itertools

    results = list(itertools.chain.from_iterable(gen_results))

    # extract content from results; None means the request failed (e.g. context
    # length exceeded) — record an empty string so the row is kept for alignment.
    num_failed = sum(1 for r in results if r is None)
    if num_failed:
        print(f"[main] {num_failed}/{len(results)} requests failed and were recorded as empty responses")
    results = np.array([("" if result is None else result.choices[0].message.content) for result in results])
    results = np.reshape(results, (-1, n_samples))

    assert results.shape == (len(chat_lst), n_samples)

    results = results.tolist()

    # add to the data frame
    dataset["responses"] = results

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {config.data.output_path}")
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
