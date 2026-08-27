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
import re

import aiohttp
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio
import time
from pprint import pprint

import pandas as pd
import pyarrow.lib
from omegaconf import OmegaConf
from openai.types import Completion
from openai.types.chat import ChatCompletion
from tqdm import tqdm

from verl.trainer.generation_server_env import (
    build_generation_server_runtime_env,
    temporarily_clear_torch_launch_env,
)
from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class

ROLLING_LORA_MODEL_NAME = "123"


def render_base_completion_prompt(messages) -> str:
    """Render a prompt without invoking the tokenizer chat template."""
    if isinstance(messages, str):
        prompt = messages
    else:
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        if not isinstance(messages, list) or not messages:
            raise ValueError("Base completion requires a prompt string or non-empty message list")
        roles = [message.get("role") for message in messages]
        if any(role != "user" for role in roles):
            raise ValueError(f"Base completion only accepts user messages, got roles={roles}")
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Base completion prompt is empty")
    return prompt + "\n"


def chat_template_kwargs_from_env() -> dict[str, bool]:
    """Return an explicit thinking-mode override for chat generation."""
    raw_value = os.environ.get("VERL_ENABLE_THINKING", "").strip().lower()
    if not raw_value:
        return {}
    if raw_value in {"1", "true", "yes", "on"}:
        return {"enable_thinking": True}
    if raw_value in {"0", "false", "no", "off"}:
        return {"enable_thinking": False}
    raise ValueError(
        "VERL_ENABLE_THINKING must be true/false when set "
        f"(got {os.environ.get('VERL_ENABLE_THINKING')!r})"
    )


def read_parquet_compat(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except pyarrow.lib.ArrowNotImplementedError:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        batches = [
            batch.to_pandas()
            for batch in parquet_file.iter_batches(batch_size=1024)
        ]
        if not batches:
            return pd.DataFrame()
        return pd.concat(batches, axis=0, ignore_index=True)


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


async def submit_request(server_address, max_retries=5, retry_base_delay=2.0, **chat_complete_request):
    base_completion = bool(chat_complete_request.pop("_base_completion", False))
    extra_headers = chat_complete_request.pop("extra_headers", {})
    for attempt in range(max_retries + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            try:
                async with session.post(
                    url=f"http://{server_address}/v1/{'completions' if base_completion else 'chat/completions'}",
                    headers={"Authorization": "Bearer token-abc123", **extra_headers},
                    json=chat_complete_request,
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("error", {}).get("message", str(data))
                        match = re.search(
                            r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
                            error_msg,
                        )
                        current_max_tokens = chat_complete_request.get("max_tokens")
                        if resp.status == 400 and match and current_max_tokens is not None:
                            max_model_len = int(match.group(1))
                            prompt_tokens = int(match.group(2))
                            adjusted_max_tokens = max(1, max_model_len - prompt_tokens)
                            if adjusted_max_tokens < int(current_max_tokens):
                                chat_complete_request["max_tokens"] = adjusted_max_tokens
                                print(
                                    f"[Adjust max_tokens] Request to {server_address}: "
                                    f"{current_max_tokens} -> {adjusted_max_tokens} "
                                    f"for prompt_tokens={prompt_tokens}, max_model_len={max_model_len}"
                                )
                                continue
                        raise RuntimeError(f"Server returned {resp.status}: {error_msg}")
                    return Completion(**data) if base_completion else ChatCompletion(**data)
            finally:
                await session.close()
        except Exception as e:
            if attempt < max_retries:
                delay = retry_base_delay * (2 ** attempt)
                print(f"[Retry {attempt + 1}/{max_retries}] Request to {server_address} failed: {e}. Retrying in {delay:.0f}s...")
                await asyncio.sleep(delay)
            else:
                raise


async def submit_indexed_request(request_index: int, server_address: str, **chat_complete_request):
    result = await submit_request(server_address, **chat_complete_request)
    return request_index, result


async def generate_per_replica(
    server_address,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_lst: list,
    progress_bar: tqdm | None = None,
    max_concurrency: int | None = None,
    deadline_epoch_seconds: float | None = None,
):
    # here we should sample n_samples for each chat_lst.
    # we use aiohttp to avoid hang in AsyncOpenAI when the number of requests is large.

    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    base_completion = os.environ.get("VERL_FORCE_BASE_COMPLETION", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    chat_complete_request = []
    env_chat_template_kwargs = chat_template_kwargs_from_env()
    for messages in chat_lst:
        request_params = dict(sampling_params)
        if base_completion:
            # These fields are valid only for the chat-completions endpoint.
            # Math evaluation supplies them when it must emulate Base prompts
            # through chat; forwarding them to /v1/completions can make the
            # OpenAI-compatible server reject an otherwise valid request.
            request_params.pop("chat_template", None)
            request_params.pop("chat_template_kwargs", None)
            request_params.pop("add_generation_prompt", None)
            request = {"model": model_path, **request_params}
            request.update({"prompt": render_base_completion_prompt(messages), "_base_completion": True})
        else:
            if env_chat_template_kwargs:
                chat_template_kwargs = dict(request_params.get("chat_template_kwargs") or {})
                chat_template_kwargs.update(env_chat_template_kwargs)
                request_params["chat_template_kwargs"] = chat_template_kwargs
            request = {"model": model_path, **request_params}
            request["messages"] = messages
        chat_complete_request.extend(dict(request) for _ in range(n_samples))

    if not chat_complete_request:
        return []

    if max_concurrency is None:
        max_concurrency = int(os.environ.get("EVAL_MAX_CONCURRENCY", os.environ.get("GENERATION_MAX_CONCURRENCY", "128")))
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def submit_with_limit(request_index: int, req: dict):
        if semaphore is None:
            return await submit_indexed_request(request_index, server_address, **req)
        async with semaphore:
            return await submit_indexed_request(request_index, server_address, **req)

    tasks = [
        asyncio.create_task(submit_with_limit(request_index, req))
        for request_index, req in enumerate(chat_complete_request)
    ]
    results = [None] * len(tasks)
    pending = set(tasks)

    try:
        while pending:
            timeout = None
            if deadline_epoch_seconds is not None:
                timeout = max(0.0, deadline_epoch_seconds - time.time())
                if timeout <= 0:
                    break

            done, pending = await asyncio.wait(
                pending,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                break
            for task in done:
                request_index, result = await task
                results[request_index] = result
                if progress_bar is not None:
                    progress_bar.update(1)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    return results


async def generate(
    server_addresses: list,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_numpy: np.ndarray,
    deadline_epoch_seconds: float | None = None,
):
    num_replicas = len(server_addresses)
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]
    assert len(server_addresses) == len(chat_sub_array)
    total_requests = len(chat_numpy) * n_samples
    deadline_kwargs = {}
    if deadline_epoch_seconds is not None:
        deadline_kwargs["deadline_epoch_seconds"] = deadline_epoch_seconds
    with tqdm(total=total_requests, desc="Generating responses", dynamic_ncols=True) as progress_bar:
        results = await asyncio.gather(
            *[
                generate_per_replica(
                    server_addresses[i],
                    model_path,
                    n_samples,
                    sampling_params,
                    chat_sub_array[i],
                    progress_bar=progress_bar,
                    **deadline_kwargs,
                )
                for i in range(num_replicas)
            ]
        )
    return results


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    with temporarily_clear_torch_launch_env():
        ray_address = os.environ.get("RAY_ADDRESS") or None
        ray.init(address=ray_address, runtime_env=build_generation_server_runtime_env())

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        n_samples = config.actor_rollout_ref.rollout.n

        if config.actor_rollout_ref.rollout.temperature == 0.0:
            assert n_samples == 1, "When temperature=0, n_samples must be 1."
        assert n_samples >= 1, "n_samples should always >= 1"
        generation_deadline = OmegaConf.select(
            config,
            "data.generation_deadline_epoch_seconds",
            default=None,
        )
        if generation_deadline is not None:
            generation_deadline = float(generation_deadline)
            print(
                "Generation deadline: "
                f"{generation_deadline:.3f} (remaining {max(0.0, generation_deadline - time.time()):.1f}s)"
            )

        sampling_params = {
            "temperature": config.actor_rollout_ref.rollout.temperature,
            "top_p": config.actor_rollout_ref.rollout.top_p,
            "top_k": config.actor_rollout_ref.rollout.top_k,
            # Empty completions cannot form KL targets. Ignore EOS for the
            # first token and retain normal EOS handling after that.
            "min_tokens": int(
                OmegaConf.select(config, "actor_rollout_ref.rollout.min_tokens", default=1)
            ),
            "max_tokens": config.actor_rollout_ref.rollout.response_length,
        }

        from omegaconf import ListConfig

        train_files = config.data.train_files
        if not isinstance(train_files, list | ListConfig):
            train_files = [train_files]

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)

        datasets = []
        for train_file in train_files:
            dataset = read_parquet_compat(train_file)
            datasets.append(dataset)

        # concat dataset
        dataset = pd.concat(datasets, axis=0, ignore_index=True)
        chat_lst = dataset[config.data.prompt_key].tolist()
        chat_lst = [chat.tolist() if hasattr(chat, "tolist") else chat for chat in chat_lst]
        chat_numpy = np.array(chat_lst)

        # start native server
        server_handles, server_addresses = asyncio.run(start_server(config))

        # run generate
        request_model = config.actor_rollout_ref.model.path
        if config.actor_rollout_ref.model.get("lora_adapter_path"):
            request_model = ROLLING_LORA_MODEL_NAME
        gen_results = asyncio.run(
            generate(
                server_addresses,
                request_model,
                n_samples,
                sampling_params,
                chat_numpy,
                deadline_epoch_seconds=generation_deadline,
            )
        )
        for server_handle in server_handles:
            ray.kill(server_handle, no_restart=True)
        ray.shutdown()

        # reshape results into a numpy array
        import itertools

        results = list(itertools.chain.from_iterable(gen_results))
        results = np.array(results, dtype=object)
        results = np.reshape(results, (-1, n_samples))
        complete_rows = np.array(
            [all(result is not None for result in row) for row in results],
            dtype=bool,
        )
        if generation_deadline is None:
            assert complete_rows.all(), "Generation completed without a deadline but some responses are missing."
        else:
            complete_count = int(complete_rows.sum())
            print(
                f"Generation deadline result: keeping {complete_count}/{len(dataset)} prompts "
                f"with all {n_samples} responses complete."
            )
            if complete_count == 0:
                raise RuntimeError(
                    "Generation deadline expired before any prompt completed all "
                    f"{n_samples} responses."
                )
            dataset = dataset.loc[complete_rows].reset_index(drop=True)
            results = results[complete_rows]

        # extract content from results
        def response_text(result) -> str:
            choice = result.choices[0]
            if hasattr(choice, "text"):
                return choice.text or ""
            return choice.message.content or ""

        results = np.array([response_text(result) for result in results.flat], dtype=object)
        results = np.reshape(results, (-1, n_samples))

        assert results.shape == (len(dataset), n_samples)

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
