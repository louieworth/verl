# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os

from recipe.opd import eval_utils
from verl.trainer.generation_server_env import (
    TORCH_LAUNCH_ENV_KEYS,
    build_generation_server_runtime_env,
)


def test_launch_generation_server_sanitizes_torchrun_env(monkeypatch):
    inherited_env = {
        "MASTER_ADDR": "10.8.30.110",
        "MASTER_PORT": "55855",
        "RANK": "0",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "4",
        "DIST_INIT_METHOD": "tcp://10.8.30.110:55855",
        "TORCHELASTIC_USE_AGENT_STORE": "True",
        "TORCHELASTIC_RUN_ID": "job-123",
    }
    for key, value in inherited_env.items():
        monkeypatch.setenv(key, value)

    observed = {}

    def fake_ray_init(*, runtime_env):
        observed["ray_init_env"] = {key: os.environ.get(key) for key in TORCH_LAUNCH_ENV_KEYS}
        observed["runtime_env"] = runtime_env

    async def fake_start_server(config):
        observed["start_server_env"] = {key: os.environ.get(key) for key in TORCH_LAUNCH_ENV_KEYS}
        observed["config"] = config
        return ["server-handle"], ["127.0.0.1:8000"]

    monkeypatch.setattr(eval_utils.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(eval_utils.ray, "init", fake_ray_init)
    monkeypatch.setattr(eval_utils, "start_server", fake_start_server)

    server_handles, server_addresses = eval_utils.launch_generation_server(
        model_path="/tmp/model",
        tokenizer_path=None,
        temperature=0.6,
        top_p=0.95,
        pass_k=1,
        nnodes=1,
        n_gpus_per_node=1,
        tensor_model_parallel_size=1,
    )

    assert server_handles == ["server-handle"]
    assert server_addresses == ["127.0.0.1:8000"]
    assert observed["runtime_env"] == build_generation_server_runtime_env()
    assert all(value is None for value in observed["ray_init_env"].values())
    assert all(value is None for value in observed["start_server_env"].values())

    for key, value in inherited_env.items():
        assert os.environ.get(key) == value
