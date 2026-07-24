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

import asyncio
import time

import numpy as np

from verl.trainer import main_generation_server


class FakeProgressBar:
    def __init__(self):
        self.updates = 0

    def update(self, value: int):
        self.updates += value


class FakeTqdm:
    def __init__(self, total: int, desc: str, dynamic_ncols: bool):
        self.total = total
        self.desc = desc
        self.dynamic_ncols = dynamic_ncols
        self.updates = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, value: int):
        self.updates += value


def test_generate_per_replica_preserves_request_order_and_updates_progress(monkeypatch):
    async def fake_submit_request(server_address, **chat_complete_request):
        content = chat_complete_request["messages"][0]["content"]
        await asyncio.sleep(0.05 - (int(content) * 0.01))
        return f"{server_address}:{content}"

    monkeypatch.setattr(main_generation_server, "submit_request", fake_submit_request)
    progress_bar = FakeProgressBar()

    results = asyncio.run(
        main_generation_server.generate_per_replica(
            server_address="server-0",
            model_path="dummy-model",
            n_samples=1,
            sampling_params={"temperature": 0.6},
            chat_lst=[
                [{"role": "user", "content": "0"}],
                [{"role": "user", "content": "1"}],
                [{"role": "user", "content": "2"}],
                [{"role": "user", "content": "3"}],
            ],
            progress_bar=progress_bar,
        )
    )

    assert results == [
        "server-0:0",
        "server-0:1",
        "server-0:2",
        "server-0:3",
    ]
    assert progress_bar.updates == 4


def test_generate_uses_shared_progress_bar(monkeypatch):
    progress_bars = []

    def fake_tqdm(total: int, desc: str, dynamic_ncols: bool):
        progress_bar = FakeTqdm(total=total, desc=desc, dynamic_ncols=dynamic_ncols)
        progress_bars.append(progress_bar)
        return progress_bar

    async def fake_generate_per_replica(
        server_address,
        model_path: str,
        n_samples: int,
        sampling_params: dict,
        chat_lst: list,
        progress_bar=None,
    ):
        if progress_bar is not None:
            progress_bar.update(len(chat_lst) * n_samples)
        return [f"{server_address}:{len(chat_lst)}"]

    monkeypatch.setattr(main_generation_server, "tqdm", fake_tqdm)
    monkeypatch.setattr(main_generation_server, "generate_per_replica", fake_generate_per_replica)

    results = asyncio.run(
        main_generation_server.generate(
            server_addresses=["server-0", "server-1"],
            model_path="dummy-model",
            n_samples=2,
            sampling_params={"temperature": 0.6},
            chat_numpy=np.array(["a", "b", "c"], dtype=object),
        )
    )

    assert results == [["server-0:2"], ["server-1:1"]]
    assert len(progress_bars) == 1
    assert progress_bars[0].total == 6
    assert progress_bars[0].desc == "Generating responses"
    assert progress_bars[0].dynamic_ncols is True
    assert progress_bars[0].updates == 6


def test_generate_per_replica_deadline_keeps_completed_requests(monkeypatch):
    async def fake_submit_request(server_address, **chat_complete_request):
        content = chat_complete_request["messages"][0]["content"]
        await asyncio.sleep(0.01 if content == "fast" else 0.5)
        return f"{server_address}:{content}"

    monkeypatch.setattr(main_generation_server, "submit_request", fake_submit_request)
    results = asyncio.run(
        main_generation_server.generate_per_replica(
            server_address="server-0",
            model_path="dummy-model",
            n_samples=2,
            sampling_params={"temperature": 0.6},
            chat_lst=[
                [{"role": "user", "content": "fast"}],
                [{"role": "user", "content": "slow"}],
            ],
            max_concurrency=4,
            deadline_epoch_seconds=time.time() + 0.1,
        )
    )

    assert results[:2] == ["server-0:fast", "server-0:fast"]
    assert results[2:] == [None, None]
