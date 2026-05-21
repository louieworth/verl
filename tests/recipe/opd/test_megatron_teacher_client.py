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

import socket
import threading

import pytest
import torch
import zmq

from recipe.gkd.megatron.teacher.client import TeacherClient
from recipe.gkd.megatron.teacher.utils import deserialize, serialize


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_teacher_client_recreates_req_socket_after_timeout(monkeypatch):
    monkeypatch.setenv("TEACHER_CLIENT_RCVTIMEO_MS", "100")
    port = _get_free_port()
    context = zmq.Context()
    ready = threading.Event()
    received_requests = []

    def server():
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 3000)
        socket.bind(f"tcp://127.0.0.1:{port}")
        ready.set()

        first = socket.recv_multipart()
        received_requests.append(deserialize(first[-1]))

        second = socket.recv_multipart()
        received_requests.append(deserialize(second[-1]))
        response = {
            "responses": [torch.tensor([3, 4], dtype=torch.int32)],
            "teacher_topk_logprobs": [torch.zeros((2, 3), dtype=torch.float32)],
            "teacher_topk_indices": [torch.ones((2, 3), dtype=torch.int32)],
        }
        socket.send_multipart([*second[:-1], serialize(response)])
        socket.close(0)

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    assert ready.wait(timeout=3)

    client = TeacherClient("127.0.0.1", port, num_microbatches=1, n_server_workers=1)
    try:
        with pytest.raises(TimeoutError):
            client.submit([[1, 2]]).result(timeout=3)

        responses, logps, indices = client.submit([[3, 4]]).result(timeout=3)
        assert responses[0].tolist() == [3, 4]
        assert logps[0].shape == (2, 3)
        assert indices[0].shape == (2, 3)
        assert len(received_requests) == 2
    finally:
        client.context.destroy(linger=0)
        context.destroy(linger=0)


def test_teacher_client_splits_request_batch_with_row_indices(monkeypatch):
    monkeypatch.setenv("TEACHER_CLIENT_RCVTIMEO_MS", "1000")
    port = _get_free_port()
    context = zmq.Context()
    ready = threading.Event()
    received_requests = []

    def server():
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 3000)
        socket.bind(f"tcp://127.0.0.1:{port}")
        ready.set()

        for response_id in (0, 1):
            request = socket.recv_multipart()
            payload = deserialize(request[-1])
            received_requests.append(payload)
            response = {
                "responses": [torch.tensor([response_id], dtype=torch.int32)],
                "teacher_topk_logprobs": [torch.full((1, 3), float(response_id), dtype=torch.float32)],
                "teacher_topk_indices": [None],
            }
            socket.send_multipart([*request[:-1], serialize(response)])
        socket.close(0)

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    assert ready.wait(timeout=3)

    client = TeacherClient("127.0.0.1", port, num_microbatches=1, n_server_workers=1, request_batch_size=1)
    try:
        responses, logps, indices = client.submit(
            [[10, 11], [20, 21, 22]],
            logprob_row_indices=[[1], [0, 2]],
        ).result(timeout=3)

        assert [x.tolist() for x in responses] == [[0], [1]]
        assert [x.tolist() for x in logps] == [[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]]
        assert indices == [None, None]
        assert [request["prompt_token_ids"] for request in received_requests] == [[[10, 11]], [[20, 21, 22]]]
        assert [request["logprob_row_indices"] for request in received_requests] == [[[1]], [[0, 2]]]
    finally:
        client.context.destroy(linger=0)
        context.destroy(linger=0)
