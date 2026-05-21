# Copyright 2025 Individual Contributor: furunding
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
import queue
import random
import threading
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime

import torch
import zmq
from codetiming import Timer

try:
    from .utils import deserialize, serialize
except ImportError:
    from utils import deserialize, serialize

DEBUG = False


def check_if_invalid(topk_logps, inputs):
    is_valid = True
    reason = ""
    for x in topk_logps:
        if x.isnan().any():
            is_valid = False
            reason = "nan"
            break
        elif x.isinf().any():
            is_valid = False
            reason = "inf"
            break
        elif (x == 0).any():
            is_valid = False
            reason = "zero"
            break
    if not is_valid:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.tolist()
        with open("teacher_debug.log", "a") as f:
            f.write("{}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write(f"{reason}\n")
            f.write(f"{str(inputs)}\n")


class TeacherClient:
    def __init__(
        self,
        server_ip,
        server_port,
        num_microbatches=1,
        max_tokens=1,
        n_server_workers=1,
        temperature=1,
        only_response=False,
        max_seq_len=None,
        recv_timeout_ms=None,
        request_batch_size=None,
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.num_microbatches = num_microbatches
        self.n_server_workers = n_server_workers
        self.max_tokens = max_tokens
        self.task_queue = queue.Queue()
        self.mutex = threading.Lock() if n_server_workers > 1 else nullcontext()
        self.context = zmq.Context()
        self.temperature = temperature
        self.only_response = only_response
        self.max_seq_len = max_seq_len
        if request_batch_size in (None, "", "null", "None"):
            self.request_batch_size = None
        else:
            self.request_batch_size = max(1, int(request_batch_size))
        if recv_timeout_ms is None:
            recv_timeout_ms = os.environ.get("TEACHER_CLIENT_RCVTIMEO_MS", "3600000")
        self.recv_timeout_ms = int(recv_timeout_ms)
        self._run()

    def bg_task(self):
        def make_socket():
            socket = self.context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
            if hasattr(zmq, "REQ_RELAXED"):
                socket.setsockopt(zmq.REQ_RELAXED, 1)
            if hasattr(zmq, "REQ_CORRELATE"):
                socket.setsockopt(zmq.REQ_CORRELATE, 1)
            socket.connect(f"tcp://{self.server_ip}:{self.server_port}")
            return socket

        while True:
            futures = []
            inputs = []
            batch = []
            socket = None
            try:
                request_options = []
                with self.mutex:
                    for _ in range(self.num_microbatches):
                        task = self.task_queue.get()
                        if len(task) == 2:
                            future, data = task
                            options = {}
                        else:
                            future, data, options = task
                        if DEBUG:
                            inputs.append(data)
                        futures.append(future)
                        request_options.append(options)
                        batch.extend(data.tolist() if isinstance(data, torch.Tensor) else data)

                if request_options:
                    first_options = {k: v for k, v in request_options[0].items() if k != "logprob_row_indices"}
                    for options in request_options:
                        comparable = {k: v for k, v in options.items() if k != "logprob_row_indices"}
                        if comparable != first_options:
                            raise RuntimeError("TeacherClient batched requests must use identical request options")
                    has_row_indices = ["logprob_row_indices" in options for options in request_options]
                    if any(has_row_indices) and not all(has_row_indices):
                        raise RuntimeError("TeacherClient batched requests must all provide logprob_row_indices or none")
                    all_logprob_row_indices = []
                    if all(has_row_indices):
                        for options in request_options:
                            all_logprob_row_indices.extend(options["logprob_row_indices"])
                    else:
                        all_logprob_row_indices = None
                else:
                    first_options = {}
                    all_logprob_row_indices = None

                max_tokens_opt = first_options.get("max_tokens", self.max_tokens)
                temperature = first_options.get("temperature", self.temperature)
                only_response = first_options.get("only_response", self.only_response)

                required = ("responses", "teacher_topk_logprobs", "teacher_topk_indices")
                response = {k: [] for k in required}
                request_batch_size = self.request_batch_size or max(1, len(batch))
                for start in range(0, len(batch), request_batch_size):
                    sub_batch = batch[start : start + request_batch_size]
                    if self.max_seq_len:
                        max_tokens = [min(max_tokens_opt, self.max_seq_len - len(prompt)) for prompt in sub_batch]
                        request = {"prompt_token_ids": sub_batch, "max_tokens": max_tokens}
                    else:
                        request = {"prompt_token_ids": sub_batch, "max_tokens": max_tokens_opt}
                    if all_logprob_row_indices is not None:
                        request["logprob_row_indices"] = all_logprob_row_indices[start : start + request_batch_size]
                    if temperature:
                        request["temperature"] = temperature
                    if only_response:
                        request["only_response"] = True

                    socket = make_socket()
                    socket.send(serialize(request))
                    raw = socket.recv()
                    sub_response = deserialize(raw)
                    socket.close(0)
                    socket = None

                    if isinstance(sub_response, dict) and sub_response.get("status") == "error":
                        reason = sub_response.get("reason", "unknown")
                        raise RuntimeError(f"Teacher error: {reason}")

                    for k in required:
                        if k not in sub_response:
                            raise RuntimeError(f"Invalid response: missing key '{k}'")
                        response[k].extend(sub_response[k])

                total = len(response["teacher_topk_logprobs"])
                if self.num_microbatches <= 0 or total % self.num_microbatches != 0:
                    raise RuntimeError(f"Size mismatch: total={total}, num_microbatches={self.num_microbatches}")

                mbs = total // self.num_microbatches
                for i, future in enumerate(futures):
                    s, e = i * mbs, (i + 1) * mbs
                    responses = response["responses"][s:e]
                    teacher_topk_logps = response["teacher_topk_logprobs"][s:e]
                    if DEBUG:
                        check_if_invalid(teacher_topk_logps, inputs[i])
                    teacher_topk_indices = response["teacher_topk_indices"][s:e]
                    future.set_result((responses, teacher_topk_logps, teacher_topk_indices))

            except zmq.Again:
                err = TimeoutError(f"Timeout waiting for server {self.server_ip}:{self.server_port}")
                for f in futures:
                    f.set_exception(err)
                if socket is not None:
                    socket.close(0)
                continue
            except Exception as e:
                for f in futures:
                    try:
                        f.set_exception(e)
                    except Exception:
                        pass
                if socket is not None:
                    socket.close(0)
                continue

    def _run(self):
        for _ in range(self.n_server_workers):
            threading.Thread(target=self.bg_task, daemon=True).start()

    def submit(self, data, **request_options):
        future = Future()
        self.task_queue.put((future, data, request_options))
        return future

    def __del__(self):
        self.context.destroy()


if __name__ == "__main__":
    gbs = 128
    n_gps = 1
    mbs = 2
    seq_len = 4096

    prompt_lens = (n_gps * gbs) * [seq_len]

    tc = TeacherClient(
        server_ip="127.0.0.1", server_port=15555, num_microbatches=gbs // mbs, n_server_workers=1, only_response=False
    )

    prompt_token_ids = []

    for pl in prompt_lens:
        prompt_token_ids.append([random.randint(1, 99999) for j in range(pl)])

    with Timer(name="get_topk_logprobs", initial_text=True):
        futures = []
        for i in range(0, n_gps * gbs, mbs):
            futures.append(tc.submit(prompt_token_ids[i : i + mbs]))

        for future in futures:
            responses, teacher_topk_logprobs, teacher_topk_indices = future.result()

            print(len(teacher_topk_logprobs), len(teacher_topk_indices))

            assert len(responses) == mbs
            assert len(teacher_topk_logprobs) == mbs
            assert len(teacher_topk_indices) == mbs

            assert all(x.shape == y.shape for x, y in zip(teacher_topk_logprobs, teacher_topk_indices, strict=False))
            out_lens = [x.shape[0] for x in teacher_topk_logprobs]
            out_dims = [x.shape[1] for x in teacher_topk_logprobs]
            assert all(out_len == seq_len for out_len in out_lens)
            assert all(out_dim == 256 for out_dim in out_dims)
            assert all(x.dtype == torch.float32 for x in teacher_topk_logprobs), [
                x.dtype for x in teacher_topk_logprobs
            ]
            assert all(x.dtype == torch.int32 for x in teacher_topk_indices)
            assert all(x.dtype == torch.int32 for x in responses)
