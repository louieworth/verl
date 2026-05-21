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

import argparse
import faulthandler
import functools
import gc
import signal
import sys

import torch
import zmq
from codetiming import Timer
from utils import deserialize, serialize

# Always-on segfault tracebacks and SIGUSR1 thread dump for debugging.
try:
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-addr", type=str, default="localhost:15556")
    parser.add_argument("--backend", type=str, default="vllm")
    parser.add_argument("--seq-len", type=int, default=3840)
    parser.add_argument("--n-logprobs", type=str, default="full_vocab")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--enable-prefix-caching", type=str, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    args = parser.parse_args()

    def parse_optional_bool(value):
        if value is None or str(value).strip() == "":
            return None
        if str(value).lower() in {"1", "true", "yes", "y"}:
            return True
        if str(value).lower() in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")

    if args.backend == "vllm":
        from vllm_engine import VLLMEngine

        engine = VLLMEngine(
            args.ckpt_path,
            args.n_logprobs,
            args.tp_size,
            max_model_len=args.seq_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=parse_optional_bool(args.enable_prefix_caching),
            enforce_eager=args.enforce_eager,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}.")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # socket.bind(f"tcp://*:{port}")
    socket.connect(f"tcp://{args.proxy_addr}")

    print("worker started...", flush=True)

    # TODO: 新增prefix_cache_hit监控

    while True:
        message = socket.recv()
        try:
            with Timer(name="deserialize", initial_text=True, logger=functools.partial(print, flush=True)):
                request = deserialize(message)
        except Exception as e:
            print("[Server Error] Deserialize failed:", str(e), flush=True)
            socket.send(serialize({"status": "error", "reason": f"Deserialize failed: {e}"}))
            continue
        if isinstance(request, dict) and "prompt_token_ids" in request:
            prompt_token_ids = request["prompt_token_ids"]
            temperature = request.get("temperature", 0.8)
            max_tokens = request.get("max_tokens", 1)
            only_response = request.get("only_response", False)
            logprob_row_indices = request.get("logprob_row_indices", None)
            if isinstance(prompt_token_ids, torch.Tensor):
                prompt_token_ids = prompt_token_ids.tolist()
            with Timer(name="get_prompt_topk_logprobs", initial_text=True, logger=functools.partial(print, flush=True)):
                ### try and sendback error
                try:
                    responses, logps, indices = engine.get_topk_logprobs(
                        prompt_token_ids,
                        temperature,
                        max_new_tokens=max_tokens,
                        only_response=only_response,
                        logprob_row_indices=logprob_row_indices,
                    )
                except Exception as e:
                    print("[Server Error] Exception occurred during generation:", str(e))
                    socket.send(serialize({"status": "error", "reason": f"Generate failed: {str(e)}"}))
                    continue
            with Timer(name="serialize", initial_text=True, logger=functools.partial(print, flush=True)):
                message = serialize(
                    {
                        "status": "ok",
                        "teacher_topk_logprobs": logps,
                        "teacher_topk_indices": indices,
                        "responses": responses,
                    }
                )
            with Timer(name="send", initial_text=True, logger=functools.partial(print, flush=True)):
                socket.send(message)

            # Drop refs to the large fp32 [seq_len, vocab_size] tensors and
            # serialized buffer before waiting for the next request. Without
            # this, Python keeps them alive in the loop frame until rebound,
            # and CUDA caching allocator never gets a chance to return memory.
            del message, responses, logps, indices
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        else:
            socket.send(serialize({"status": "error", "reason": "invalid request format."}))


if __name__ == "__main__":
    main()
