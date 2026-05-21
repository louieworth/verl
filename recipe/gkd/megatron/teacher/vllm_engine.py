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
import os
import random
from typing import NamedTuple

import numpy as np
import torch
from codetiming import Timer
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

# from vllm.v1.outputs import LogprobsTensors
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsTensors as VllmLogprobsTensors
from vllm.v1.sample.sampler import Sampler


def _patch_transformers_tokenizer_compat():
    """Bridge vLLM 0.11's tokenizer cache with transformers 5 tokenizers.

    vLLM's get_cached_tokenizer reads all_special_tokens_extended, but the
    TokenizersBackend classes in this environment expose only
    all_special_tokens. This local worker-process patch keeps the official
    vLLM path intact and only supplies the missing read-only property.
    """
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)


def _update_prompt_logprobs(
    self,
    prompt_logprobs_tensors,
) -> None:
    """Update with prompt logprobs from EngineCore.

    Args:
        prompt_logprobs_tensors: tuple containing the prompt logprobs
                                tensors.

    """

    # Prompt logprobs are enabled.
    assert self.num_prompt_logprobs is not None
    assert self.prompt_logprobs is not None

    self.prompt_logprobs.append(prompt_logprobs_tensors)


def _update_sample_logprobs(self, logprobs_lists) -> None:
    """Update with sample logprobs from EngineCore.

    Outer lists are only of len > 1 if EngineCore made
    >1 tokens in prior step (e.g. in spec decoding).

    Args:
        logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

    """

    assert self.num_logprobs is not None
    assert self.logprobs is not None
    assert self.cumulative_logprob is not None

    # token_ids_lst, logprobs_lst, ranks_lst = logprobs_lists

    # for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst,
    #                                         token_ids_lst):

    #     # Detokenize (non-incrementally).
    #     decoded_tokens = NONES if self.tokenizer is None else (
    #         convert_ids_list_to_tokens(self.tokenizer, token_ids))

    #     # Sampler puts the sampled logprob in first.
    #     sampled_token_logprob = logprobs[0]
    #     self.cumulative_logprob += sampled_token_logprob

    #     # Update with the Logprob dictionary for this pos.
    #     self.logprobs.append(
    #         self._make_logprob_dict(
    #             logprobs,
    #             token_ids,
    #             decoded_tokens,
    #             rank,
    #             self.num_logprobs,
    #         ))
    self.logprobs.append(logprobs_lists)


LogprobsProcessor._update_prompt_logprobs = _update_prompt_logprobs
LogprobsProcessor._update_sample_logprobs = _update_sample_logprobs


def _patch_vllm_dense_full_vocab_sample_logprobs():
    """Return full-vocab generated-token logprobs in vocab-column order.

    vLLM's normal sampled-token path returns ``[sampled_token, topk_sorted...]``.
    For OPD full-vocab KL we need dense vocab-order columns. ``num_logprobs=-1``
    is our full-vocab sentinel, so bypass topk/gather and keep logprobs dense.
    """

    original = Sampler.gather_logprobs
    if getattr(original, "_opd_dense_full_vocab", False):
        return

    def gather_logprobs(logprobs: torch.Tensor, num_logprobs: int, token_ids: torch.Tensor) -> VllmLogprobsTensors:
        if int(num_logprobs) < 0 or int(num_logprobs) >= int(logprobs.size(-1)):
            empty_ids = torch.empty((logprobs.size(0), 0), dtype=torch.int32, device=logprobs.device)
            ranks = torch.zeros((logprobs.size(0),), dtype=torch.int32, device=logprobs.device)
            return VllmLogprobsTensors(empty_ids, logprobs, ranks)
        return original(logprobs, num_logprobs, token_ids)

    gather_logprobs._opd_dense_full_vocab = True
    Sampler.gather_logprobs = staticmethod(gather_logprobs)


_patch_vllm_dense_full_vocab_sample_logprobs()


class LogprobsTensors(NamedTuple):
    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs]
    selected_token_ranks: torch.Tensor

    def tolists(self):
        return LogprobsTensors(
            logprob_token_ids=self.logprob_token_ids.cpu(),
            logprobs=self.logprobs.cpu(),
            selected_token_ranks=self.selected_token_ranks.cpu(),
        )

    @staticmethod
    def empty_cpu(num_positions: int, num_tokens_per_position: int) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty((num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu")
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(num_positions, dtype=torch.int32, device="cpu")
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )

    def slice(self, start: int, end: int):
        return LogprobsTensors(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.selected_token_ranks[start:end],
        )


# outputs.LogprobsTensors = LogprobsTensors
# def tolists(self):
#     return self


# LogprobsTensors.tolists = tolists
# setattr(LogprobsTensors, "slice", slice)


FULL_VOCAB_LOGPROBS_VALUES = {"-", "-1", "all", "full", "full_vocab", "vocab"}


def _resolve_vocab_size(ckpt_path):
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    resolved = getattr(config, "vocab_size", None)
    if resolved is None:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        resolved = len(tokenizer)
    return int(resolved)


def is_full_vocab_logprobs(n_logprobs) -> bool:
    if isinstance(n_logprobs, int):
        return n_logprobs < 0
    return str(n_logprobs).strip().lower() in FULL_VOCAB_LOGPROBS_VALUES


def resolve_n_logprobs(n_logprobs, ckpt_path):
    if isinstance(n_logprobs, int):
        resolved = n_logprobs
    else:
        value = str(n_logprobs).strip().lower()
        if value in FULL_VOCAB_LOGPROBS_VALUES:
            resolved = _resolve_vocab_size(ckpt_path)
        else:
            resolved = int(value)

    if resolved < 0:
        raise ValueError(f"n_logprobs must be non-negative or one of {sorted(FULL_VOCAB_LOGPROBS_VALUES)}")
    return resolved


class VLLMEngine:
    def __init__(
        self,
        ckpt_path,
        n_logprobs=0,
        tp_size=1,
        max_model_len=None,
        max_num_batched_tokens=None,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=None,
        enforce_eager=False,
        disable_custom_all_reduce=False,
    ):
        self.full_vocab_logprobs = is_full_vocab_logprobs(n_logprobs)
        self.vocab_size = _resolve_vocab_size(ckpt_path) if self.full_vocab_logprobs else None
        self.n_logprobs = -1 if self.full_vocab_logprobs else resolve_n_logprobs(n_logprobs, ckpt_path)
        llm_max_logprobs = self.vocab_size if self.full_vocab_logprobs else self.n_logprobs
        self.normalize_full_vocab_logprobs = str(
            os.environ.get("OPD_NORMALIZE_TEACHER_FULL_VOCAB", "false")
        ).lower() in {"1", "true", "yes", "y"}
        print(f"Using n_logprobs={self.n_logprobs}", flush=True)
        if self.full_vocab_logprobs:
            print(f"Using raw full-vocab logprobs with vocab_size={self.vocab_size}", flush=True)
            if self.normalize_full_vocab_logprobs:
                print("Normalizing full-vocab logprobs before returning them", flush=True)
        print(f"Using gpu_memory_utilization={gpu_memory_utilization}", flush=True)
        _patch_transformers_tokenizer_compat()
        llm_kwargs = {}
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if max_num_batched_tokens is not None and max_num_batched_tokens > 0:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
            llm_kwargs["enable_chunked_prefill"] = True
            print(f"Using max_num_batched_tokens={max_num_batched_tokens}", flush=True)
        else:
            llm_kwargs["enable_chunked_prefill"] = False
        if enable_prefix_caching is not None:
            llm_kwargs["enable_prefix_caching"] = enable_prefix_caching
            print(f"Using enable_prefix_caching={enable_prefix_caching}", flush=True)
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
            print("Using enforce_eager=True", flush=True)
        if disable_custom_all_reduce:
            llm_kwargs["disable_custom_all_reduce"] = True
            print("Using disable_custom_all_reduce=True", flush=True)
        # self.llm = LLM(ckpt_path, tensor_parallel_size=tp_size, trust_remote_code=True,
        #                enable_chunked_prefill=False, distributed_executor_backend="ray",
        #                max_logprobs=n_logprobs, gpu_memory_utilization=0.7)
        self.llm = LLM(
            ckpt_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            max_logprobs=llm_max_logprobs,
            gpu_memory_utilization=gpu_memory_utilization,
            **llm_kwargs,
        )

    def get_topk_logprobs(
        self,
        prompt_token_ids,
        temperature=0.8,
        max_new_tokens=1,
        only_response=False,
        logprob_row_indices=None,
    ):
        def extract_prompt_logprobs(prompt_logprobs):
            if isinstance(prompt_logprobs, (list, tuple)) and len(prompt_logprobs) > 1:
                return prompt_logprobs[1]
            return prompt_logprobs

        def align_full_vocab_logprob_dim(tensor):
            if not self.full_vocab_logprobs:
                return tensor
            if tensor.size(-1) < self.vocab_size:
                raise RuntimeError(
                    "vLLM returned fewer full-vocab logprob columns than the tokenizer vocab: "
                    f"logprob_dim={tensor.size(-1)}, vocab_size={self.vocab_size}"
                )
            if tensor.size(-1) > self.vocab_size:
                tensor = tensor[..., : self.vocab_size]
            return tensor

        def as_2d_tensor(value):
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.from_numpy(np.asarray(value))
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor

        def select_matrix_rows(value, rows=None):
            tensor = as_2d_tensor(value)
            if rows is not None:
                row_idx = torch.as_tensor(rows, dtype=torch.long, device=tensor.device)
                tensor = tensor.index_select(0, row_idx)
            return align_full_vocab_logprob_dim(tensor)

        def token_ids_are_empty(token_ids) -> bool:
            if token_ids is None:
                return True
            if isinstance(token_ids, torch.Tensor):
                return token_ids.numel() == 0
            return np.asarray(token_ids).size == 0

        def chunk_num_rows(value) -> int:
            if isinstance(value, torch.Tensor):
                return 1 if value.ndim == 1 else int(value.shape[0])
            array = np.asarray(value)
            return 1 if array.ndim == 1 else int(array.shape[0])

        def dense_response_chunks_to_tensor(logprobs, rows=None):
            row_idx = np.asarray(rows, dtype=np.int64) if rows is not None else None
            chunks = []
            cursor = 0
            for item in logprobs:
                values = item.logprobs
                n_rows = chunk_num_rows(values)
                local_rows = None
                if row_idx is not None:
                    left = int(np.searchsorted(row_idx, cursor, side="left"))
                    right = int(np.searchsorted(row_idx, cursor + n_rows, side="left"))
                    if left == right:
                        cursor += n_rows
                        continue
                    local_rows = row_idx[left:right] - cursor
                if isinstance(values, torch.Tensor):
                    tensor = values
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                    if local_rows is not None:
                        tensor = tensor.index_select(
                            0,
                            torch.as_tensor(local_rows, dtype=torch.long, device=tensor.device),
                        )
                    chunks.append(tensor)
                else:
                    array = np.asarray(values)
                    if array.ndim == 1:
                        array = array[None, :]
                    if local_rows is not None:
                        array = array[local_rows]
                    chunks.append(array)
                cursor += n_rows

            if not chunks:
                return torch.empty((0, self.vocab_size), dtype=torch.float32)
            if all(isinstance(chunk, torch.Tensor) for chunk in chunks):
                return align_full_vocab_logprob_dim(torch.cat(chunks, dim=0))
            if any(isinstance(chunk, torch.Tensor) for chunk in chunks):
                chunks = [chunk.detach().cpu().numpy() if isinstance(chunk, torch.Tensor) else chunk for chunk in chunks]
            return align_full_vocab_logprob_dim(torch.from_numpy(np.concatenate(chunks, axis=0)))

        def extract_response_logprobs(logprobs, rows=None):
            if self.full_vocab_logprobs and logprobs:
                first_token_ids = getattr(logprobs[0], "logprob_token_ids", None)
                if token_ids_are_empty(first_token_ids):
                    return dense_response_chunks_to_tensor(logprobs, rows)

            chunks = []
            for item in logprobs:
                tensor = as_2d_tensor(item.logprobs)
                token_ids = getattr(item, "logprob_token_ids", None)
                if self.full_vocab_logprobs and not token_ids_are_empty(token_ids):
                    token_ids = torch.as_tensor(token_ids, device=tensor.device)
                    if token_ids.ndim == 1:
                        token_ids = token_ids.unsqueeze(0)
                    # Fallback for an unpatched vLLM sampled-token path:
                    # [sampled_token, topk_sorted...] -> dense vocab order.
                    if token_ids.size(-1) == tensor.size(-1) and tensor.size(-1) > self.vocab_size:
                        token_ids = token_ids[:, 1:]
                        tensor = tensor[:, 1:]
                    dense = tensor.new_full((tensor.size(0), self.vocab_size), float("-inf"))
                    valid = (token_ids >= 0) & (token_ids < self.vocab_size)
                    dense.scatter_(1, token_ids.clamp(0, self.vocab_size - 1), tensor.masked_fill(~valid, float("-inf")))
                    tensor = dense
                tensor = align_full_vocab_logprob_dim(tensor)
                chunks.append(tensor)
            if not chunks:
                return torch.empty((0, self.vocab_size), dtype=torch.float32)
            tensor = torch.cat(chunks, dim=0)
            return slice_logprob_rows(tensor, rows)

        def slice_logprob_rows(tensor, rows):
            if rows is None:
                return tensor
            row_idx = torch.as_tensor(rows, dtype=torch.long, device=tensor.device)
            return tensor.index_select(0, row_idx)

        def normalize_full_vocab_logprobs(tensor):
            if not self.full_vocab_logprobs:
                return tensor
            tensor = align_full_vocab_logprob_dim(tensor)
            if not self.normalize_full_vocab_logprobs:
                return tensor
            tensor = tensor.float()
            return tensor - torch.logsumexp(tensor, dim=-1, keepdim=True)

        def make_sampling_params(i=None):
            return SamplingParams(
                temperature=temperature,
                top_p=0.95,
                detokenize=False,
                logprobs=self.n_logprobs,
                prompt_logprobs=None if only_response else self.n_logprobs,
                max_tokens=max_new_tokens[i] if (i is not None) else max_new_tokens,
            )

        if isinstance(max_new_tokens, list):
            assert len(prompt_token_ids) == len(max_new_tokens)
            sampling_params = [make_sampling_params(i) for i in range(len(max_new_tokens))]
        else:
            sampling_params = make_sampling_params()

        prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        responses, teacher_topk_logprobs, teacher_topk_indices = [], [], []
        for output_idx, output in enumerate(outputs):
            rows = logprob_row_indices[output_idx] if logprob_row_indices is not None else None
            responses.append(torch.tensor(output.outputs[0].token_ids, dtype=torch.int32))
            if self.full_vocab_logprobs:
                if only_response:
                    selected_logprobs = extract_response_logprobs(output.outputs[0].logprobs, rows).contiguous()
                else:
                    prompt_logprobs = extract_prompt_logprobs(output.prompt_logprobs)
                    prompt_n_rows = chunk_num_rows(prompt_logprobs.logprobs)
                    if rows is None:
                        prompt_topk_logprobs = select_matrix_rows(prompt_logprobs.logprobs).contiguous()
                        response_topk_logprobs = extract_response_logprobs(output.outputs[0].logprobs).contiguous()
                    else:
                        prompt_rows = [row for row in rows if row < prompt_n_rows]
                        response_rows = [row - prompt_n_rows for row in rows if row >= prompt_n_rows]
                        prompt_topk_logprobs = select_matrix_rows(prompt_logprobs.logprobs, prompt_rows).contiguous()
                        response_topk_logprobs = extract_response_logprobs(
                            output.outputs[0].logprobs, response_rows
                        ).contiguous()
                    if prompt_topk_logprobs.numel() == 0:
                        selected_logprobs = response_topk_logprobs
                    elif response_topk_logprobs.numel() == 0:
                        selected_logprobs = prompt_topk_logprobs
                    else:
                        selected_logprobs = torch.vstack([prompt_topk_logprobs, response_topk_logprobs])
                teacher_topk_logprobs.append(
                    normalize_full_vocab_logprobs(selected_logprobs).to(torch.bfloat16).contiguous()
                )
                teacher_topk_indices.append(None)
            elif self.n_logprobs > 0:
                # Per-token vLLM logprobs/indices arrive as a Python list of
                # 1-D numpy arrays (one per token). Constructing the 2-D
                # tensor naively with torch.tensor(list_of_ndarrays) hits the
                # well-known vLLM slow path (10s of seconds for 16k tokens ×
                # 152k vocab — was THE bottleneck, not GPU).
                # Fix: build a contiguous numpy array first with np.array(),
                # then a single from_numpy() / cast pass. ~10-50× faster.
                _lp_list = [x.logprobs[0] for x in output.outputs[0].logprobs]
                _idx_list = [x.logprob_token_ids[0] for x in output.outputs[0].logprobs]
                _lp_np = np.array(_lp_list, dtype=np.float32)  # [seq_len, top_k+1]
                _idx_np = np.array(_idx_list, dtype=np.int32)
                # bf16 transport: downcast logprobs (indices stay int32 —
                # vocab > 65535 so int16 won't fit). _fp32_loss_inputs()
                # upcasts back to fp32 inside the KL op for numerical safety.
                response_topk_logprobs = torch.from_numpy(_lp_np)[:, 1:].to(torch.bfloat16).contiguous()
                response_topk_indices = torch.from_numpy(_idx_np)[:, 1:].contiguous()
                if only_response:
                    teacher_topk_logprobs.append(slice_logprob_rows(response_topk_logprobs, rows))
                    teacher_topk_indices.append(slice_logprob_rows(response_topk_indices, rows))
                else:
                    prompt_topk_logprobs = output.prompt_logprobs[1].logprobs[:, 1:].to(torch.bfloat16)
                    prompt_topk_indices = output.prompt_logprobs[1].logprob_token_ids[:, 1:].to(torch.int32)
                    all_logprobs = torch.vstack([prompt_topk_logprobs, response_topk_logprobs])
                    all_indices = torch.vstack([prompt_topk_indices, response_topk_indices])
                    teacher_topk_logprobs.append(slice_logprob_rows(all_logprobs, rows))
                    teacher_topk_indices.append(slice_logprob_rows(all_indices, rows))

        return responses, teacher_topk_logprobs, teacher_topk_indices

    # def get_response_and_topk_logprobs(self, prompt_token_ids, max_tokens=64):
    #     sampling_params = SamplingParams(temperature=0.8, top_p=0.95, detokenize=False,
    #                                      logprobs=self.n_logprobs, max_tokens=max_tokens)

    #     outputs = self.llm.generate(prompt_token_ids=prompt_token_ids,
    #                                 sampling_params=sampling_params)

    #     student_topk_logprobs, student_topk_indices = [], []
    #     for output in outputs:
    #         student_topk_logprobs.append([])
    #         student_topk_indices.append([])
    #         for logprob_list in output.outputs[0].logprobs:
    #             student_topk_logprobs[-1].extend(logprob_list.logprobs)
    #             student_topk_indices[-1].extend(logprob_list.logprob_token_ids)

    #     return student_topk_logprobs, student_topk_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM logprob")
    parser.add_argument("model_dir", help="Model directory")
    parser.add_argument("--tp-size", type=int, default=1, help="TP size")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Test batch size")
    parser.add_argument("--seq-len", "-s", type=int, default=3840, help="Test sequence length")
    parser.add_argument("--n-logprobs", type=str, default="full_vocab", help="Use '-' for full vocab")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--token-file", "-t", type=str, help="Input token file")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_dir)
    print(f"Reading configs from {args.model_dir}: {config.vocab_size=}")

    prompt_token_ids = []
    if args.token_file:
        # Init input with tokenid file
        from get_batch import get_batch

        prompt_token_ids = get_batch()
    else:
        # Init input randomly
        prompt_lens = args.batch_size * [args.seq_len]
        for pl in prompt_lens:
            prompt_token_ids.append([random.randint(1, config.vocab_size - 1000) for j in range(pl)])

    engine = VLLMEngine(
        ckpt_path=args.model_dir,
        n_logprobs=args.n_logprobs,
        tp_size=args.tp_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )

    with Timer(name="get_topk_logprobs", initial_text=True):
        responses, teacher_topk_logprobs, teacher_topk_indices = engine.get_topk_logprobs(
            prompt_token_ids, temperature=0.7, max_new_tokens=1, only_response=True
        )
    # debug
    import ipdb

    ipdb.set_trace()
