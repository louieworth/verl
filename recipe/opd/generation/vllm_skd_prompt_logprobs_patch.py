#!/usr/bin/env python3
"""Repo-local vLLM patch for SKD prompt-logprobs verification.

SKD only needs teacher prompt logprobs for the draft proposal suffix.
Vanilla vLLM materializes prompt logprobs for the entire prompt, then
Pythonizes every position in the engine process. For long OPD rollouts this
dominates runtime. This patch keeps vanilla behavior unless
VLLM_SKD_PROMPT_LOGPROBS_TAIL is set to a positive integer.
"""

from __future__ import annotations

import os
from typing import Any

_APPLIED = False
_ORIGINAL = None


def _tail_size() -> int:
    try:
        return int(os.environ.get("VLLM_SKD_PROMPT_LOGPROBS_TAIL", "0") or 0)
    except ValueError:
        return 0


def apply_patch() -> bool:
    global _APPLIED, _ORIGINAL
    if _APPLIED:
        return True

    tail = _tail_size()
    if tail <= 0:
        return False

    import torch
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _ORIGINAL = GPUModelRunner._get_prompt_logprobs_dict

    def _skd_tail_get_prompt_logprobs_dict(
        self: Any,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        tail = _tail_size()
        if tail <= 0 or not num_prompt_logprobs_dict:
            return _ORIGINAL(self, hidden_states, num_scheduled_tokens)

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}
        completed_prefill_reqs: list[str] = []

        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                continue

            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            tail_rows = min(tail, max(0, num_prompt_tokens - 1))
            if tail_rows <= 0:
                continue

            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    tail_rows, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                num_logits = num_tokens
            else:
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                continue

            # Vanilla row r corresponds to target prompt token r + 1.
            # Keep only target token positions in the final `tail_rows` prompt
            # tokens, which are exactly the SKD draft proposal suffix.
            tail_target_start = num_prompt_tokens - tail_rows
            row_start = max(start_idx, tail_target_start - 1)
            row_end = min(start_idx + num_logits, num_prompt_tokens - 1)
            if row_end <= row_start:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            local_start = row_start - start_idx
            local_end = row_end - start_idx
            prompt_hidden_states = hidden_states[
                offset + local_start : offset + local_end
            ]

            logits = self.model.compute_logits(prompt_hidden_states)
            tgt_token_ids = prompt_token_ids[row_start + 1 : row_end + 1]
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            dst_start = row_start - (tail_target_start - 1)
            dst_end = dst_start + (row_end - row_start)
            chunk_slice = slice(dst_start, dst_end)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(
                logprobs, non_blocking=True
            )
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    GPUModelRunner._get_prompt_logprobs_dict = _skd_tail_get_prompt_logprobs_dict
    _APPLIED = True
    return True
