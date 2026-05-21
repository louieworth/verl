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

"""
Utility functions for teacher model knowledge distillation.

Functions:
    get_teacher_knowledge: Retrieve teacher model's top-k predictions and log probabilities.
"""

import time
from types import SimpleNamespace

import numpy as np
import torch

from verl import DataProto

teacher_topk_logps_padded, teacher_topk_indices_padded = None, None


def _default_distill_loss_mask(batch: DataProto, attention_mask: torch.Tensor) -> torch.Tensor:
    if "distill_loss_mask" in batch.batch.keys():
        loss_mask = batch.batch["distill_loss_mask"].to(torch.bool).clone()
    else:
        response_length = batch.meta_info.get("response_length")
        if response_length is None and "responses" in batch.batch.keys():
            response_length = batch.batch["responses"].size(1)
        loss_mask = attention_mask.clone()
        if response_length is not None:
            loss_mask[:, : (-int(response_length) - 1)] = False

    loss_mask &= attention_mask
    if "is_padded_sample" in batch.batch.keys():
        loss_mask[batch.batch["is_padded_sample"].to(torch.bool)] = False
    return loss_mask


def _as_numpy_payload(tensor: torch.Tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy()
    return tensor.numpy()


def get_teacher_knowledge(batch: DataProto, teacher_client, n_server_workers=1, is_async=False):
    """
    Retrieve teacher model's top-k predictions and log probabilities for knowledge distillation.

    Args:
        batch (DataProto): Input batch containing input_ids and attention_mask
        teacher_client: Client for communicating with teacher model
        n_server_workers (int): Number of parallel workers for teacher model inference
        is_async (bool): Whether to use asynchronous processing

    Returns:
        If is_async=True: SimpleNamespace with get() method to process futures
        If is_async=False: Processed DataProto containing teacher knowledge

    Raises:
        RuntimeError: If teacher model request fails
    """

    input_ids = []
    attention_mask = batch.batch["attention_mask"].to(torch.bool)
    loss_mask = _default_distill_loss_mask(batch, attention_mask)
    loss_row_masks = []
    logprob_row_indices = []
    # response_length = batch.meta_info["response_length"]

    for ids, mask, row_loss_mask in zip(batch.batch["input_ids"], attention_mask, loss_mask, strict=False):
        input_ids.append(ids[mask].tolist())
        active_loss_mask = row_loss_mask[mask]
        loss_row_masks.append(active_loss_mask)
        logprob_row_indices.append(active_loss_mask.nonzero(as_tuple=False).flatten().tolist())

    all_teacher_topk_logps = []
    all_teacher_topk_indices = []

    batch_size = len(input_ids)
    n_chunks = max(1, min(n_server_workers, batch_size))
    micro_batch_size = (batch_size + n_chunks - 1) // n_chunks
    futures = []
    tik1 = time.time()
    tok1 = tik1

    def cb(future):
        nonlocal tok1
        tok1 = max(tok1, time.time())

    for i in range(0, batch_size, micro_batch_size):
        fut = teacher_client.submit(
            input_ids[i : i + micro_batch_size],
            logprob_row_indices=logprob_row_indices[i : i + micro_batch_size],
        )
        fut.add_done_callback(cb)
        futures.append(fut)

    def handle_futures():
        for future in futures:
            try:
                _, teacher_topk_logps, teacher_topk_indices = future.result()
            except Exception as e:
                raise RuntimeError(f"Teacher request failed: {e}") from e

            all_teacher_topk_logps.extend(teacher_topk_logps)
            all_teacher_topk_indices.extend(teacher_topk_indices)

        tik2 = time.time()
        # teacher_topk_logps = [x.to(params_dtype) for x in all_teacher_topk_logps]
        # teacher_topk_indices = [x.to(params_dtype) for x in all_teacher_topk_indices]
        teacher_topk_logps, teacher_topk_indices = all_teacher_topk_logps, all_teacher_topk_indices
        has_indices = bool(teacher_topk_indices) and teacher_topk_indices[0] is not None

        batch_size = attention_mask.size(0)
        real_seq_lens = torch.tensor([x.size(0) for x in teacher_topk_logps], dtype=torch.int32)
        teacher_loss_lens = torch.zeros(batch_size, dtype=torch.int32)
        logps_payload = np.empty((batch_size,), dtype=object)
        indices_payload = np.empty((batch_size,), dtype=object) if has_indices else None

        for i in range(batch_size):
            row_active_loss_mask = loss_row_masks[i]
            expected_loss_rows = int(row_active_loss_mask.sum().item())
            if teacher_topk_logps[i].size(0) == expected_loss_rows:
                selected_logps = teacher_topk_logps[i]
            elif row_active_loss_mask.numel() == teacher_topk_logps[i].size(0):
                selected_logps = teacher_topk_logps[i][row_active_loss_mask.to(teacher_topk_logps[i].device)]
            else:
                raise RuntimeError(
                    "Teacher logprob rows do not align with active input tokens: "
                    f"row={i}, active_tokens={row_active_loss_mask.numel()}, "
                    f"loss_tokens={expected_loss_rows}, teacher_rows={teacher_topk_logps[i].size(0)}"
                )
            teacher_loss_lens[i] = selected_logps.size(0)
            logps_payload[i] = _as_numpy_payload(selected_logps)
            if has_indices:
                if teacher_topk_indices[i].size(0) == expected_loss_rows:
                    selected_indices = teacher_topk_indices[i]
                elif row_active_loss_mask.numel() == teacher_topk_indices[i].size(0):
                    selected_indices = teacher_topk_indices[i][row_active_loss_mask.to(teacher_topk_indices[i].device)]
                else:
                    raise RuntimeError(
                        "Teacher index rows do not align with active input tokens: "
                        f"row={i}, active_tokens={row_active_loss_mask.numel()}, "
                        f"loss_tokens={expected_loss_rows}, teacher_rows={teacher_topk_indices[i].size(0)}"
                    )
                indices_payload[i] = _as_numpy_payload(selected_indices)

        output_batch = DataProto.from_single_dict(
            data={"real_seq_lens": real_seq_lens, "teacher_loss_lens": teacher_loss_lens},
        )

        # Per-sample compact payload: each object row is [loss_tokens_i, topk].
        # Full-vocab no longer pads prompt/padding tokens into [batch, seq, vocab].
        non_tensor = {"teacher_topk_logps": logps_payload}
        if has_indices:
            non_tensor["teacher_topk_indices"] = indices_payload
        output_batch.non_tensor_batch.update(non_tensor)

        tok2 = time.time()

        output_batch.meta_info["timing"] = {"get_teacher_knowledge": (tok1 - tik1) + (tok2 - tik2)}

        return output_batch

    if is_async:
        return SimpleNamespace(get=handle_futures)
    else:
        return handle_futures()


if __name__ == "__main__":
    batch = DataProto.load_from_disk("gen_batch_output")
    from teacher import TeacherClient

    teacher_client = TeacherClient(server_ip="10.215.192.141", server_port=15555)
    output_batch = get_teacher_knowledge(batch, 2, teacher_client)
    output_batch_chunks = output_batch.chunk(2)

    for data in output_batch_chunks:
        topk = data.meta_info["topk"]
        seq_lens = data.batch["seq_lens"]
        teacher_topk_logps = data.batch["teacher_topk_logps"].view(-1, topk)
        teacher_topk_indices = data.batch["teacher_topk_indices"].view(-1, topk)

        attention_mask = data.batch["attention_mask"]
        batch_size, sequence_length = attention_mask.size(0), attention_mask.size(1)
        teacher_topk_logps_padded = torch.zeros(batch_size, sequence_length, topk, dtype=teacher_topk_logps.dtype)
        teacher_topk_indices_padded = torch.zeros(batch_size, sequence_length, topk, dtype=teacher_topk_indices.dtype)

        teacher_topk_logps_padded[attention_mask] = teacher_topk_logps[: seq_lens.sum()]
        teacher_topk_indices_padded[attention_mask] = teacher_topk_indices[: seq_lens.sum()]

        data.batch["teacher_topk_logps"] = teacher_topk_logps_padded
        data.batch["teacher_topk_indices"] = teacher_topk_indices_padded

        assert (data.batch["teacher_topk_logps"] == data.batch["teacher_topk_logps_padded"]).all()
        assert (data.batch["teacher_topk_indices"] == data.batch["teacher_topk_indices_padded"]).all()
