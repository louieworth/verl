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

import torch

from verl import DataProto


PAIR_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_mask",
)


def build_log_prob_batch(batch: DataProto) -> tuple[DataProto, torch.Tensor]:
    """Flatten chosen/rejected tensors into the standard log-prob RPC batch format."""
    tensors = {}
    masks = []
    for key in PAIR_KEYS:
        chosen_key = f"chosen_{key}"
        rejected_key = f"rejected_{key}"
        if chosen_key not in batch.batch or rejected_key not in batch.batch:
            raise KeyError(f"Missing required DPO pair key(s): {chosen_key}, {rejected_key}")
        tensors[key] = torch.cat([batch.batch[chosen_key], batch.batch[rejected_key]], dim=0)
        if key == "response_mask":
            masks.append(tensors[key])

    log_prob_batch = DataProto.from_dict(
        tensors={
            "input_ids": tensors["input_ids"],
            "attention_mask": tensors["attention_mask"],
            "position_ids": tensors["position_ids"],
            "responses": tensors["responses"],
        }
    )
    return log_prob_batch, masks[0]


def compute_sequence_log_probs(
    token_log_probs: torch.Tensor, response_mask: torch.Tensor, average_log_prob: bool = False
) -> torch.Tensor:
    if token_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"Token log prob shape {token_log_probs.shape} and response mask shape {response_mask.shape} must match"
        )
    masked_log_probs = token_log_probs * response_mask.to(token_log_probs.dtype)
    sequence_log_probs = masked_log_probs.sum(dim=-1)
    if average_log_prob:
        token_count = response_mask.sum(dim=-1).clamp_min(1).to(token_log_probs.dtype)
        return sequence_log_probs / token_count
    return sequence_log_probs


def build_dpo_update_proto(
    batch: DataProto,
    beta: float,
    loss_type: str,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
    simpo_gamma: float = 0.5,
    reference_chosen_logps: torch.Tensor | None = None,
    reference_rejected_logps: torch.Tensor | None = None,
) -> DataProto:
    tensors = {}
    for prefix in ("chosen", "rejected"):
        for key in ("input_ids", "attention_mask", "position_ids", "labels"):
            batch_key = f"{prefix}_{key}"
            if batch_key not in batch.batch:
                raise KeyError(f"Missing required DPO update key: {batch_key}")
            tensors[batch_key] = batch.batch[batch_key]

    if reference_chosen_logps is not None:
        tensors["reference_chosen_logps"] = reference_chosen_logps
    if reference_rejected_logps is not None:
        tensors["reference_rejected_logps"] = reference_rejected_logps

    meta_info = {
        "dpo_beta": beta,
        "dpo_loss_type": loss_type,
        "dpo_label_smoothing": label_smoothing,
        "reference_free": reference_free,
        "simpo_gamma": simpo_gamma,
        "global_token_num": int(
            batch.batch["chosen_attention_mask"].sum().item() + batch.batch["rejected_attention_mask"].sum().item()
        ),
    }
    return DataProto.from_dict(tensors=tensors, meta_info=meta_info)
