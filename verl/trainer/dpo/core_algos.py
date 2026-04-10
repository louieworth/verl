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
import torch.nn.functional as F


def normalize_dpo_loss_type(loss_type: str) -> str:
    return loss_type.lower()


def use_average_sequence_log_probs(loss_type: str) -> bool:
    """IPO and SimPO use length-normalized sequence scores."""
    return normalize_dpo_loss_type(loss_type) in {"ipo", "simpo"}


def requires_reference_model(loss_type: str, reference_free: bool) -> bool:
    return (not reference_free) and normalize_dpo_loss_type(loss_type) != "simpo"


def get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    """Compute sequence log probabilities for a batch of labels."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"Logits shape {logits.shape[:-1]} and labels shape {labels.shape} must align")

    labels = labels.contiguous().to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token_logps = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token_logps = per_token_logps.view(shift_logits.size(0), shift_logits.size(1))

    valid_mask = shift_labels != -100
    masked_logps = per_token_logps * valid_mask
    sequence_logps = masked_logps.sum(dim=-1)

    if average_log_prob:
        token_count = valid_mask.sum(dim=-1)
        return sequence_logps / torch.clamp(token_count, min=1)
    return sequence_logps


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor | None,
    reference_rejected_logps: torch.Tensor | None,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
    simpo_gamma: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a mean DPO-family loss and return scalar statistics."""
    loss_type = normalize_dpo_loss_type(loss_type)
    effective_reference_free = not requires_reference_model(loss_type, reference_free)
    pi_logratios = policy_chosen_logps - policy_rejected_logps

    if effective_reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)
    else:
        if reference_chosen_logps is None or reference_rejected_logps is None:
            raise ValueError("Reference log probabilities are required unless reference_free=True")
        ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "simpo":
        logits = pi_logratios - (simpo_gamma / beta)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    else:
        raise ValueError(f"Unsupported DPO loss_type: {loss_type}")

    chosen_rewards = beta * (
        policy_chosen_logps - (torch.zeros_like(policy_chosen_logps) if effective_reference_free else reference_chosen_logps)
    )
    rejected_rewards = beta * (
        policy_rejected_logps
        - (torch.zeros_like(policy_rejected_logps) if effective_reference_free else reference_rejected_logps)
    )

    stats = {
        "loss": losses.mean(),
        "logits": logits.mean(),
        "accuracy": (logits > 0).float().mean(),
        "margin": (chosen_rewards - rejected_rewards).mean(),
        "policy_logratio": pi_logratios.mean(),
        "reference_logratio": ref_logratios.mean(),
        "chosen_reward": chosen_rewards.mean(),
        "rejected_reward": rejected_rewards.mean(),
    }
    return losses.mean(), stats
