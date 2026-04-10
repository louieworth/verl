from collections import defaultdict

import numpy as np
import torch

from verl.utils.model import compute_position_id_with_mask


_POINTWISE_SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "labels",
}

_POINTWISE_RESPONSE_KEYS = {
    "responses",
    "response_mask",
}


def _compute_pointwise_prompt_start(data_list: list[dict]) -> int:
    if not data_list:
        return 0

    first = data_list[0]
    required_keys = {"input_ids", "attention_mask", "responses"}
    if not required_keys.issubset(first):
        return 0

    response_length = int(first["responses"].shape[0])
    total_seq_length = int(first["input_ids"].shape[0])
    fixed_prompt_length = total_seq_length - response_length

    batch_prompt_length = 0
    for sample in data_list:
        if not required_keys.issubset(sample):
            return 0
        if int(sample["responses"].shape[0]) != response_length:
            return 0
        if int(sample["input_ids"].shape[0]) != total_seq_length:
            return 0

        prompt_attention_mask = sample["attention_mask"][:fixed_prompt_length]
        batch_prompt_length = max(batch_prompt_length, int(prompt_attention_mask.sum().item()))

    return max(fixed_prompt_length - batch_prompt_length, 0)


def _compute_pointwise_response_length(data_list: list[dict]) -> int:
    if not data_list:
        return 0

    first = data_list[0]
    if "response_mask" not in first:
        return 0

    fixed_response_length = int(first["response_mask"].shape[0])
    batch_response_length = 0
    for sample in data_list:
        if "response_mask" not in sample or int(sample["response_mask"].shape[0]) != fixed_response_length:
            return fixed_response_length
        batch_response_length = max(batch_response_length, int(sample["response_mask"].sum().item()))

    return max(batch_response_length, 1)


def pointwise_dynamic_prompt_collate_fn(data_list: list[dict]) -> dict:
    """Stack point-wise DPO samples while trimming prompt and response padding per batch."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    prompt_start = _compute_pointwise_prompt_start(data_list)
    response_length = _compute_pointwise_response_length(data_list)

    total_seq_length = int(data_list[0]["input_ids"].shape[0]) if data_list and "input_ids" in data_list[0] else 0
    fixed_response_length = int(data_list[0]["responses"].shape[0]) if data_list and "responses" in data_list[0] else 0
    fixed_prompt_length = total_seq_length - fixed_response_length
    seq_end = fixed_prompt_length + response_length

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key in _POINTWISE_SEQUENCE_KEYS:
                    val = val[prompt_start:seq_end]
                elif key in _POINTWISE_RESPONSE_KEYS:
                    val = val[:response_length]
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    if "attention_mask" in tensors and "position_ids" in tensors:
        tensors["position_ids"] = compute_position_id_with_mask(tensors["attention_mask"]).to(torch.long)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}
