import torch

from verl import DataProto


PAIR_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_mask",
)

POINT_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_mask",
)


def build_pair_log_prob_batch(batch: DataProto) -> tuple[DataProto, torch.Tensor]:
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


def build_point_log_prob_batch(batch: DataProto) -> tuple[DataProto, torch.Tensor]:
    tensors = {}
    response_mask = None
    for key in POINT_KEYS:
        if key not in batch.batch:
            raise KeyError(f"Missing required point-wise DPO key: {key}")
        tensors[key] = batch.batch[key]
        if key == "response_mask":
            response_mask = tensors[key]

    log_prob_batch = DataProto.from_dict(
        tensors={
            "input_ids": tensors["input_ids"],
            "attention_mask": tensors["attention_mask"],
            "position_ids": tensors["position_ids"],
            "responses": tensors["responses"],
        }
    )
    return log_prob_batch, response_mask


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


def build_prospect_dpo_update_proto(
    batch: DataProto,
    beta: float,
    positive_kappa: float,
    negative_kappa: float,
    reference_logps: torch.Tensor | None,
    average_log_prob: bool = False,
    reference_free: bool = False,
    sft_coef: float = 0.0,
    negative_scale: float = 1.0,
    global_class_mean: bool = False,
    use_feedback_weights: bool = True,
) -> DataProto:
    tensors = {}
    for key in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "label",
        "s_dwell",
        "p_ctr",
    ):
        if key not in batch.batch:
            raise KeyError(f"Missing required Prospect-DPO update key: {key}")
        tensors[key] = batch.batch[key]
    if reference_logps is not None:
        tensors["reference_logps"] = reference_logps

    meta_info = {
        "dpo_beta": beta,
        "dpo_loss_type": "prospect_dpo",
        "reference_free": bool(reference_free),
        "prospect_dpo_positive_kappa": positive_kappa,
        "prospect_dpo_negative_kappa": negative_kappa,
        "prospect_dpo_sft_coef": sft_coef,
        "prospect_dpo_negative_scale": negative_scale,
        "prospect_dpo_use_feedback_weights": bool(use_feedback_weights),
        "average_log_prob": bool(average_log_prob),
        "global_token_num": int(batch.batch["attention_mask"].sum().item()),
    }
    if global_class_mean:
        positive_fraction = float((batch.batch["label"] > 0.5).float().mean().item())
        meta_info["prospect_dpo_class_fractions"] = (positive_fraction, 1.0 - positive_fraction)
    return DataProto.from_dict(tensors=tensors, meta_info=meta_info)


def build_single_wise_dpo_update_proto(
    batch: DataProto,
    beta: float,
    reference_logps: torch.Tensor | None,
    average_log_prob: bool = False,
    reference_free: bool = False,
) -> DataProto:
    tensors = {}
    for key in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "label",
    ):
        if key not in batch.batch:
            raise KeyError(f"Missing required single-wise DPO update key: {key}")
        tensors[key] = batch.batch[key]
    if reference_logps is not None:
        tensors["reference_logps"] = reference_logps

    meta_info = {
        "dpo_beta": beta,
        "dpo_loss_type": "single_wise_dpo",
        "reference_free": bool(reference_free),
        "average_log_prob": bool(average_log_prob),
        "global_token_num": int(batch.batch["attention_mask"].sum().item()),
    }
    return DataProto.from_dict(tensors=tensors, meta_info=meta_info)
