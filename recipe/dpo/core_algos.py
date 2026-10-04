import math

import torch
import torch.nn.functional as F


def normalize_dpo_loss_type(loss_type: str) -> str:
    return loss_type.lower()


def is_prospect_dpo_loss(loss_type: str) -> bool:
    return normalize_dpo_loss_type(loss_type) == "prospect_dpo"


def is_single_wise_dpo_loss(loss_type: str) -> bool:
    return normalize_dpo_loss_type(loss_type) == "single_wise_dpo"


def is_pointwise_dpo_loss(loss_type: str) -> bool:
    normalized_loss_type = normalize_dpo_loss_type(loss_type)
    return normalized_loss_type in {"prospect_dpo", "single_wise_dpo"}


def use_average_sequence_log_probs(loss_type: str) -> bool:
    """IPO and SimPO use length-normalized sequence scores."""
    return normalize_dpo_loss_type(loss_type) in {"ipo", "simpo"}


def requires_reference_model(loss_type: str, reference_free: bool) -> bool:
    normalized_loss_type = normalize_dpo_loss_type(loss_type)
    return (not reference_free) and normalized_loss_type != "simpo"


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


def _validate_positive_kappa(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _validate_nonnegative_kappa(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


def compute_sigmoid_feedback_weight(
    signal: torch.Tensor,
    kappa: float,
    *,
    name: str,
    allow_zero: bool = False,
) -> torch.Tensor:
    """Compute kappa * sigmoid(signal) after clipping the feedback signal to [0, 1]."""
    kappa = (
        _validate_nonnegative_kappa(kappa, name)
        if allow_zero
        else _validate_positive_kappa(kappa, name)
    )
    return kappa * torch.sigmoid(signal.float().clamp(0.0, 1.0))


def compute_prospect_dpo_weights(
    *,
    s_dwell: torch.Tensor,
    p_ctr: torch.Tensor,
    positive_kappa: float,
    negative_kappa: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the positive and negative sigmoid feedback weights."""
    alpha = compute_sigmoid_feedback_weight(
        s_dwell,
        positive_kappa,
        name="positive_kappa",
    )
    lambda_weight = compute_sigmoid_feedback_weight(
        p_ctr,
        negative_kappa,
        name="negative_kappa",
        allow_zero=True,
    )
    return alpha, lambda_weight


def compute_prospect_dpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor | None,
    labels: torch.Tensor,
    beta: float,
    s_dwell: torch.Tensor,
    p_ctr: torch.Tensor,
    positive_kappa: float,
    negative_kappa: float,
    reference_free: bool = False,
    sft_coef: float = 0.0,
    negative_scale: float = 1.0,
    class_fractions: tuple[float, float] | None = None,
    use_feedback_weights: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Signed sigmoid objective, optionally anchored by positive response NLL.

    With class_fractions from the full update batch, sample-weighted gradient
    accumulation and data-parallel averaging yield exact class means even
    when individual micro-batches contain only one class. Defaults retain the
    historical objective. NLL uses the same sequence logp convention as rewards;
    the ablation launchers require average_log_prob=True.
    Disabling use_feedback_weights fixes alpha and lambda to one without
    changing the sigmoid objective or the reductions.
    """
    sft_coef = _validate_nonnegative_kappa(sft_coef, "sft_coef")
    negative_scale = _validate_nonnegative_kappa(negative_scale, "negative_scale")
    if class_fractions is not None:
        if len(class_fractions) != 2 or any(
            not math.isfinite(f) or not 0.0 <= f <= 1.0 for f in class_fractions
        ) or not math.isclose(sum(class_fractions), 1.0, abs_tol=1e-6):
            raise ValueError("class_fractions must contain two nonnegative fractions summing to one")
    if reference_free:
        rewards = (beta * policy_logps).float()
    else:
        if reference_logps is None:
            raise ValueError("Prospect-DPO requires reference log probabilities unless reference_free=True")
        rewards = (beta * (policy_logps - reference_logps)).float()

    labels = labels.float()
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    positive_mask_f = positive_mask.float()
    negative_mask_f = negative_mask.float()

    if use_feedback_weights:
        alpha, lambda_weight = compute_prospect_dpo_weights(
            s_dwell=s_dwell,
            p_ctr=p_ctr,
            positive_kappa=positive_kappa,
            negative_kappa=negative_kappa,
        )
    else:
        # Strict weight ablation: keep the sigmoid objective and its class
        # reductions, replacing both feedback transforms with unit weights.
        alpha = torch.ones_like(rewards)
        lambda_weight = torch.ones_like(rewards)
    alpha = alpha.to(rewards.device)
    lambda_weight = lambda_weight.to(rewards.device)

    positive_losses = torch.sigmoid(-(alpha * rewards))
    negative_losses = torch.sigmoid(lambda_weight * rewards)

    # Mask-weighted means (no Python-level `torch.any` branching). This keeps
    # the autograd graph shape identical across ranks even when a micro-batch
    # on a given rank is all-positive or all-negative — required for FSDP
    # allgather/reduce-scatter to stay in sync across ranks. A previous version
    # used `if torch.any(mask) else zero`, which produced different NCCL
    # collective sequences per rank and deterministically deadlocked at
    # the first backward pass (`_ALLGATHER_BASE` hang, seq ~6452 on 8× H100).
    def _masked_mean(x: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        denom = mask_f.sum().clamp(min=1.0)
        return (x * mask_f).sum() / denom

    if class_fractions is None:
        positive_loss = _masked_mean(positive_losses, positive_mask_f)
        negative_loss = _masked_mean(negative_losses, negative_mask_f)
        positive_nll = _masked_mean(-policy_logps.float(), positive_mask_f)
    else:
        # Zero-fraction branches have zero numerator; keep their graph intact.
        positive_fraction, negative_fraction = class_fractions
        positive_denom = max(policy_logps.numel() * positive_fraction, 1e-12)
        negative_denom = max(policy_logps.numel() * negative_fraction, 1e-12)
        positive_loss = (positive_losses * positive_mask_f).sum() / positive_denom
        negative_loss = (negative_losses * negative_mask_f).sum() / negative_denom
        positive_nll = (-policy_logps.float() * positive_mask_f).sum() / positive_denom
    total_loss = positive_loss + negative_scale * negative_loss + sft_coef * positive_nll

    signed_margin = torch.where(positive_mask, rewards, -rewards)
    positive_reward = _masked_mean(rewards, positive_mask_f)
    negative_reward = _masked_mean(rewards, negative_mask_f)
    alpha_mean = _masked_mean(alpha, positive_mask_f)
    lambda_mean = _masked_mean(lambda_weight, negative_mask_f)

    stats = {
        "loss": total_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "positive_nll": positive_nll,
        "negative_scale": rewards.new_tensor(negative_scale),
        "accuracy": (signed_margin > 0).float().mean(),
        "margin": signed_margin.mean(),
        "reward": rewards.mean(),
        "positive_reward": positive_reward,
        "negative_reward": negative_reward,
        "alpha": alpha_mean,
        "lambda": lambda_mean,
        "positive_fraction": positive_mask_f.mean(),
    }
    return total_loss, stats


def prospect_negative_scale_at_step(scale: float, warmup_steps: int, step: int) -> float:
    """Linear negative warmup using the absolute next optimizer step (1-based)."""
    scale = _validate_nonnegative_kappa(scale, "negative_scale")
    if warmup_steps < 0 or step < 0:
        raise ValueError("warmup_steps and step must be nonnegative")
    return scale * (min(step / warmup_steps, 1.0) if warmup_steps else 1.0)


def compute_single_wise_dpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor | None,
    labels: torch.Tensor,
    beta: float,
    reference_free: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if reference_free:
        rewards = (beta * policy_logps).float()
    else:
        if reference_logps is None:
            raise ValueError("Single-wise DPO requires reference log probabilities unless reference_free=True")
        rewards = (beta * (policy_logps - reference_logps)).float()

    labels = labels.float().clamp(0.0, 1.0)
    losses = F.binary_cross_entropy_with_logits(rewards, labels, reduction="none")

    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    positive_mask_f = positive_mask.float()
    negative_mask_f = negative_mask.float()

    def _masked_mean(x: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        denom = mask_f.sum().clamp(min=1.0)
        return (x * mask_f).sum() / denom

    # Mask-weighted means keep the autograd graph identical across ranks
    # even when a micro-batch happens to be all-positive or all-negative.
    # Required to avoid FSDP allgather desync at backward (see prospect_dpo
    # loss for the original deadlock signature).
    positive_loss = _masked_mean(losses, positive_mask_f)
    negative_loss = _masked_mean(losses, negative_mask_f)

    signed_margin = torch.where(positive_mask, rewards, -rewards)
    positive_reward = _masked_mean(rewards, positive_mask_f)
    negative_reward = _masked_mean(rewards, negative_mask_f)

    stats = {
        "loss": losses.mean(),
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "accuracy": (signed_margin > 0).float().mean(),
        "margin": signed_margin.mean(),
        "reward": rewards.mean(),
        "positive_reward": positive_reward,
        "negative_reward": negative_reward,
        "positive_fraction": positive_mask_f.mean(),
    }
    return losses.mean(), stats
