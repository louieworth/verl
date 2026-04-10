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
    if normalized_loss_type in {"prospect_dpo", "single_wise_dpo"}:
        return True
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


def compute_prospect_dpo_alpha(
    s_dwell: torch.Tensor,
    alpha_tau: float,
    alpha_k: float,
) -> torch.Tensor:
    s_dwell = s_dwell.float().clamp(0.0, 1.0)
    return torch.sigmoid(alpha_k * (s_dwell - alpha_tau))


def compute_prospect_dpo_lambda(
    p_ctr: torch.Tensor,
    lambda_max: float,
    lambda_gamma: float,
) -> torch.Tensor:
    p_ctr = p_ctr.float().clamp(0.0, 1.0)
    return 1.0 + (lambda_max - 1.0) * torch.pow(p_ctr, lambda_gamma)


def compute_prospect_dpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor | None,
    labels: torch.Tensor,
    beta: float,
    s_dwell: torch.Tensor,
    p_ctr: torch.Tensor,
    alpha_tau: float,
    alpha_k: float,
    lambda_max: float,
    lambda_gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if reference_logps is None:
        raise ValueError("Prospect-DPO requires reference log probabilities")

    rewards = (beta * (policy_logps - reference_logps)).float()
    labels = labels.float()
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask

    alpha = compute_prospect_dpo_alpha(s_dwell=s_dwell, alpha_tau=alpha_tau, alpha_k=alpha_k).to(rewards.device)
    lambda_weight = compute_prospect_dpo_lambda(
        p_ctr=p_ctr,
        lambda_max=lambda_max,
        lambda_gamma=lambda_gamma,
    ).to(rewards.device)

    positive_losses = torch.sigmoid(-(alpha * rewards))
    negative_losses = torch.sigmoid(lambda_weight * rewards)

    zero = rewards.new_zeros(())
    positive_loss = positive_losses[positive_mask].mean() if torch.any(positive_mask) else zero
    negative_loss = negative_losses[negative_mask].mean() if torch.any(negative_mask) else zero
    total_loss = positive_loss + negative_loss

    signed_margin = torch.where(positive_mask, rewards, -rewards)
    positive_reward = rewards[positive_mask].mean() if torch.any(positive_mask) else zero
    negative_reward = rewards[negative_mask].mean() if torch.any(negative_mask) else zero
    alpha_mean = alpha[positive_mask].mean() if torch.any(positive_mask) else zero
    lambda_mean = lambda_weight[negative_mask].mean() if torch.any(negative_mask) else zero

    stats = {
        "loss": total_loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "accuracy": (signed_margin > 0).float().mean() if signed_margin.numel() > 0 else zero,
        "margin": signed_margin.mean() if signed_margin.numel() > 0 else zero,
        "reward": rewards.mean() if rewards.numel() > 0 else zero,
        "positive_reward": positive_reward,
        "negative_reward": negative_reward,
        "alpha": alpha_mean,
        "lambda": lambda_mean,
        "positive_fraction": positive_mask.float().mean() if positive_mask.numel() > 0 else zero,
    }
    return total_loss, stats


def compute_single_wise_dpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor | None,
    labels: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if reference_logps is None:
        raise ValueError("Single-wise DPO requires reference log probabilities")

    rewards = (beta * (policy_logps - reference_logps)).float()
    labels = labels.float().clamp(0.0, 1.0)
    losses = F.binary_cross_entropy_with_logits(rewards, labels, reduction="none")

    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    zero = rewards.new_zeros(())
    positive_loss = losses[positive_mask].mean() if torch.any(positive_mask) else zero
    negative_loss = losses[negative_mask].mean() if torch.any(negative_mask) else zero

    signed_margin = torch.where(positive_mask, rewards, -rewards)
    positive_reward = rewards[positive_mask].mean() if torch.any(positive_mask) else zero
    negative_reward = rewards[negative_mask].mean() if torch.any(negative_mask) else zero

    stats = {
        "loss": losses.mean(),
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "accuracy": (signed_margin > 0).float().mean() if signed_margin.numel() > 0 else zero,
        "margin": signed_margin.mean() if signed_margin.numel() > 0 else zero,
        "reward": rewards.mean() if rewards.numel() > 0 else zero,
        "positive_reward": positive_reward,
        "negative_reward": negative_reward,
        "positive_fraction": positive_mask.float().mean() if positive_mask.numel() > 0 else zero,
    }
    return losses.mean(), stats
