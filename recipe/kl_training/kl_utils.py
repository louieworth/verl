#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# KL Divergence Computation Utilities
# Implements 4 variants of KL divergence for token-level training

import torch
import torch.nn.functional as F
from typing import Literal


def compute_reverse_kl_monte_carlo(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Compute Reverse KL divergence using Monte Carlo estimation.

    Formula: D_KL[P || Q] where P = student, Q = teacher
    KL = E_P[log(P/Q)] = E_P[log P - log Q]

    Monte Carlo surrogate evaluated on student-sampled tokens:
    KL = E_P[Q/P - 1 - log(Q/P)]
       = E_P[exp(log_q - log_p) - 1 - (log_q - log_p)]

    This is the most memory-efficient variant, only computing KL for sampled tokens.

    Args:
        student_logprobs: Log probabilities from student model [batch, seq_len]
        teacher_logprobs: Log probabilities from teacher model [batch, seq_len]
        mask: Attention mask [batch, seq_len]
        reduction: How to reduce the KL values

    Returns:
        KL divergence value(s)

    Reference:
        LMOps/minillm/minillm/utils.py:get_rev_kl()
    """
    log_ratio = (teacher_logprobs - student_logprobs) * mask
    # Numerical stability: clamp log_ratio to prevent overflow in exp()
    # When log_ratio > 20, exp(log_ratio) can overflow (exp(20) ≈ 5e8)
    # When log_ratio < -100, exp(log_ratio) underflows to 0 (handled gracefully)
    log_ratio_clamped = log_ratio.clamp(max=20)
    kl = log_ratio_clamped.float().exp() - 1 - log_ratio

    if reduction == "mean":
        return (kl * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (kl * mask).sum()
    else:  # none
        return kl


def compute_forward_kl_monte_carlo(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Compute Forward KL divergence using Monte Carlo estimation.

    Formula: D_KL[Q || P] where Q = teacher, P = student
    KL = E_Q[log(Q/P)] = E_Q[log Q - log P]

    Teacher-sampled surrogate:
    KL = E_Q[log Q - log P]
    d/dtheta KL = d/dtheta E_Q[-log P]

    The teacher term is constant with respect to student parameters, so the
    token-level partial-gradient objective reduces to the teacher-forced NLL.

    Args:
        teacher_logprobs: Log probabilities from teacher model [batch, seq_len]
        student_logprobs: Log probabilities from student model [batch, seq_len]
        mask: Attention mask [batch, seq_len]
        reduction: How to reduce the KL values

    Returns:
        KL divergence value(s)
    """
    del teacher_logprobs
    # For forward KL with Monte Carlo, we use the negative log likelihood
    # of teacher's samples under student's distribution
    kl = -student_logprobs

    if reduction == "mean":
        return (kl * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (kl * mask).sum()
    else:  # none
        return kl


def compute_reverse_kl_full_vocab(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute Reverse KL divergence over full vocabulary.

    Formula: D_KL[P || Q] = sum_i P(i) * log(P(i) / Q(i))
    where P = student, Q = teacher, i ranges over all tokens in vocabulary

    This is more accurate than Monte Carlo but requires more memory.
    Uses mixed precision (fp32 for softmax, fp16/bf16 for logits) for efficiency.

    Args:
        student_logits: Logits from student model [batch, seq_len, vocab_size]
        teacher_logits: Logits from teacher model [batch, seq_len, vocab_size]
        mask: Attention mask [batch, seq_len]
        reduction: How to reduce the KL values
        temperature: Temperature for softmax (default 1.0)

    Returns:
        KL divergence value(s)
    """
    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Compute log probabilities in fp32 for numerical stability
    student_logprobs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1)

    # Compute student probabilities
    student_probs = student_logprobs.exp()

    # Reverse KL: sum over vocab of P(i) * log(P(i) / Q(i))
    kl_per_position = (student_probs * (student_logprobs - teacher_logprobs)).sum(dim=-1)

    if reduction == "mean":
        return (kl_per_position * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (kl_per_position * mask).sum()
    else:  # none
        return kl_per_position


def compute_forward_kl_full_vocab(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute Forward KL divergence over full vocabulary.

    Formula: D_KL[Q || P] = sum_i Q(i) * log(Q(i) / P(i))
    where Q = teacher, P = student, i ranges over all tokens in vocabulary

    This encourages the student to cover all modes of the teacher distribution.

    Args:
        teacher_logits: Logits from teacher model [batch, seq_len, vocab_size]
        student_logits: Logits from student model [batch, seq_len, vocab_size]
        mask: Attention mask [batch, seq_len]
        reduction: How to reduce the KL values
        temperature: Temperature for softmax (default 1.0)

    Returns:
        KL divergence value(s)
    """
    # Apply temperature scaling
    teacher_logits = teacher_logits / temperature
    student_logits = student_logits / temperature

    # Compute log probabilities in fp32 for numerical stability
    teacher_logprobs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logprobs = F.log_softmax(student_logits.float(), dim=-1)

    # Compute teacher probabilities
    teacher_probs = teacher_logprobs.exp()

    # Forward KL: sum over vocab of Q(i) * log(Q(i) / P(i))
    kl_per_position = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)

    if reduction == "mean":
        return (kl_per_position * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (kl_per_position * mask).sum()
    else:  # none
        return kl_per_position


def compute_kl_divergence(
    student_logits_or_logprobs: torch.Tensor,
    teacher_logits_or_logprobs: torch.Tensor,
    mask: torch.Tensor,
    kl_type: Literal["reverse", "forward"] = "reverse",
    kl_method: Literal["monte_carlo", "full_vocab"] = "monte_carlo",
    is_logprobs: bool = False,
    reduction: Literal["mean", "sum", "none"] = "mean",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Unified interface for computing KL divergence.

    Args:
        student_logits_or_logprobs: Student model outputs
        teacher_logits_or_logprobs: Teacher model outputs
        mask: Attention mask
        kl_type: "reverse" or "forward"
        kl_method: "monte_carlo" or "full_vocab"
        is_logprobs: Whether inputs are log probabilities (True) or logits (False)
        reduction: How to reduce the KL values
        temperature: Temperature for softmax (only used for full_vocab)

    Returns:
        KL divergence value(s)
    """
    if kl_method == "monte_carlo":
        if not is_logprobs:
            raise ValueError("Monte Carlo KL requires log probabilities, not logits")

        if kl_type == "reverse":
            return compute_reverse_kl_monte_carlo(
                student_logits_or_logprobs,
                teacher_logits_or_logprobs,
                mask,
                reduction,
            )
        else:  # forward
            return compute_forward_kl_monte_carlo(
                teacher_logits_or_logprobs,
                student_logits_or_logprobs,
                mask,
                reduction,
            )

    else:  # full_vocab
        if is_logprobs:
            raise ValueError("Full vocab KL requires logits, not log probabilities")

        if kl_type == "reverse":
            return compute_reverse_kl_full_vocab(
                student_logits_or_logprobs,
                teacher_logits_or_logprobs,
                mask,
                reduction,
                temperature,
            )
        else:  # forward
            return compute_forward_kl_full_vocab(
                teacher_logits_or_logprobs,
                student_logits_or_logprobs,
                mask,
                reduction,
                temperature,
            )
