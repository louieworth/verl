#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Unit tests for KL utilities

import torch
import pytest
from recipe.kl_training.kl_utils import (
    compute_reverse_kl_monte_carlo,
    compute_reverse_kl_full_vocab,
    compute_forward_kl_monte_carlo,
    compute_forward_kl_full_vocab,
)


def test_reverse_kl_monte_carlo_identical():
    """Test that KL is 0 when distributions are identical."""
    batch_size, seq_len = 2, 10
    logprobs = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)

    kl = compute_reverse_kl_monte_carlo(logprobs, logprobs, mask)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-5)


def test_reverse_kl_monte_carlo_non_negative():
    """Test that KL is always non-negative."""
    batch_size, seq_len = 2, 10
    student_logprobs = torch.randn(batch_size, seq_len)
    teacher_logprobs = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)

    kl = compute_reverse_kl_monte_carlo(student_logprobs, teacher_logprobs, mask)
    assert kl >= 0


def test_reverse_kl_monte_carlo_with_mask():
    """Test that masking works correctly."""
    batch_size, seq_len = 2, 10
    student_logprobs = torch.randn(batch_size, seq_len)
    teacher_logprobs = torch.randn(batch_size, seq_len)

    # Mask out second half
    mask = torch.ones(batch_size, seq_len)
    mask[:, seq_len//2:] = 0

    kl = compute_reverse_kl_monte_carlo(student_logprobs, teacher_logprobs, mask)
    assert kl >= 0


def test_forward_kl_monte_carlo_identical():
    """Teacher-forced forward KL surrogate reduces to NLL."""
    batch_size, seq_len = 2, 10
    logits = torch.randn(batch_size, seq_len, 32)
    logprobs = torch.log_softmax(logits, dim=-1)[..., 0]
    mask = torch.ones(batch_size, seq_len)

    kl = compute_forward_kl_monte_carlo(logprobs, logprobs, mask)
    expected = (-logprobs * mask).sum() / mask.sum()
    assert torch.allclose(kl, expected, atol=1e-5)


def test_forward_kl_monte_carlo_non_negative():
    """Teacher-forced NLL is non-negative for valid log-probabilities."""
    batch_size, seq_len = 2, 10
    teacher_logprobs = torch.log_softmax(torch.randn(batch_size, seq_len, 16), dim=-1)[..., 0]
    student_logprobs = torch.log_softmax(torch.randn(batch_size, seq_len, 16), dim=-1)[..., 0]
    mask = torch.ones(batch_size, seq_len)

    kl = compute_forward_kl_monte_carlo(teacher_logprobs, student_logprobs, mask)
    assert kl >= 0


def test_reverse_kl_full_vocab_identical():
    """Test that full vocab KL is 0 when distributions are identical."""
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    mask = torch.ones(batch_size, seq_len)

    kl = compute_reverse_kl_full_vocab(logits, logits, mask)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-4)


def test_reverse_kl_full_vocab_non_negative():
    """Test that full vocab KL is always non-negative."""
    batch_size, seq_len, vocab_size = 2, 10, 100
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    mask = torch.ones(batch_size, seq_len)

    kl = compute_reverse_kl_full_vocab(student_logits, teacher_logits, mask)
    assert kl >= 0


def test_reverse_kl_full_vocab_with_temperature():
    """Test that temperature scaling works."""
    batch_size, seq_len, vocab_size = 2, 10, 100
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    mask = torch.ones(batch_size, seq_len)

    kl_t1 = compute_reverse_kl_full_vocab(student_logits, teacher_logits, mask, temperature=1.0)
    kl_t2 = compute_reverse_kl_full_vocab(student_logits, teacher_logits, mask, temperature=2.0)

    # Different temperatures should give different KL values
    assert not torch.allclose(kl_t1, kl_t2)


def test_forward_kl_full_vocab_identical():
    """Test that forward full vocab KL is 0 when distributions are identical."""
    batch_size, seq_len, vocab_size = 2, 10, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    mask = torch.ones(batch_size, seq_len)

    kl = compute_forward_kl_full_vocab(logits, logits, mask)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-4)


def test_forward_kl_full_vocab_non_negative():
    """Test that forward full vocab KL is always non-negative."""
    batch_size, seq_len, vocab_size = 2, 10, 100
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    mask = torch.ones(batch_size, seq_len)

    kl = compute_forward_kl_full_vocab(teacher_logits, student_logits, mask)
    assert kl >= 0


def test_reduction_modes():
    """Test different reduction modes."""
    batch_size, seq_len = 2, 10
    student_logprobs = torch.randn(batch_size, seq_len)
    teacher_logprobs = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)

    kl_mean = compute_reverse_kl_monte_carlo(student_logprobs, teacher_logprobs, mask, reduction="mean")
    kl_sum = compute_reverse_kl_monte_carlo(student_logprobs, teacher_logprobs, mask, reduction="sum")
    kl_none = compute_reverse_kl_monte_carlo(student_logprobs, teacher_logprobs, mask, reduction="none")

    assert kl_mean.shape == torch.Size([])
    assert kl_sum.shape == torch.Size([])
    assert kl_none.shape == torch.Size([batch_size, seq_len])


def test_asymmetry():
    """Test that reverse and forward KL are different (asymmetric)."""
    batch_size, seq_len = 2, 10
    p_logprobs = torch.randn(batch_size, seq_len)
    q_logprobs = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)

    reverse_kl = compute_reverse_kl_monte_carlo(p_logprobs, q_logprobs, mask)
    forward_kl = compute_forward_kl_monte_carlo(q_logprobs, p_logprobs, mask)

    # They should be different (KL is asymmetric)
    assert not torch.allclose(reverse_kl, forward_kl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
