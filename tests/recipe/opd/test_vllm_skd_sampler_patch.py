#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace

import torch

from recipe.opd.generation import vllm_skd_sampler_patch as patch


class FakeRs:
    PLACEHOLDER_TOKEN_ID = -1
    GREEDY_TEMPERATURE = 0

    @staticmethod
    def expand_batch_to_tokens(x, cu_num_tokens, num_tokens, replace_from=0, replace_to=0):
        values = x.clone()
        values = torch.where(values == replace_from, torch.full_like(values, replace_to), values)
        out = []
        start = 0
        for end, value in zip(cu_num_tokens.tolist(), values):
            out.extend([value] * (end - start))
            start = end
        return torch.stack(out) if out else values.new_empty((num_tokens,))


def test_skd_support_accepts_top_k_and_rejects_outside_support():
    logits = torch.tensor(
        [
            [4.0, 3.0, 1.0],
            [0.0, 5.0, 1.0],
        ]
    )
    draft_ids = torch.tensor([1, 0])
    accept = patch._draft_in_teacher_support(
        FakeRs,
        logits,
        draft_ids,
        torch.tensor([2]),
        SimpleNamespace(all_greedy=True),
        top_k=2,
        top_p=1.0,
    )
    assert accept.tolist() == [True, False]


def test_skd_sample_output_discards_after_first_rejection():
    output = patch._skd_sample_output(
        FakeRs,
        draft_token_ids=torch.tensor([7, 8, 9]),
        num_draft_tokens=[3],
        max_spec_len=3,
        accept_mask=torch.tensor([True, False, True]),
        replacement_token_ids=torch.tensor([17, 18, 19]),
    )
    assert output.tolist() == [[7, 18, -1, -1]]
