"""Repo-local vLLM SKD sampler monkeypatch.

This intentionally does not edit site-packages. Import ``install`` before
creating a vLLM engine to replace vLLM's speculative rejection sampler with the
SKD top-k/top-p accept rule when vLLM speculative decoding is active.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "y", "on"}


def _accept_config() -> tuple[int, float]:
    top_k = int(os.environ.get("VLLM_SKD_ACCEPT_TOP_K", "25"))
    top_p = float(os.environ.get("VLLM_SKD_ACCEPT_TOP_P", "1.0"))
    return top_k, top_p


def _expand_temperature(rs: Any, sampling_metadata: Any,
                        cu_num_tokens: torch.Tensor,
                        num_tokens: int) -> torch.Tensor | None:
    temperature = getattr(sampling_metadata, "temperature", None)
    if temperature is None or getattr(sampling_metadata, "all_greedy", False):
        return None
    return rs.expand_batch_to_tokens(
        temperature,
        cu_num_tokens,
        num_tokens,
        replace_from=rs.GREEDY_TEMPERATURE,
        replace_to=1,
    )


def _draft_in_teacher_support(
    rs: Any,
    accept_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    sampling_metadata: Any,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    num_tokens, vocab_size = accept_logits.shape
    if num_tokens == 0:
        return torch.empty((0,), dtype=torch.bool, device=accept_logits.device)

    support = torch.ones((num_tokens,), dtype=torch.bool, device=accept_logits.device)

    if top_k and top_k > 0 and top_k < vocab_size:
        topk_ids = torch.topk(accept_logits, k=top_k, dim=-1).indices
        support &= (topk_ids == draft_token_ids.unsqueeze(-1)).any(dim=-1)

    if top_p is not None and 0.0 < top_p < 1.0:
        logits = accept_logits
        temperature = _expand_temperature(
            rs, sampling_metadata, cu_num_draft_tokens, num_tokens
        )
        if temperature is not None:
            logits = logits / temperature.unsqueeze(-1)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
        token_rank = torch.argmax(
            (sorted_ids == draft_token_ids.unsqueeze(-1)).to(torch.int64),
            dim=-1,
        )
        token_prob = probs.gather(1, draft_token_ids.unsqueeze(-1)).squeeze(-1)
        token_cum = sorted_probs.cumsum(dim=-1).gather(
            1, token_rank.unsqueeze(-1)
        ).squeeze(-1)
        support &= (token_cum - token_prob) < top_p

    return support


def _sample_teacher_replacements(
    rs: Any,
    target_probs: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    if getattr(sampling_metadata, "all_greedy", False):
        return target_probs.argmax(dim=-1).to(torch.int32)

    sampled = torch.multinomial(target_probs, num_samples=1).squeeze(-1).to(torch.int32)
    if getattr(sampling_metadata, "all_random", False):
        return sampled

    temperature = getattr(sampling_metadata, "temperature", None)
    if temperature is None:
        return sampled
    is_greedy = rs.expand_batch_to_tokens(
        temperature == rs.GREEDY_TEMPERATURE,
        cu_num_draft_tokens,
        target_probs.shape[0],
    )
    greedy = target_probs.argmax(dim=-1).to(torch.int32)
    return torch.where(is_greedy, greedy, sampled)


def _skd_sample_output(
    rs: Any,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    accept_mask: torch.Tensor,
    replacement_token_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size = len(num_draft_tokens)
    output = torch.full(
        (batch_size, max_spec_len + 1),
        rs.PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=draft_token_ids.device,
    )
    start = 0
    for req_idx, count in enumerate(num_draft_tokens):
        rejected = False
        for pos in range(int(count)):
            if rejected:
                break
            token_index = start + pos
            if bool(accept_mask[token_index].item()):
                token_id = draft_token_ids[token_index]
            else:
                token_id = replacement_token_ids[token_index]
                rejected = True
            output[req_idx, pos] = token_id
        start += int(count)
    return output


def install() -> bool:
    """Install the patch. Returns True when the old V1 sampler was patched."""
    if not _truthy(os.environ.get("VLLM_SKD_SAMPLER")):
        return False

    import vllm.v1.sample.rejection_sampler as rs  # pylint: disable=import-outside-toplevel

    if getattr(rs, "OPD_SKD_NATIVE", False):
        return True

    sampler_cls = rs.RejectionSampler
    if getattr(sampler_cls, "_opd_skd_patched", False):
        return True

    original_forward = sampler_cls.forward

    def skd_forward(self, metadata, draft_probs, logits, sampling_metadata):  # type: ignore[no-untyped-def]
        del draft_probs
        if sampling_metadata.max_num_logprobs:
            raise RuntimeError(
                "VLLM_SKD_SAMPLER does not support vLLM speculative logprobs; "
                "unset logprobs for SKD rollout generation."
            )
        assert metadata.max_spec_len <= rs.MAX_SPEC_LEN
        assert logits is not None

        target_logits_indices = metadata.target_logits_indices
        raw_target_logits = logits[target_logits_indices].to(torch.float32)
        accept_logits = self.apply_logits_processors(
            raw_target_logits.clone(), sampling_metadata, metadata
        )
        replacement_logits = rs.apply_sampling_constraints(
            accept_logits.clone(),
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
        replacement_probs = replacement_logits.softmax(dim=-1, dtype=torch.float32)
        replacement_token_ids = _sample_teacher_replacements(
            rs, replacement_probs, metadata.cu_num_draft_tokens, sampling_metadata
        )

        top_k, top_p = _accept_config()
        accept_mask = _draft_in_teacher_support(
            rs,
            accept_logits,
            metadata.draft_token_ids,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
            top_k,
            top_p,
        )
        output_token_ids = _skd_sample_output(
            rs,
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            accept_mask,
            replacement_token_ids,
        )
        return rs.SamplerOutput(sampled_token_ids=output_token_ids, logprobs_tensors=None)

    sampler_cls._opd_skd_original_forward = original_forward
    sampler_cls.forward = skd_forward
    sampler_cls._opd_skd_patched = True
    return True


def installed_vllm_draft_model_is_supported() -> bool:
    """Return whether this vLLM import path supports plain draft_model."""
    try:
        from pathlib import Path
        import vllm.config.speculative as speculative  # pylint: disable=import-outside-toplevel
        import vllm.v1.worker.gpu_model_runner as gpu_model_runner  # pylint: disable=import-outside-toplevel

        config_source = Path(speculative.__file__).read_text(encoding="utf-8")
        runner_source = Path(gpu_model_runner.__file__).read_text(encoding="utf-8")
    except Exception:
        return False

    draft_idx = config_source.find('self.method = "draft_model"')
    config_blocks_plain_draft = (
        draft_idx >= 0
        and "raise NotImplementedError" in config_source[draft_idx:draft_idx + 800]
    )
    runner_has_plain_draft = (
        "PlainDraftProposer" in runner_source
        and 'method == "draft_model"' in runner_source
    )
    return (not config_blocks_plain_draft) and runner_has_plain_draft


def unsupported_draft_model_message(student_model_path: str) -> str:
    return (
        "Installed vLLM does not support plain draft_model speculative decoding "
        f"for student model {student_model_path!r}. The repo-local SKD sampler "
        "patch is installed before engine creation, but this vLLM build only has "
        "EAGLE/MTP/Medusa/ngram/suffix proposers; Qwen student-as-draft needs a "
        "draft-model proposer in vLLM before the internal sampler path can run."
    )


__all__ = [
    "install",
    "installed_vllm_draft_model_is_supported",
    "unsupported_draft_model_message",
]
