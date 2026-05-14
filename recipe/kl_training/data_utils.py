#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Data preparation for KL divergence training with verl FSDP engines.

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator

# Instruction following format from existing pipeline
instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


PROMPT_TEMPLATE_REVERSE_KL_TEACHER = """
{PROBLEM}

Here is a reference solution:
{EXPERT_SOLUTION}

After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:
"""

PROMPT_TEMPLATE_REVERSE_KL_STUDENT = """
{PROBLEM}
"""

PROMPT_TEMPLATE_FORWARD_KL_WITH_INITIAL_RESPONSE = """
Your task is to rewrite your mathematical solution using the reference solution as guidance.

**Problem:**
{PROBLEM}

**Reference Solution:**
{EXPERT_SOLUTION}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Review the reference solution to understand the target reasoning and method
2. Rewrite your solution so it is consistent with the reference solution
3. Keep useful parts of your original structure and style when appropriate
4. Output ONLY the rewritten solution
"""


def build_teacher_prompt(
    problem: str,
    expert_solution: str,
    *,
    initial_response: str = "",
    use_initial_response: bool = False,
) -> str:
    """Build the teacher-side prompt for either rewrite-only or rewrite-with-initial-response mode."""
    if use_initial_response:
        prompt = (
            PROMPT_TEMPLATE_FORWARD_KL_WITH_INITIAL_RESPONSE.replace("{PROBLEM}", problem)
            .replace("{INITIAL_RESPONSE}", initial_response)
            .replace("{EXPERT_SOLUTION}", expert_solution)
            .strip()
        )
    else:
        prompt = (
            PROMPT_TEMPLATE_REVERSE_KL_TEACHER.replace("{PROBLEM}", problem)
            .replace("{EXPERT_SOLUTION}", expert_solution)
            .strip()
        )
    return prompt + " " + instruction_following


_INIT_BLOCK_MARKER = "**Your Initial Solution:**"
_INIT_BLOCK_END_MARKER = "**Instructions:**"
_TRUNC_NOTICE = "\n[... initial solution truncated ...]\n"


class KLTrainingDataset(Dataset):
    """Builds student/teacher sequences for KL training under verl's no-padding format."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        kl_type: str = "reverse",
        max_length: int = 20480,
        max_samples: int | None = None,
        corrected_responses_path: str | None = None,
        use_initial_response: bool = False,
        prompt_truncation: bool = False,
        log_difficulty_buckets: bool = False,
    ):
        self.tokenizer = tokenizer
        self.kl_type = kl_type
        self.max_length = max_length
        self.use_initial_response = use_initial_response
        self.prompt_truncation = prompt_truncation
        self.log_difficulty_buckets = log_difficulty_buckets
        if prompt_truncation:
            print(
                "[KLTrainingDataset] prompt_truncation=ON: when prompt+response > max_length, "
                "the **Your Initial Solution:** block will be truncated from its tail to "
                "preserve the response (only applies to teacher-side correction prompts)."
            )

        print(f"Loading data from: {data_path}")
        self.data = datasets.load_dataset("parquet", data_files=data_path, split="train")

        self.corrected_responses = None
        if kl_type == "forward" and corrected_responses_path:
            print(f"Loading legacy rewrite targets from: {corrected_responses_path}")
            self.corrected_responses = datasets.load_dataset("parquet", data_files=corrected_responses_path, split="train")

        if max_samples is not None:
            print(f"Limiting to {max_samples} samples for testing")
            self.data = self.data.select(range(min(max_samples, len(self.data))))

        print(f"Dataset loaded: {len(self.data)} samples")

        # T4 sanity check: scan the reward column so a misconfigured run
        # (e.g. forward KL on un-backfilled stage2 parquet) fails LOUDLY
        # instead of silently routing every sample into the "hard" bucket.
        if self.log_difficulty_buckets:
            rewards = []
            missing = 0
            for ex in self.data:
                ei = ex.get("extra_info") or {}
                r = ei.get("reward") if isinstance(ei, dict) else None
                if r is None:
                    missing += 1
                else:
                    try:
                        rewards.append(float(r))
                    except (TypeError, ValueError):
                        missing += 1
            n = len(self.data)
            n_easy = sum(1 for r in rewards if r >= 1.0)
            n_hard = len(rewards) - n_easy
            print(
                f"[KLTrainingDataset] T4 difficulty buckets: "
                f"easy={n_easy} ({n_easy / n:.1%}), "
                f"hard={n_hard} ({n_hard / n:.1%}), "
                f"missing={missing} ({missing / n:.1%})"
            )
            if missing >= 0.5 * n:
                raise RuntimeError(
                    f"T4 (log_difficulty_buckets=True) requires extra_info.reward to be "
                    f"populated, but {missing}/{n} ({missing/n:.0%}) rows are missing it. "
                    f"\n  - For stage1 parquet: run "
                    f"`python -m recipe.kl_training.score_stage1_reward --parquet <path>`."
                    f"\n  - For stage2 parquet (forward KL): run "
                    f"`python -m recipe.kl_training.backfill_stage2_reward_from_stage1 "
                    f"--stage1_parquet <stage1.parquet> --stage2_parquet <stage2.parquet> "
                    f"--output <stage2_with_reward.parquet>` and pass DATA_PATH=<output>."
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # JSD trains on stage1 (student rollout) with reverse-KL style prompts
        # (student sees just the problem; teacher gets the expert-solution hint).
        if self.kl_type in ("reverse", "jsd"):
            return self._prepare_reverse_kl_item(idx)
        return self._prepare_forward_kl_item(idx)

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _truncate_initial_response(self, prompt: str, response: str) -> str:
        """Trim the **Your Initial Solution:** block so that the full prompt+response
        fits within ``max_length``. Returns the modified prompt; caller still applies
        a final right-side safety cut. Returns the prompt unchanged if (a) it already
        fits, (b) the markers are not present (e.g. student-side or non-correction
        prompt), or (c) trimming the block alone is insufficient.
        """
        prompt_ids = self._encode_text(prompt + "\n")
        response_ids = self._encode_text(response)
        if len(prompt_ids) + len(response_ids) <= self.max_length:
            return prompt

        init_start_marker = prompt.find(_INIT_BLOCK_MARKER)
        init_end_marker = prompt.find(_INIT_BLOCK_END_MARKER, init_start_marker + 1) if init_start_marker >= 0 else -1
        if init_start_marker < 0 or init_end_marker <= init_start_marker:
            return prompt

        block_text_start = init_start_marker + len(_INIT_BLOCK_MARKER)
        block_text = prompt[block_text_start:init_end_marker]
        block_ids = self._encode_text(block_text)
        if not block_ids:
            return prompt

        # Tokens we need to drop from the prompt side, leaving a small margin so
        # that the post-truncation re-encode still fits (re-tokenization can be
        # off by a few tokens vs. the slice arithmetic).
        margin = 16
        overflow = len(prompt_ids) + len(response_ids) - self.max_length + margin
        keep_tokens = max(0, len(block_ids) - overflow)
        if keep_tokens >= len(block_ids):
            return prompt  # nothing to do

        truncated_block = self.tokenizer.decode(block_ids[:keep_tokens], skip_special_tokens=True)
        new_prompt = (
            prompt[:block_text_start]
            + truncated_block
            + _TRUNC_NOTICE
            + prompt[init_end_marker:]
        )
        return new_prompt

    def _build_sequence(self, prompt: str, response: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.prompt_truncation:
            prompt = self._truncate_initial_response(prompt, response)
        prompt_ids = self._encode_text(prompt + "\n")
        response_ids = self._encode_text(response)
        full_ids = (prompt_ids + response_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        position_ids = torch.arange(len(full_ids), dtype=torch.long)
        return input_ids, position_ids, prompt_len

    def _build_prediction_mask(self, seq_len: int, prompt_len: int, response_len: int) -> torch.Tensor:
        mask = torch.zeros(seq_len, dtype=torch.long)
        if response_len <= 0 or seq_len == 0:
            return mask
        start = max(prompt_len - 1, 0)
        end = min(start + response_len, seq_len)
        mask[start:end] = 1
        return mask

    def _difficulty_bucket(self, item: dict) -> int:
        """Binary difficulty bucket from stage1 reward.

        Returns 1 if stage1 reward >= 1 ("easy" — base model solved it),
        else 0 ("hard"). Caller must guard with ``self.log_difficulty_buckets``
        (we validate at dataset init that reward is populated when the flag is
        on, so reward==None reaching here would be a bug).
        """
        ei = item.get("extra_info") or {}
        reward = ei.get("reward")
        if reward is None:
            return 0
        try:
            return 1 if float(reward) >= 1.0 else 0
        except (TypeError, ValueError):
            return 0

    def _prepare_item(self, student_prompt: str, teacher_prompt: str, response: str) -> dict[str, torch.Tensor]:
        student_input_ids, student_position_ids, student_prompt_len = self._build_sequence(student_prompt, response)
        teacher_input_ids, teacher_position_ids, teacher_prompt_len = self._build_sequence(teacher_prompt, response)

        student_response = student_input_ids[student_prompt_len:]
        teacher_response = teacher_input_ids[teacher_prompt_len:]
        common_len = min(student_response.numel(), teacher_response.numel())

        if common_len <= 0:
            raise ValueError("No aligned response tokens remain after prompt construction and truncation.")

        student_response = student_response[:common_len]
        teacher_response = teacher_response[:common_len]
        if not torch.equal(student_response, teacher_response):
            raise ValueError("Student and teacher response tokens diverged after truncation.")

        student_loss_mask = self._build_prediction_mask(
            seq_len=student_input_ids.numel(),
            prompt_len=student_prompt_len,
            response_len=common_len,
        )
        teacher_loss_mask = self._build_prediction_mask(
            seq_len=teacher_input_ids.numel(),
            prompt_len=teacher_prompt_len,
            response_len=common_len,
        )

        return {
            "student_input_ids": student_input_ids,
            "student_position_ids": student_position_ids,
            "student_loss_mask": student_loss_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_position_ids": teacher_position_ids,
            "teacher_loss_mask": teacher_loss_mask,
        }

    def _prepare_reverse_kl_item(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        extra_info = item.get("extra_info", {})
        problem = extra_info.get("problem", "")
        expert_solution = extra_info.get("expert_cot", "")

        responses = item.get("responses", [""])
        response = responses[0] if isinstance(responses, list) else responses
        if not response:
            raise ValueError("Reverse KL requires stage1 responses in the 'responses' field.")

        student_prompt = PROMPT_TEMPLATE_REVERSE_KL_STUDENT.replace("{PROBLEM}", problem).strip()
        student_prompt = student_prompt + " " + instruction_following

        teacher_prompt = build_teacher_prompt(
            problem,
            expert_solution,
            initial_response=response,
            use_initial_response=self.use_initial_response,
        )

        out = self._prepare_item(student_prompt=student_prompt, teacher_prompt=teacher_prompt, response=response)
        if self.log_difficulty_buckets:
            out["difficulty_bucket"] = torch.tensor([self._difficulty_bucket(item)], dtype=torch.long)
        return out

    def _prepare_forward_kl_item(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        extra_info = item.get("extra_info", {})
        problem = extra_info.get("problem", "")
        expert_solution = extra_info.get("expert_cot", "")
        initial_response = extra_info.get("initial_response", "")

        if self.corrected_responses is not None:
            corrected_item = self.corrected_responses[idx]
            if not initial_response:
                responses = item.get("responses", [""])
                initial_response = responses[0] if isinstance(responses, list) else responses
            rewritten_responses = corrected_item.get("responses", [""])
            target_response = rewritten_responses[0] if isinstance(rewritten_responses, list) else rewritten_responses
        else:
            rewritten_responses = item.get("responses", [""])
            target_response = rewritten_responses[0] if isinstance(rewritten_responses, list) else rewritten_responses
            if self.use_initial_response and not initial_response:
                raise ValueError(
                    "Forward KL single-file mode requires extra_info['initial_response'] in the stage2 parquet."
                )

        student_prompt = problem + " " + instruction_following
        teacher_prompt = build_teacher_prompt(
            problem,
            expert_solution,
            initial_response=initial_response,
            use_initial_response=self.use_initial_response,
        )

        out = self._prepare_item(
            student_prompt=student_prompt,
            teacher_prompt=teacher_prompt,
            response=target_response,
        )
        if self.log_difficulty_buckets:
            out["difficulty_bucket"] = torch.tensor([self._difficulty_bucket(item)], dtype=torch.long)
        return out


def create_kl_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    kl_type: str = "reverse",
    batch_size: int = 1,
    max_length: int = 20480,
    max_samples: int | None = None,
    corrected_responses_path: str | None = None,
    use_initial_response: bool = False,
    num_workers: int = 4,
    local_rank: int = -1,
    world_size: int = 1,
    prompt_truncation: bool = False,
    log_difficulty_buckets: bool = False,
) -> DataLoader:
    dataset = KLTrainingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        kl_type=kl_type,
        max_length=max_length,
        max_samples=max_samples,
        corrected_responses_path=corrected_responses_path,
        use_initial_response=use_initial_response,
        prompt_truncation=prompt_truncation,
        log_difficulty_buckets=log_difficulty_buckets,
    )

    sampler = None
    shuffle = True
    if local_rank != -1 and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=False,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=SFTTensorCollator(DatasetPadMode.NO_PADDING),
        num_workers=num_workers,
        pin_memory=True,
    )
