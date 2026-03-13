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

PROMPT_TEMPLATE_FORWARD_KL_CORRECT = """
Your task is to correct your wrong mathematical solution using the expert solution as reference.

**Problem:**
{PROBLEM}

**Your Initial Solution (Wrong):**
{INITIAL_RESPONSE}

**Expert Solution (Correct):**
{EXPERT_SOLUTION}

**Correction Strategy:**
1. First, try to MINIMALLY EDIT your initial solution:
   - Keep your original structure, style, and flow
   - Only change specific wrong steps/numbers/equations
   - Preserve your original wording and explanations where correct

2. If minimal editing is NOT feasible (e.g., fundamental approach error):
   - Then rewrite using the expert solution's approach
   - But still try to maintain your original style and format

**Key Principles:**
- Prefer MINIMAL EDITS over complete rewrites
- Stay as close as possible to your original solution style
- Only use the expert solution to identify and fix specific errors
- Output ONLY the corrected solution, no meta-commentary

Please provide your corrected solution:
"""


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
    ):
        self.tokenizer = tokenizer
        self.kl_type = kl_type
        self.max_length = max_length
        self.use_initial_response = use_initial_response

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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.kl_type == "reverse":
            return self._prepare_reverse_kl_item(idx)
        return self._prepare_forward_kl_item(idx)

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_sequence(self, prompt: str, response: str) -> tuple[torch.Tensor, torch.Tensor, int]:
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

        teacher_prompt = (
            PROMPT_TEMPLATE_REVERSE_KL_TEACHER.replace("{PROBLEM}", problem)
            .replace("{EXPERT_SOLUTION}", expert_solution)
            .strip()
        )
        teacher_prompt = teacher_prompt + " " + instruction_following

        return self._prepare_item(student_prompt=student_prompt, teacher_prompt=teacher_prompt, response=response)

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
        if self.use_initial_response:
            teacher_prompt = (
                PROMPT_TEMPLATE_FORWARD_KL_CORRECT.replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
                .replace("{EXPERT_SOLUTION}", expert_solution)
                .strip()
            )
        else:
            teacher_prompt = (
                PROMPT_TEMPLATE_REVERSE_KL_TEACHER.replace("{PROBLEM}", problem)
                .replace("{EXPERT_SOLUTION}", expert_solution)
                .strip()
            )
        teacher_prompt = teacher_prompt + " " + instruction_following

        return self._prepare_item(
            student_prompt=student_prompt,
            teacher_prompt=teacher_prompt,
            response=target_response,
        )


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
) -> DataLoader:
    dataset = KLTrainingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        kl_type=kl_type,
        max_length=max_length,
        max_samples=max_samples,
        corrected_responses_path=corrected_responses_path,
        use_initial_response=use_initial_response,
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
