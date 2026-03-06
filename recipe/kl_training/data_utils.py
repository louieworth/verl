#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Data Preparation for KL Divergence Training
# Aligns with existing pipeline prompt format from stage1_prepare.py and stage2_prepare_v2.py

import os
import torch
import datasets
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Literal
from transformers import PreTrainedTokenizer

# Instruction following format from existing pipeline
instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


# ============================================================================
# Reverse KL Prompt Templates
# ============================================================================
# Reverse KL: Teacher sees expert solution as guidance, student sees only problem
# Goal: Learn to generate good reasoning directly (not correction)
#
# Two variants:
# 1. Without initial response (cleaner, teacher only sees expert solution)
# 2. With initial response (teacher sees both initial attempt and expert solution)

# Variant 1: Teacher sees only expert solution (no initial response)
PROMPT_TEMPLATE_REVERSE_KL_TEACHER_V1 = """
{PROBLEM}

Here is a reference solution:
{EXPERT_SOLUTION}

After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:
"""

# Variant 2: Teacher sees initial response + expert solution
# NOTE: This is now unified with Forward KL - both use the same "modify based on reference" format
PROMPT_TEMPLATE_REVERSE_KL_TEACHER_V2 = """
Your task is to modify your mathematical solution based on the reference solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Reference Solution:**
{EXPERT_SOLUTION}

**Instructions:**
1. Review the reference solution to understand the correct approach
2. Revise your initial solution to match the reference solution's method
3. Keep your original style and structure where possible
4. Output ONLY the modified solution."""

PROMPT_TEMPLATE_REVERSE_KL_STUDENT = """
{PROBLEM}
"""


# ============================================================================
# Forward KL Prompt Template
# ============================================================================
# Forward KL: Learn to modify solutions based on reference
# NOTE: Now unified with Reverse KL Variant 2 - same prompt format

PROMPT_TEMPLATE_FORWARD_KL = """
Your task is to modify your mathematical solution based on the reference solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Reference Solution:**
{EXPERT_SOLUTION}

**Instructions:**
1. Review the reference solution to understand the correct approach
2. Revise your initial solution to match the reference solution's method
3. Keep your original style and structure where possible
4. Output ONLY the modified solution."""

PROMPT_TEMPLATE_REVERSE_KL_STUDENT = """
{PROBLEM}
"""



class KLTrainingDataset(Dataset):
    """
    Dataset for KL Divergence Training.

    Supports two modes:
    1. Reverse KL: Teacher sees (problem + expert solution), Student sees (problem only)
       - Variant 1 (use_initial_response=False): Teacher sees only expert solution
       - Variant 2 (use_initial_response=True): Teacher sees initial response + expert solution
    2. Forward KL: Both see (problem + initial response + expert solution), learn to modify

    Expert solutions are read from extra_info['expert_cot'] in the data file.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        kl_type: Literal["reverse", "forward"] = "reverse",
        max_length: int = 20480,
        max_samples: Optional[int] = None,
        corrected_responses_path: Optional[str] = None,
        use_initial_response: bool = False,
    ):
        """
        Args:
            data_path: Path to stage1 generation results (parquet), contains expert_cot in extra_info
            tokenizer: Tokenizer for encoding text
            kl_type: "reverse" or "forward"
            max_length: Maximum sequence length
            max_samples: Limit number of samples (for testing)
            corrected_responses_path: Path to stage2 corrected responses (for forward KL)
            use_initial_response: For reverse KL, whether teacher sees initial response (Variant 2)
        """
        self.tokenizer = tokenizer
        self.kl_type = kl_type
        self.max_length = max_length
        self.use_initial_response = use_initial_response

        # Load dataset
        print(f"Loading data from: {data_path}")
        self.data = datasets.load_dataset("parquet", data_files=data_path, split="train")

        # For forward KL, load corrected responses
        if kl_type == "forward":
            if corrected_responses_path is None:
                raise ValueError("Forward KL requires corrected_responses_path")
            print(f"Loading corrected responses from: {corrected_responses_path}")
            self.corrected_responses = datasets.load_dataset("parquet", data_files=corrected_responses_path, split="train")
        else:
            self.corrected_responses = None

        # Limit samples if specified
        if max_samples is not None:
            print(f"Limiting to {max_samples} samples for testing")
            self.data = self.data.select(range(min(max_samples, len(self.data))))

        print(f"Dataset loaded: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
        - student_input_ids: Input for student model
        - student_attention_mask: Attention mask for student
        - teacher_input_ids: Input for teacher model
        - teacher_attention_mask: Attention mask for teacher
        - labels: Target labels (for forward KL, this is the corrected response)
        """
        if self.kl_type == "reverse":
            return self._prepare_reverse_kl_item(idx)
        else:
            return self._prepare_forward_kl_item(idx)

    def _prepare_reverse_kl_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Prepare data for Reverse KL training.

        Key insight: We need to compute KL divergence over the RESPONSE part,
        not the prompt part. So we:
        1. Build full sequences (prompt + response)
        2. Create labels that mask out the prompt part

        Two variants:
        - Variant 1 (use_initial_response=False): Teacher sees Problem + Expert Solution only
        - Variant 2 (use_initial_response=True): Teacher sees Problem + Initial Response + Expert Solution

        Student always sees: Problem only
        Response (y'): Expert solution (what we want student to learn)
        """
        item = self.data[idx]

        # Extract fields
        extra_info = item.get('extra_info', {})
        problem = extra_info.get('problem', '')
        expert_solution = extra_info.get('expert_cot', '')

        # The response we want student to learn is the expert solution
        response = expert_solution

        # Build student prompt (problem only + instruction)
        student_prompt = PROMPT_TEMPLATE_REVERSE_KL_STUDENT.replace("{PROBLEM}", problem).strip()
        student_prompt = student_prompt + " " + instruction_following

        # Build teacher prompt based on variant
        if self.use_initial_response:
            # Variant 2: Teacher sees initial response + expert solution
            responses = item.get('responses', [''])
            initial_response = responses[0] if isinstance(responses, list) else responses

            teacher_prompt = (
                PROMPT_TEMPLATE_REVERSE_KL_TEACHER_V2
                .replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
                .replace("{EXPERT_SOLUTION}", expert_solution)
                .strip()
            )
        else:
            # Variant 1: Teacher sees only expert solution
            teacher_prompt = (
                PROMPT_TEMPLATE_REVERSE_KL_TEACHER_V1
                .replace("{PROBLEM}", problem)
                .replace("{EXPERT_SOLUTION}", expert_solution)
                .strip()
            )

        teacher_prompt = teacher_prompt + " " + instruction_following

        # Build full sequences (prompt + response)
        student_full_text = student_prompt + "\n" + response
        teacher_full_text = teacher_prompt + "\n" + response

        # Tokenize full sequences
        student_encoding = self.tokenizer(
            student_full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        teacher_encoding = self.tokenizer(
            teacher_full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize just the prompts to find prompt lengths
        student_prompt_encoding = self.tokenizer(
            student_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        teacher_prompt_encoding = self.tokenizer(
            teacher_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        student_prompt_len = student_prompt_encoding["input_ids"].shape[1]
        teacher_prompt_len = teacher_prompt_encoding["input_ids"].shape[1]

        # Create labels: copy input_ids, but mask out prompt part with -100
        student_labels = student_encoding["input_ids"].clone().squeeze(0)
        student_labels[:student_prompt_len] = -100

        teacher_labels = teacher_encoding["input_ids"].clone().squeeze(0)
        teacher_labels[:teacher_prompt_len] = -100

        return {
            "student_input_ids": student_encoding["input_ids"].squeeze(0),
            "student_attention_mask": student_encoding["attention_mask"].squeeze(0),
            "student_labels": student_labels,
            "student_prompt_len": student_prompt_len,
            "teacher_input_ids": teacher_encoding["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher_encoding["attention_mask"].squeeze(0),
            "teacher_labels": teacher_labels,
            "teacher_prompt_len": teacher_prompt_len,
        }

    def _prepare_forward_kl_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Prepare data for Forward KL training.

        Key insight: We need to compute KL divergence over the RESPONSE part,
        not the prompt part. So we:
        1. Build full sequences (prompt + response)
        2. Create labels that mask out the prompt part

        Student sees: Problem only
        Teacher sees: Problem + Initial Response + Expert Solution (modification context)
        Response (y'): Corrected response
        """
        item = self.data[idx]
        corrected_item = self.corrected_responses[idx]

        # Extract fields
        extra_info = item.get('extra_info', {})
        problem = extra_info.get('problem', '')
        expert_solution = extra_info.get('expert_cot', '')

        # Get initial response (from stage1 generation)
        responses = item.get('responses', [''])
        initial_response = responses[0] if isinstance(responses, list) else responses

        # Get corrected response (target y')
        corrected_responses = corrected_item.get('responses', [''])
        corrected_response = corrected_responses[0] if isinstance(corrected_responses, list) else corrected_responses

        # Build student prompt (problem only + instruction)
        student_prompt = problem + " " + instruction_following

        # Build teacher prompt (modification context - unified with Reverse KL Variant 2)
        teacher_prompt = (
            PROMPT_TEMPLATE_FORWARD_KL
            .replace("{PROBLEM}", problem)
            .replace("{INITIAL_RESPONSE}", initial_response)
            .replace("{EXPERT_SOLUTION}", expert_solution)
            .strip()
        )
        teacher_prompt = teacher_prompt + " " + instruction_following

        # The response we want student to learn is the corrected response
        response = corrected_response

        # Build full sequences (prompt + response)
        student_full_text = student_prompt + "\n" + response
        teacher_full_text = teacher_prompt + "\n" + response

        # Tokenize full sequences
        student_encoding = self.tokenizer(
            student_full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        teacher_encoding = self.tokenizer(
            teacher_full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize just the prompts to find prompt lengths
        student_prompt_encoding = self.tokenizer(
            student_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        teacher_prompt_encoding = self.tokenizer(
            teacher_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        student_prompt_len = student_prompt_encoding["input_ids"].shape[1]
        teacher_prompt_len = teacher_prompt_encoding["input_ids"].shape[1]

        # Create labels: copy input_ids, but mask out prompt part with -100
        student_labels = student_encoding["input_ids"].clone().squeeze(0)
        student_labels[:student_prompt_len] = -100

        teacher_labels = teacher_encoding["input_ids"].clone().squeeze(0)
        teacher_labels[:teacher_prompt_len] = -100

        return {
            "student_input_ids": student_encoding["input_ids"].squeeze(0),
            "student_attention_mask": student_encoding["attention_mask"].squeeze(0),
            "student_labels": student_labels,
            "student_prompt_len": student_prompt_len,
            "teacher_input_ids": teacher_encoding["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher_encoding["attention_mask"].squeeze(0),
            "teacher_labels": teacher_labels,
            "teacher_prompt_len": teacher_prompt_len,
        }


def create_kl_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    kl_type: Literal["reverse", "forward"] = "reverse",
    batch_size: int = 4,
    max_length: int = 20480,
    max_samples: Optional[int] = None,
    corrected_responses_path: Optional[str] = None,
    use_initial_response: bool = False,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for KL training.

    Args:
        data_path: Path to stage1 generation results (contains expert_cot in extra_info)
        tokenizer: Tokenizer
        kl_type: "reverse" or "forward"
        batch_size: Batch size
        max_length: Maximum sequence length
        max_samples: Limit samples (for testing)
        corrected_responses_path: Path to corrected responses (for forward KL)
        use_initial_response: For reverse KL, whether teacher sees initial response (Variant 2)
        num_workers: Number of data loading workers

    Returns:
        DataLoader

    NOTE: Expert solutions are read from extra_info['expert_cot'] in the stage1 data,
          so no separate expert_solutions_path is needed.
    """
    dataset = KLTrainingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        kl_type=kl_type,
        max_length=max_length,
        max_samples=max_samples,
        corrected_responses_path=corrected_responses_path,
        use_initial_response=use_initial_response,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
