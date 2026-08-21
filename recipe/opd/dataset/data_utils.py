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
code_instruction_following = (
    "You will be given a programming problem. Write a correct Python program that solves it. "
    "Return only raw Python source code. Do not use Markdown fences or explanations."
)


def build_student_prompt(problem: str, task: str = "math") -> str:
    """Render the canonical plain Base-model prompt used for rollout and loss."""
    problem = problem.strip()
    suffix = code_instruction_following if task == "code" else instruction_following
    return f"{problem}\n\n{suffix}"



# =============================================================================
# Teacher / student prompt templates (training-time)
# =============================================================================
# The 4 teacher prompts form a 2×2 grid:
#
#                          | use_initial_response = False  | use_initial_response = True
#   ─────────────────────  | ───────────────────────────── | ─────────────────────────────────
#   distill_mode = "opsd"  | vanilla OPSD: π_T(·|x, y*)    | refine OPSD:  π_T(·|x, y*, y_o)
#   distill_mode = "opd"   | vanilla OPD:  π_T(·|x)        | refine OPD:   π_T(·|x, y_o)
#
# The student prompt is always just π_S(·|x) (problem only). All 4 teacher
# templates that embed initial_response share the same **Your Initial
# Solution:** / **Instructions:** markers so the prompt-truncation logic
# below works uniformly.
# =============================================================================


# Student prompt — used in all (distill_mode, use_initial_response) combos.
PROMPT_TEMPLATE_STUDENT = """
{PROBLEM}
"""

# vanilla OPSD: teacher sees problem + expert solution (no initial response)
# and rewrites the expert solution in its own words.
PROMPT_TEMPLATE_OPSD_VANILLA_TEACHER = """
Given the expert solution below, rewrite the mathematical solution in your own words while preserving the correct reasoning and final answer.

**Problem:**
{PROBLEM}

**Expert Solution:**
{EXPERT_SOLUTION}

**Instructions:**
1. Use the expert solution as the authoritative reasoning reference
2. Rewrite the complete solution clearly and correctly
3. Preserve the expert solution's final answer
4. Output ONLY the rewritten solution
"""

# refine OPSD: teacher sees problem + expert solution + initial response.
PROMPT_TEMPLATE_OPSD_REFINE_TEACHER = """
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

# vanilla OPD: teacher sees only the problem (no y*, no y_o).
PROMPT_TEMPLATE_OPD_VANILLA_TEACHER = """
{PROBLEM}
"""

# refine OPD: teacher sees problem + initial response (no expert reference).
# Same instruction shape as refine OPSD, minus the expert-anchor language.
PROMPT_TEMPLATE_OPD_REFINE_TEACHER = """
Your task is to rewrite your mathematical solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Preserve the overall structure and reasoning path of your original solution
2. Identify and fix errors in computation or logic
3. Keep correct intermediate steps and meaningful work
4. Output ONLY the rewritten solution
"""



PROMPT_TEMPLATE_CODE_STUDENT = """
{PROBLEM}
"""

PROMPT_TEMPLATE_CODE_OPSD_VANILLA_TEACHER = """
{PROBLEM}

Here is a reference Python solution:
```python
{EXPERT_SOLUTION}
```

After understanding the reference solution, write your own correct Python solution below.
"""

PROMPT_TEMPLATE_CODE_OPSD_REFINE_TEACHER = """
Your task is to rewrite your Python solution using the reference solution as guidance.

**Problem:**
{PROBLEM}

**Reference Solution:**
```python
{EXPERT_SOLUTION}
```

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Fix correctness issues and edge cases
2. Preserve useful parts of the original approach when appropriate
3. Output ONLY the rewritten Python solution
"""

PROMPT_TEMPLATE_CODE_OPD_VANILLA_TEACHER = """
{PROBLEM}
"""

PROMPT_TEMPLATE_CODE_OPD_REFINE_TEACHER = """
Your task is to rewrite your Python solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Fix correctness issues and edge cases
2. Preserve useful parts of the original approach when appropriate
3. Output ONLY the rewritten Python solution
"""

def build_teacher_prompt(
    problem: str,
    expert_solution: str,
    *,
    initial_response: str = "",
    use_initial_response: bool = False,
    distill_mode: str = "opsd",
    task: str = "math",
) -> str:
    """Build the teacher-side prompt by (distill_mode × use_initial_response):

        opsd + False → vanilla OPSD: π_T(·|x, y*)
        opsd + True  → refine  OPSD: π_T(·|x, y*, y_o)
        opd  + False → vanilla OPD:  π_T(·|x)
        opd  + True  → refine  OPD:  π_T(·|x, y_o)
    """
    if task == "code":
        suffix = code_instruction_following
        if distill_mode == "opd":
            if use_initial_response:
                prompt = (
                    PROMPT_TEMPLATE_CODE_OPD_REFINE_TEACHER
                    .replace("{PROBLEM}", problem)
                    .replace("{INITIAL_RESPONSE}", initial_response)
                    .strip()
                )
            else:
                prompt = PROMPT_TEMPLATE_CODE_OPD_VANILLA_TEACHER.replace("{PROBLEM}", problem).strip()
        else:
            if use_initial_response:
                prompt = (
                    PROMPT_TEMPLATE_CODE_OPSD_REFINE_TEACHER
                    .replace("{PROBLEM}", problem)
                    .replace("{INITIAL_RESPONSE}", initial_response)
                    .replace("{EXPERT_SOLUTION}", expert_solution)
                    .strip()
                )
            else:
                prompt = (
                    PROMPT_TEMPLATE_CODE_OPSD_VANILLA_TEACHER
                    .replace("{PROBLEM}", problem)
                    .replace("{EXPERT_SOLUTION}", expert_solution)
                    .strip()
                )
    elif distill_mode == "opd":
        suffix = instruction_following
        if use_initial_response:
            prompt = (
                PROMPT_TEMPLATE_OPD_REFINE_TEACHER
                .replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
                .strip()
            )
        else:
            prompt = PROMPT_TEMPLATE_OPD_VANILLA_TEACHER.replace("{PROBLEM}", problem).strip()
    else:  # opsd
        suffix = instruction_following
        if use_initial_response:
            prompt = (
                PROMPT_TEMPLATE_OPSD_REFINE_TEACHER
                .replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
                .replace("{EXPERT_SOLUTION}", expert_solution)
                .strip()
            )
        else:
            prompt = (
                PROMPT_TEMPLATE_OPSD_VANILLA_TEACHER
                .replace("{PROBLEM}", problem)
                .replace("{EXPERT_SOLUTION}", expert_solution)
                .strip()
            )
    return f"{prompt.strip()}\n\n{suffix}"


_INIT_BLOCK_MARKER = "**Your Initial Solution:**"
_INIT_BLOCK_END_MARKER = "**Instructions:**"
_TRUNC_NOTICE = "\n[... initial solution truncated ...]\n"
_REF_TRUNC_NOTICE = "\n[... reference solution truncated ...]\n"


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
        distill_mode: str = "opsd",
        task: str = "math",
    ):
        self.tokenizer = tokenizer
        self.kl_type = kl_type
        self.max_length = max_length
        self.use_initial_response = use_initial_response
        self.prompt_truncation = prompt_truncation
        self.log_difficulty_buckets = log_difficulty_buckets
        self.distill_mode = distill_mode
        self.task = task
        if prompt_truncation:
            print(
                "[KLTrainingDataset] prompt_truncation=ON: when prompt+response > max_length, "
                "long **Your Initial Solution:** and reference-solution blocks will be truncated "
                "from their tail to preserve response tokens."
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
                    f"`python -m recipe.opd.dataset.score_stage1_reward --parquet <path>`."
                    f"\n  - For stage2 parquet (forward KL): run "
                    f"`python -m recipe.opd.dataset.backfill_stage2_reward_from_stage1 "
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

    def _truncate_prompt_block(
        self,
        prompt: str,
        response: str,
        start_marker: str,
        end_markers: tuple[str, ...],
        notice: str,
    ) -> str:
        prompt_ids = self._encode_text(prompt + "\n")
        response_ids = self._encode_text(response)
        if len(prompt_ids) + len(response_ids) <= self.max_length:
            return prompt

        start = prompt.find(start_marker)
        if start < 0:
            return prompt
        block_text_start = start + len(start_marker)
        end_candidates = [prompt.find(marker, block_text_start) for marker in end_markers]
        end_candidates = [pos for pos in end_candidates if pos > block_text_start]
        end = min(end_candidates) if end_candidates else -1
        if end <= block_text_start:
            return prompt

        block_text = prompt[block_text_start:end]
        block_ids = self._encode_text(block_text)
        if not block_ids:
            return prompt

        margin = 16
        overflow = len(prompt_ids) + len(response_ids) - self.max_length + margin
        keep_tokens = max(0, len(block_ids) - overflow)
        if keep_tokens >= len(block_ids):
            return prompt

        truncated_block = self.tokenizer.decode(block_ids[:keep_tokens], skip_special_tokens=True)
        return prompt[:block_text_start] + truncated_block + notice + prompt[end:]

    def _truncate_initial_response(self, prompt: str, response: str) -> str:
        return self._truncate_prompt_block(
            prompt,
            response,
            _INIT_BLOCK_MARKER,
            (_INIT_BLOCK_END_MARKER,),
            _TRUNC_NOTICE,
        )

    def _truncate_reference_solution(self, prompt: str, response: str) -> str:
        prompt = self._truncate_prompt_block(
            prompt,
            response,
            "**Reference Solution:**",
            (_INIT_BLOCK_MARKER, _INIT_BLOCK_END_MARKER),
            _REF_TRUNC_NOTICE,
        )
        prompt = self._truncate_prompt_block(
            prompt,
            response,
            "Here is a reference Python solution:",
            ("After understanding",),
            _REF_TRUNC_NOTICE,
        )
        return self._truncate_prompt_block(
            prompt,
            response,
            "Here is a reference solution:",
            ("After understanding",),
            _REF_TRUNC_NOTICE,
        )

    def _build_sequence(self, prompt: str, response: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.prompt_truncation:
            prompt = self._truncate_initial_response(prompt, response)
            prompt = self._truncate_reference_solution(prompt, response)
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

        if self.task == "code":
            student_prompt = build_student_prompt(problem, task="code")
        else:
            student_prompt = build_student_prompt(problem, task="math")

        teacher_prompt = build_teacher_prompt(
            problem,
            expert_solution,
            initial_response=response,
            use_initial_response=self.use_initial_response,
            distill_mode=self.distill_mode,
            task=self.task,
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

        if self.task == "code":
            student_prompt = build_student_prompt(problem, task="code")
        else:
            student_prompt = build_student_prompt(problem, task="math")
        teacher_prompt = ""
        if extra_info.get("teacher_prompt_contract") == "srd_generation_prompt_v1":
            raw_prompt = item.get("prompt")
            if hasattr(raw_prompt, "tolist"):
                raw_prompt = raw_prompt.tolist()
            if isinstance(raw_prompt, (list, tuple)) and len(raw_prompt) == 1:
                message = raw_prompt[0]
                if isinstance(message, dict) and message.get("role") == "user":
                    teacher_prompt = message.get("content", "")
            if not isinstance(teacher_prompt, str) or not teacher_prompt.strip():
                raise ValueError("TRD row is missing its exact generation teacher prompt")
        else:
            teacher_prompt = build_teacher_prompt(
                problem,
                expert_solution,
                initial_response=initial_response,
                use_initial_response=self.use_initial_response,
                distill_mode=self.distill_mode,
                task=self.task,
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
    distill_mode: str = "opsd",
    task: str = "math",
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
        distill_mode=distill_mode,
        task=task,
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
