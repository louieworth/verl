"""Plain completion adapters for Qwen Base-model training.

The default verl RL/SFT datasets render message lists with a chat template.
OPD experiments in this recipe intentionally train Base checkpoints with the
same plain-text prompt used by the KL pipeline and external evaluators.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.profiler import simple_timer


def render_plain_prompt(messages: str | list[dict[str, Any]]) -> str:
    """Return a single Base-model prompt without chat-template tokens."""
    if isinstance(messages, str):
        text = messages
    else:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("Base completion prompt must be a non-empty string or message list")
        unsupported = [message.get("role") for message in messages if message.get("role") != "user"]
        if unsupported:
            raise ValueError(
                "Base completion rollout accepts user prompt content only; "
                f"found unsupported roles: {unsupported}"
            )
        contents = [message.get("content") for message in messages]
        if any(not isinstance(content, str) for content in contents):
            raise TypeError("Base completion prompt content must be text")
        text = "\n".join(contents)
    text = text.strip()
    if not text:
        raise ValueError("Base completion prompt is empty")
    # One explicit separator is part of the training contract. KL and SFT
    # tokenize ``prompt + \"\\n\"`` before the target, so rollout must sample
    # from the identical prefix rather than from the stripped prompt.
    return text + "\n"


@register("base_completion_agent")
class BaseCompletionAgentLoop(SingleTurnAgentLoop):
    """Single-turn rollout that tokenizes the prompt directly."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        if kwargs.get("tools") or kwargs.get("images") or kwargs.get("videos"):
            raise ValueError("BaseCompletionAgentLoop does not support tools or multimodal inputs")

        prompt = render_plain_prompt(kwargs["raw_prompt"])
        prompt_ids = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.tokenizer.encode(prompt, add_special_tokens=False),
        )
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"Rendered Base prompt has {len(prompt_ids)} tokens, exceeding cap {self.prompt_length}"
            )

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        metrics.setdefault("num_preempted", output.num_preempted if output.num_preempted is not None else -1)
        response_ids = output.token_ids[: self.response_length]
        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
        )
        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result


class BaseCompletionSFTDataset(Dataset):
    """SFT dataset for ``prompt``/``response`` plain-completion parquet rows."""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        if processor is not None:
            raise ValueError("BaseCompletionSFTDataset is text-only")
        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]
        local_files = [copy_local_path_from_hdfs(path, verbose=True) for path in parquet_files]
        self.dataframe = pd.concat([pd.read_parquet(Path(path)) for path in local_files], ignore_index=True)
        if max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]
        required = {"prompt", "response"}
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Base completion SFT parquet is missing columns: {sorted(missing)}")
        self.tokenizer = tokenizer
        self.max_length = int(config.get("max_length", 18432))
        self.pad_mode = config.get("pad_mode", DatasetPadMode.NO_PADDING)
        self.truncation = config.get("truncation", "error")
        if self.pad_mode not in (DatasetPadMode.NO_PADDING, DatasetPadMode.RIGHT, "no_padding", "right"):
            raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")
        if self.truncation != "error":
            raise ValueError("Base completion SFT requires truncation=error; filter data during preparation")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[item]
        prompt = str(row["prompt"]).strip()
        response = str(row["response"]).strip()
        if not prompt or not response:
            raise ValueError(f"Empty prompt/response at SFT row {item}")
        prompt_ids = self.tokenizer.encode(prompt + "\n", add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            response_ids.append(self.tokenizer.eos_token_id)
        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        if input_ids.numel() > self.max_length:
            raise ValueError(
                f"SFT row {item} has {input_ids.numel()} tokens, exceeding max_length={self.max_length}"
            )
        loss_mask = torch.cat(
            [torch.zeros(len(prompt_ids), dtype=torch.long), torch.ones(len(response_ids), dtype=torch.long)]
        )
        position_ids = torch.arange(input_ids.numel(), dtype=torch.long)
        if self.pad_mode in (DatasetPadMode.RIGHT, "right") and input_ids.numel() < self.max_length:
            pad = self.max_length - input_ids.numel()
            pad_id = self.tokenizer.pad_token_id or 0
            attention_mask = torch.cat(
                [torch.ones(input_ids.numel(), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]
            )
            input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=pad_id)
            loss_mask = torch.nn.functional.pad(loss_mask, (0, pad), value=0)
            position_ids = torch.nn.functional.pad(position_ids, (0, pad), value=0)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}
