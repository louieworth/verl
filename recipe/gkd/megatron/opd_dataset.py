# Copyright 2025 Individual Contributor: furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
from torch.utils.data import Dataset

from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.torch_functional import postprocess_data


class TokenizedPromptDataset(Dataset):
    """Compatibility adapter for the Megatron GKD recipe.

    Current verl RLHFDataset returns raw chat prompts and leaves tokenization to
    newer agent-loop rollout paths. recipe/gkd/megatron still uses the older
    tensorized prompt contract, so this wrapper adds input_ids, attention_mask,
    position_ids, and raw_prompt_ids while preserving the original row fields.
    """

    def __init__(self, dataset: Dataset, tokenizer, config, opd_config=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.opd_config = opd_config or {}
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.instruction_following = str(self.opd_config.get("instruction_following", "")).strip()
        self.append_instruction_to_prompt = bool(self.opd_config.get("append_instruction_to_prompt", False))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _append_instruction(self, content: str) -> str:
        if not self.append_instruction_to_prompt or not self.instruction_following:
            return content
        if self.instruction_following in content:
            return content
        return content.rstrip() + " " + self.instruction_following

    def _format_raw_prompt(self, raw_prompt):
        if not self.append_instruction_to_prompt:
            return raw_prompt
        if isinstance(raw_prompt, str):
            return self._append_instruction(raw_prompt)
        if isinstance(raw_prompt, list):
            formatted = [dict(message) for message in raw_prompt]
            target_idx = None
            for i, message in enumerate(formatted):
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    target_idx = i
            if target_idx is None:
                return raw_prompt
            formatted[target_idx]["content"] = self._append_instruction(formatted[target_idx]["content"])
            return formatted
        return raw_prompt

    def __len__(self):
        return len(self.dataset)

    def _tokenize_prompt(self, raw_prompt):
        if isinstance(raw_prompt, list):
            apply_kwargs = dict(self.apply_chat_template_kwargs)
            apply_kwargs.pop("tokenize", None)
            apply_kwargs.pop("return_dict", None)
            apply_kwargs.pop("return_tensors", None)
            token_ids = self.tokenizer.apply_chat_template(
                raw_prompt,
                add_generation_prompt=True,
                tokenize=True,
                **apply_kwargs,
            )
            token_ids = normalize_token_ids(token_ids)
        elif isinstance(raw_prompt, str):
            token_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        else:
            raise TypeError(f"Unsupported prompt type for Megatron GKD: {type(raw_prompt)}")
        return token_ids

    def __getitem__(self, idx):
        row = dict(self.dataset[idx])
        raw_prompt = row.get("raw_prompt")
        if raw_prompt is None:
            raw_prompt = row.get(self.config.get("prompt_key", "prompt"))
        raw_prompt = self._format_raw_prompt(raw_prompt)
        row["raw_prompt"] = raw_prompt
        token_ids = self._tokenize_prompt(raw_prompt)
        raw_prompt_ids = list(token_ids)

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        input_ids, attention_mask = postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        attention_mask = attention_mask.to(torch.bool)
        position_ids = compute_position_id_with_mask(attention_mask)

        row["input_ids"] = input_ids.squeeze(0)
        row["attention_mask"] = attention_mask.squeeze(0)
        row["position_ids"] = position_ids.squeeze(0)
        row["raw_prompt_ids"] = raw_prompt_ids
        return row
