# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer import normalize_token_ids


class DPOPairDataset(Dataset):
    """Static pairwise preference dataset for offline DPO."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.config = config
        self.processor = processor
        self.max_samples = max_samples

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/dpo"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.chosen_key = config.get("chosen_key", "chosen")
        self.rejected_key = config.get("rejected_key", "rejected")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 1024)
        self.add_eos = config.get("add_eos", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.prompt_truncation = config.get("prompt_truncation", "left")
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        self._download()
        self._read_files()

    def _download(self):
        from verl.utils.fs import copy_to_local

        for i, data_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=data_file, cache_dir=self.cache_dir, use_shm=False)

    def _read_files(self):
        dataframes = []
        for data_file in self.data_files:
            if data_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
            elif data_file.endswith(".json") or data_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=data_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} samples out of {total}")

    def __len__(self):
        return len(self.dataframe)

    def _tokenize_prompt(self, prompt) -> list[int]:
        if isinstance(prompt, list):
            apply_kwargs = dict(self.apply_chat_template_kwargs)
            tokenized_prompt = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                **apply_kwargs,
            )
            token_ids = normalize_token_ids(tokenized_prompt)
        elif isinstance(prompt, str):
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")
        return self._truncate_prompt(token_ids)

    def _tokenize_response(self, response: str) -> list[int]:
        if not isinstance(response, str):
            raise TypeError(f"Unsupported response type: {type(response)}")
        token_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        if self.add_eos and self.eos_token_id is not None:
            token_ids = token_ids + [self.eos_token_id]
        return token_ids[: self.max_response_length]

    def _truncate_prompt(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) <= self.max_prompt_length:
            return token_ids
        if self.prompt_truncation == "left":
            return token_ids[-self.max_prompt_length :]
        if self.prompt_truncation == "right":
            return token_ids[: self.max_prompt_length]
        raise ValueError(f"Unsupported prompt_truncation: {self.prompt_truncation}")

    def _left_pad(self, token_ids: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
        pad_length = length - len(token_ids)
        padded = [self.pad_token_id] * pad_length + token_ids
        mask = [0] * pad_length + [1] * len(token_ids)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def _right_pad(self, token_ids: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
        pad_length = length - len(token_ids)
        padded = token_ids + [self.pad_token_id] * pad_length
        mask = [1] * len(token_ids) + [0] * pad_length
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def _build_pair_tensors(self, prompt_ids: list[int], response_ids: list[int], prefix: str) -> dict[str, torch.Tensor]:
        prompt_tensor, prompt_mask = self._left_pad(prompt_ids, self.max_prompt_length)
        response_tensor, response_mask = self._right_pad(response_ids, self.max_response_length)
        input_ids = torch.cat([prompt_tensor, response_tensor], dim=0)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=0)
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0].to(torch.long)

        labels = torch.full((self.max_prompt_length + self.max_response_length,), -100, dtype=torch.long)
        valid_response_tokens = int(response_mask.sum().item())
        if valid_response_tokens > 0:
            labels[self.max_prompt_length : self.max_prompt_length + valid_response_tokens] = response_tensor[
                :valid_response_tokens
            ]

        return {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attention_mask": attention_mask,
            f"{prefix}_position_ids": position_ids,
            f"{prefix}_responses": response_tensor,
            f"{prefix}_response_mask": response_mask,
            f"{prefix}_labels": labels,
        }

    def __getitem__(self, item):
        row = self.dataframe[item]
        prompt = row[self.prompt_key]
        prompt_ids = self._tokenize_prompt(prompt)
        chosen_ids = self._tokenize_response(row[self.chosen_key])
        rejected_ids = self._tokenize_response(row[self.rejected_key])

        sample = {}
        sample.update(self._build_pair_tensors(prompt_ids, chosen_ids, "chosen"))
        sample.update(self._build_pair_tensors(prompt_ids, rejected_ids, "rejected"))

        if "data_source" in row:
            sample["data_source"] = row["data_source"]
        if self.return_raw_chat and isinstance(prompt, list):
            sample["raw_prompt"] = prompt

        return sample
