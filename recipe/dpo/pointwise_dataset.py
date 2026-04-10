import copy
import os
from contextlib import contextmanager

import datasets
import numpy as np
import pyarrow as pa
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask
from verl.utils.tokenizer import normalize_token_ids

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


@contextmanager
def suppress_hf_datasets_progress():
    disable_fn = getattr(datasets, "disable_progress_bars", None)
    enable_fn = getattr(datasets, "enable_progress_bars", None)

    if disable_fn is None or enable_fn is None:
        yield
        return

    disable_fn()
    try:
        yield
    finally:
        enable_fn()


class PointwiseDPODataset(Dataset):
    """Static point-wise preference dataset for Prospect-DPO."""

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
        self.response_key = config.get("response_key", "response")
        self.label_key = config.get("label_key", "label")
        self.s_dwell_key = config.get("s_dwell_key", "s_dwell")
        self.p_ctr_key = config.get("p_ctr_key", "p_ctr")
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
        self.reference_logps_key = config.get("reference_logps_key", "reference_logps")
        self._has_reference_logps = False

        self._download()
        self._read_files()

    def _download(self):
        from verl.utils.fs import copy_to_local

        for i, data_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=data_file, cache_dir=self.cache_dir, use_shm=False)

    def _read_files(self):
        dataframes = []
        total_rows = 0
        with suppress_hf_datasets_progress():
            progress = tqdm(
                self.data_files,
                desc="Loading offline dataset",
                total=len(self.data_files),
                unit="file",
                dynamic_ncols=True,
            )
            for file_idx, data_file in enumerate(progress, start=1):
                file_name = os.path.basename(data_file)
                rows_hint = self._maybe_get_rows_hint(data_file)
                progress.set_postfix(current=file_name, rows_hint=rows_hint or "?")
                if data_file.endswith(".parquet"):
                    dataframe = self._load_parquet_with_progress(
                        data_file,
                        file_idx=file_idx,
                        total_files=len(self.data_files),
                        file_name=file_name,
                        rows_hint=rows_hint,
                    )
                else:
                    dataframe = self._load_data_file(data_file)
                file_rows = len(dataframe)
                total_rows += file_rows
                progress.set_postfix(file_rows=file_rows, total_rows=total_rows)
                dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        self._has_reference_logps = self.reference_logps_key in self.dataframe.column_names
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

    def _load_data_file(self, data_file: str) -> datasets.Dataset:
        if data_file.endswith(".json") or data_file.endswith(".jsonl"):
            return datasets.load_dataset("json", data_files=data_file)["train"]
        raise ValueError(f"Unsupported file format: {data_file}")

    def _load_parquet_with_progress(
        self,
        data_file: str,
        *,
        file_idx: int,
        total_files: int,
        file_name: str,
        rows_hint: int | None,
    ) -> datasets.Dataset:
        if pq is None:
            return datasets.load_dataset("parquet", data_files=data_file)["train"]

        parquet = pq.ParquetFile(data_file)
        tables: list[pa.Table] = []
        rows_read = 0
        progress = tqdm(
            range(parquet.num_row_groups),
            desc=f"Loading parquet {file_idx}/{total_files}",
            total=parquet.num_row_groups,
            unit="group",
            dynamic_ncols=True,
            leave=False,
        )
        for row_group_idx in progress:
            table = parquet.read_row_group(row_group_idx)
            tables.append(table)
            rows_read += table.num_rows
            progress.set_postfix(file=file_name, rows=rows_read, rows_hint=rows_hint or parquet.metadata.num_rows)
        progress.close()
        return datasets.Dataset(pa.concat_tables(tables))

    def _maybe_get_rows_hint(self, data_file: str) -> int | None:
        if not data_file.endswith(".parquet") or pq is None:
            return None
        try:
            return pq.ParquetFile(data_file).metadata.num_rows
        except Exception:
            return None

    def __len__(self):
        return len(self.dataframe)

    def has_reference_logps(self) -> bool:
        return self._has_reference_logps

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

    def _build_response_tensors(self, prompt_ids: list[int], response_ids: list[int]) -> dict[str, torch.Tensor]:
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
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_tensor,
            "response_mask": response_mask,
            "labels": labels,
        }

    def __getitem__(self, item):
        row = self.dataframe[item]
        prompt = row[self.prompt_key]
        prompt_ids = self._tokenize_prompt(prompt)
        response_ids = self._tokenize_response(row[self.response_key])

        sample = self._build_response_tensors(prompt_ids, response_ids)
        sample["label"] = torch.tensor(float(row[self.label_key]), dtype=torch.float32)
        sample["s_dwell"] = torch.tensor(float(row[self.s_dwell_key]), dtype=torch.float32)
        sample["p_ctr"] = torch.tensor(float(row[self.p_ctr_key]), dtype=torch.float32)
        if self.reference_logps_key in row:
            reference_logps = row[self.reference_logps_key]
            if reference_logps is not None and not (
                isinstance(reference_logps, (float, np.floating)) and np.isnan(reference_logps)
            ):
                sample["reference_logps"] = torch.tensor(float(reference_logps), dtype=torch.float32)

        if "data_source" in row:
            sample["data_source"] = row["data_source"]
        if self.return_raw_chat and isinstance(prompt, list):
            sample["raw_prompt"] = prompt

        return sample
