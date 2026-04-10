from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler
from torchdata.stateful_dataloader.stateful import Stateful


def _get_dataset_lengths(data_source) -> Sequence[int]:
    if hasattr(data_source, "get_sample_lengths"):
        return data_source.get_sample_lengths()
    if hasattr(data_source, "sample_lengths"):
        return data_source.sample_lengths
    raise AttributeError("Dataset must provide get_sample_lengths() or sample_lengths for length bucketing.")


class _StatefulLengthBucketSamplerIterator(Iterator[int], Stateful):
    _GENERATOR = "generator"
    _YIELDED = "yielded"

    def __init__(self, sampler: "LengthBucketSampler"):
        self.sampler = sampler
        self.generator_state = self.sampler.generator.get_state() if self.sampler.generator is not None else None
        self.yielded = 0
        self.next_yielded = None
        self.indices = self.sampler._build_index_order()

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.yielded >= len(self.indices):
            raise StopIteration
        idx = self.indices[self.yielded]
        self.yielded += 1
        return idx

    def state_dict(self) -> dict:
        state = {self._YIELDED: self.yielded}
        if self.generator_state is not None:
            state[self._GENERATOR] = self.generator_state
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        self.next_yielded = state_dict[self._YIELDED]
        generator_state = state_dict.get(self._GENERATOR)
        if generator_state is not None and self.sampler.generator is not None:
            self.generator_state = generator_state
            self.sampler.generator.set_state(generator_state)
        self.indices = self.sampler._build_index_order()
        self.yielded = self.next_yielded
        self.next_yielded = None


class LengthBucketSampler(Sampler[int]):
    """Length-aware sampler that keeps similarly-sized examples in the same batches."""

    def __init__(
        self,
        data_source,
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
        bucket_size_multiplier: int = 50,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if bucket_size_multiplier <= 0:
            raise ValueError(f"bucket_size_multiplier must be positive, got {bucket_size_multiplier}")

        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size_multiplier = bucket_size_multiplier
        self.lengths = _get_dataset_lengths(data_source)
        self.num_samples = len(self.lengths)

        self.generator = None
        if shuffle:
            if seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        return _StatefulLengthBucketSamplerIterator(self)

    def _build_index_order(self) -> list[int]:
        if self.shuffle:
            assert self.generator is not None
            shuffled = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            shuffled = list(range(self.num_samples))

        bucket_size = self.batch_size * self.bucket_size_multiplier
        ordered: list[int] = []
        for start in range(0, self.num_samples, bucket_size):
            bucket = shuffled[start : start + bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)

            if self.shuffle and len(bucket) > self.batch_size:
                batches = [bucket[i : i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]
                batch_order = torch.randperm(len(batches), generator=self.generator).tolist()
                for batch_idx in batch_order:
                    ordered.extend(batches[batch_idx])
            else:
                ordered.extend(bucket)

        return ordered
