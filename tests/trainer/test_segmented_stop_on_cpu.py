# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import inspect
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import _validate_stop_at_step as validate_ppo_stop
from verl.trainer.sft_trainer import SFTTrainer
from verl.trainer.sft_trainer import _validate_stop_at_step as validate_sft_stop


@pytest.mark.parametrize("validator", [validate_ppo_stop, validate_sft_stop])
def test_stop_at_step_validation(validator):
    assert validator(None, resumed_global_step=3, total_training_steps=10) is None
    assert validator(8, resumed_global_step=3, total_training_steps=10) == 8
    assert validator(3, resumed_global_step=3, total_training_steps=10) == 3

    with pytest.raises(ValueError, match="exceeds total_training_steps"):
        validator(11, resumed_global_step=3, total_training_steps=10)
    with pytest.raises(ValueError, match="behind resumed global step"):
        validator(2, resumed_global_step=3, total_training_steps=10)

    for invalid_stop in (0, -1, True, 1.0, 1.5, "three"):
        with pytest.raises(ValueError):
            validator(invalid_stop, resumed_global_step=0, total_training_steps=10)


class _FakeEngine:
    def is_mp_src_rank_with_outputs(self):
        return False


class _FakeTrainingClient:
    def __init__(self):
        self.trained_batches = 0

    def train_batch(self, data):
        self.trained_batches += 1
        return None

    def start_profile(self):
        raise AssertionError("profiling should not start in this test")

    def stop_profile(self):
        raise AssertionError("profiling should not stop in this test")


class _FakeSampler:
    def set_epoch(self, epoch):
        self.epoch = epoch


class _FakeCheckpointHandler:
    def __init__(self):
        self.saved_steps = []

    def save_checkpoint(self, step):
        self.saved_steps.append(step)


def _build_sft_trainer(*, resumed_step: int, stop_at_step: int):
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"stop_at_step": stop_at_step, "total_epochs": 5},
            "model": {"use_remove_padding": False},
            "data": {
                "use_dynamic_bsz": False,
                "max_token_len_per_gpu": 32,
                "micro_batch_size_per_gpu": 1,
                "pad_mode": "right",
            },
        }
    )
    trainer.engine = _FakeEngine()
    trainer.resume_global_step = resumed_step
    trainer.total_training_steps = 8
    trainer.steps_per_epoch = 2
    trainer.train_sampler = _FakeSampler()
    trainer.train_dataloader = [{"sample": 1}, {"sample": 2}]
    trainer.val_dataloader = None
    trainer.model_config = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    trainer.global_batch_size = 1
    trainer.device_name = "cpu"
    trainer.start_profile_step = -1
    trainer.end_profile_step = -1
    trainer.save_freq = -1
    trainer.test_freq = -1
    trainer.rank = 0
    trainer.training_client = _FakeTrainingClient()
    trainer.ckpt_handler = _FakeCheckpointHandler()
    trainer._get_batch_seqlens = lambda data: [1]
    return trainer


def test_sft_segment_endpoint_saves_and_returns_without_shortening_schedule(monkeypatch):
    import verl.trainer.sft_trainer as sft_module

    monkeypatch.setattr(sft_module, "aggressive_empty_cache", lambda **kwargs: None)
    monkeypatch.setattr(sft_module, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(sft_module, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(sft_module, "NonTensorData", lambda value: value)
    monkeypatch.setattr(sft_module.tu, "get_tensordict", lambda tensor_dict, non_tensor_dict: tensor_dict)
    monkeypatch.setattr(sft_module.tu, "assign_non_tensor", lambda data, **kwargs: None)

    trainer = _build_sft_trainer(resumed_step=2, stop_at_step=3)
    trainer.fit()

    assert trainer.training_client.trained_batches == 1
    assert trainer.ckpt_handler.saved_steps == [3]
    assert trainer.total_training_steps == 8


def test_sft_resume_at_segment_endpoint_is_a_noop(monkeypatch):
    import verl.trainer.sft_trainer as sft_module

    monkeypatch.setattr(sft_module, "log_with_rank", lambda *args, **kwargs: None)
    trainer = _build_sft_trainer(resumed_step=3, stop_at_step=3)

    trainer.fit()

    assert trainer.training_client.trained_batches == 0
    assert trainer.ckpt_handler.saved_steps == []


def test_ppo_segment_checkpoint_releases_rollout_after_metrics():
    source = inspect.getsource(RayPPOTrainer.fit)

    collect_metrics = source.index("metrics.update(compute_data_metrics")
    release_batch = source.index("del batch, batch_dict")
    terminal_save = source.index(
        'with marked_timer("save_checkpoint", timing_raw, color="green"):',
        release_batch,
    )

    assert collect_metrics < release_batch < terminal_save
