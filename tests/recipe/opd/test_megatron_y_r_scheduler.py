# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from recipe.gkd.megatron.ray_trainer import OnPolicyDistillTrainer


class _ReadyFuture:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def test_three_step_off_scheduler_streams_samples_and_packs_actor_batches():
    trainer = object.__new__(OnPolicyDistillTrainer)
    trainer.opd_y_mode = "y_r"
    trainer._policy_version = 0
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 8},
            "trainer": {
                "actor_update_batch_size": 4,
                "teacher_inflight_samples": 8,
            },
            "actor_rollout_ref": {
                "actor": {"gradient_accumulation_steps": 2},
                "teacher": {"request_batch_size": 4},
            },
        }
    )
    sync_calls = []
    seen_samples = []
    packed_samples = []

    def sync_rollout_weights_if_pending(timing):
        timing["sync_rollout_weights"] = 0.25
        sync_calls.append("sync")

    def finish_sample(item, epoch, batch_dict, ready_queue, request_chunk_size, rollout_worker_index):
        assert request_chunk_size == 4
        del rollout_worker_index
        sample_id = int(batch_dict["id"].item())
        seen_samples.append(sample_id)
        item.completed_at = item.submitted_at + 1.0
        sample = SimpleNamespace(epoch=epoch, sample_id=sample_id, timing={"generate_y_r_and_teacher_topk": 0.5})
        ready_queue.put((item, sample, None))

    def pack_samples(records, **kwargs):
        del kwargs
        ids = [record[1].sample_id for record in records]
        packed_samples.append(ids)
        teacher = SimpleNamespace(meta_info={"timing": {}, "metrics": {"three_step_stream/policy_lag": 0}})
        return 0, f"batch-{ids}", f"gen-{ids}", teacher, {}

    trainer.sync_rollout_weights_if_pending = sync_rollout_weights_if_pending
    trainer._run_three_step_streaming_sample = finish_sample
    trainer._pack_three_step_streaming_samples = pack_samples

    scheduler = trainer.three_step_off_scheduler(
        iter([(0, {"id": torch.arange(8), "x": torch.zeros(8, 1)})])
    )
    first = next(scheduler)
    second = next(scheduler)

    assert sync_calls == ["sync"]
    assert sorted(seen_samples) == list(range(8))
    assert len(packed_samples) == 2
    assert sorted(packed_samples[0] + packed_samples[1]) == list(range(8))
    assert first[1].startswith("batch-")
    assert second[1].startswith("batch-")
    assert first[3].meta_info["metrics"]["three_step_stream/policy_lag"] == 0
    assert second[3].meta_info["metrics"]["three_step_stream/policy_lag"] == 0

def test_three_step_off_scheduler_requires_effective_batch_to_cover_rollout_window():
    trainer = object.__new__(OnPolicyDistillTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 8},
            "trainer": {"actor_update_batch_size": 4},
            "actor_rollout_ref": {
                "actor": {"gradient_accumulation_steps": 1},
                "teacher": {"request_batch_size": 4},
            },
        }
    )

    try:
        trainer._three_step_streaming_config()
    except ValueError as exc:
        assert "EFFECTIVE_BATCH_SIZE" in str(exc)
    else:
        raise AssertionError("streaming three_step_off should reject EBS smaller than rollout window")

def test_y_r_generation_prompt_respects_teacher_training_prompt():
    trainer = object.__new__(OnPolicyDistillTrainer)
    trainer.opd_config = {"teacher_training_prompt": "vanilla"}
    captured = []

    trainer._decode_response = lambda response_ids: "student response"
    trainer._raw_problem_at = lambda batch, idx: "problem"
    trainer._expert_solution_at = lambda batch, idx: "expert"

    def build_prompt(**kwargs):
        captured.append(kwargs["use_initial_response"])
        return "prompt"

    trainer._build_teacher_prompt_content = build_prompt
    trainer._tokenize_chat_prompt = lambda content: [1, 2, 3]

    y_o_output = SimpleNamespace(batch={"responses": [torch.tensor([1, 2, 3])]})
    prompts = trainer._build_yr_generation_prompts(SimpleNamespace(), y_o_output)

    assert prompts == [[1, 2, 3]]
    assert captured == [False]

    trainer.opd_config = {"teacher_training_prompt": "refine"}
    captured.clear()
    trainer._build_yr_generation_prompts(SimpleNamespace(), y_o_output)
    assert captured == [True]


def test_bounded_lag_y_r_scheduler_tracks_policy_versions():
    trainer = object.__new__(OnPolicyDistillTrainer)
    trainer.opd_y_mode = "y_r"
    trainer._policy_version = 0
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 4},
            "trainer": {
                "actor_update_batch_size": 4,
                "teacher_inflight_samples": 8,
                "max_policy_lag": 1,
            },
            "actor_rollout_ref": {
                "actor": {"gradient_accumulation_steps": 2},
                "teacher": {"request_batch_size": 4},
            },
        }
    )
    submitted = []

    def start_rollout(epoch, batch_dict, sync_before_generation=True, sync_timing=None):
        del sync_before_generation
        del sync_timing
        submitted.append(batch_dict["id"])
        return SimpleNamespace(epoch=epoch, batch_dict=batch_dict)

    def start_y_r_generation(rollout_future, *, request_chunk_size=None):
        assert request_chunk_size == 4
        return SimpleNamespace(rollout_future=rollout_future)

    def score_y_r(y_r_generation_future, *, return_full_result=False, request_chunk_size=None):
        assert request_chunk_size == 4
        assert return_full_result is True
        rollout_future = y_r_generation_future.rollout_future
        batch_id = rollout_future.batch_dict["id"]
        teacher = SimpleNamespace(meta_info={"timing": {}})
        return _ReadyFuture((rollout_future.epoch, f"batch-{batch_id}", f"gen-{batch_id}", teacher))

    trainer._async_gen_next_batch = start_rollout
    trainer._async_generate_y_r = start_y_r_generation
    trainer._async_score_y_r_generation = score_y_r

    scheduler = trainer.bounded_lag_y_r_scheduler((0, {"id": i, "x": torch.zeros(4, 1)}) for i in range(2))
    first = next(scheduler)
    trainer._policy_version = 1
    second = next(scheduler)

    assert sorted(submitted) == [0, 1]
    assert first[1].startswith("batch-")
    assert first[2].startswith("gen-")
    assert first[3].meta_info["metrics"]["bounded_lag/policy_lag"] == 0
    assert second[3].meta_info["metrics"]["bounded_lag/train_policy_version"] == 1
    assert second[3].meta_info["metrics"]["bounded_lag/policy_lag"] == 1


def test_bounded_lag_y_r_scheduler_runs_y_o_in_background():
    trainer = object.__new__(OnPolicyDistillTrainer)
    trainer.opd_y_mode = "y_r"
    trainer._policy_version = 0
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 4},
            "trainer": {
                "actor_update_batch_size": 4,
                "teacher_inflight_samples": 8,
                "max_policy_lag": 1,
            },
            "actor_rollout_ref": {
                "actor": {"gradient_accumulation_steps": 2},
                "teacher": {"request_batch_size": 4},
            },
        }
    )
    submitted = []

    def start_rollout(epoch, batch_dict, sync_before_generation=True, sync_timing=None):
        del sync_timing
        assert sync_before_generation is False
        submitted.append(batch_dict["id"])
        return SimpleNamespace(epoch=epoch, batch_dict=batch_dict)

    def finish_item(item, epoch, batch_dict, ready_queue, request_chunk_size):
        assert request_chunk_size == 4
        item.completed_at = item.submitted_at + 1.0
        teacher = SimpleNamespace(meta_info={"timing": {}})
        rollout_future = trainer._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
        ready_queue.put((item, (rollout_future.epoch, f"batch-{item.item_id}", f"gen-{item.item_id}", teacher), None))

    trainer._async_gen_next_batch = start_rollout
    trainer._run_bounded_lag_item = finish_item

    scheduler = trainer.bounded_lag_y_r_scheduler((0, {"id": i, "x": torch.zeros(4, 1)}) for i in range(2))
    next(scheduler)
    next(scheduler)

    assert sorted(submitted) == [0, 1]


def test_bounded_lag_y_r_scheduler_drops_stale_samples():
    trainer = object.__new__(OnPolicyDistillTrainer)
    trainer.opd_y_mode = "y_r"
    trainer._policy_version = 0
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 4},
            "trainer": {
                "actor_update_batch_size": 4,
                "teacher_inflight_samples": 8,
                "max_policy_lag": 1,
            },
            "actor_rollout_ref": {
                "actor": {"gradient_accumulation_steps": 1},
                "teacher": {"request_batch_size": 4},
            },
        }
    )

    def start_rollout(epoch, batch_dict, sync_before_generation=True, sync_timing=None):
        del sync_before_generation
        del sync_timing
        return SimpleNamespace(epoch=epoch, batch_dict=batch_dict)

    def start_y_r_generation(rollout_future, *, request_chunk_size=None):
        del request_chunk_size
        return SimpleNamespace(rollout_future=rollout_future)

    def score_y_r(y_r_generation_future, *, return_full_result=False, request_chunk_size=None):
        del request_chunk_size
        assert return_full_result is True
        rollout_future = y_r_generation_future.rollout_future
        batch_id = rollout_future.batch_dict["id"]
        teacher = SimpleNamespace(meta_info={"timing": {}})
        return _ReadyFuture((rollout_future.epoch, f"batch-{batch_id}", f"gen-{batch_id}", teacher))

    trainer._async_gen_next_batch = start_rollout
    trainer._async_generate_y_r = start_y_r_generation
    trainer._async_score_y_r_generation = score_y_r

    scheduler = trainer.bounded_lag_y_r_scheduler((0, {"id": i, "x": torch.zeros(4, 1)}) for i in range(2))
    first = next(scheduler)
    trainer._policy_version = 2

    assert first[3].meta_info["metrics"]["bounded_lag/policy_lag"] == 0
    try:
        next(scheduler)
    except StopIteration:
        pass
    else:
        raise AssertionError("stale bounded-lag sample should be dropped instead of yielded")
