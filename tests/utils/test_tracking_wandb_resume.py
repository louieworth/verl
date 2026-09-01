# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import sys

from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.utils.wandb_metadata import merge_opd_wandb_config


class FakeWandb:
    def __init__(self):
        self.init_kwargs = None
        self.finished = False
        self.defined_metrics = []
        self.logged = []
        self.run = object()

    class Table:
        def __init__(self, columns, data=None):
            self.columns = columns
            self.data = list(data or [])

        def add_data(self, *row):
            self.data.append(list(row))

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def finish(self, **kwargs):
        self.finished = True

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, *args, **kwargs):
        if args:
            assert len(args) == 1
            entry = dict(args[0])
            entry.update(kwargs)
            self.logged.append(entry)
        else:
            self.logged.append(kwargs)


def test_wandb_metadata_is_available_from_the_installed_verl_namespace(monkeypatch):
    monkeypatch.setenv("OPD_TASK", "math")
    assert merge_opd_wandb_config({"native": True}) == {
        "native": True,
        "opd_experiment": {"task": "math"},
    }


def test_wandb_metadata_includes_teacher_thinking_only_for_opd(monkeypatch):
    monkeypatch.setenv("OPD_FAMILY", "opd")
    monkeypatch.setenv("TEACHER_MODEL", "Qwen3-30B-A3B-Instruct-2507")
    monkeypatch.setenv("TEACHER_MODEL_PATH", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    monkeypatch.setenv("TEACHER_ENABLE_THINKING", "false")
    monkeypatch.setenv("TEACHER_THINKING_NAME_TAG", "teacher-thinking-false")
    monkeypatch.setenv("TEACHER_SUPERVISION_RENDER_MODE", "plain_completion")
    monkeypatch.setenv("TEACHER_ROLLOUT_RENDER_MODE", "plain_completion")
    config = merge_opd_wandb_config({})["opd_experiment"]
    assert config["teacher_model"] == "Qwen3-30B-A3B-Instruct-2507"
    assert config["teacher_model_path"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert config["teacher_enable_thinking"] is False
    assert config["teacher_thinking_name_tag"] == "teacher-thinking-false"
    assert config["teacher_supervision_render_mode"] == "plain_completion"
    assert config["teacher_rollout_render_mode"] == "plain_completion"


def test_wandb_metadata_omits_teacher_thinking_for_opsd(monkeypatch):
    monkeypatch.setenv("OPD_FAMILY", "opsd")
    monkeypatch.setenv("TEACHER_ENABLE_THINKING", "false")
    monkeypatch.setenv("TEACHER_THINKING_NAME_TAG", "teacher-thinking-false")
    monkeypatch.setenv("TEACHER_SUPERVISION_RENDER_MODE", "base_completion")
    monkeypatch.setenv("TEACHER_ROLLOUT_RENDER_MODE", "base_completion")
    config = merge_opd_wandb_config({})["opd_experiment"]
    assert not any("thinking" in key for key in config)
    assert "teacher_supervision_render_mode" not in config
    assert "teacher_rollout_render_mode" not in config


def test_tracking_passes_resume_metadata_but_ignores_wandb_tags(monkeypatch):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_RUN_ID", "shared_run_123")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    monkeypatch.setenv("WANDB_TAGS", "task=math,family=opd,variant=clip")
    monkeypatch.setenv("WANDB_GROUP", "clip-1B")
    monkeypatch.setenv("WANDB_JOB_TYPE", "opd-clip")
    monkeypatch.setenv("OPD_TASK", "math")
    monkeypatch.setenv("OPD_FAMILY", "opd")

    tracking = Tracking("trd", "math-opd", default_backend="wandb")

    assert fake_wandb.init_kwargs["id"] == "shared_run_123"
    assert fake_wandb.init_kwargs["resume"] == "allow"
    assert "tags" not in fake_wandb.init_kwargs
    assert fake_wandb.init_kwargs["group"] == "clip-1B"
    assert fake_wandb.init_kwargs["job_type"] == "opd-clip"
    assert fake_wandb.init_kwargs["config"]["opd_experiment"] == {
        "task": "math",
        "family": "opd",
    }
    assert (("*",), {"step_metric": "global_step"}) in fake_wandb.defined_metrics

    tracking.log({"train/loss": 0.25}, step=17)
    assert fake_wandb.logged == [
        {
            "data": {
                "train/loss": 0.25,
                "train/global_step": 17,
                "global_step": 17,
            }
        }
    ]
    ValidationGenerationsLogger().log_generations_to_wandb(
        [["question", "answer", 1.0]], step=17
    )
    generation_log = fake_wandb.logged[-1]
    assert generation_log["global_step"] == 17
    assert "step" not in generation_log
    tracking.logger.clear()


def test_tracking_adds_common_grpo_training_aliases(monkeypatch):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_RUN_ID", "shared_grpo_run")

    tracking = Tracking("opd-code", "code-grpo", default_backend="wandb")
    tracking.log(
        {
            "actor/pg_loss": 0.2,
            "actor/lr": 1e-6,
            "actor/grad_norm": 0.8,
            "actor/kl_loss": 0.01,
            "actor/entropy": 0.4,
            "critic/score/mean": 0.7,
            "critic/rewards/mean": 0.68,
            "response_length/mean": 512.0,
            "perf/time_per_step": 3.5,
            "perf/mfu/actor": 0.42,
            "training/epoch": 2,
        },
        step=9,
    )

    payload = fake_wandb.logged[-1]["data"]
    assert payload["train/loss"] == 0.2
    assert payload["train/learning_rate"] == 1e-6
    assert payload["train/grad_norm"] == 0.8
    assert payload["train/reward"] == 0.7
    assert payload["train/reward_after_kl"] == 0.68
    assert payload["train/kl_loss"] == 0.01
    assert payload["train/entropy"] == 0.4
    assert payload["train/response_tokens"] == 512.0
    assert payload["train/step_time_sec"] == 3.5
    assert payload["train/mfu"] == 0.42
    assert payload["train/epoch"] == 2
    assert payload["train/global_step"] == 9
    assert payload["global_step"] == 9
    tracking.logger.clear()


def test_tracking_keeps_wandb_init_compatible_without_resume_environment(monkeypatch):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)
    monkeypatch.delenv("WANDB_TAGS", raising=False)
    monkeypatch.delenv("WANDB_GROUP", raising=False)
    monkeypatch.delenv("WANDB_JOB_TYPE", raising=False)
    monkeypatch.delenv("OPD_TASK", raising=False)
    monkeypatch.delenv("OPD_FAMILY", raising=False)

    tracking = Tracking("trd", "math-grpo", default_backend="wandb")

    assert "id" not in fake_wandb.init_kwargs
    assert "resume" not in fake_wandb.init_kwargs
    tracking.log({"train/loss": 0.5}, step=3)
    assert fake_wandb.logged == [{"data": {"train/loss": 0.5}, "step": 3}]
    ValidationGenerationsLogger().log_generations_to_wandb(
        [["question", "answer", 1.0]], step=3
    )
    assert fake_wandb.logged[-1]["step"] == 3
    assert "global_step" not in fake_wandb.logged[-1]
    tracking.logger.clear()
