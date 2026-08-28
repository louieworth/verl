# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import math
from types import SimpleNamespace

import pytest
import torch

from recipe.opd.config import KLTrainingConfig
from recipe.opd.experiment_tracking import (
    build_eval_metrics_payload,
    compute_eval_milestone_fractions,
    compute_eval_milestones,
    deterministic_wandb_run_id,
    initialize_wandb_run,
    log_eval_metrics,
    parse_eval_fractions,
    resolve_wandb_run_state,
)
from recipe.opd.kl_trainer import KLTrainer, token_entropy_from_logits


class FakeWandb:
    def __init__(self):
        self.init_kwargs = None
        self.defined_metrics = []
        self.logged = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, payload, **kwargs):
        self.logged.append((payload, kwargs))


def test_eval_fractions_parse_sort_and_deduplicate():
    assert parse_eval_fractions(".75,0.25,.5,1,0.50") == (0.25, 0.5, 0.75, 1.0)


@pytest.mark.parametrize("value", ["0", "1.01", "nan", "0.25,,1", []])
def test_eval_fractions_reject_invalid_values(value):
    with pytest.raises(ValueError):
        parse_eval_fractions(value)


def test_eval_milestones_use_ceil_and_deduplicate_short_runs():
    assert compute_eval_milestones(10, "0.25,.5,.75,1") == (3, 5, 8, 10)
    assert compute_eval_milestones(58) == (15, 29, 44, 58)
    assert compute_eval_milestones(37) == (10, 19, 28, 37)
    assert compute_eval_milestones(2) == (1, 2)
    assert compute_eval_milestones(1) == (1,)
    assert compute_eval_milestone_fractions(2) == {1: 0.5, 2: 1.0}


def test_wandb_run_id_is_deterministic_and_persisted(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    state_path = tmp_path / "shared" / "wandb_run.json"
    expected = deterministic_wandb_run_id("trd", "opd-math-1.7b-vanilla")

    first = resolve_wandb_run_state(
        project="trd",
        run_name="display-segment-1",
        run_identity="opd-math-1.7b-vanilla",
        state_path=state_path,
    )
    second = resolve_wandb_run_state(
        project="trd",
        run_name="display-segment-2",
        run_identity="opd-math-1.7b-vanilla",
        state_path=state_path,
    )

    assert first.run_id == expected == second.run_id
    assert json.loads(state_path.read_text(encoding="utf-8"))["run_id"] == expected


def test_wandb_state_rejects_conflicting_explicit_id(tmp_path):
    state_path = tmp_path / "wandb_run.json"
    resolve_wandb_run_state(
        project="trd",
        run_identity="experiment-a",
        state_path=state_path,
        explicit_run_id="run_a",
    )

    with pytest.raises(ValueError, match="run id mismatch"):
        resolve_wandb_run_state(
            project="trd",
            run_identity="experiment-a",
            state_path=state_path,
            explicit_run_id="run_b",
        )


def test_wandb_state_rejects_conflicting_experiment_identity(tmp_path):
    state_path = tmp_path / "wandb_run.json"
    resolve_wandb_run_state(
        project="trd",
        run_identity="experiment-a",
        state_path=state_path,
    )

    with pytest.raises(ValueError, match="run identity mismatch"):
        resolve_wandb_run_state(
            project="trd",
            run_identity="experiment-b",
            state_path=state_path,
        )


def test_initialize_wandb_resumes_persisted_run(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)
    monkeypatch.delenv("WANDB_TAGS", raising=False)
    monkeypatch.delenv("WANDB_GROUP", raising=False)
    monkeypatch.delenv("WANDB_JOB_TYPE", raising=False)
    monkeypatch.delenv("OPD_TASK", raising=False)
    fake_wandb = FakeWandb()
    state = initialize_wandb_run(
        fake_wandb,
        project="trd",
        run_name="opd-math",
        state_path=tmp_path / "wandb_run.json",
        mode="disabled",
        config={"response_length": 16384},
    )

    assert fake_wandb.init_kwargs == {
        "project": "trd",
        "id": state.run_id,
        "name": "opd-math",
        "resume": "allow",
        "mode": "disabled",
        "config": {"response_length": 16384},
    }
    assert (("global_step",), {}) in fake_wandb.defined_metrics
    assert (("*",), {"step_metric": "global_step"}) in fake_wandb.defined_metrics


def test_initialize_wandb_attaches_group_and_config_but_no_tags(monkeypatch, tmp_path):
    monkeypatch.setenv("OPD_TASK", "math")
    monkeypatch.setenv("OPD_FAMILY", "opd")
    monkeypatch.setenv("OPD_VARIANT", "clip")
    monkeypatch.setenv("MODEL_ALIAS", "Qwen3-1.7B-Base")
    monkeypatch.setenv("MAX_RESPONSE_LENGTH", "16384")
    monkeypatch.setenv("KL_TOKEN_CLIP", "0.05")
    monkeypatch.setenv("MODEL_ARTIFACT_POLICY", "milestone_hf_deferred_eval")
    monkeypatch.setenv("PIPELINE_EPHEMERAL_MODELS", "true")
    monkeypatch.setenv("PIPELINE_DEFER_MILESTONE_EVALS", "true")
    monkeypatch.setenv("WANDB_TAGS", "task=math,family=opd,variant=clip,task=math")
    monkeypatch.setenv("WANDB_GROUP", "clip-1B")
    monkeypatch.setenv("WANDB_JOB_TYPE", "opd-clip")

    fake_wandb = FakeWandb()
    initialize_wandb_run(
        fake_wandb,
        project="opd-math",
        run_name="opd-math-clip",
        state_path=tmp_path / "wandb_run.json",
        mode="disabled",
        config={"native": "kept"},
    )

    assert "tags" not in fake_wandb.init_kwargs
    assert fake_wandb.init_kwargs["group"] == "clip-1B"
    assert fake_wandb.init_kwargs["job_type"] == "opd-clip"
    assert fake_wandb.init_kwargs["config"]["native"] == "kept"
    assert fake_wandb.init_kwargs["config"]["opd_experiment"] == {
        "task": "math",
        "family": "opd",
        "variant": "clip",
        "model": "Qwen3-1.7B-Base",
        "train_response_length": 16384,
        "kl_token_clip": 0.05,
        "model_artifact_policy": "milestone_hf_deferred_eval",
        "pipeline_ephemeral_models": True,
        "pipeline_defer_milestone_evals": True,
    }


def test_eval_metrics_are_flattened_and_logged_at_explicit_step():
    metrics = {
        "aime26": {"avg@16": 0.25, "pass@16": 0.5},
        "macro": {"avg@16": 0.4},
    }
    assert build_eval_metrics_payload(metrics, task="math") == {
        "eval/math/aime26/avg@16": 0.25,
        "eval/math/aime26/pass@16": 0.5,
        "eval/math/macro/avg@16": 0.4,
    }

    fake_wandb = FakeWandb()
    payload = log_eval_metrics(
        fake_wandb,
        metrics,
        task="math",
        milestone_fraction=0.5,
        global_step=37,
    )

    assert payload["global_step"] == 37
    assert payload["eval/global_step"] == 37
    assert payload["eval/milestone_fraction"] == 0.5
    assert fake_wandb.logged == [(payload, {"commit": True})]


def test_kl_config_defaults_to_strict_micro_batch_one_and_trd():
    config = KLTrainingConfig(eval_fractions="1,.25,.5,.75")

    assert config.wandb_project == "trd"
    assert config.use_dynamic_bsz is False
    assert config.micro_batch_size_per_gpu == 1
    assert config.eval_fractions == (0.25, 0.5, 0.75, 1.0)


def test_kl_config_validates_experiment_total_steps():
    assert KLTrainingConfig(wandb_total_training_steps=40).wandb_total_training_steps == 40
    with pytest.raises(ValueError, match="wandb_total_training_steps"):
        KLTrainingConfig(wandb_total_training_steps=0)


def test_kl_trainer_finishes_wandb_with_failure_exit_code(monkeypatch):
    finish_calls = []
    trainer = object.__new__(KLTrainer)
    trainer.sync_only = False
    trainer.is_logging = True
    fake_wandb = SimpleNamespace(
        run=object(),
        finish=lambda **kwargs: finish_calls.append(kwargs),
    )
    monkeypatch.setattr("recipe.opd.kl_trainer.HAS_WANDB", True)
    monkeypatch.setattr("recipe.opd.kl_trainer.wandb", fake_wandb)

    trainer.finish_tracking(exit_code=1)

    assert finish_calls == [{"exit_code": 1}]


def test_kl_trainer_common_meta_uses_batching_config():
    trainer = object.__new__(KLTrainer)
    trainer.tokenizer = SimpleNamespace(pad_token_id=7)
    trainer.config = SimpleNamespace(
        use_remove_padding=True,
        use_dynamic_bsz=False,
        max_token_len_per_gpu=18432,
        micro_batch_size_per_gpu=1,
    )

    meta = trainer._common_meta(return_logits=True)

    assert meta["use_dynamic_bsz"] is False
    assert meta["micro_batch_size_per_gpu"] == 1
    assert meta["max_token_len_per_gpu"] == 18432


def test_opd_student_entropy_is_full_vocab_categorical_entropy():
    logits = torch.zeros(3, 4)
    entropy = token_entropy_from_logits(logits)
    assert entropy.shape == (3,)
    assert torch.allclose(entropy, torch.full((3,), math.log(4)), atol=1e-6)


def test_kl_step_summary_reports_student_entropy():
    summary = KLTrainer._summarize_step_output(
        {
            "metrics": {
                "kl_num": [2.0],
                "student_nll_num": [3.0],
                "teacher_nll_num": [4.0],
                "student_entropy_num": [6.0, 4.0],
                "student_entropy_den": [2.0, 3.0],
                "response_tokens": [5.0],
            }
        }
    )
    assert summary["student_entropy"] == 2.0


def test_kl_trainer_engine_config_uses_batching_config(monkeypatch):
    trainer = object.__new__(KLTrainer)
    trainer.config = SimpleNamespace(
        fsdp_strategy="fsdp2",
        fsdp_size=-1,
        ulysses_sequence_parallel_size=1,
        use_dynamic_bsz=False,
        max_token_len_per_gpu=18432,
        micro_batch_size_per_gpu=1,
        use_remove_padding=True,
        use_torch_compile=False,
        param_offload=False,
        optimizer_offload=False,
        offload_policy=False,
        bf16=True,
    )
    trainer._build_wrap_policy = lambda _: {}
    monkeypatch.setattr(
        "recipe.opd.kl_trainer.FSDPEngineConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    engine_config = trainer._build_engine_config("Qwen/Qwen3-1.7B-Base", forward_only=False)

    assert engine_config.use_dynamic_bsz is False
    assert engine_config.micro_batch_size_per_gpu == 1
    assert engine_config.infer_micro_batch_size_per_gpu == 1
