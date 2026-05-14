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

import subprocess
from types import SimpleNamespace

import pytest

from recipe.opd.kl_trainer import KLTrainer


def build_trainer(tmp_path, rank: int):
    trainer = object.__new__(KLTrainer)
    trainer.rank = rank
    trainer.config = SimpleNamespace(
        model_save_dir=str(tmp_path),
        student_model_path="Qwen/Qwen3-4B-Instruct-2507",
    )
    trainer._find_latest_checkpoint = lambda: str(tmp_path / "global_step_27")
    return trainer


def test_export_hf_model_rank0_success(monkeypatch, tmp_path):
    trainer = build_trainer(tmp_path, rank=0)
    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd
        recorded["check"] = check

    monkeypatch.setattr("recipe.opd.kl_trainer.subprocess.run", fake_run)
    monkeypatch.setattr("recipe.opd.kl_trainer.dist.broadcast_object_list", lambda obj, src: None)

    trainer._export_hf_model()

    assert recorded["check"] is True
    assert recorded["cmd"][:4] == [recorded["cmd"][0], "-m", "verl.model_merger", "merge"]
    assert "--target_dir" in recorded["cmd"]
    target_dir_index = recorded["cmd"].index("--target_dir")
    assert recorded["cmd"][target_dir_index + 1] == str(tmp_path / "hf_merged")


def test_export_hf_model_rank0_failure_is_rethrown_after_broadcast(monkeypatch, tmp_path):
    trainer = build_trainer(tmp_path, rank=0)

    def fake_run(cmd, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr("recipe.opd.kl_trainer.subprocess.run", fake_run)
    monkeypatch.setattr("recipe.opd.kl_trainer.dist.broadcast_object_list", lambda obj, src: None)

    with pytest.raises(RuntimeError, match="FSDP checkpoint export failed on rank 0"):
        trainer._export_hf_model()


def test_export_hf_model_non_rank0_raises_rank0_failure(monkeypatch, tmp_path):
    trainer = build_trainer(tmp_path, rank=1)

    def fake_broadcast(obj, src):
        obj[0] = "CalledProcessError: Command '['python', '-m', 'verl.model_merger']' returned non-zero exit status 1."

    monkeypatch.setattr("recipe.opd.kl_trainer.dist.broadcast_object_list", fake_broadcast)

    with pytest.raises(RuntimeError, match="FSDP checkpoint export failed on rank 0"):
        trainer._export_hf_model()
