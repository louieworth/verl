from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe.opd import export_checkpoint
from recipe.opd.export_checkpoint import ensure_lora_metadata


def test_lora_metadata_is_persisted_and_validated(tmp_path):
    ensure_lora_metadata(tmp_path, rank=64, alpha=128)
    assert json.loads((tmp_path / "lora_train_meta.json").read_text()) == {
        "lora_alpha": 128,
        "r": 64,
        "task_type": "CAUSAL_LM",
    }
    ensure_lora_metadata(tmp_path, rank=64, alpha=128)
    with pytest.raises(ValueError, match="LoRA metadata mismatch"):
        ensure_lora_metadata(tmp_path, rank=32, alpha=128)


def test_lora_metadata_is_not_required_for_full_finetuning(tmp_path):
    ensure_lora_metadata(tmp_path, rank=0, alpha=0)
    assert not (tmp_path / "lora_train_meta.json").exists()


def _args(checkpoint_dir, target_dir, *, rank=0, alpha=0):
    return Namespace(
        local_dir=str(checkpoint_dir),
        target_dir=str(target_dir),
        base_model="base-model",
        lora_rank=rank,
        lora_alpha=alpha,
        trust_remote_code=False,
    )


def test_reexport_uses_clean_staging_and_atomically_replaces_target(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    target_dir = tmp_path / "target"
    checkpoint_dir.mkdir()
    target_dir.mkdir()
    marker = target_dir / "opd_export.json"
    marker.write_text('{"stale": true}\n', encoding="utf-8")
    (target_dir / "old_weights.safetensors").write_text("old", encoding="utf-8")
    stale_adapter = target_dir / "lora_adapter"
    stale_adapter.mkdir()
    (stale_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(export_checkpoint, "parse_args", lambda: _args(checkpoint_dir, target_dir))

    observed = {}

    def fake_run(command, **_kwargs):
        staging_dir = Path(command[command.index("--target_dir") + 1])
        observed["unmerged"] = staging_dir
        assert staging_dir != target_dir
        assert staging_dir.parent == target_dir.parent
        assert not (staging_dir / "lora_adapter").exists()
        assert json.loads(marker.read_text(encoding="utf-8")) == {"stale": True}
        (staging_dir / "new_weights.safetensors").write_text("new", encoding="utf-8")

    def fake_strict_load(staging_dir, _trust_remote_code):
        assert staging_dir == observed["unmerged"]
        assert (staging_dir / "new_weights.safetensors").read_text(encoding="utf-8") == "new"
        # The previously committed export stays visible through validation.
        assert (target_dir / "old_weights.safetensors").read_text(encoding="utf-8") == "old"

    monkeypatch.setattr(export_checkpoint.subprocess, "run", fake_run)
    monkeypatch.setattr(export_checkpoint, "strict_load", fake_strict_load)

    export_checkpoint.main()
    assert json.loads(marker.read_text(encoding="utf-8"))["schema_version"] == export_checkpoint.EXPORT_SCHEMA
    assert (target_dir / "new_weights.safetensors").read_text(encoding="utf-8") == "new"
    assert not (target_dir / "old_weights.safetensors").exists()
    assert not (target_dir / "lora_adapter").exists()
    assert not observed["unmerged"].exists()


def test_failed_reexport_preserves_previous_committed_target(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    target_dir = tmp_path / "target"
    checkpoint_dir.mkdir()
    target_dir.mkdir()
    marker = target_dir / "opd_export.json"
    marker.write_text('{"committed": true}\n', encoding="utf-8")
    old_weights = target_dir / "model.safetensors"
    old_weights.write_text("old", encoding="utf-8")

    monkeypatch.setattr(export_checkpoint, "parse_args", lambda: _args(checkpoint_dir, target_dir))

    def failed_merge(command, **_kwargs):
        staging_dir = Path(command[command.index("--target_dir") + 1])
        (staging_dir / "partial.safetensors").write_text("partial", encoding="utf-8")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(export_checkpoint.subprocess, "run", failed_merge)

    with pytest.raises(subprocess.CalledProcessError):
        export_checkpoint.main()

    assert json.loads(marker.read_text(encoding="utf-8")) == {"committed": True}
    assert old_weights.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".target.unmerged.*"))


def test_stale_adapter_cannot_satisfy_new_lora_export(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    target_dir = tmp_path / "target"
    checkpoint_dir.mkdir()
    target_dir.mkdir()
    marker = target_dir / "opd_export.json"
    marker.write_text('{"committed": true}\n', encoding="utf-8")
    stale_adapter = target_dir / "lora_adapter"
    stale_adapter.mkdir()
    (stale_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        export_checkpoint,
        "parse_args",
        lambda: _args(checkpoint_dir, target_dir, rank=64, alpha=128),
    )

    def merger_without_adapter(command, **_kwargs):
        staging_dir = Path(command[command.index("--target_dir") + 1])
        (staging_dir / "model.safetensors").write_text("new", encoding="utf-8")

    monkeypatch.setattr(export_checkpoint.subprocess, "run", merger_without_adapter)

    with pytest.raises(RuntimeError, match="no adapter was exported"):
        export_checkpoint.main()

    assert json.loads(marker.read_text(encoding="utf-8")) == {"committed": True}
    assert (stale_adapter / "adapter_config.json").is_file()
    assert not list(tmp_path.glob(".target.unmerged.*"))


def test_lora_merge_writes_to_distinct_directory_without_touching_source(monkeypatch, tmp_path):
    source_dir = tmp_path / "unmerged"
    output_dir = tmp_path / "merged"
    adapter_dir = source_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True)
    output_dir.mkdir()

    shard_one = source_dir / "model-00001-of-00002.safetensors"
    shard_two = source_dir / "model-00002-of-00002.safetensors"
    index = source_dir / "model.safetensors.index.json"
    shard_one.write_text("source-one", encoding="utf-8")
    shard_two.write_text("source-two", encoding="utf-8")
    index.write_text("source-index", encoding="utf-8")
    (source_dir / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path, **_kwargs):
            assert Path(path) == source_dir
            return object()

    class FakeMergedModel:
        def save_pretrained(self, path, *, safe_serialization):
            destination = Path(path)
            assert destination == output_dir
            assert destination != source_dir
            assert safe_serialization is True
            assert shard_one.read_text(encoding="utf-8") == "source-one"
            assert shard_two.read_text(encoding="utf-8") == "source-two"
            (destination / "model.safetensors").write_text("merged", encoding="utf-8")

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, _base, path, *, is_trainable):
            assert Path(path) == adapter_dir
            assert is_trainable is False
            return cls()

        def merge_and_unload(self, *, safe_merge):
            assert safe_merge is True
            return FakeMergedModel()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bfloat16"))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=FakeAutoModel),
    )

    assert export_checkpoint.merge_lora_adapter(source_dir, output_dir, trust_remote_code=False)
    assert shard_one.read_text(encoding="utf-8") == "source-one"
    assert shard_two.read_text(encoding="utf-8") == "source-two"
    assert index.read_text(encoding="utf-8") == "source-index"
    assert not (output_dir / shard_one.name).exists()
    assert not (output_dir / shard_two.name).exists()
    assert not (output_dir / index.name).exists()
    assert (output_dir / "model.safetensors").read_text(encoding="utf-8") == "merged"
    assert (output_dir / "tokenizer.json").read_text(encoding="utf-8") == "tokenizer"
    assert (output_dir / "lora_adapter" / "adapter_config.json").is_file()


def test_lora_merge_rejects_in_place_output(tmp_path):
    with pytest.raises(ValueError, match="source and output directories must differ"):
        export_checkpoint.merge_lora_adapter(tmp_path, tmp_path, trust_remote_code=False)
