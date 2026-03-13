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

from types import SimpleNamespace

import pytest

from verl.model_merger.base_model_merger import BaseModelMerger, generate_config_from_args


class DummyMerger(BaseModelMerger):
    def merge_and_save(self):
        raise NotImplementedError

    def cleanup(self):
        return None


class DummyAsset:
    def __init__(self):
        self.saved_paths = []

    def save_pretrained(self, path):
        self.saved_paths.append(path)


def build_dummy_merger(tmp_path):
    merger = object.__new__(DummyMerger)
    merger.config = SimpleNamespace(target_dir=str(tmp_path), trust_remote_code=False)
    merger.hf_model_config_path = "/tmp/source-model"
    return merger


def test_generate_config_from_args_prefers_explicit_hf_model_config_path(tmp_path):
    args = SimpleNamespace(
        operation="merge",
        backend="fsdp",
        target_dir=str(tmp_path / "merged"),
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=False,
        is_value_model=False,
        local_dir=str(tmp_path / "checkpoint"),
        hf_model_config_path=str(tmp_path / "base-model"),
        use_cpu_initialization=False,
    )

    config = generate_config_from_args(args)

    assert config.hf_model_config_path == str(tmp_path / "base-model")


def test_generate_config_from_args_defaults_to_checkpoint_huggingface_dir(tmp_path):
    args = SimpleNamespace(
        operation="merge",
        backend="fsdp",
        target_dir=str(tmp_path / "merged"),
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=False,
        is_value_model=False,
        local_dir=str(tmp_path / "checkpoint"),
        hf_model_config_path=None,
        use_cpu_initialization=False,
    )

    config = generate_config_from_args(args)

    assert config.hf_model_config_path == str(tmp_path / "checkpoint" / "huggingface")


def test_save_hf_processing_assets_saves_processor_and_tokenizer(monkeypatch, tmp_path):
    merger = build_dummy_merger(tmp_path)
    processor = DummyAsset()
    tokenizer = DummyAsset()

    monkeypatch.setattr("verl.model_merger.base_model_merger.hf_processor", lambda *args, **kwargs: processor)
    monkeypatch.setattr("verl.model_merger.base_model_merger.hf_tokenizer", lambda *args, **kwargs: tokenizer)

    merger.save_hf_processing_assets()

    assert processor.saved_paths == [str(tmp_path)]
    assert tokenizer.saved_paths == [str(tmp_path)]


def test_save_hf_processing_assets_warns_and_skips_missing_tokenizer(monkeypatch, tmp_path):
    merger = build_dummy_merger(tmp_path)

    monkeypatch.setattr("verl.model_merger.base_model_merger.hf_processor", lambda *args, **kwargs: None)

    def raise_missing_tokenizer(*args, **kwargs):
        raise TypeError("expected str, bytes or os.PathLike object, not NoneType")

    monkeypatch.setattr("verl.model_merger.base_model_merger.hf_tokenizer", raise_missing_tokenizer)

    with pytest.warns(UserWarning, match="--hf_model_config_path"):
        merger.save_hf_processing_assets()
