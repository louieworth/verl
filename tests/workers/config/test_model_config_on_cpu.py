# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright Amazon.com and/or its affiliates
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
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from verl.workers.config.model import HFModelConfig, resolve_remove_padding_for_model


class TestHFModelConfigCPU:
    model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")  # Just a path string, not loaded

    def test_target_modules_accepts_list_via_omegaconf(self):
        """
        Test that target_modules field accepts both string and list values
        when merging OmegaConf configs (simulates CLI override behavior).

        The purpose is to ensure we can pass
        actor_rollout_ref.model.target_modules='["k_proj","o_proj","down_proj","q_proj"]'
        """

        # Create structured config from the dataclass defaults
        # This is what omega_conf_to_dataclass does internally
        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        # Simulate CLI override with target_modules as a list
        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"],
            }
        )

        # This merge should NOT raise ValidationError
        # Before the fix (target_modules: str), this would fail with:
        # "Cannot convert 'ListConfig' to string"
        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)

        # Verify the list was merged correctly
        assert list(merged.target_modules) == ["k_proj", "o_proj", "q_proj", "v_proj"]

    def test_target_modules_accepts_none_via_omegaconf(self):
        """Test that target_modules still accepts None values."""

        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": None,
            }
        )

        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)
        assert merged.target_modules is None

    def test_target_modules_accepts_string_via_omegaconf(self):
        """Test that target_modules still accepts string values."""

        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": "all-linear",
            }
        )

        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)
        assert merged.target_modules == "all-linear"

    @patch("verl.workers.config.model.AutoConfig.from_pretrained")
    @patch("verl.workers.config.model.get_generation_config")
    @patch("verl.workers.config.model.copy_to_local")
    def test_target_modules_raises_on_invalid_type(
        self, mock_copy_to_local, mock_get_generation_config, mock_auto_config_from_pretrained
    ):
        """Test that __post_init__ raises TypeError for invalid target_modules types."""
        mock_copy_to_local.side_effect = lambda path, use_shm=False: path
        mock_get_generation_config.return_value = None
        mock_auto_config_from_pretrained.return_value = SimpleNamespace(
            model_type="qwen2",
            tie_word_embeddings=False,
            architectures=["Qwen2ForCausalLM"],
        )

        base_config = OmegaConf.structured(HFModelConfig)
        invalid_cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "load_tokenizer": False,
                "target_modules": [1, 2, 3],  # list of ints instead of strings
            }
        )
        merged_config = OmegaConf.merge(base_config, invalid_cli_config)
        with pytest.raises(TypeError):
            OmegaConf.to_object(merged_config)

    def test_resolve_remove_padding_keeps_supported_models_enabled(self):
        assert resolve_remove_padding_for_model("qwen2", True) is True
        assert resolve_remove_padding_for_model(None, True) is True
        assert resolve_remove_padding_for_model("qwen3_5", False) is False

    def test_resolve_remove_padding_disables_qwen35(self):
        with pytest.warns(UserWarning, match="qwen3_5"):
            assert resolve_remove_padding_for_model("qwen3_5", True) is False
