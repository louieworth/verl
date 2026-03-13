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

import json

from omegaconf import OmegaConf

from verl.trainer.main_eval import format_eval_results, save_eval_results
from verl.trainer.ppo.reward import get_custom_reward_fn


def test_get_custom_reward_fn_supports_legacy_top_level_config(monkeypatch):
    def fake_load_extern_object(module_path, object_name):
        assert module_path == "dummy_path.py"
        assert object_name == "compute_score"
        return lambda data_source, response, ground_truth: 1.0

    monkeypatch.setattr("verl.utils.import_utils.load_extern_object", fake_load_extern_object)

    config = OmegaConf.create(
        {
            "custom_reward_function": {
                "path": "dummy_path.py",
                "name": "compute_score",
            }
        }
    )

    reward_fn = get_custom_reward_fn(config)

    assert reward_fn is not None
    assert reward_fn("aime24", "resp", "gt") == 1.0


def test_get_custom_reward_fn_supports_nested_reward_config(monkeypatch):
    def fake_load_extern_object(module_path, object_name):
        assert module_path == "dummy_nested.py"
        assert object_name == "compute_score_nested"
        return lambda data_source, response, ground_truth: 2.0

    monkeypatch.setattr("verl.utils.import_utils.load_extern_object", fake_load_extern_object)

    config = OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {
                    "path": "dummy_nested.py",
                    "name": "compute_score_nested",
                }
            }
        }
    )

    reward_fn = get_custom_reward_fn(config)

    assert reward_fn is not None
    assert reward_fn("aime24", "resp", "gt") == 2.0


def test_save_eval_results_merges_into_results_json(tmp_path):
    output_json_path = tmp_path / "results.json"
    save_eval_results({"test_score/aime24": 0.5}, str(output_json_path), "model_a", pass_k=1)
    save_eval_results({"test_score/math500": 0.75}, str(output_json_path), "model_a", pass_k=1)

    with open(output_json_path) as f:
        results = json.load(f)

    assert results == {
        "model_a": {
            "aime24_pass1_generation_pass_1": 0.5,
            "math500_pass1_generation_pass_1": 0.75,
        }
    }


def test_format_eval_results_uses_expected_key_shape():
    assert format_eval_results({"test_score/aime24": 0.25}, pass_k=3) == {
        "aime24_pass3_generation_pass_3": 0.25
    }
