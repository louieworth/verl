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

from recipe.opd.config import KLTrainingConfig
from recipe.opd.dataset.data_utils import build_teacher_prompt
from recipe.opd.kl_trainer import build_eval_model_name, build_eval_tag


def test_build_teacher_prompt_rewrite_mode_omits_initial_response():
    prompt = build_teacher_prompt(
        "Solve x^2=1",
        "x=1 or x=-1",
        initial_response="x=1",
        use_initial_response=False,
    )

    assert "Here is a reference solution" in prompt
    assert "Your Initial Solution:" not in prompt


def test_build_teacher_prompt_with_initial_response_mode_includes_initial_response():
    prompt = build_teacher_prompt(
        "Solve x^2=1",
        "x=1 or x=-1",
        initial_response="x=1",
        use_initial_response=True,
    )

    assert "Your Initial Solution:" in prompt
    assert "x=1" in prompt
    assert "Reference Solution:" in prompt
    assert "correct your wrong mathematical solution" not in prompt


def test_eval_name_includes_prompt_mode_suffix():
    rewrite_config = KLTrainingConfig(
        kl_type="forward",
        kl_method="monte_carlo",
        student_model_path="Qwen/Qwen3-4B-Instruct-2507",
        base_model_name="Qwen3-4B-Instruct-2507",
        epoch_index=2,
        use_initial_response=False,
    )
    correction_config = KLTrainingConfig(
        kl_type="reverse",
        kl_method="monte_carlo",
        student_model_path="Qwen/Qwen3-4B-Instruct-2507",
        base_model_name="Qwen3-4B-Instruct-2507",
        epoch_index=2,
        use_initial_response=True,
    )

    assert build_eval_tag(rewrite_config) == "kl_forward_monte_carlo_rewrite"
    assert build_eval_model_name(rewrite_config) == "Qwen3-4B-Instruct-2507_kl_forward_monte_carlo_rewrite_epoch2"
    assert build_eval_tag(correction_config) == "kl_reverse_monte_carlo_correction"
    assert build_eval_model_name(correction_config) == "Qwen3-4B-Instruct-2507_kl_reverse_monte_carlo_correction_epoch2"
