from types import SimpleNamespace

import pytest

from recipe.code_evaluation.export_metrics import build_metrics_payload
from recipe.code_evaluation.extract_evalplus_metrics import is_correct
from recipe.code_evaluation.results_json_to_csv import build_rows, fieldnames
from recipe.code_evaluation.run_lcb_base import build_base_prompt


def test_code_metrics_schema_and_macro():
    entry = {
        "model_path": "/model",
        "humaneval_plus_avg16": 0.25,
        "humaneval_plus_pass16": 0.5,
        "humaneval_plus_num_problems": 164,
        "mbpp_plus_avg16": 0.5,
        "mbpp_plus_pass16": 0.75,
        "mbpp_plus_num_problems": 378,
        "livecodebench_v6_avg16": 0.0,
        "livecodebench_v6_pass16": 0.25,
    }
    payload = build_metrics_payload(
        entry=entry,
        model_name="Qwen3-4B_base",
        model_path="/model",
        datasets=["humaneval_plus", "mbpp_plus", "livecodebench_v6"],
        step=100,
        n_samples=16,
        prompt_length=2048,
        response_length=16384,
        temperature=0.6,
        top_p=0.95,
        seed=42,
    )
    assert payload["schema_version"] == "opd_eval_metrics/v1"
    assert payload["sampling"]["prompt_format"] == "base_completion"
    assert payload["sampling"]["temperature"] == 0.6
    assert payload["sampling"]["top_p"] == 0.95
    assert payload["macro"] == {"avg@16": 0.25, "pass@16": 0.5}
    assert payload["wandb"]["eval/code/macro/pass@16"] == 0.5


def test_code_export_rejects_noncanonical_sample_count():
    with pytest.raises(ValueError, match="n_samples=16"):
        build_metrics_payload(
            entry={},
            model_name="model",
            model_path="",
            datasets=["humaneval_plus"],
            step=None,
            n_samples=4,
            prompt_length=2048,
            response_length=16384,
            temperature=0.6,
            top_p=0.95,
            seed=42,
        )


def test_code_metrics_export_keeps_available_dataset_when_suite_is_partial():
    payload = build_metrics_payload(
        entry={
            "humaneval_plus_avg16": 0.25,
            "humaneval_plus_pass16": 0.5,
        },
        model_name="model",
        model_path="/model",
        datasets=["humaneval_plus", "mbpp_plus", "livecodebench_v6"],
        step=10,
        n_samples=16,
        prompt_length=2048,
        response_length=16384,
        temperature=0.6,
        top_p=0.95,
        seed=42,
    )
    assert payload["datasets"] == {
        "humaneval_plus": {"avg@16": 0.25, "pass@16": 0.5}
    }
    assert payload["macro"] == {"avg@16": 0.25, "pass@16": 0.5}


def test_lcb_base_prompt_requests_raw_source_without_markdown_fences():
    problem = SimpleNamespace(question_content="Add two integers.", starter_code="", question_id="x")
    prompt = build_base_prompt(problem)
    assert prompt.startswith("Problem:\nAdd two integers.")
    assert "raw Python source code" in prompt
    assert "```python```" not in prompt
    assert "<|im_start|>" not in prompt


def test_code_csv_includes_macro_at_16():
    rows = build_rows(
        {
            "Qwen3-4B_CODE_base": {
                "humaneval_plus_avg16": 0.1,
                "humaneval_plus_pass16": 0.2,
                "macro_avg16": 0.3,
                "macro_pass16": 0.4,
            }
        }
    )
    assert rows[0]["macro_Avg@16"] == "0.3"
    assert rows[0]["macro_Pass@16"] == "0.4"
    assert "macro_Avg@16" in fieldnames(16)


def test_evalplus_plus_metric_requires_both_base_and_plus_pass():
    assert is_correct({"base_status": "pass", "plus_status": "pass"})
    assert not is_correct({"base_status": "pass", "plus_status": "fail"})
