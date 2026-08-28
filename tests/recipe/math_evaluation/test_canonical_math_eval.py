import sys
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from recipe.math_evaluation import compute_pass_at_k_from_gen as aggregates
from recipe.math_evaluation import log_metrics_wandb as wandb_logger
from recipe.math_evaluation.compute_score import compute_score_data_source
from recipe.math_evaluation.eval_utils import (
    BASE_COMPLETION_CHAT_TEMPLATE,
    DEFAULT_EVAL_DATASETS,
    EVAL_DATASET_ROOTS,
    build_generation_cache_provenance,
    completion_text,
    generation_cache_manifest_path,
    generation_parquet_is_complete,
    render_base_prompt,
    write_generation_cache_manifest,
)
from recipe.math_evaluation.log_metrics_wandb import load_wandb_payload
from recipe.math_evaluation.results_json_to_csv import SUMMARY_COLUMNS, rows_for_results


def test_canonical_dataset_defaults_and_routes():
    assert DEFAULT_EVAL_DATASETS == ("aime25", "aime26", "hmmt26", "amobench")
    assert EVAL_DATASET_ROOTS["aime25"].endswith("aime25_test.parquet")
    assert EVAL_DATASET_ROOTS["aime26"].endswith("aime26_test.parquet")
    assert EVAL_DATASET_ROOTS["hmmt26"].endswith("hmmt26_test.parquet")
    assert "beyondaime" not in DEFAULT_EVAL_DATASETS
    assert compute_score_data_source("aime26", "The answer is \\boxed{42}.", "42") == 1
    assert compute_score_data_source("MathArena/hmmt_feb_2026", "\\boxed{7}", "7") == 1


def test_base_prompt_renderer_has_no_chat_markers():
    prompt = render_base_prompt([{"role": "user", "content": "problem\ninstruction"}])
    assert prompt == "problem\ninstruction\n"
    assert "im_start" not in BASE_COMPLETION_CHAT_TEMPLATE
    assert "assistant" not in BASE_COMPLETION_CHAT_TEMPLATE


def test_math_generation_accepts_plain_completion_and_chat_responses():
    plain = SimpleNamespace(choices=[SimpleNamespace(text="plain")])
    chat = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="chat"))])
    assert completion_text(plain) == "plain"
    assert completion_text(chat) == "chat"


def _write_generation(path, response_count=16):
    pd.DataFrame(
        [
            {
                "data_source": "aime26",
                "reward_model": {"ground_truth": "1"},
                "responses": ["1", *(["0"] * (response_count - 1))],
            },
            {
                "data_source": "aime26",
                "reward_model": {"ground_truth": "1"},
                "responses": ["0"] * response_count,
            },
        ]
    ).to_parquet(path)


def test_avg16_pass16_are_strict_and_distinct(tmp_path, monkeypatch):
    path = tmp_path / "generation.parquet"
    _write_generation(path)
    monkeypatch.setattr(
        aggregates,
        "compute_score_data_source",
        lambda _source, response, ground_truth: response == ground_truth,
    )
    result = aggregates.aggregate_bench(str(path), n_samples=16)
    assert result == {"avg": 1 / 32, "pass": 0.5, "num_problems": 2, "n_samples": 16}

    payload = aggregates.build_metrics_payload(
        model_name="model",
        model_path="/model",
        step=25,
        aggregates={"aime25": result, "aime26": result, "hmmt26": result, "amobench": result},
        n_samples=16,
        prompt_length=2048,
        response_length=16384,
        temperature=1.0,
        top_p=0.7,
        seed=42,
    )
    assert payload["schema_version"] == "opd_eval_metrics/v1"
    assert payload["step"] == 25
    assert payload["macro"] == {"avg@16": 1 / 32, "pass@16": 0.5}
    assert payload["wandb"]["eval/math/macro/pass@16"] == 0.5


def test_incomplete_generation_is_not_reported_as_pass16(tmp_path):
    path = tmp_path / "generation.parquet"
    _write_generation(path, response_count=15)
    with pytest.raises(ValueError, match="requires exactly 16"):
        aggregates.aggregate_bench(str(path), n_samples=16)


def test_math_scorer_failures_are_not_silently_reported_as_zero(tmp_path, monkeypatch):
    path = tmp_path / "generation.parquet"
    _write_generation(path)
    monkeypatch.setattr(
        aggregates,
        "compute_score_data_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken scorer")),
    )
    with pytest.raises(RuntimeError, match="Scorer failed for row 0, sample 0"):
        aggregates.aggregate_bench(str(path), n_samples=16)


def test_canonical_math_generation_requires_all_problems(tmp_path):
    path = tmp_path / "generation.parquet"
    _write_generation(path)
    with pytest.raises(ValueError, match="requires 30"):
        aggregates.aggregate_bench(str(path), n_samples=16, dataset_name="aime26")


def test_generation_cache_sidecar_binds_checkpoint_dataset_sampling_and_output(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EVAL_MAX_MODEL_LEN", "18432")
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"unit"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"unit-weights")

    dataset_path = tmp_path / "source.parquet"
    generation_path = tmp_path / "generation.parquet"
    pd.DataFrame([{"data_source": "toy"}, {"data_source": "toy"}]).to_parquet(dataset_path)
    _write_generation(generation_path)

    def provenance(**overrides):
        values = {
            "model_path": str(model_path),
            "tokenizer_path": None,
            "dataset_path": str(dataset_path),
            "dataset_name": "toy",
            "prompt_key": "prompt",
            "pass_k": 16,
            "temperature": 1.0,
            "top_p": 0.7,
            "max_tokens": 16384,
            "prompt_length": 2048,
            "seed": 42,
            "force_base_prompt": True,
        }
        values.update(overrides)
        return build_generation_cache_provenance(**values)

    expected = provenance()
    assert not generation_parquet_is_complete(
        str(generation_path),
        str(dataset_path),
        "toy",
        16,
        expected_provenance=expected,
    )
    write_generation_cache_manifest(str(generation_path), expected)
    assert generation_parquet_is_complete(
        str(generation_path),
        str(dataset_path),
        "toy",
        16,
        expected_provenance=expected,
    )
    with open(generation_cache_manifest_path(str(generation_path)), encoding="utf-8") as handle:
        sidecar = json.load(handle)
    assert sidecar["provenance"]["sampling"]["max_tokens"] == 16384
    assert sidecar["provenance"]["prompt"]["contract"].startswith("plain_base_completion")
    assert len(sidecar["generation_sha256"]) == 64

    assert not generation_parquet_is_complete(
        str(generation_path),
        str(dataset_path),
        "toy",
        16,
        expected_provenance=provenance(seed=43),
    )

    changed_dataset_path = tmp_path / "changed_source.parquet"
    pd.DataFrame([{"data_source": "toy"}, {"data_source": "changed"}]).to_parquet(
        changed_dataset_path
    )
    assert not generation_parquet_is_complete(
        str(generation_path),
        str(dataset_path),
        "toy",
        16,
        expected_provenance=provenance(dataset_path=str(changed_dataset_path)),
    )

    (model_path / "config.json").write_text('{"model_type":"changed"}', encoding="utf-8")
    assert not generation_parquet_is_complete(
        str(generation_path),
        str(dataset_path),
        "toy",
        16,
        expected_provenance=provenance(),
    )

    # Even with matching provenance, mutating the cached outputs invalidates
    # the sidecar's generation hash.
    _write_generation(generation_path, response_count=16)
    changed = pd.read_parquet(generation_path)
    changed.at[0, "responses"] = ["different"] * 16
    changed.to_parquet(generation_path)
    assert not generation_parquet_is_complete(
        str(generation_path),
        str(dataset_path),
        "toy",
        16,
        expected_provenance=expected,
    )


def test_math_csv_uses_canonical_suite_and_macro():
    entry = {
        "aime25_avg16": 0.05,
        "aime25_pass16": 0.15,
        "aime26_avg16": 0.1,
        "aime26_pass16": 0.2,
        "hmmt26_avg16": 0.3,
        "hmmt26_pass16": 0.4,
        "amobench_avg16": 0.5,
        "amobench_pass16": 0.6,
        "macro_avg16": 0.3,
        "macro_pass16": 0.4,
    }
    row = rows_for_results({"Qwen3-4B_base": entry}, SUMMARY_COLUMNS)[0]
    assert row["aime25_avg@16"] == "0.05"
    assert row["aime26_avg@16"] == "0.1"
    assert row["hmmt26_pass@16"] == "0.4"
    assert row["macro_avg@16"] == "0.3"
    assert "beyondaime_avg@16" not in row


def test_wandb_payload_uses_stable_schema_without_credentials(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        '{"schema_version":"opd_eval_metrics/v1","task":"math","step":25,'
        '"milestone_fraction":0.5,"wandb":{'
        '"eval/math/aime26/avg@16":0.5,"eval/math/aime26/pass@16":0.75,'
        '"eval/math/hmmt26/avg@16":0.5,"eval/math/hmmt26/pass@16":0.75,'
        '"eval/math/amobench/avg@16":0.5,"eval/math/amobench/pass@16":0.75,'
        '"eval/math/macro/avg@16":0.5,"eval/math/macro/pass@16":0.75}}'
    )
    task, payload = load_wandb_payload(str(path), 0.5)
    assert task == "math"
    assert payload["eval/math/macro/pass@16"] == 0.75
    assert payload["eval/milestone_fraction"] == 0.5
    assert all("api" not in key.lower() for key in payload)
    path.write_text(
        '{"schema_version":"opd_eval_metrics/v1","task":"math","step":0,'
        '"milestone_fraction":null,"wandb":{'
        '"eval/math/aime26/avg@16":0.5,"eval/math/aime26/pass@16":0.75,'
        '"eval/math/hmmt26/avg@16":0.5,"eval/math/hmmt26/pass@16":0.75,'
        '"eval/math/amobench/avg@16":0.5,"eval/math/amobench/pass@16":0.75,'
        '"eval/math/macro/avg@16":0.5,"eval/math/macro/pass@16":0.75}}'
    )
    _, base_payload = load_wandb_payload(str(path), None, global_step=0)
    assert "eval/milestone_fraction" not in base_payload


def test_wandb_payload_rejects_wrong_step_or_fraction(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        '{"schema_version":"opd_eval_metrics/v1","task":"math","step":15,'
        '"milestone_fraction":0.25,"wandb":{'
        '"eval/math/macro/avg@16":0.5,"eval/math/macro/pass@16":0.75}}'
    )
    with pytest.raises(ValueError, match="does not match requested global_step"):
        load_wandb_payload(str(path), 0.25, global_step=29)
    with pytest.raises(ValueError, match="does not match requested"):
        load_wandb_payload(str(path), 0.5, global_step=15)


def test_wandb_payload_accepts_partial_eval_suite(tmp_path):
    path = tmp_path / "partial_metrics.json"
    path.write_text(
        '{"schema_version":"opd_eval_metrics/v1","task":"math","step":15,'
        '"milestone_fraction":0.25,"wandb":{'
        '"eval/math/aime26/avg@16":0.5,"eval/math/aime26/pass@16":0.75}}'
    )
    task, payload = load_wandb_payload(str(path), 0.25, global_step=15)
    assert task == "math"
    assert payload["eval/math/aime26/avg@16"] == 0.5
    assert payload["eval/math/aime26/pass@16"] == 0.75
    assert "eval/math/hmmt26/avg@16" not in payload


def test_wandb_logger_uses_custom_global_step_axis(tmp_path, monkeypatch):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        '{"schema_version":"opd_eval_metrics/v1","task":"math","step":25,'
        '"milestone_fraction":0.25,"wandb":{'
        '"eval/math/aime26/avg@16":0.5,"eval/math/aime26/pass@16":0.75,'
        '"eval/math/hmmt26/avg@16":0.5,"eval/math/hmmt26/pass@16":0.75,'
        '"eval/math/amobench/avg@16":0.5,"eval/math/amobench/pass@16":0.75,'
        '"eval/math/macro/avg@16":0.5,"eval/math/macro/pass@16":0.75}}'
    )

    calls = {"definitions": []}

    class FakeRun:
        def define_metric(self, *args, **kwargs):
            calls["definitions"].append((args, kwargs))

        def log(self, payload):
            calls["payload"] = payload

        def finish(self):
            calls["finished"] = True

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_metrics_wandb.py",
            "--metrics_file",
            str(metrics_file),
            "--project",
            "trd",
            "--run_id",
            "run-1",
            "--global_step",
            "25",
            "--milestone_fraction",
            "0.25",
        ],
    )
    wandb_logger.main()
    assert calls["init"]["resume"] == "allow"
    assert calls["init"]["mode"] == "offline"
    assert calls["definitions"] == [
        (("global_step",), {}),
        (("*",), {"step_metric": "global_step"}),
    ]
    assert calls["payload"]["global_step"] == 25
    assert calls["finished"]
