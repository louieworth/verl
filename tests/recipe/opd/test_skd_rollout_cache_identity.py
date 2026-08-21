from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
RUNNER = ROOT / "recipe/opd/run/run_kl_training.sh"

SKD_FIELDS = {
    "skd_gamma": ("SKD_GAMMA", "7"),
    "skd_accept_top_k": ("SKD_ACCEPT_TOP_K", "31"),
    "skd_accept_top_p": ("SKD_ACCEPT_TOP_P", "0.91"),
    "skd_student_temperature": ("SKD_STUDENT_TEMPERATURE", "0.72"),
    "skd_student_top_p": ("SKD_STUDENT_TOP_P", "0.92"),
    "skd_teacher_temperature": ("SKD_TEACHER_TEMPERATURE", "0.23"),
    "skd_teacher_top_p": ("SKD_TEACHER_TOP_P", "0.93"),
    "skd_rollout_batch_size": ("SKD_ROLLOUT_BATCH_SIZE", "96"),
    "skd_pipeline_lanes": ("SKD_PIPELINE_LANES", "3"),
    "skd_parallel_student_teacher": ("SKD_PARALLEL_STUDENT_TEACHER", "false"),
    "skd_teacher_prompt_contract": ("SKD_TEACHER_PROMPT_CONTRACT", "opsd_x_y_star_v1"),
    "skd_teacher_prompt_length": ("SKD_TEACHER_PROMPT_LENGTH", "6144"),
    "skd_rollout_max_model_len": ("SKD_ROLLOUT_MAX_MODEL_LEN", "22528"),
}


def _shell_function(name: str) -> str:
    source = RUNNER.read_text(encoding="utf-8")
    match = re.search(rf"(?m)^{re.escape(name)}\(\) \{{\n", source)
    assert match, f"missing shell function {name}"
    end = re.search(r"(?m)^}\n", source[match.end() :])
    assert end, f"unterminated shell function {name}"
    return source[match.start() : match.end() + end.end()]


def _run_bash(body: str, *, functions: tuple[str, ...], env: dict[str, str] | None = None):
    script = "set -euo pipefail\n" + "\n".join(_shell_function(name) for name in functions) + body
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )


def _signature(mode: str, **overrides: str) -> tuple[str, str]:
    assignments = {
        "Y_O_ROLLOUT_MODE": mode,
        **{env_name: value for env_name, value in SKD_FIELDS.values()},
        **overrides,
    }
    shell_assignments = "\n".join(f"{name}={value!r}" for name, value in assignments.items())
    # The signature function has many non-SKD inputs. Give all otherwise-unset
    # variables an empty value without weakening nounset checks in the function.
    function_source = _shell_function("compute_gen_results_signature")
    referenced = set(re.findall(r"\$([A-Z][A-Z0-9_]*)", function_source))
    referenced.update(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", function_source))
    empty_defaults = "\n".join(f": \"${{{name}:=}}\"" for name in sorted(referenced))
    result = _run_bash(
        f"\n{empty_defaults}\n{shell_assignments}\n"
        "compute_gen_results_signature\n"
        "printf '%s\\n__DETAILS__\\n%s\\n' \"$GEN_RESULTS_RUN_SIGNATURE\" \"$GEN_RESULTS_RUN_SIGNATURE_CONTENT\"\n",
        functions=("compute_gen_results_signature",),
    )
    assert result.returncode == 0, result.stderr
    digest, details = result.stdout.split("\n__DETAILS__\n", 1)
    return digest, details


@pytest.mark.parametrize("mode", ("skd", "skd_vllm", "skd_vllm_internal"))
def test_skd_signature_contains_every_generation_parameter(mode: str):
    digest, details = _signature(mode)
    changed_digest, _ = _signature(mode, SKD_GAMMA="8")

    assert digest != changed_digest
    for key, (_, value) in SKD_FIELDS.items():
        assert f"{key}={value}\n" in details


def test_non_skd_signature_ignores_skd_environment_values():
    digest, details = _signature("student")
    changed_digest, changed_details = _signature(
        "student",
        SKD_GAMMA="999",
        SKD_ACCEPT_TOP_K="1",
        SKD_PARALLEL_STUDENT_TEACHER="true",
    )

    assert digest == changed_digest
    assert details == changed_details
    assert not any(f"{key}=" in details for key in SKD_FIELDS)


def _base_metadata(mode: str, *, include_skd: bool) -> dict[str, str]:
    metadata = {
        "task": "math",
        "distill_mode": "opd",
        "prompt_contract_version": "plain_base_completion_v3",
        "student_model_name": "student",
        "student_model_path": "/models/student",
        "teacher_model_name": "teacher",
        "teacher_model_path": "/models/teacher",
        "y_mode": "y_o",
        "y_o_rollout_mode": mode,
        "trajectory_model_path": "",
        "train_data_path": "/data/train.parquet",
        "max_samples": "all",
        "multi_step": "4",
        "base_prompt_length": "2048",
        "max_response_length": "16384",
        "rollout_temperature": "1.0",
        "rollout_top_p": "1.0",
        "rollout_top_k": "-1",
    }
    if include_skd:
        metadata.update({key: value for key, (_, value) in SKD_FIELDS.items()})
    return metadata


def _metadata_matches(path: Path, mode: str) -> bool:
    globals_ = {
        "TASK": "math",
        "DISTILL_MODE": "opd",
        "PROMPT_CONTRACT_VERSION": "plain_base_completion_v3",
        "MODEL_NAME": "student",
        "MODEL_PATH": "/models/student",
        "TEACHER_MODEL_NAME": "teacher",
        "TEACHER_MODEL_PATH": "/models/teacher",
        "Y_O_ROLLOUT_MODE": mode,
        "TRAJECTORY_MODEL_PATH": "",
        "TRAIN_DATA_PATH": "/data/train.parquet",
        "MAX_SAMPLES": "all",
        "MULTI_STEP": "4",
        "BASE_PROMPT_LENGTH": "2048",
        "MAX_RESPONSE_LENGTH": "16384",
        "ROLLOUT_TEMPERATURE": "1.0",
        "ROLLOUT_TOP_P": "1.0",
        "ROLLOUT_TOP_K": "-1",
        "STAGE2_PROMPT_LENGTH": "6144",
        **{env_name: value for env_name, value in SKD_FIELDS.values()},
    }
    assignments = "\n".join(f"{name}={value!r}" for name, value in globals_.items())
    result = _run_bash(
        f"\n{assignments}\nmetadata_matches_step1_reuse_context {str(path)!r} y_o\n",
        functions=("gen_results_metadata_value", "metadata_matches_step1_reuse_context"),
    )
    return result.returncode == 0


def _write_metadata(path: Path, metadata: dict[str, str]) -> None:
    path.write_text("".join(f"{key}: {value}\n" for key, value in metadata.items()), encoding="utf-8")


def test_step1_reuse_strictly_matches_each_skd_parameter(tmp_path: Path):
    metadata = _base_metadata("skd_vllm", include_skd=True)
    path = tmp_path / "run_metadata.yaml"
    _write_metadata(path, metadata)
    assert _metadata_matches(path, "skd_vllm")

    for key in SKD_FIELDS:
        mismatched = dict(metadata)
        mismatched[key] = "different"
        _write_metadata(path, mismatched)
        assert not _metadata_matches(path, "skd_vllm"), key


def test_step1_reuse_does_not_require_skd_metadata_for_non_skd_mode(tmp_path: Path):
    path = tmp_path / "run_metadata.yaml"
    _write_metadata(path, _base_metadata("student", include_skd=False))
    assert _metadata_matches(path, "student")


def test_skd_metadata_is_emitted_by_run_and_training_yaml():
    source = RUNNER.read_text(encoding="utf-8")
    block = re.search(r'SKD_ROLLOUT_METADATA_YAML="\$\(cat <<EOF\n(.*?)\nEOF\n\)"', source, re.DOTALL)
    assert block is not None
    for key, (env_name, _) in SKD_FIELDS.items():
        assert f"{key}: ${env_name}" in block.group(1)
    assert source.count("${SKD_ROLLOUT_METADATA_YAML}") == 2
