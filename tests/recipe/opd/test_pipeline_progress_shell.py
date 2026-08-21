from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[3]
RUNNER = ROOT / "recipe/opd/run/run_kl_training.sh"
HF_VALIDATION = ROOT / "recipe/opd/run/hf_export_validation.sh"


def _shell_function(name: str) -> str:
    source = RUNNER.read_text(encoding="utf-8")
    match = re.search(rf"(?m)^{re.escape(name)}\(\) \{{\n", source)
    assert match, f"missing shell function {name}"
    end = re.search(r"(?m)^}\n", source[match.end() :])
    assert end, f"unterminated shell function {name}"
    return source[match.start() : match.end() + end.end()]


def _run_shell(functions: tuple[str, ...], body: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    function_source = "\n".join(_shell_function(name) for name in functions)
    script = f"""
set -euo pipefail
source {HF_VALIDATION}
PIPELINE_DONE_MARKER_SCHEMA=opd_pipeline_update/v1
PYTHON_BIN={sys.executable}
pipeline_progress_dir() {{ printf '%s\\n' "$TEST_PROGRESS_DIR"; }}
format_pipeline_batch_id() {{ printf '%05d' "$1"; }}
find_latest_fsdp_checkpoint() {{ find "$1" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort | tail -1; }}
{function_source}
{body}
"""
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, **env},
    )


def _write_marker(path: Path, **overrides: object) -> None:
    values: dict[str, object] = {
        "schema": "opd_pipeline_update/v1",
        "step": 1,
        "status": "rolling_temp",
        "model_path": "",
        "optimizer_step_offset": 0,
        "segment_optimizer_steps": 2,
        "global_optimizer_step": 2,
    }
    values.update(overrides)
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")


MARKER_VALIDATORS = (
    "read_pipeline_done_marker_field",
    "pipeline_done_marker",
    "pipeline_done_marker_structurally_valid",
    "pipeline_fsdp_checkpoint_complete",
    "pipeline_done_model_artifact_complete",
    "pipeline_done_marker_semantically_valid",
)


def test_explicit_ms_balances_every_math_row_across_exactly_40_steps():
    result = _run_shell(
        ("pipeline_partition_start_size",),
        """
for step in $(seq 1 40); do
    pipeline_partition_start_size "$step"
done
""",
        {
            "TEST_PROGRESS_DIR": "/tmp/unused-progress",
            "PIPELINE_STEP_MODE": "explicit_steps",
            "PIPELINE_AUTO_CHUNK_SIZE": "736",
            "PIPELINE_BALANCED_REMAINDER": "32",
            "PIPELINE_BALANCED_INCREMENT": "8",
            "TOTAL_TRAIN_SAMPLES": "29696",
        },
    )
    assert result.returncode == 0, result.stderr

    partitions = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    assert len(partitions) == 40
    assert [size for _, size in partitions[:32]] == [744] * 32
    assert [size for _, size in partitions[32:]] == [736] * 8
    assert partitions[0][0] == 0
    assert all(
        next_start == start + size
        for (start, size), (next_start, _) in zip(partitions, partitions[1:])
    )
    assert partitions[-1][0] + partitions[-1][1] == 29696


def test_default_batch_keeps_short_final_tail():
    result = _run_shell(
        ("pipeline_partition_start_size",),
        "pipeline_partition_start_size 1; pipeline_partition_start_size 2",
        {
            "TEST_PROGRESS_DIR": "/tmp/unused-progress",
            "PIPELINE_STEP_MODE": "default_batch",
            "PIPELINE_AUTO_CHUNK_SIZE": "512",
            "PIPELINE_BALANCED_REMAINDER": "0",
            "TOTAL_TRAIN_SAMPLES": "1000",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["0 512", "512 488"]


def test_marker_schema_and_all_fsdp_ranks_are_required(tmp_path: Path):
    progress = tmp_path / "progress"
    progress.mkdir()
    checkpoint = tmp_path / "global_step_2"
    checkpoint.mkdir()
    (checkpoint / "fsdp_config.json").write_text(json.dumps({"world_size": 2}), encoding="utf-8")
    for rank in range(2):
        for kind in ("model", "optim", "extra_state"):
            (checkpoint / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"state")
    marker = progress / "step00001.done"
    _write_marker(marker, model_path=checkpoint)

    result = _run_shell(
        MARKER_VALIDATORS,
        "pipeline_done_marker_structurally_valid 1; pipeline_done_marker_semantically_valid 1",
        {"TEST_PROGRESS_DIR": str(progress)},
    )
    assert result.returncode == 0, result.stderr

    (checkpoint / "extra_state_world_size_2_rank_1.pt").unlink()
    result = _run_shell(
        MARKER_VALIDATORS,
        "pipeline_done_marker_structurally_valid 1; ! pipeline_done_marker_semantically_valid 1",
        {"TEST_PROGRESS_DIR": str(progress)},
    )
    assert result.returncode == 0, result.stderr

    marker.write_text(f"step=1\nstatus=rolling_temp\nmodel_path={checkpoint}\n", encoding="utf-8")
    result = _run_shell(
        MARKER_VALIDATORS,
        "! pipeline_done_marker_structurally_valid 1",
        {"TEST_PROGRESS_DIR": str(progress)},
    )
    assert result.returncode == 0, result.stderr


def test_kept_marker_requires_strict_hf_artifact_and_cannot_use_frontier(tmp_path: Path):
    progress = tmp_path / "progress"
    progress.mkdir()
    model = tmp_path / "hf_model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "opd_export.json").write_text("{}\n", encoding="utf-8")
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights")
    _write_marker(progress / "step00001.done", status="kept", model_path=model)

    result = _run_shell(
        MARKER_VALIDATORS + ("pipeline_done_marker_can_use_frontier",),
        "pipeline_done_marker_semantically_valid 1; ! pipeline_done_marker_can_use_frontier kept; "
        "pipeline_done_marker_can_use_frontier rolling_temp",
        {"TEST_PROGRESS_DIR": str(progress)},
    )
    assert result.returncode == 0, result.stderr

    weights.unlink()
    result = _run_shell(
        MARKER_VALIDATORS,
        "! pipeline_done_marker_semantically_valid 1",
        {"TEST_PROGRESS_DIR": str(progress)},
    )
    assert result.returncode == 0, result.stderr


def test_marker_commit_uses_atomic_temp_rename_and_preserves_old_marker_on_failure(tmp_path: Path):
    progress = tmp_path / "progress"
    progress.mkdir()
    marker = progress / "step00002.done"
    common_stubs = """
RESIDENT_STUDENT_ROLLOUT=false
PIPELINE_FULL_BATCH_OPTIMIZER_STEPS=2
PIPELINE_TAIL_OPTIMIZER_STEPS=2
pipeline_optimizer_step_offset() { echo 4; }
pipeline_checkpoint_for_update() { echo /checkpoint/global_step_2; }
pipeline_should_keep_update() { return 1; }
model_dir_for_global_update() { echo /models/step2; }
persist_pipeline_model_for_resume() { echo /models/step2/hf_merged; }
pipeline_resume_model_dir() { echo /resume/other/hf_merged; }
pipeline_done_model_artifact_complete() { return 0; }
pipeline_update_resume_checkpoint_complete() { return 0; }
write_pipeline_latest_model_state() { :; }
"""
    result = _run_shell(
        ("pipeline_done_marker", "mark_pipeline_update_done"),
        common_stubs + "\nmark_pipeline_update_done 2 4 4",
        {"TEST_PROGRESS_DIR": str(progress)},
    )
    assert result.returncode == 0, result.stderr
    fields = marker.read_text(encoding="utf-8").splitlines()
    assert fields[0] == "schema=opd_pipeline_update/v1"
    assert len(fields) == 7
    assert not list(progress.glob("*.tmp.*"))

    marker.write_text("old-committed-marker\n", encoding="utf-8")
    result = _run_shell(
        ("pipeline_done_marker", "mark_pipeline_update_done"),
        common_stubs + "\nmv() { return 1; }\n! mark_pipeline_update_done 2 4 4",
        {"TEST_PROGRESS_DIR": str(progress)},
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == "old-committed-marker\n"
    assert not list(progress.glob("*.tmp.*"))


def test_resident_temp_prune_never_deletes_future_frontier(tmp_path: Path):
    temp_root = tmp_path / "temp"
    for step in (1, 3, 5):
        (temp_root / f"step{step:05d}").mkdir(parents=True)
    result = _run_shell(
        ("prune_pipeline_temp_checkpoints",),
        "prune_pipeline_temp_checkpoints 3 10",
        {
            "TEST_PROGRESS_DIR": str(tmp_path / "unused"),
            "RESIDENT_STUDENT_ROLLOUT": "true",
            "PIPELINE_TEMP_MODEL_DIR": str(temp_root),
        },
    )
    assert result.returncode == 0, result.stderr
    assert not (temp_root / "step00001").exists()
    assert (temp_root / "step00003").is_dir()
    assert (temp_root / "step00005").is_dir()


def test_persist_resume_model_recovers_move_before_marker_crash(tmp_path: Path):
    source_model = tmp_path / "canonical" / "hf_merged"
    resume_model = tmp_path / "resume" / "step00001" / "hf_merged"
    resume_model.mkdir(parents=True)
    (resume_model / "config.json").write_text("{}\n", encoding="utf-8")
    (resume_model / "opd_export.json").write_text("{}\n", encoding="utf-8")
    (resume_model / "model.safetensors").write_bytes(b"weights")
    result = _run_shell(
        ("persist_pipeline_model_for_resume",),
        f"""
PIPELINE_STORE_RESUME_MODEL_IN_GEN_RESULTS=true
pipeline_should_keep_update() {{ return 1; }}
pipeline_resume_model_dir() {{ printf '%s\\n' {resume_model}; }}
cleanup_old_gen_results_resume_models() {{ :; }}
recovered="$(persist_pipeline_model_for_resume 1 4 {source_model})"
[ "$recovered" = {resume_model} ]
""",
        {"TEST_PROGRESS_DIR": str(tmp_path / "unused")},
    )
    assert result.returncode == 0, result.stderr
    assert resume_model.is_dir()


def test_multistep_async_export_is_explicitly_rejected():
    source = RUNNER.read_text(encoding="utf-8")
    guard = re.search(
        r'if \[ "\$MULTI_STEP" -gt 0 \]; then.*?ASYNC_HF_EXPORT.*?not crash-safe',
        source,
        flags=re.DOTALL,
    )
    assert guard is not None


def test_code_eval_resume_requires_current_signature_directory(tmp_path: Path):
    output_dir = tmp_path / "eval"
    signature = "n16_t0.6_p0.95_prompt2048_response16384_seed42_base"
    current_result = output_dir / "evalplus" / signature / "humaneval" / "run_eval_results.json"
    current_result.parent.mkdir(parents=True)
    current_result.write_text("{}\n", encoding="utf-8")
    results_file = tmp_path / "results.json"
    results_file.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "humaneval_plus_avg16": 0.25,
                    "humaneval_plus_pass16": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_shell(
        ("eval_results_complete", "eval_missing_datasets"),
        'eval_results_complete checkpoint "$TEST_EVAL_DIR"; '
        '[ -z "$(eval_missing_datasets checkpoint "$TEST_EVAL_DIR")" ]',
        {
            "TEST_PROGRESS_DIR": str(tmp_path / "unused"),
            "TASK": "code",
            "RESULTS_FILE": str(results_file),
            "EVAL_DATASETS": "humaneval_plus",
            "PASS_K": "16",
            "CODE_EVAL_SIGNATURE": signature,
            "TEST_EVAL_DIR": str(output_dir),
        },
    )
    assert result.returncode == 0, result.stderr

    current_result.unlink()
    legacy_result = output_dir / "evalplus" / "humaneval" / "legacy_eval_results.json"
    legacy_result.parent.mkdir(parents=True)
    legacy_result.write_text("{}\n", encoding="utf-8")
    result = _run_shell(
        ("eval_results_complete", "eval_missing_datasets"),
        '! eval_results_complete checkpoint "$TEST_EVAL_DIR"; '
        '[ "$(eval_missing_datasets checkpoint "$TEST_EVAL_DIR")" = humaneval_plus ]',
        {
            "TEST_PROGRESS_DIR": str(tmp_path / "unused"),
            "TASK": "code",
            "RESULTS_FILE": str(results_file),
            "EVAL_DATASETS": "humaneval_plus",
            "PASS_K": "16",
            "CODE_EVAL_SIGNATURE": signature,
            "TEST_EVAL_DIR": str(output_dir),
        },
    )
    assert result.returncode == 0, result.stderr


def test_math_eval_resume_requires_only_canonical_avg16_and_pass16(tmp_path: Path):
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "aime26_pass16_generation.parquet").write_bytes(b"parquet")
    results_file = tmp_path / "results.json"
    results_file.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "aime26_avg_pass1_generation_pass_16": 0.25,
                    "aime26_pass16_generation_pass_16": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_shell(
        ("eval_results_complete", "eval_missing_datasets"),
        'eval_results_complete checkpoint "$TEST_EVAL_DIR"; '
        '[ -z "$(eval_missing_datasets checkpoint "$TEST_EVAL_DIR")" ]',
        {
            "TEST_PROGRESS_DIR": str(tmp_path / "unused"),
            "TASK": "math",
            "RESULTS_FILE": str(results_file),
            "EVAL_DATASETS": "aime26",
            "PASS_K": "16",
            "TEST_EVAL_DIR": str(output_dir),
        },
    )
    assert result.returncode == 0, result.stderr
