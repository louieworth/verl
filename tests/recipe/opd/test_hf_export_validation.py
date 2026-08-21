from __future__ import annotations

import subprocess
from pathlib import Path


VALIDATOR = Path(__file__).parents[3] / "recipe/opd/run/hf_export_validation.sh"


def _is_complete(export_dir: Path) -> bool:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; hf_export_complete "$2"',
            "hf-export-test",
            str(VALIDATOR),
            str(export_dir),
        ],
        check=False,
    )
    return result.returncode == 0


def _reached_step(checkpoint_root: Path, expected_step: str) -> bool:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; require_checkpoint_step "$2" "$3"',
            "checkpoint-step-test",
            str(VALIDATOR),
            str(checkpoint_root),
            expected_step,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _prune_checkpoints(checkpoint_root: Path, keep_step: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; prune_old_step_checkpoints "$2" "$3"',
            "checkpoint-prune-test",
            str(VALIDATOR),
            str(checkpoint_root),
            keep_step,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_hf_export_requires_commit_marker_and_nonempty_weights(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert not _is_complete(tmp_path)

    (tmp_path / "opd_export.json").write_text("{}\n", encoding="utf-8")
    assert _is_complete(tmp_path)

    (tmp_path / "model.safetensors").write_bytes(b"")
    assert not _is_complete(tmp_path)


def test_hf_export_rejects_config_only_partial_directory(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    assert not _is_complete(tmp_path)


def test_hf_export_accepts_completed_directory_symlink(tmp_path: Path):
    export_dir = tmp_path / "model"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (export_dir / "opd_export.json").write_text("{}\n", encoding="utf-8")
    (export_dir / "model.safetensors").write_bytes(b"weights")
    link = tmp_path / "final_model"
    link.symlink_to(export_dir, target_is_directory=True)
    assert _is_complete(link)


def test_requested_checkpoint_endpoint_must_be_reached(tmp_path: Path):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("3\n", encoding="utf-8")
    assert _reached_step(tmp_path, "3")
    assert not _reached_step(tmp_path, "15")
    assert _reached_step(tmp_path, "null")


def test_segmented_checkpoint_pruning_keeps_only_resume_endpoint(tmp_path: Path):
    for step in (3, 7, 11):
        step_dir = tmp_path / f"global_step_{step}"
        step_dir.mkdir()
        (step_dir / "shard.bin").write_bytes(b"state")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("11\n", encoding="utf-8")

    assert _prune_checkpoints(tmp_path, "11").returncode == 0
    assert not (tmp_path / "global_step_3").exists()
    assert not (tmp_path / "global_step_7").exists()
    assert (tmp_path / "global_step_11/shard.bin").is_file()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "11\n"


def test_segmented_checkpoint_pruning_rejects_missing_keep_step(tmp_path: Path):
    old = tmp_path / "global_step_3"
    old.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3\n", encoding="utf-8")
    assert _prune_checkpoints(tmp_path, "11").returncode != 0
    assert old.is_dir()


def test_segmented_checkpoint_pruning_rejects_tracker_mismatch_without_deleting(tmp_path: Path):
    old = tmp_path / "global_step_3"
    keep = tmp_path / "global_step_11"
    old.mkdir()
    keep.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3\n", encoding="utf-8")

    assert _prune_checkpoints(tmp_path, "11").returncode != 0
    assert old.is_dir()
    assert keep.is_dir()
