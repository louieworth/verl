from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[3]
DISPATCHER = ROOT / "recipe/opd/run/run_experiment.sh"
KL_RUNNER = ROOT / "recipe/opd/run/run_kl_training.sh"
GRPO_RUNNER = ROOT / "recipe/opd/run/grpo/_run_qwen3_grpo_8h100.sh"


def _shell_function(name: str) -> str:
    source = DISPATCHER.read_text(encoding="utf-8")
    match = re.search(rf"(?m)^{re.escape(name)}\(\) \{{\n", source)
    assert match, f"missing shell function {name}"
    end = re.search(r"(?m)^}\n", source[match.end() :])
    assert end, f"unterminated shell function {name}"
    return source[match.start() : match.end() + end.end()]


def test_shared_dispatcher_enforces_ephemeral_models_and_no_wandb_tags():
    source = DISPATCHER.read_text(encoding="utf-8")
    assert 'export MODEL_ARTIFACT_POLICY="ephemeral_eval_only"' in source
    assert 'export PIPELINE_EPHEMERAL_MODELS="true"' in source
    assert 'export SAVE_MERGED_MODEL="false"' in source
    assert "canonical_wandb_tags" not in source
    assert "unset WANDB_TAGS" in source


def test_canonical_grpo_disables_ordinary_step_checkpoints():
    source = DISPATCHER.read_text(encoding="utf-8")
    grpo_case = source[source.index("        grpo)") :]

    assert "export SAVE_FREQ=-1" in grpo_case
    assert "export SAVE_AT_END=false" in grpo_case


def test_hf_exports_run_only_after_training_processes_exit():
    kl_source = KL_RUNNER.read_text(encoding="utf-8")
    assert "--save_merged_model false" in kl_source
    assert "--async_hf_export false" in kl_source
    assert "--run_eval_after_training false" in kl_source
    assert "--save_merged_model $save_merged_this_update" not in kl_source
    torchrun_exit = kl_source.index(') 2>&1 | tee "$current_output_dir/logs/training_')
    shell_export = kl_source.index(
        'export_latest_fsdp_checkpoint_after_training "$current_model_save_dir"',
        torchrun_exit,
    )
    resident_stop = kl_source.index("stop_managed_resident_y_o_server", torchrun_exit)
    assert torchrun_exit < resident_stop < shell_export

    grpo_source = GRPO_RUNNER.read_text(encoding="utf-8")
    training_exit = grpo_source.index('"${TRAIN_COMMAND[@]}" 2>&1 | tee')
    shell_export = grpo_source.index('current_hf_model="$(latest_hf_checkpoint', training_exit)
    assert training_exit < shell_export


def test_ephemeral_cleanup_is_confined_to_run_root(tmp_path: Path):
    run_root = tmp_path / "run"
    inside = run_root / "models" / "step_15"
    outside = tmp_path / "outside" / "step_15"
    inside.mkdir(parents=True)
    outside.mkdir(parents=True)
    function_source = _shell_function("remove_ephemeral_model_path")
    script = f"""
set -euo pipefail
RUN_ROOT={shlex.quote(str(run_root))}
PYTHON_BIN=python3
{function_source}
remove_ephemeral_model_path {shlex.quote(str(inside))}
if remove_ephemeral_model_path {shlex.quote(str(outside))}; then
    exit 9
fi
"""
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert not inside.exists()
    assert outside.is_dir()
