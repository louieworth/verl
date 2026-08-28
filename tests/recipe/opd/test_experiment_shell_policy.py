from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[3]
DISPATCHER = ROOT / "recipe/opd/run/run_experiment.sh"
KL_RUNNER = ROOT / "recipe/opd/run/run_kl_training.sh"
TRAINING_ENTRYPOINT = ROOT / "recipe/opd/run_training.py"
GRPO_RUNNER = ROOT / "recipe/opd/run/grpo/_run_qwen3_grpo_8h100.sh"
GENERATION_SERVER = ROOT / "verl/trainer/main_generation_server.py"
VLLM_SERVER = ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"


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
    assert 'export RESIDENT_STUDENT_ROLLOUT="false"' in source
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


def test_failed_training_flushes_wandb_before_hard_exit():
    source = TRAINING_ENTRYPOINT.read_text(encoding="utf-8")
    failure_handler = source[source.index("    except BaseException:") :]

    finish = failure_handler.index("trainer.finish_tracking(exit_code=1)")
    hard_exit = failure_handler.index("os._exit(1)")
    assert finish < hard_exit


def test_ephemeral_resume_exports_evaluates_then_cleans_without_retraining():
    source = KL_RUNNER.read_text(encoding="utf-8")
    start = source.index("backfill_completed_pipeline_milestone_eval() {")
    end = source.index("\nformat_duration_seconds() {", start)
    backfill = source[start:end]

    checkpoint = backfill.index('pipeline_ephemeral_training_checkpoint_complete "$update"')
    export = backfill.index('export_latest_fsdp_checkpoint_after_training "$ephemeral_model_dir"')
    evaluate = backfill.index("run_post_training_eval_if_needed", export)
    cleanup = backfill.index('cleanup_ephemeral_hf_export "$ephemeral_model_dir/hf_merged"')
    assert checkpoint < export < evaluate < cleanup

    finalize_start = source.index("finalize_resumed_pipeline_update() {")
    finalize_end = source.index("\narchive_or_delete_pipeline_path() {", finalize_start)
    finalize = source[finalize_start:finalize_end]
    assert 'resumed_model_dir="$(pipeline_temp_model_save_dir "$update")"' in finalize
    assert 'run_single_rollout_optimizer_milestone_evals \\\n            "$resumed_model_dir"' in finalize


def test_training_never_runs_inline_resident_sync():
    source = KL_RUNNER.read_text(encoding="utf-8")
    run_epoch_start = source.index("run_epoch() {")
    run_epoch_source = source[run_epoch_start:]

    assert "--sync_resident_rollout false" in run_epoch_source
    assert "sync_resident_rollout_this_update" not in source


def test_isolated_student_rollout_uses_rolling_lora_adapter():
    source = KL_RUNNER.read_text(encoding="utf-8")

    assert 'RESIDENT_STUDENT_ROLLOUT="false"' in source
    assert 'student|skd|skd_vllm)' in source
    assert 'rollout_lora_adapter_path="$prev_ckpt/lora_adapter"' in source
    assert "actor_rollout_ref.model.lora_adapter_path=$student_lora_adapter_path" in source

    generation_source = GENERATION_SERVER.read_text(encoding="utf-8")
    assert "request_model = ROLLING_LORA_MODEL_NAME" in generation_source
    assert "ray.kill(server_handle, no_restart=True)" in generation_source

    vllm_source = VLLM_SERVER.read_text(encoding="utf-8")
    assert 'lora_args["lora_modules"]' in vllm_source
    assert 'f"{VLLM_LORA_NAME}={self.model_config.lora_adapter_path}"' in vllm_source


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
