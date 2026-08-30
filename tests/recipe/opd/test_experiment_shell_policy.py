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
WANDB_GUARDIAN = ROOT / "recipe/opd/wandb_run_guardian.py"
WANDB_LIFECYCLE = ROOT / "recipe/opd/run/wandb_run_lifecycle.sh"


def _shell_function(name: str) -> str:
    source = DISPATCHER.read_text(encoding="utf-8")
    match = re.search(rf"(?m)^{re.escape(name)}\(\) \{{\n", source)
    assert match, f"missing shell function {name}"
    end = re.search(r"(?m)^}\n", source[match.end() :])
    assert end, f"unterminated shell function {name}"
    return source[match.start() : match.end() + end.end()]


def test_every_math_leaf_launcher_uses_canonical_dispatcher():
    launcher_roots = [
        ROOT / "recipe/opd/scripts_math/Baselines",
        ROOT / "recipe/opd/scripts_math/OPD",
        ROOT / "recipe/opd/scripts_math/OPSD",
    ]
    launchers = sorted(path for root in launcher_roots for path in root.glob("*/*.sh"))

    assert len(launchers) == 41
    for launcher in launchers:
        source = launcher.read_text(encoding="utf-8")
        assert 'scripts_math/lib/launch_common.sh' in source, launcher


def test_shared_dispatcher_defers_math_training_eval_and_disables_wandb_tags():
    source = DISPATCHER.read_text(encoding="utf-8")
    assert "math/opd/*|math/opsd/*)" in source
    assert "math/baseline/sft|math/baseline/grpo)" in source
    assert 'export MODEL_ARTIFACT_POLICY="milestone_hf_deferred_eval"' in source
    assert 'export PIPELINE_DEFER_MILESTONE_EVALS="true"' in source
    assert 'export MODEL_ARTIFACT_POLICY="ephemeral_eval_only"' in source
    assert 'export PIPELINE_DEFER_MILESTONE_EVALS="false"' in source
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


def test_deferred_math_opd_saves_milestones_then_evaluates_after_training():
    source = KL_RUNNER.read_text(encoding="utf-8")
    run_epoch_start = source.index("run_epoch() {")
    run_epoch_end = source.index("\neval_results_complete() {", run_epoch_start)
    run_epoch = source[run_epoch_start:run_epoch_end]

    assert '[ "$PIPELINE_DEFER_MILESTONE_EVALS" = "true" ]' in run_epoch
    assert 'run_eval_this_update="false"' in run_epoch
    assert 'deferred_hf_export_dir="$current_final_model_save_dir/hf_merged"' in run_epoch
    torchrun_exit = run_epoch.index(') 2>&1 | tee "$current_output_dir/logs/training_')
    durable_export = run_epoch.index('"$current_model_save_dir" "$deferred_hf_export_dir"')
    assert torchrun_exit < durable_export

    training_loop = source.index('for EPOCH in $(seq 1 "$TOTAL_EPOCHS")')
    deferred_eval = source.rindex("run_deferred_pipeline_milestone_evals", training_loop)
    assert training_loop < deferred_eval


def test_deferred_math_baseline_exports_all_models_before_ordered_eval():
    deferred = _shell_function("run_deferred_math_baseline")

    training_loop = deferred.index('echo "Training and exporting all baseline milestones before evaluation"')
    inline_eval_disabled = deferred.index("export RUN_EVAL_AFTER_TRAINING=false", training_loop)
    runner = deferred.index('bash "$runner" "$@"', inline_eval_disabled)
    exported_model = deferred.index('hf_export_complete "$model_dir"', runner)
    eval_phase = deferred.index('echo "All baseline milestone models are ready, starting ordered evaluation"')
    eval_call = deferred.index('run_saved_math_baseline_eval "$BASELINE_MODELS_DIR/step_${step}"', eval_phase)

    assert training_loop < inline_eval_disabled < runner < exported_model < eval_phase < eval_call
    assert 'remove_ephemeral_model_path "$BASELINE_CHECKPOINT_DIR"' in deferred
    assert "cleanup_completed_baseline_models" not in deferred


def test_deferred_math_baseline_runtime_order(tmp_path):
    events = tmp_path / "events.txt"
    runner = tmp_path / "runner.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "train:%s\\n" "$STOP_AT_STEP" >> "$EVENTS_FILE"\n'
        'mkdir -p "$MODELS_DIR/step_${STOP_AT_STEP}"\n'
        ': > "$MODELS_DIR/step_${STOP_AT_STEP}/complete"\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)

    script = f"""
set -euo pipefail
{_shell_function("configure_baseline_milestone")}
{_shell_function("run_deferred_math_baseline")}
RUN_ROOT={shlex.quote(str(tmp_path / "run"))}
BASELINE_MODELS_DIR="$RUN_ROOT/models"
BASELINE_CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
MODELS_DIR="$BASELINE_MODELS_DIR"
GLOBAL_PROMPT_BATCH_SIZE=512
EXPERIMENT_ID=baseline-test
EVENTS_FILE={shlex.quote(str(events))}
export MODELS_DIR EVENTS_FILE
training_milestones() {{
    printf '%s\n' '15 0.25 58' '29 0.5 58' '44 0.75 58' '58 1.0 58'
}}
hf_export_complete() {{
    [ -f "$1/complete" ]
}}
run_saved_math_baseline_eval() {{
    mkdir -p "$(dirname "$EVAL_RESULTS_FILE")"
    printf 'eval:%s\n' "$EVAL_STEP" >> "$EVENTS_FILE"
}}
remove_ephemeral_model_path() {{
    printf 'cleanup\n' >> "$EVENTS_FILE"
}}
run_deferred_math_baseline train.parquet {shlex.quote(str(runner))}
"""
    subprocess.run(["bash", "-c", script], cwd=ROOT, check=True)

    assert events.read_text(encoding="utf-8").splitlines() == [
        "train:15",
        "train:29",
        "train:44",
        "train:58",
        "eval:15",
        "eval:29",
        "eval:44",
        "eval:58",
        "cleanup",
    ]


def test_failed_training_flushes_wandb_before_hard_exit():
    source = TRAINING_ENTRYPOINT.read_text(encoding="utf-8")
    failure_handler = source[source.index("    except BaseException:") :]

    finish = failure_handler.index("trainer.finish_tracking(exit_code=1)")
    hard_exit = failure_handler.index("os._exit(1)")
    assert finish < hard_exit


def test_wandb_run_finishes_only_from_pipeline_exit_traps():
    dispatcher = DISPATCHER.read_text(encoding="utf-8")
    kl_runner = KL_RUNNER.read_text(encoding="utf-8")
    lifecycle = WANDB_LIFECYCLE.read_text(encoding="utf-8")
    guardian = WANDB_GUARDIAN.read_text(encoding="utf-8")

    assert 'start_wandb_run_guardian "$RUN_ROOT"' in dispatcher
    assert "trap on_baseline_exit EXIT" in dispatcher
    assert 'start_wandb_run_guardian "$MODEL_SAVE_BASE_DIR"' in kl_runner
    assert "trap on_training_exit EXIT" in kl_runner
    assert 'finish_wandb_run_guardian "$status"' in dispatcher
    assert 'finish_wandb_run_guardian "$status"' in kl_runner
    assert "export WANDB_SHARED_RUN=1" in lifecycle
    assert 'run.finish(exit_code=exit_code)' in guardian


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
    assert "shutdown_rollout_servers(rollout_servers)" in generation_source
    assert "worker_actors = [worker for replica in rollout_servers for worker in replica.workers]" in generation_source

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
