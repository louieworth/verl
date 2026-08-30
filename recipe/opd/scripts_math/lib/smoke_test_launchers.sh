#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$OPD_ROOT/../.." && pwd)"
SMOKE_TMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$SMOKE_TMP_DIR"' EXIT

write_qwen3_config() {
    local output_dir="$1"
    local hidden_size="$2"
    local num_layers="$3"
    local provenance="$4"
    mkdir -p "$output_dir"
    printf '%s\n' \
        "{\"architectures\":[\"Qwen3ForCausalLM\"],\"model_type\":\"qwen3\",\"hidden_size\":$hidden_size,\"num_hidden_layers\":$num_layers,\"_name_or_path\":\"$provenance\"}" \
        > "$output_dir/config.json"
    : > "$output_dir/model.safetensors.index.json"
    : > "$output_dir/tokenizer_config.json"
    : > "$output_dir/tokenizer.json"
}

launchers=()
while IFS= read -r launcher; do
    launchers+=("$launcher")
done < <(
    find "$OPD_ROOT/scripts_math" "$OPD_ROOT/script_code" \
        -mindepth 3 -maxdepth 3 -type f -name '*.sh' | sort
)
if [ "${#launchers[@]}" -ne 80 ]; then
    echo "ERROR: expected 80 leaf launchers, found ${#launchers[@]}" >&2
    exit 1
fi

shell_files=()
while IFS= read -r shell_file; do
    shell_files+=("$shell_file")
done < <(
    find "$OPD_ROOT/scripts_math/Baselines" "$OPD_ROOT/scripts_math/OPD" "$OPD_ROOT/scripts_math/OPSD" \
         "$OPD_ROOT/script_code" "$OPD_ROOT/scripts_math/lib" \
         -type f -name '*.sh' | sort
)
for shell_file in "${shell_files[@]}" "$OPD_ROOT/scripts_math/run_matrix.sh"; do
    bash -n "$shell_file"
done

grep -Fxq '        export MODEL_ARTIFACT_POLICY="ephemeral_eval_only"' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fxq '        export MODEL_ARTIFACT_POLICY="milestone_hf_deferred_eval"' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fxq '        export PIPELINE_DEFER_MILESTONE_EVALS="true"' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fxq 'export PIPELINE_EPHEMERAL_MODELS="true"' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fxq 'export SAVE_MERGED_MODEL="false"' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fxq 'export RESIDENT_STUDENT_ROLLOUT="false"' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fq 'export SAVE_FREQ=-1' "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fq 'export SAVE_AT_END=false' "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fxq 'unset WANDB_TAGS' "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fq -- '--save_merged_model false' "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
grep -Fq -- '--async_hf_export false' "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
grep -Fq -- '--run_eval_after_training false' "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
grep -Fq -- '--sync_resident_rollout false' "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
grep -Fq 'export_latest_fsdp_checkpoint_after_training "$current_model_save_dir"' \
    "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
grep -Fq 'run_eval_with_wandb.py' \
    "$REPO_ROOT/recipe/math_evaluation/benchmark_kl_model.sh"
grep -Fq 'EVAL_WANDB_WRAPPED' \
    "$REPO_ROOT/recipe/math_evaluation/benchmark_kl_model.sh"
grep -Fq 'source "$REPO_ROOT/recipe/opd/run/wandb_run_lifecycle.sh"' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fq 'trap on_baseline_exit EXIT' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fq 'start_wandb_run_guardian "$RUN_ROOT"' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh"
grep -Fq 'source "$RUN_KL_SCRIPT_DIR/wandb_run_lifecycle.sh"' \
    "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
grep -Fq 'trap on_training_exit EXIT' \
    "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
grep -Fq 'start_wandb_run_guardian "$MODEL_SAVE_BASE_DIR"' \
    "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
grep -Fq 'stop_managed_resident_y_o_server' "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"
if grep -Fq -- '--save_merged_model $save_merged_this_update' \
    "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"; then
    echo "ERROR: OPD still launches HF export inside live training ranks" >&2
    exit 1
fi
if grep -Fq 'sync_resident_rollout_this_update' \
    "$REPO_ROOT/recipe/opd/run/run_kl_training.sh"; then
    echo "ERROR: OPD still syncs resident rollout inside live training ranks" >&2
    exit 1
fi
if rg -q 'canonical_wandb_tags|metadata\["tags"\]' \
    "$REPO_ROOT/recipe/opd/run/run_experiment.sh" \
    "$REPO_ROOT/verl/utils/wandb_metadata.py"; then
    echo "ERROR: W&B tag construction is still enabled" >&2
    exit 1
fi

for launcher in "${launchers[@]}"; do
    grep -Fq 'scripts_math/lib/launch_common.sh' "$launcher"
    grep -q '^export OPD_TASK=' "$launcher"
    grep -q '^export OPD_FAMILY=' "$launcher"
    grep -q '^export OPD_VARIANT=' "$launcher"
    grep -q '^export OPD_MODEL_SIZE=' "$launcher"
    grep -q '^export MODEL_PATH=' "$launcher"

    relative_launcher="${launcher#"$OPD_ROOT/"}"
    IFS=/ read -r launcher_tree family_dir model_dir launcher_file <<<"$relative_launcher"
    case "$launcher_tree" in
        scripts_math) expected_task=math ;;
        script_code) expected_task=code ;;
        *) echo "ERROR: unexpected launcher tree: $relative_launcher" >&2; exit 1 ;;
    esac
    case "$family_dir" in
        Baselines) expected_family=baseline ;;
        OPD) expected_family=opd ;;
        OPSD) expected_family=opsd ;;
        *) echo "ERROR: unexpected launcher family: $relative_launcher" >&2; exit 1 ;;
    esac
    launcher_stem="${launcher_file%.sh}"
    expected_teacher_thinking=false
    case "$launcher_stem" in
        *_thinking)
            expected_variant="${launcher_stem%_thinking}"
            expected_teacher_thinking=true
            ;;
        *) expected_variant="$launcher_stem" ;;
    esac
    grep -Fxq "export OPD_TASK=\"$expected_task\"" "$launcher"
    grep -Fxq "export OPD_FAMILY=\"$expected_family\"" "$launcher"
    grep -Fxq "export OPD_VARIANT=\"$expected_variant\"" "$launcher"
    grep -Fxq "export OPD_MODEL_SIZE=\"$model_dir\"" "$launcher"
    grep -Fxq 'export MULTI_STEP="${MULTI_STEP:-0}"' "$launcher"
    if [ "$expected_task" = code ] && [ "$expected_family" = baseline ] && [ "$expected_variant" = grpo ]; then
        grep -Fxq 'export CODE_GRPO_REWARD_CONTRACT="deepcoder_binary_15_longest_v1"' "$launcher"
        grep -Fxq 'export CODE_GRPO_MAX_TEST_CASES="15"' "$launcher"
        grep -Fxq 'export CODE_GRPO_EXEC_TIMEOUT_SECONDS="10"' "$launcher"
    fi
    if [ "$expected_family" = baseline ] && [ "$expected_variant" = grpo ]; then
        grep -Fxq 'export ROLLOUT_N="${ROLLOUT_N:-8}"' "$launcher"
    fi
    case "$family_dir" in
        OPD)
            grep -Fxq 'export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}"' "$launcher"
            grep -Fxq 'export DISTILL_MODE="opd"' "$launcher"
            grep -Fxq "export TEACHER_ENABLE_THINKING=\"\${TEACHER_ENABLE_THINKING:-$expected_teacher_thinking}\"" "$launcher"
            ;;
        OPSD)
            grep -Fxq 'export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$MODEL_PATH}"' "$launcher"
            grep -Fxq 'export DISTILL_MODE="opsd"' "$launcher"
            ! grep -q '^export TEACHER_ENABLE_THINKING=' "$launcher"
            ;;
    esac
done

if opsd_thinking_output="$(
    TEACHER_ENABLE_THINKING=true DRY_RUN=1 \
        bash "$OPD_ROOT/scripts_math/OPSD/1B/trd.sh" 2>&1
)"; then
    echo "ERROR: OPSD accepted TEACHER_ENABLE_THINKING=true" >&2
    exit 1
fi
grep -q 'OPSD uses a frozen Base self-teacher and cannot enable teacher thinking mode' \
    <<<"$opsd_thinking_output"

opd_nonthinking_output="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPD/1B/trd.sh")"
grep -q 'teacher thinking:    false' <<<"$opd_nonthinking_output"
grep -q 'teacher supervision: plain_completion' <<<"$opd_nonthinking_output"
grep -q 'teacher rollout:     plain_completion' <<<"$opd_nonthinking_output"
grep -q 'W&B project/run:     opd-math / opd-math-trd-Qwen3-1.7B-Base-teacher-thinking-false-ms0-' <<<"$opd_nonthinking_output"
grep -q 'run root:            .*teacher-thinking-false' <<<"$opd_nonthinking_output"
grep -q 'results file:        .*teacher-thinking-false' <<<"$opd_nonthinking_output"
grep -q 'eval output:         .*teacher-thinking-false' <<<"$opd_nonthinking_output"
opd_thinking_output="$(
    TEACHER_ENABLE_THINKING=true DRY_RUN=1 \
        bash "$OPD_ROOT/scripts_math/OPD/1B/trd.sh"
)"
grep -q 'teacher thinking:    true' <<<"$opd_thinking_output"
grep -q 'teacher supervision: qwen3_chat_thinking' <<<"$opd_thinking_output"
grep -q 'teacher rollout:     qwen3_chat_thinking' <<<"$opd_thinking_output"

vanilla_thinking_output="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPD/1B/vanilla_thinking.sh")"
grep -q 'task/family/variant: math/opd/vanilla' <<<"$vanilla_thinking_output"
grep -q 'teacher thinking:    true' <<<"$vanilla_thinking_output"
grep -q 'teacher supervision: qwen3_chat_thinking' <<<"$vanilla_thinking_output"
grep -q 'teacher rollout:     qwen3_chat_thinking' <<<"$vanilla_thinking_output"

trd_thinking_output="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPD/1B/trd_thinking.sh")"
grep -q 'task/family/variant: math/opd/trd' <<<"$trd_thinking_output"
grep -q 'teacher thinking:    true' <<<"$trd_thinking_output"
grep -q 'teacher supervision: qwen3_chat_thinking' <<<"$trd_thinking_output"
grep -q 'teacher rollout:     qwen3_chat_thinking' <<<"$trd_thinking_output"
grep -q 'W&B project/run:     opd-math / opd-math-trd-Qwen3-1.7B-Base-teacher-thinking-true-ms0-' <<<"$opd_thinking_output"
grep -q 'run root:            .*teacher-thinking-true' <<<"$opd_thinking_output"
grep -q 'results file:        .*teacher-thinking-true' <<<"$opd_thinking_output"
grep -q 'eval output:         .*teacher-thinking-true' <<<"$opd_thinking_output"

if rg -q 'basename.*caller|variant=.*basename|family_dir=' "$OPD_ROOT/scripts_math/lib/launch_common.sh"; then
    echo "ERROR: shared launcher still infers experiment identity from a path or filename" >&2
    exit 1
fi
grep -q 'configure_wandb_auth' "$OPD_ROOT/scripts_math/lib/launch_common.sh"
grep -q 'prepare_code_experiment' "$OPD_ROOT/scripts_math/lib/launch_common.sh"
grep -q '^export WANDB_API_KEY=' "$OPD_ROOT/scripts_math/lib/launch_common.sh"

math_topk="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPD/1B/top_k.sh")"
grep -q 'task: math' <<<"$math_topk"
grep -q 'model: Qwen/Qwen3-1.7B-Base' <<<"$math_topk"
grep -q 'teacher: Qwen/Qwen3-14B' <<<"$math_topk"
grep -q 'teacher thinking:    false' <<<"$math_topk"
grep -q 'teacher supervision: plain_completion' <<<"$math_topk"
grep -q 'teacher rollout:     plain_completion' <<<"$math_topk"
grep -q 'KL objective:        reverse/full_vocab' <<<"$math_topk"
grep -q 'rollout mode:        student' <<<"$math_topk"
grep -q 'loss support top-k:  32' <<<"$math_topk"
grep -q 'token KL clip:       0' <<<"$math_topk"
grep -q 'Canonical OPD experiment' <<<"$math_topk"
grep -q 'train/eval response: 16384/16384' <<<"$math_topk"
grep -q 'W&B project/run:     opd-math / opd-math-top_k-Qwen3-1.7B-Base-teacher-thinking-false-ms0-' <<<"$math_topk"
grep -q 'W&B group/job:       top_k-1B / opd-top_k' <<<"$math_topk"
grep -q 'W&B tags:            disabled' <<<"$math_topk"
grep -q 'model artifacts:     milestone_hf_deferred_eval' <<<"$math_topk"
if grep -q 'Preparing canonical code' <<<"$math_topk"; then
    echo "ERROR: a math dry-run triggered code bootstrap" >&2
    exit 1
fi

code_clip="$(DRY_RUN=1 bash "$OPD_ROOT/script_code/OPSD/4B/clip.sh")"
grep -q 'task: code' <<<"$code_clip"
grep -q 'teacher: Qwen/Qwen3-4B-Base' <<<"$code_clip"
grep -q 'KL objective:        forward/full_vocab' <<<"$code_clip"
grep -q 'rollout mode:        student' <<<"$code_clip"
grep -q 'loss support top-k:  0' <<<"$code_clip"
grep -q 'token KL clip:       0.05' <<<"$code_clip"
grep -q 'Canonical OPD experiment' <<<"$code_clip"
grep -q 'W&B project/run:     opsd-code / opsd-code-clip-Qwen3-4B-Base-ms0-' <<<"$code_clip"
if grep -q 'teacher-thinking' <<<"$code_clip"; then
    echo "ERROR: OPSD exposed an OPD-only teacher thinking dimension" >&2
    exit 1
fi
if grep -q 'Preparing canonical code' <<<"$code_clip"; then
    echo "ERROR: a code dry-run triggered code bootstrap" >&2
    exit 1
fi

code_grpo="$(DRY_RUN=1 bash "$OPD_ROOT/script_code/Baselines/1B/grpo.sh")"
grep -q 'GRPO group size:     8 responses/prompt' <<<"$code_grpo"
grep -q 'code reward:         deepcoder_binary_15_longest_v1' <<<"$code_grpo"
grep -q 'reward test suite:   top 15 by input length; binary all-pass; timeout=10s' <<<"$code_grpo"
grep -q 'W&B project/run:     opd-code / baseline-code-grpo-Qwen3-1.7B-Base-ms0-' <<<"$code_grpo"
grep -q 'W&B tags:            disabled' <<<"$code_grpo"
grep -q 'model artifacts:     ephemeral_eval_only' <<<"$code_grpo"

math_skd="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPD/1B/skd.sh")"
grep -q 'task/family/variant: math/opd/skd' <<<"$math_skd"
grep -q 'KL objective:        forward/full_vocab' <<<"$math_skd"
grep -q 'rollout mode:        skd_vllm' <<<"$math_skd"
grep -q 'token KL clip:       0' <<<"$math_skd"
grep -q 'SKD draft/accept:    gamma=5 top_k=25 top_p=1.0' <<<"$math_skd"
grep -q 'teacher_T=0.2 teacher_p=1.0' <<<"$math_skd"

math_opsd_8b="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPSD/8B/trd.sh")"
grep -q 'family: opsd' <<<"$math_opsd_8b"
grep -q 'model: Qwen/Qwen3-8B-Base' <<<"$math_opsd_8b"
grep -q 'teacher: Qwen/Qwen3-8B-Base' <<<"$math_opsd_8b"
grep -q 'model artifacts:     milestone_hf_deferred_eval' <<<"$math_opsd_8b"

math_opd_8b="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPD/8B/trd.sh")"
grep -q 'family: opd' <<<"$math_opd_8b"
grep -q 'model: Qwen/Qwen3-8B-Base' <<<"$math_opd_8b"
grep -q 'teacher: Qwen/Qwen3-14B' <<<"$math_opd_8b"

if OPD_EXPERIMENT_RUNNER="$SMOKE_TMP_DIR/missing.sh" \
    DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPD/1B/top_k.sh" >/dev/null 2>&1; then
    echo "ERROR: dry-run accepted a missing canonical dispatcher" >&2
    exit 1
fi

explicit_identity="$(OPD_TASK=code DRY_RUN=1 bash "$OPD_ROOT/scripts_math/Baselines/1B/base.sh")"
grep -q 'task/family/variant: math/baseline/base' <<<"$explicit_identity"

if OPD_TASK=math OPD_FAMILY=opd OPD_VARIANT=clip OPD_MODEL_SIZE=1B \
    MODEL_PATH=Qwen/Qwen3-1.7B-Base TEACHER_MODEL_PATH=Qwen/Qwen3-14B \
    MULTI_STEP=0 \
    DISTILL_MODE=opd KL_TYPE=reverse KL_METHOD=full_vocab BETA=0 \
    Y_MODE=y_o TEACHER_TRAINING_PROMPT=vanilla Y_O_ROLLOUT_MODE=student \
    TOP_K=0 KL_TOKEN_CLIP=0.05 DRY_RUN=1 \
        bash "$REPO_ROOT/recipe/opd/run/run_experiment.sh" >/dev/null 2>&1; then
    echo "ERROR: canonical dispatcher accepted a clip launcher with reverse KL" >&2
    exit 1
fi

local_1b="$SMOKE_TMP_DIR/Qwen3 1.7B Base"
local_4b="$SMOKE_TMP_DIR/Qwen3-4B-Base"
local_8b="$SMOKE_TMP_DIR/Qwen3-8B-Base"
local_teacher_14b="$SMOKE_TMP_DIR/Qwen3-14B"
local_instruct="$SMOKE_TMP_DIR/Qwen3-1.7B-Instruct"
write_qwen3_config "$local_1b" 2048 28 Qwen/Qwen3-1.7B-Base
write_qwen3_config "$local_4b" 2560 36 Qwen/Qwen3-4B-Base
write_qwen3_config "$local_8b" 4096 36 Qwen/Qwen3-8B-Base
write_qwen3_config "$local_teacher_14b" 5120 40 Qwen/Qwen3-14B
write_qwen3_config "$local_instruct" 2048 28 Qwen/Qwen3-1.7B-Instruct
local_1b="$(cd "$local_1b" && pwd -P)"
local_4b="$(cd "$local_4b" && pwd -P)"
local_8b="$(cd "$local_8b" && pwd -P)"
local_teacher_14b="$(cd "$local_teacher_14b" && pwd -P)"
local_instruct="$(cd "$local_instruct" && pwd -P)"

local_opd_paths="$({
    MODEL_PATH="$local_1b" \
    TEACHER_MODEL_PATH="$local_teacher_14b" \
    DRY_RUN=1 \
        bash "$OPD_ROOT/scripts_math/OPD/1B/top_k.sh"
})"
grep -Fq "model: $local_1b" <<<"$local_opd_paths"
grep -Fq "teacher: $local_teacher_14b" <<<"$local_opd_paths"
grep -q 'Canonical OPD experiment' <<<"$local_opd_paths"

local_opsd_paths="$(MODEL_PATH="$local_1b" DRY_RUN=1 \
    bash "$OPD_ROOT/scripts_math/OPSD/1B/top_k.sh")"
grep -Fq "model: $local_1b" <<<"$local_opsd_paths"
grep -Fq "teacher: $local_1b" <<<"$local_opsd_paths"

if MODEL_PATH="$local_1b" TEACHER_MODEL_PATH="$local_teacher_14b" DRY_RUN=1 \
    bash "$OPD_ROOT/scripts_math/OPSD/1B/top_k.sh" >/dev/null 2>&1; then
    echo "ERROR: OPSD launcher accepted an external teacher instead of its step-0 Base model" >&2
    exit 1
fi

if MODEL_PATH="$local_4b" DRY_RUN=1 \
    bash "$OPD_ROOT/scripts_math/Baselines/1B/base.sh" >/dev/null 2>&1; then
    echo "ERROR: 1B launcher accepted a local 4B model" >&2
    exit 1
fi
if MODEL_PATH="$local_instruct" DRY_RUN=1 \
    bash "$OPD_ROOT/scripts_math/Baselines/1B/base.sh" >/dev/null 2>&1; then
    echo "ERROR: Base launcher accepted an explicit Instruct model" >&2
    exit 1
fi
if MODEL_PATH=Qwen/Qwen3-1.7B-Instruct DRY_RUN=1 \
    bash "$OPD_ROOT/scripts_math/Baselines/1B/base.sh" >/dev/null 2>&1; then
    echo "ERROR: Base launcher accepted a non-default remote model ID" >&2
    exit 1
fi

if DRY_RUN=1 bash "$OPD_ROOT/scripts_math/OPD/1B/vanilla.sh" trainer.foo=bar >/dev/null 2>&1; then
    echo "ERROR: OPD launcher silently accepted an unused positional override" >&2
    exit 1
fi
sft_override="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/Baselines/1B/sft.sh" trainer.foo=bar)"
grep -q 'trainer.foo=bar' <<<"$sft_override"
grep -q 'Canonical OPD experiment' <<<"$sft_override"
grep -q 'model artifacts:     milestone_hf_deferred_eval' <<<"$sft_override"

grpo_baseline="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/Baselines/8B/grpo.sh")"
grep -q 'task/family/variant: math/baseline/grpo' <<<"$grpo_baseline"
grep -q 'model artifacts:     milestone_hf_deferred_eval' <<<"$grpo_baseline"

matrix_override="$(
    DRY_RUN=1 \
    MATRIX_FAMILIES=baseline \
    MATRIX_MODELS=1B \
    MATRIX_VARIANTS=sft \
        bash "$OPD_ROOT/scripts_math/run_matrix.sh" trainer.foo=bar
)"
grep -q 'trainer.foo=bar' <<<"$matrix_override"
grep -q 'Matrix complete: launched=1 failed=0 dry_run=1' <<<"$matrix_override"

math_matrix="$(DRY_RUN=1 bash "$OPD_ROOT/scripts_math/run_matrix.sh")"
code_matrix="$(DRY_RUN=1 bash "$OPD_ROOT/script_code/run_matrix.sh")"
grep -q 'Matrix complete: launched=39 failed=0 dry_run=1' <<<"$math_matrix"
grep -q 'Matrix complete: launched=39 failed=0 dry_run=1' <<<"$code_matrix"

echo "Launcher smoke tests passed (${#launchers[@]} leaf launchers)."
