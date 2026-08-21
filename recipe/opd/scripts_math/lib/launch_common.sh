#!/usr/bin/env bash
# Shared validator/dispatcher for the math (scripts_math/) and code
# (script_code/) OPD/OPSD experiment matrices. Every experiment identity and
# algorithm knob is declared in the leaf wrapper; this file never infers an
# experiment from the wrapper's path or filename.

set -euo pipefail

LAUNCH_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LAUNCH_COMMON_DIR/model_path_validation.sh"

# Shared by every recipe/opd/scripts_math and recipe/opd/script_code launcher.
# Authentication comes from the caller's environment and is never stored in Git.
export WANDB_API_KEY="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"

launcher_error() {
    echo "ERROR: $*" >&2
    exit 2
}

launcher_print_command() {
    printf '  command:'
    printf ' %q' "$@"
    printf '\n'
}

configure_wandb_auth() {
    case "${WANDB_MODE:-online}" in
        offline|disabled) return 0 ;;
    esac
    if [ -z "${WANDB_API_KEY:-}" ]; then
        launcher_error "WANDB_API_KEY is required for online runs; export it in the calling shell or set WANDB_MODE=offline"
    fi
    case "$WANDB_API_KEY" in
        wandb_v1_*) ;;
        *) launcher_error "WANDB_API_KEY does not have the expected wandb_v1_ format" ;;
    esac
    export WANDB_API_KEY
}

prepare_code_experiment() {
    local repo_root="$1"
    local prepare_script="$repo_root/recipe/opd/script_code/prepare_code.sh"
    [ -x "$prepare_script" ] || launcher_error "code bootstrap script is missing or not executable: $prepare_script"
    echo "Preparing canonical code train/eval data and execution runtime..."
    bash "$prepare_script" all
    export OPD_CODE_BOOTSTRAP_READY=1
}

launch_experiment() {
    local repo_root
    local task family model_size variant
    local default_model_path model_path model_alias teacher_model_path teacher_model_alias
    local expected_hidden_size expected_num_layers
    local dry_run="${DRY_RUN:-0}"
    local -a forwarded_args=()
    local arg

    repo_root="$(cd "$LAUNCH_COMMON_DIR/../../../.." && pwd)"
    task="${OPD_TASK:?leaf launcher must explicitly set OPD_TASK}"
    family="${OPD_FAMILY:?leaf launcher must explicitly set OPD_FAMILY}"
    variant="${OPD_VARIANT:?leaf launcher must explicitly set OPD_VARIANT}"
    model_size="${OPD_MODEL_SIZE:?leaf launcher must explicitly set OPD_MODEL_SIZE}"

    for arg in "$@"; do
        case "$arg" in
            --dry-run) dry_run=1 ;;
            *) forwarded_args+=("$arg") ;;
        esac
    done
    case "$dry_run" in
        1|true|TRUE|yes|YES) dry_run=1 ;;
        0|false|FALSE|no|NO) dry_run=0 ;;
        *) launcher_error "DRY_RUN must be 0/1 or false/true (got: $dry_run)" ;;
    esac

    case "$task" in
        math|code) ;;
        *) launcher_error "OPD_TASK must be math or code (got: $task)" ;;
    esac
    case "$family" in
        baseline|opd|opsd) ;;
        *) launcher_error "OPD_FAMILY must be baseline, opd, or opsd (got: $family)" ;;
    esac

    case "$model_size" in
        1B)
            default_model_path="Qwen/Qwen3-1.7B-Base"
            model_alias="Qwen3-1.7B-Base"
            expected_hidden_size=2048
            expected_num_layers=28
            ;;
        4B)
            default_model_path="Qwen/Qwen3-4B-Base"
            model_alias="Qwen3-4B-Base"
            expected_hidden_size=2560
            expected_num_layers=36
            ;;
        8B)
            default_model_path="Qwen/Qwen3-8B-Base"
            model_alias="Qwen3-8B-Base"
            expected_hidden_size=4096
            expected_num_layers=36
            ;;
        *) launcher_error "unsupported model size: $model_size" ;;
    esac
    model_path="$(
        opd_resolve_base_model_path \
            "${MODEL_PATH:-$default_model_path}" \
            "$default_model_path" \
            "$expected_hidden_size" \
            "$expected_num_layers" \
            MODEL_PATH
    )" || return $?

    case "$family" in
        baseline)
            case "$variant" in
                base|sft|grpo) ;;
                *) launcher_error "baseline supports only base, sft, and grpo (got: $variant)" ;;
            esac
            ;;
        opd|opsd)
            case "$variant" in
                vanilla|top_k|clip|skd|trd) ;;
                *) launcher_error "$family supports only vanilla, top_k, clip, skd, and trd (got: $variant)" ;;
            esac
            ;;
    esac

    # Re-export the leaf's literal declarations for the canonical dispatcher.
    export OPD_TASK="$task" OPD_FAMILY="$family" OPD_VARIANT="$variant" OPD_MODEL_SIZE="$model_size"
    export MODEL_PATH="$model_path"
    export MODEL_ALIAS="$model_alias"
    export MODEL_NAME="$model_alias"
    export STUDENT_MODEL="$model_alias"

    teacher_model_path=""
    teacher_model_alias=""
    case "$family" in
        opd)
            teacher_model_path="$(
                opd_resolve_base_model_path \
                    "${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}" \
                    Qwen/Qwen3-14B \
                    5120 \
                    40 \
                    TEACHER_MODEL_PATH
            )" || return $?
            teacher_model_alias="Qwen3-14B"
            export TEACHER_MODEL_PATH="$teacher_model_path"
            export TEACHER_MODEL="$teacher_model_alias"
            export OPD_TEACHER_MODEL_PATH="$teacher_model_path"
            ;;
        opsd)
            teacher_model_path="$(
                opd_resolve_base_model_path \
                    "${TEACHER_MODEL_PATH:-$model_path}" \
                    "$default_model_path" \
                    "$expected_hidden_size" \
                    "$expected_num_layers" \
                    TEACHER_MODEL_PATH
            )" || return $?
            if [ "$teacher_model_path" != "$model_path" ]; then
                launcher_error "opsd is self-distillation: TEACHER_MODEL_PATH must equal MODEL_PATH"
            fi
            teacher_model_alias="$model_alias"
            export TEACHER_MODEL_PATH="$teacher_model_path"
            export TEACHER_MODEL="$teacher_model_alias"
            unset OPD_TEACHER_MODEL_PATH 2>/dev/null || true
            ;;
        baseline)
            unset TEACHER_MODEL_PATH TEACHER_MODEL OPD_TEACHER_MODEL_PATH 2>/dev/null || true
            ;;
    esac

    export DRY_RUN="$dry_run"
    local runner="${OPD_EXPERIMENT_RUNNER:-$repo_root/recipe/opd/run/run_experiment.sh}"
    [ -f "$runner" ] || launcher_error "canonical experiment runner is missing: $runner"

    if [ "$dry_run" = 1 ]; then
        echo "Experiment launcher dry run"
        echo "  task: $task"
        echo "  family: $family"
        echo "  variant: $variant"
        echo "  model size: $model_size"
        echo "  model: $model_path"
        if [ -n "$teacher_model_path" ]; then
            echo "  teacher: $teacher_model_path"
        fi
        launcher_print_command bash "$runner" "${forwarded_args[@]}"
        DRY_RUN=1 bash "$runner" "${forwarded_args[@]}"
        return 0
    fi

    configure_wandb_auth
    if [ "$task" = code ]; then
        prepare_code_experiment "$repo_root"
    fi

    exec bash "$runner" "${forwarded_args[@]}"
}

launch_experiment "$@"
