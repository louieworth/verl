#!/usr/bin/env bash

WANDB_GUARDIAN_PID=""
WANDB_GUARDIAN_READY_FILE=""
WANDB_GUARDIAN_CONTROL_FILE=""

start_wandb_run_guardian() {
    local state_root="$1"
    local lifecycle_dir="$state_root/wandb_lifecycle"
    local log_file="$lifecycle_dir/guardian.log"
    local attempt_id="${AWS_BATCH_JOB_ATTEMPT:-${SLURM_JOB_ID:-$$}}"
    local wait_count

    [ -n "${WANDB_RUN_ID:-}" ] || return 0
    case "${WANDB_MODE:-online}" in
        online|shared) ;;
        *) return 0 ;;
    esac

    mkdir -p "$lifecycle_dir"
    WANDB_GUARDIAN_READY_FILE="$lifecycle_dir/ready_${attempt_id}"
    WANDB_GUARDIAN_CONTROL_FILE="$lifecycle_dir/control_${attempt_id}.json"
    rm -f -- "$WANDB_GUARDIAN_READY_FILE" "$WANDB_GUARDIAN_CONTROL_FILE"
    export WANDB_SHARED_RUN=1

    "$PYTHON_BIN" recipe/opd/wandb_run_guardian.py \
        --ready-file "$WANDB_GUARDIAN_READY_FILE" \
        --control-file "$WANDB_GUARDIAN_CONTROL_FILE" \
        >>"$log_file" 2>&1 &
    WANDB_GUARDIAN_PID=$!

    for wait_count in $(seq 1 120); do
        if [ -s "$WANDB_GUARDIAN_READY_FILE" ]; then
            echo "W&B pipeline run is active (guardian pid=$WANDB_GUARDIAN_PID)"
            return 0
        fi
        if ! kill -0 "$WANDB_GUARDIAN_PID" 2>/dev/null; then
            wait "$WANDB_GUARDIAN_PID" || true
            echo "ERROR: W&B pipeline guardian exited during startup: $log_file" >&2
            return 1
        fi
        sleep 1
    done

    kill "$WANDB_GUARDIAN_PID" 2>/dev/null || true
    wait "$WANDB_GUARDIAN_PID" 2>/dev/null || true
    WANDB_GUARDIAN_PID=""
    echo "ERROR: timed out starting W&B pipeline guardian: $log_file" >&2
    return 1
}

finish_wandb_run_guardian() {
    local exit_code="${1:-1}"
    local temporary_control

    [ -n "$WANDB_GUARDIAN_PID" ] || return 0
    temporary_control="${WANDB_GUARDIAN_CONTROL_FILE}.tmp.$$"
    printf '{"exit_code":%s}\n' "$exit_code" >"$temporary_control"
    mv -f -- "$temporary_control" "$WANDB_GUARDIAN_CONTROL_FILE"
    wait "$WANDB_GUARDIAN_PID"
    WANDB_GUARDIAN_PID=""
}
