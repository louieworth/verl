#!/usr/bin/env bash

# config.json is emitted before the expensive merge/reload validation finishes,
# so it cannot serve as a durable completion marker. export_checkpoint.py writes
# opd_export.json atomically only after a strict Transformers reload succeeds.
hf_export_complete() {
    local export_dir="$1"

    [ -d "$export_dir" ] || return 1
    [ -s "$export_dir/config.json" ] || return 1
    [ -s "$export_dir/opd_export.json" ] || return 1
    # -H follows a command-line symlink such as RUN_ROOT/final_model while
    # preserving the normal non-recursive behavior inside the export.
    find -H "$export_dir" \
        -maxdepth 1 \
        -type f \
        -name 'model*.safetensors' \
        -size +0c \
        -print -quit 2>/dev/null | grep -q .
}

# Segmented SFT/GRPO runners start a fresh trainer process at every evaluation
# milestone.  The trainer's in-memory checkpoint retention state therefore
# cannot delete checkpoints created by an earlier process.  Once the current
# endpoint has been strictly exported (and, when enabled, evaluated), keep only
# its sharded checkpoint for the next segment's optimizer/scheduler resume.
prune_old_step_checkpoints() {
    local checkpoint_root="$1"
    local keep_step="$2"
    local keep_dir step_dir step_name tracker_file tracked_step

    if [ -z "$checkpoint_root" ] || [ ! -d "$checkpoint_root" ]; then
        echo "ERROR: refusing to prune unsafe checkpoint root: '$checkpoint_root'" >&2
        return 1
    fi
    checkpoint_root="$(realpath -e -- "$checkpoint_root")" || return 1
    if [ "$checkpoint_root" = "/" ]; then
        echo "ERROR: refusing to prune filesystem root" >&2
        return 1
    fi
    if ! [[ "$keep_step" =~ ^[0-9]+$ ]]; then
        echo "ERROR: checkpoint step to keep must be a non-negative integer, got: $keep_step" >&2
        return 1
    fi
    tracker_file="$checkpoint_root/latest_checkpointed_iteration.txt"
    if [ ! -s "$tracker_file" ]; then
        echo "ERROR: refusing to prune without checkpoint tracker: $tracker_file" >&2
        return 1
    fi
    tracked_step="$(tr -d '[:space:]' < "$tracker_file")"
    if ! [[ "$tracked_step" =~ ^[0-9]+$ ]] || [ "$tracked_step" -ne "$keep_step" ]; then
        echo "ERROR: refusing to prune: tracker step '$tracked_step' does not match keep step $keep_step" >&2
        return 1
    fi
    keep_dir="$checkpoint_root/global_step_${keep_step}"
    if [ ! -d "$keep_dir" ]; then
        echo "ERROR: checkpoint to retain is missing: $keep_dir" >&2
        return 1
    fi

    while IFS= read -r -d '' step_dir; do
        [ "$step_dir" = "$keep_dir" ] && continue
        step_name="$(basename "$step_dir")"
        if ! [[ "$step_name" =~ ^global_step_[0-9]+$ ]]; then
            echo "ERROR: refusing to prune unexpected checkpoint path: $step_dir" >&2
            return 1
        fi
        echo "Pruning superseded sharded checkpoint: $step_dir" >&2
        rm -rf -- "$step_dir"
    done < <(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print0)
}

require_checkpoint_step() {
    local checkpoint_root="$1"
    local expected_step="$2"
    local tracker_file="$checkpoint_root/latest_checkpointed_iteration.txt"
    local actual_step

    # Direct/non-segmented invocations may leave the endpoint unspecified.
    [ -n "$expected_step" ] && [ "$expected_step" != "null" ] || return 0
    if ! [[ "$expected_step" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: expected checkpoint step must be a positive integer, got: $expected_step" >&2
        return 1
    fi
    if [ ! -s "$tracker_file" ]; then
        echo "ERROR: checkpoint tracker is missing after training: $tracker_file" >&2
        return 1
    fi
    actual_step="$(tr -d '[:space:]' < "$tracker_file")"
    if ! [[ "$actual_step" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid checkpoint tracker value '$actual_step' in $tracker_file" >&2
        return 1
    fi
    if [ "$actual_step" -ne "$expected_step" ]; then
        echo "ERROR: training stopped at step $actual_step, before requested endpoint $expected_step" >&2
        echo "       Refusing to export/evaluate or mark this milestone complete." >&2
        return 1
    fi
}
