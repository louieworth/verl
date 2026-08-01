#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LEARNING_RATE="${LEARNING_RATE:-1e-7}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
IDLE_MEMORY_LIMIT_MIB="${IDLE_MEMORY_LIMIT_MIB:-1024}"
IDLE_UTILIZATION_LIMIT="${IDLE_UTILIZATION_LIMIT:-5}"
IDLE_CONSECUTIVE_CHECKS="${IDLE_CONSECUTIVE_CHECKS:-3}"
SCHEDULER_ROOT="${SCHEDULER_ROOT:-/data2/tmp/table1_4_sft_pipeline}"
SCHEDULER_LOG="${SCHEDULER_LOG:-$SCHEDULER_ROOT/scheduler.log}"

mkdir -p "$SCHEDULER_ROOT"
exec > >(tee -a "$SCHEDULER_LOG") 2>&1
exec 9>"$SCHEDULER_ROOT/scheduler.lock"
if ! flock -n 9; then
    echo "ERROR: another Table 1-4 SFT scheduler is already running." >&2
    exit 1
fi

NORMALIZED_VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$NORMALIZED_VISIBLE_GPUS"
if [ "${#VISIBLE_GPU_IDS[@]}" -lt 4 ]; then
    echo "ERROR: at least four selected GPUs are required, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
for gpu_id in "${VISIBLE_GPU_IDS[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: GPU IDs must be numeric, got: $gpu_id" >&2
        exit 1
    fi
done

echo "Waiting for GPUs: $CUDA_VISIBLE_DEVICES"
echo "Idle threshold: memory<=${IDLE_MEMORY_LIMIT_MIB}MiB utilization<=${IDLE_UTILIZATION_LIMIT}% for ${IDLE_CONSECUTIVE_CHECKS} checks"

idle_streak=0
while [ "$idle_streak" -lt "$IDLE_CONSECUTIVE_CHECKS" ]; do
    if ! gpu_state="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] nvidia-smi failed: $gpu_state" >&2
        idle_streak=0
        sleep "$CHECK_INTERVAL_SECONDS"
        continue
    fi
    busy_gpus="$(
        printf '%s\n' "$gpu_state" |
            awk -F',' -v selected=",$NORMALIZED_VISIBLE_GPUS," -v memory_limit="$IDLE_MEMORY_LIMIT_MIB" -v utilization_limit="$IDLE_UTILIZATION_LIMIT" '
                {
                    gpu_index = $1
                    memory_used = $2
                    utilization = $3
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu_index)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", memory_used)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", utilization)
                    if (index(selected, "," gpu_index ",") && ((memory_used + 0) > memory_limit || (utilization + 0) > utilization_limit)) {
                        printf "%s(mem=%sMiB,util=%s%%) ", gpu_index, memory_used, utilization
                    }
                }
            '
    )"
    if [ -z "$busy_gpus" ]; then
        idle_streak=$((idle_streak + 1))
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] selected GPUs idle ($idle_streak/$IDLE_CONSECUTIVE_CHECKS)"
    else
        idle_streak=0
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] waiting: $busy_gpus"
    fi
    if [ "$idle_streak" -lt "$IDLE_CONSECUTIVE_CHECKS" ]; then
        sleep "$CHECK_INTERVAL_SECONDS"
    fi
done

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] GPUs are stably idle; starting Table 1-4 pipeline."
unset NUM_GPUS SFT_DRY_RUN RUN_EVAL_AFTER_TRAINING
exec bash "$SCRIPT_DIR/run_table1_4_missing_sft_sequence.sh"
