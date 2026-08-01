#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NORMALIZED_VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$NORMALIZED_VISIBLE_GPUS"
if [ "${#VISIBLE_GPU_IDS[@]}" -lt 7 ]; then
    echo "ERROR: this sequence needs at least seven visible GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

wait_for_gpu_release() {
    if [ "${SFT_DRY_RUN:-false}" = "true" ]; then
        return
    fi
    local idle_streak=0 busy_gpus gpu_state
    while [ "$idle_streak" -lt 2 ]; do
        gpu_state="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
        busy_gpus="$(
            printf '%s\n' "$gpu_state" |
                awk -F',' -v selected=",$NORMALIZED_VISIBLE_GPUS," '
                    {
                        gpu_index = $1
                        memory_used = $2
                        utilization = $3
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu_index)
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", memory_used)
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", utilization)
                        if (index(selected, "," gpu_index ",") && ((memory_used + 0) > 1024 || (utilization + 0) > 5)) {
                            printf "%s(mem=%sMiB,util=%s%%) ", gpu_index, memory_used, utilization
                        }
                    }
                '
        )"
        if [ -z "$busy_gpus" ]; then
            idle_streak=$((idle_streak + 1))
        else
            idle_streak=0
            echo "Waiting for GPU release: $busy_gpus"
        fi
        if [ "$idle_streak" -lt 2 ]; then
            sleep 5
        fi
    done
}

TRAINING_SCRIPTS=(
    "$SCRIPT_DIR/qwen3_1_7b_taco_sft.sh"
    "$SCRIPT_DIR/qwen3_4b_instruct_taco_sft.sh"
    "$SCRIPT_DIR/qwen3_8b_taco_sft.sh"
    "$SCRIPT_DIR/qwen3_1_7b_deepscaler_sft.sh"
)

echo "Table 1-4 missing SFT sequence"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Learning rate: ${LEARNING_RATE:-1e-7}"
echo "Training order: 1.7B TACO -> 4B TACO -> 8B TACO -> 1.7B DeepScaleR"
echo "Evaluation order: 1.7B math, then four code suites in parallel"

for index in "${!TRAINING_SCRIPTS[@]}"; do
    script="${TRAINING_SCRIPTS[$index]}"
    number=$((index + 1))
    start_time="$(date +%s)"
    echo
    echo "[train $number/${#TRAINING_SCRIPTS[@]}] START $(date -u '+%Y-%m-%dT%H:%M:%SZ') $script"
    RUN_EVAL_AFTER_TRAINING=false bash "$script" "$@"
    elapsed=$(( $(date +%s) - start_time ))
    echo "[train $number/${#TRAINING_SCRIPTS[@]}] DONE  $(date -u '+%Y-%m-%dT%H:%M:%SZ') elapsed=${elapsed}s"
    wait_for_gpu_release
done

echo
echo "[math eval] START $(date -u '+%Y-%m-%dT%H:%M:%SZ') Qwen3-1.7B"
RUN_EVAL_AFTER_TRAINING=true bash "$SCRIPT_DIR/qwen3_1_7b_deepscaler_sft.sh" "$@"
echo "[math eval] DONE  $(date -u '+%Y-%m-%dT%H:%M:%SZ') Qwen3-1.7B"
wait_for_gpu_release

if [ "${SFT_DRY_RUN:-false}" = "true" ]; then
    echo "Dry run complete; parallel code evaluation launch skipped."
    exit 0
fi

CODE_EVAL_LOG_DIR="${CODE_EVAL_LOG_DIR:-/data2/tmp/table1_4_sft_code_eval_logs}"
mkdir -p "$CODE_EVAL_LOG_DIR"
EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-12}"
eval_pids=()
eval_labels=()

launch_eval() {
    local gpu_id="$1" label="$2" script="$3"
    echo "[code eval] START $(date -u '+%Y-%m-%dT%H:%M:%SZ') gpu=$gpu_id label=$label"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    NUM_GPUS=1 \
    CODE_EVAL_GEN_TP=1 \
    EVALPLUS_EXTRA_ARGS="--parallel $EVALPLUS_PARALLEL" \
    RUN_EVAL_AFTER_TRAINING=true \
        bash "$script" >"$CODE_EVAL_LOG_DIR/${label}.log" 2>&1 &
    eval_pids+=("$!")
    eval_labels+=("$label")
}

launch_eval "${VISIBLE_GPU_IDS[0]}" qwen3_1_7b_taco "$SCRIPT_DIR/qwen3_1_7b_taco_sft.sh"
launch_eval "${VISIBLE_GPU_IDS[1]}" qwen3_4b_taco "$SCRIPT_DIR/qwen3_4b_instruct_taco_sft.sh"
launch_eval "${VISIBLE_GPU_IDS[2]}" qwen3_8b_taco "$SCRIPT_DIR/qwen3_8b_taco_sft.sh"

eval_failures=0
for index in "${!eval_pids[@]}"; do
    if wait "${eval_pids[$index]}"; then
        echo "[code eval] DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ') label=${eval_labels[$index]}"
    else
        echo "[code eval] FAILED $(date -u '+%Y-%m-%dT%H:%M:%SZ') label=${eval_labels[$index]} log=$CODE_EVAL_LOG_DIR/${eval_labels[$index]}.log" >&2
        eval_failures=$((eval_failures + 1))
    fi
done

if [ "$eval_failures" -ne 0 ]; then
    echo "ERROR: $eval_failures code evaluation job(s) failed" >&2
    exit 1
fi

BASE_LCB_LOG_DIR="${BASE_LCB_LOG_DIR:-/data2/tmp/qwen3_8b_base_lcb_resume_logs}"
mkdir -p "$BASE_LCB_LOG_DIR"
base_lcb_pids=()
base_lcb_labels=()

for shard_index in 2 3 4 5 6 7 8; do
    gpu_offset=$((shard_index - 2))
    gpu_id="${VISIBLE_GPU_IDS[$gpu_offset]}"
    label="shard_${shard_index}"
    echo "[base LCB] RESUME $(date -u '+%Y-%m-%dT%H:%M:%SZ') gpu=$gpu_id shard=$shard_index"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    NUM_GPUS=1 \
    NGPUS_PER_NODE=1 \
    GEN_TP=1 \
    SHARD_INDEX="$shard_index" \
    PYTHON_BIN="${PYTHON_BIN:-/data2/conda/envs/verl/bin/python}" \
        bash "$SCRIPT_DIR/eval_qwen3_8b_base_lcb_shard.sh" \
        >"$BASE_LCB_LOG_DIR/${label}.log" 2>&1 &
    base_lcb_pids+=("$!")
    base_lcb_labels+=("$label")
done

base_lcb_failures=0
for index in "${!base_lcb_pids[@]}"; do
    if wait "${base_lcb_pids[$index]}"; then
        echo "[base LCB] DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ') label=${base_lcb_labels[$index]}"
    else
        echo "[base LCB] FAILED $(date -u '+%Y-%m-%dT%H:%M:%SZ') label=${base_lcb_labels[$index]} log=$BASE_LCB_LOG_DIR/${base_lcb_labels[$index]}.log" >&2
        base_lcb_failures=$((base_lcb_failures + 1))
    fi
done

if [ "$base_lcb_failures" -ne 0 ]; then
    echo "ERROR: $base_lcb_failures Base LCB shard job(s) failed" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/data2/conda/envs/verl/bin/python}" \
    bash "$SCRIPT_DIR/wait_for_base_lcb_shards_and_aggregate.sh"

SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-/data2/tmp/table1_4_sft_pipeline/completed_results.md}"
"${PYTHON_BIN:-/data2/conda/envs/verl/bin/python}" \
    "$SCRIPT_DIR/summarize_table1_4_sft_results.py" | tee "$SUMMARY_OUTPUT"

echo "All missing SFT training and evaluation jobs completed."
