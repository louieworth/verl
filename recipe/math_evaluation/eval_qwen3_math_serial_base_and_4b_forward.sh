#!/usr/bin/env bash
#
# Serial math evaluation:
#   1. Qwen3-1.7B base on the 3 canonical math datasets with pass@16.
#   2. Qwen3-4B-Instruct-2507 OPD_MATH forward y_o/y_r on Amobench with pass@128.
#
# Usage:
#   bash recipe/math_evaluation/eval_qwen3_math_serial_base_and_4b_forward.sh
#
# Common overrides:
#   NGPUS_PER_NODE=8 GEN_TP=8 bash recipe/math_evaluation/eval_qwen3_math_serial_base_and_4b_forward.sh
#   QWEN3_1_7B_BASE_MODEL_PATH=/path/to/model bash recipe/math_evaluation/eval_qwen3_math_serial_base_and_4b_forward.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARK_SH="$SCRIPT_DIR/benchmark_kl_model.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
GEN_TP="${GEN_TP:-${NGPUS_PER_NODE}}"
EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-$VERL_ROOT/data/eval_dataset/math}"
EVAL_PROMPT_LENGTH="${EVAL_PROMPT_LENGTH:-2048}"
EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-16384}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-18432}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$VERL_ROOT/outputs/math_evaluation}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/eval_qwen3_math_serial_base_and_4b_forward_${RUN_ID}.log}"

QWEN3_1_7B_BASE_MODEL_PATH="${QWEN3_1_7B_BASE_MODEL_PATH:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"

QWEN3_4B_FORWARD_Y_O_MODEL_PATH="${QWEN3_4B_FORWARD_Y_O_MODEL_PATH:-/data/data/jiangli/models/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260531-163323/epoch1/ms1/batch00001/hf_merged}"
QWEN3_4B_FORWARD_Y_R_MODEL_PATH="${QWEN3_4B_FORWARD_Y_R_MODEL_PATH:-/data/data/jiangli/models/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms1_20260601-001538/epoch1/ms1/batch00001/hf_merged}"

BASE_DATASETS="${BASE_DATASETS:-aime25 aime26 hmmt26 amobench}"
AMOBENCH_DATASETS="${AMOBENCH_DATASETS:-amobench}"

BASE_RESULTS_FILE="${BASE_RESULTS_FILE:-$VERL_ROOT/results/base/math/Qwen3-1.7B_pass16.json}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$VERL_ROOT/gen_results/eval/math/base/Qwen3-1.7B_pass16}"

Y_O_MODEL_NAME="${Y_O_MODEL_NAME:-Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260531-163323_LORA_amobench_pass128}"
Y_O_RESULTS_FILE="${Y_O_RESULTS_FILE:-$VERL_ROOT/results/OPD/math/Qwen3-4B-Instruct-2507_amobench_pass128.json}"
Y_O_OUTPUT_DIR="${Y_O_OUTPUT_DIR:-$VERL_ROOT/gen_results/eval/math/Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260531-163323_LORA_amobench_pass128}"

Y_R_MODEL_NAME="${Y_R_MODEL_NAME:-Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms1_20260601-001538_LORA_amobench_pass128}"
Y_R_RESULTS_FILE="${Y_R_RESULTS_FILE:-$Y_O_RESULTS_FILE}"
Y_R_OUTPUT_DIR="${Y_R_OUTPUT_DIR:-$VERL_ROOT/gen_results/eval/math/Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms1_20260601-001538_LORA_amobench_pass128}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_dir() {
    local path="$1"
    local label="$2"
    if [ ! -d "$path" ]; then
        log "ERROR: $label not found: $path"
        exit 1
    fi
}

run_eval() {
    local stage_name="$1"
    local model_path="$2"
    local model_name="$3"
    local base_model_name="$4"
    local datasets="$5"
    local pass_k="$6"
    local output_dir="$7"
    local results_file="$8"
    local write_pass16="$9"

    require_dir "$model_path" "$stage_name model"
    mkdir -p "$output_dir" "$(dirname "$results_file")"

    log "START $stage_name"
    log "  model_path=$model_path"
    log "  model_name=$model_name"
    log "  datasets=$datasets"
    log "  pass_k=$pass_k"
    log "  eval_lengths=prompt:$EVAL_PROMPT_LENGTH response:$EVAL_RESPONSE_LENGTH max_model:$EVAL_MAX_MODEL_LEN"
    log "  output_dir=$output_dir"
    log "  results_file=$results_file"

    PYTHON_BIN="$PYTHON_BIN" \
    NGPUS_PER_NODE="$NGPUS_PER_NODE" \
    NNODES="$NNODES" \
    GEN_TP="$GEN_TP" \
    EVAL_DATASETS_DIR="$EVAL_DATASETS_DIR" \
    EVAL_PROMPT_LENGTH="$EVAL_PROMPT_LENGTH" \
    EVAL_RESPONSE_LENGTH="$EVAL_RESPONSE_LENGTH" \
    EVAL_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN" \
    DATASETS="$datasets" \
    PASS_K="$pass_k" \
    EVAL_BASE_MODEL_NAME="$base_model_name" \
    EVAL_MODEL_NAME="$model_name" \
    EVAL_OUTPUT_DIR="$output_dir" \
    EVAL_RESULTS_FILE="$results_file" \
    WRITE_PASS16_AGGREGATES="$write_pass16" \
    WRITE_RESULTS_CSV=true \
    bash "$BENCHMARK_SH" "$model_path"

    log "DONE $stage_name"
}

main() {
    cd "$VERL_ROOT"
    require_dir "$SCRIPT_DIR" "math evaluation script dir"
    require_dir "$EVAL_DATASETS_DIR" "eval datasets dir"

    log "Logging to $LOG_FILE"
    log "Serial order: Qwen3-1.7B base pass@16 -> Qwen3-4B-Instruct y_o amobench pass@128 -> y_r amobench pass@128"

    run_eval \
        "qwen3_1_7b_base_3_math_pass16" \
        "$QWEN3_1_7B_BASE_MODEL_PATH" \
        "Qwen3-1.7B_base_pass16" \
        "Qwen3-1.7B" \
        "$BASE_DATASETS" \
        "16" \
        "$BASE_OUTPUT_DIR" \
        "$BASE_RESULTS_FILE" \
        "true"

    run_eval \
        "qwen3_4b_instruct_forward_y_o_amobench_pass128" \
        "$QWEN3_4B_FORWARD_Y_O_MODEL_PATH" \
        "$Y_O_MODEL_NAME" \
        "Qwen3-4B-Instruct-2507" \
        "$AMOBENCH_DATASETS" \
        "128" \
        "$Y_O_OUTPUT_DIR" \
        "$Y_O_RESULTS_FILE" \
        "false"

    run_eval \
        "qwen3_4b_instruct_forward_y_r_amobench_pass128" \
        "$QWEN3_4B_FORWARD_Y_R_MODEL_PATH" \
        "$Y_R_MODEL_NAME" \
        "Qwen3-4B-Instruct-2507" \
        "$AMOBENCH_DATASETS" \
        "128" \
        "$Y_R_OUTPUT_DIR" \
        "$Y_R_RESULTS_FILE" \
        "false"

    log "All requested evaluations completed."
}

main "$@" 2>&1 | tee -a "$LOG_FILE"
