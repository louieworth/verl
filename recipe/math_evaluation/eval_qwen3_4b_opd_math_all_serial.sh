#!/usr/bin/env bash
#
# Serial math evaluation for Qwen3-4B-Instruct-2507 base and all saved OPD_MATH models.
#
# Usage:
#   bash recipe/math_evaluation/eval_qwen3_4b_opd_math_all_serial.sh
#
# Common overrides:
#   PASS_K=16 DATASETS="aime25 aime26 hmmt26 amobench" \
#     bash recipe/math_evaluation/eval_qwen3_4b_opd_math_all_serial.sh
#   EVAL_RESPONSE_LENGTH=16384 EVAL_MAX_MODEL_LEN=18432 \
#     bash recipe/math_evaluation/eval_qwen3_4b_opd_math_all_serial.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARK_SH="$SCRIPT_DIR/benchmark_kl_model.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
GEN_TP="${GEN_TP:-${NGPUS_PER_NODE}}"
EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-$VERL_ROOT/data/eval_dataset/math}"

DATASETS="${DATASETS:-aime25 aime26 hmmt26 amobench}"
PASS_K="${PASS_K:-16}"

EVAL_PROMPT_LENGTH="${EVAL_PROMPT_LENGTH:-2048}"
EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-16384}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-18432}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$VERL_ROOT/outputs/math_evaluation}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/eval_qwen3_4b_opd_math_all_serial_${RUN_ID}.log}"

RESULTS_FILE="${RESULTS_FILE:-$VERL_ROOT/results/OPD/math/Qwen3-4B-Instruct-2507_opd_math_all_pass${PASS_K}.json}"
GEN_OUTPUT_BASE_DIR="${GEN_OUTPUT_BASE_DIR:-$VERL_ROOT/gen_results/eval/math}"

BASE_MODEL_NAME="Qwen3-4B-Instruct-2507_base_pass${PASS_K}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"

MODEL_1_NAME="Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260531-163323_LORA_pass${PASS_K}"
MODEL_1_PATH="${MODEL_1_PATH:-/data/data/jiangli/models/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260531-163323/epoch1/ms1/batch00001/hf_merged}"

MODEL_2_NAME="Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms1_20260601-001538_LORA_pass${PASS_K}"
MODEL_2_PATH="${MODEL_2_PATH:-/data/data/jiangli/models/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms1_20260601-001538/epoch1/ms1/batch00001/hf_merged}"

MODEL_3_NAME="Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_o_kl_forward_full_vocab_clip01_vanilla_ms1_20260601-091503_LORA_pass${PASS_K}"
MODEL_3_PATH="${MODEL_3_PATH:-/data/data/jiangli/models/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip01_vanilla_ms1_20260601-091503/epoch1/ms1/batch00001/hf_merged}"

MODEL_4_NAME="Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_o_kl_reverse_full_vocab_clip0_vanilla_ms1_20260601-140742_LORA_pass${PASS_K}"
MODEL_4_PATH="${MODEL_4_PATH:-/data/data/jiangli/models/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_o_kl_reverse_full_vocab_clip0_vanilla_ms1_20260601-140742/epoch1/ms1/batch00001/hf_merged}"

MODEL_5_NAME="Qwen3-4B-Instruct-2507_OPD_MATH_teacherQwen3-8B_y_o_kl_reverse_full_vocab_clip0_topk32_vanilla_ms1_20260601-190226_LORA_pass${PASS_K}"
MODEL_5_PATH="${MODEL_5_PATH:-/data/data/jiangli/models/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_o_kl_reverse_full_vocab_clip0_topk32_vanilla_ms1_20260601-190226/epoch1/ms1/batch00001/hf_merged}"

mkdir -p "$LOG_DIR" "$(dirname "$RESULTS_FILE")"

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

run_one() {
    local stage_name="$1"
    local model_name="$2"
    local model_path="$3"
    local output_dir="$GEN_OUTPUT_BASE_DIR/$model_name"

    require_dir "$model_path" "$stage_name model"
    mkdir -p "$output_dir"

    log "START $stage_name"
    log "  model_path=$model_path"
    log "  model_name=$model_name"
    log "  datasets=$DATASETS"
    log "  pass_k=$PASS_K"
    log "  eval_lengths=prompt:$EVAL_PROMPT_LENGTH response:$EVAL_RESPONSE_LENGTH max_model:$EVAL_MAX_MODEL_LEN"
    log "  output_dir=$output_dir"
    log "  results_file=$RESULTS_FILE"

    PYTHON_BIN="$PYTHON_BIN" \
    NGPUS_PER_NODE="$NGPUS_PER_NODE" \
    NNODES="$NNODES" \
    GEN_TP="$GEN_TP" \
    EVAL_DATASETS_DIR="$EVAL_DATASETS_DIR" \
    EVAL_PROMPT_LENGTH="$EVAL_PROMPT_LENGTH" \
    EVAL_RESPONSE_LENGTH="$EVAL_RESPONSE_LENGTH" \
    EVAL_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN" \
    DATASETS="$DATASETS" \
    PASS_K="$PASS_K" \
    EVAL_BASE_MODEL_NAME="Qwen3-4B-Instruct-2507" \
    EVAL_MODEL_NAME="$model_name" \
    EVAL_OUTPUT_DIR="$output_dir" \
    EVAL_RESULTS_FILE="$RESULTS_FILE" \
    WRITE_PASS16_AGGREGATES=true \
    WRITE_RESULTS_CSV=true \
    bash "$BENCHMARK_SH" "$model_path"

    log "DONE $stage_name"
}

main() {
    cd "$VERL_ROOT"
    require_dir "$SCRIPT_DIR" "math evaluation script dir"
    require_dir "$EVAL_DATASETS_DIR" "eval datasets dir"

    log "Logging to $LOG_FILE"
    log "Serial order: base -> forward clip0 y_o -> forward clip0 y_r -> forward clip01 y_o -> reverse clip0 y_o -> reverse clip0 topk32 y_o"

    run_one "qwen3_4b_instruct_base" "$BASE_MODEL_NAME" "$BASE_MODEL_PATH"
    run_one "qwen3_4b_opd_math_forward_clip0_y_o" "$MODEL_1_NAME" "$MODEL_1_PATH"
    run_one "qwen3_4b_opd_math_forward_clip0_y_r" "$MODEL_2_NAME" "$MODEL_2_PATH"
    run_one "qwen3_4b_opd_math_forward_clip01_y_o" "$MODEL_3_NAME" "$MODEL_3_PATH"
    run_one "qwen3_4b_opd_math_reverse_clip0_y_o" "$MODEL_4_NAME" "$MODEL_4_PATH"
    run_one "qwen3_4b_opd_math_reverse_clip0_topk32_y_o" "$MODEL_5_NAME" "$MODEL_5_PATH"

    log "All requested Qwen3-4B OPD_MATH evaluations completed."
}

main "$@" 2>&1 | tee -a "$LOG_FILE"
