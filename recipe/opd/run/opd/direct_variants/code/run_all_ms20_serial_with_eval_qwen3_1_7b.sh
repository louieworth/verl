#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export TASK="code"

# Qwen3-1.7B student variant. Teacher remains Qwen3-8B unless overridden.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen3-8B}"

export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"

RUNNER_LOG_DIR="${RUNNER_LOG_DIR:-$VERL_ROOT/outputs/OPD/code/direct_variants_ms20_serial_logs_qwen3_1_7b}"
mkdir -p "$RUNNER_LOG_DIR"
RUNNER_LOG="$RUNNER_LOG_DIR/run_all_ms20_serial_with_eval_qwen3_1_7b_$(date +%Y%m%d_%H%M%S).log"

VARIANTS=(
    "direct_forward_y_r_ms1_qwen3_4b_8b.sh"
    "direct_forward_y_o_ms20_qwen3_4b_8b.sh"
    "direct_forward_y_r_ms20_qwen3_4b_8b.sh"
    "direct_forward_clip01_y_o_ms20_qwen3_4b_8b.sh"
    "direct_reverse_y_o_ms20_qwen3_4b_8b.sh"
    "direct_reverse_topk32_y_o_ms20_qwen3_4b_8b.sh"
)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUNNER_LOG"
}

log "Starting serial ms20 code direct variant run for Qwen3-1.7B"
log "TASK=$TASK"
log "MODEL_PATH=$MODEL_PATH"
log "MODEL_NAME=$MODEL_NAME"
log "TEACHER_MODEL_PATH=$TEACHER_MODEL_PATH"
log "MAX_LENGTH=${MAX_LENGTH:-auto}"
log "MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-auto}"
log "PIPELINE_RESUME_MODE=$PIPELINE_RESUME_MODE"
log "Log: $RUNNER_LOG"

for variant in "${VARIANTS[@]}"; do
    script="$SCRIPT_DIR/$variant"
    if [ ! -x "$script" ]; then
        log "ERROR: missing or non-executable variant script: $script"
        exit 1
    fi

    log "=========================================="
    log "START $variant"
    log "=========================================="
    bash "$script" 2>&1 | tee -a "$RUNNER_LOG"
    log "=========================================="
    log "DONE  $variant"
    log "=========================================="
done

log "All ms20 code direct variants completed for Qwen3-1.7B"
