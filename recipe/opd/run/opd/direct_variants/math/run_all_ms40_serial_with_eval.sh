#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export TASK="math"

export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"

RUNNER_LOG_DIR="${RUNNER_LOG_DIR:-$VERL_ROOT/outputs/OPD/math/direct_variants_ms40_serial_logs}"
mkdir -p "$RUNNER_LOG_DIR"
RUNNER_LOG="$RUNNER_LOG_DIR/run_all_ms40_serial_with_eval_$(date +%Y%m%d_%H%M%S).log"

VARIANTS=(
    "direct_forward_clip01_y_o_ms40_qwen3_4b_8b.sh"
    "direct_reverse_y_o_ms40_qwen3_4b_8b.sh"
    "direct_reverse_topk32_y_o_ms40_qwen3_4b_8b.sh"
)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUNNER_LOG"
}

log "Starting serial ms40 math direct variant run"
log "TASK=$TASK"
log "RUN_EVAL_AFTER_TRAINING=$RUN_EVAL_AFTER_TRAINING"
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

log "All ms40 math direct variants completed"
