#!/usr/bin/env bash
#
# Serial EvalPlus eval for two Qwen3-1.7B OPD_CODE reverse variants.
#
# Usage:
#   bash recipe/code_evaluation/eval_qwen3_1_7b_reverse_lcb_v6.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARK_SH="$SCRIPT_DIR/benchmark_code_model.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
GEN_TP="${GEN_TP:-${NGPUS_PER_NODE}}"
PASS_K="${PASS_K:-16}"

DATASETS="${DATASETS:-humaneval_plus mbpp_plus}"
GEN_OUTPUT_BASE_DIR="${GEN_OUTPUT_BASE_DIR:-$VERL_ROOT/gen_results/eval/code}"
RESULTS_FILE="${RESULTS_FILE:-$VERL_ROOT/results/OPD/code/Qwen3-1.7B_reverse_evalplus.json}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$VERL_ROOT/outputs/OPD/code/evalplus_logs_qwen3_1_7b}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/eval_qwen3_1_7b_reverse_evalplus_${RUN_ID}.log}"

TOPK32_MODEL_NAME="Qwen3-1.7B_OPD_CODE_teacherQwen3-8B_y_o_kl_reverse_full_vocab_clip0_topk32_vanilla_ms1_20260603-042450_LORA"
VANILLA_MODEL_NAME="Qwen3-1.7B_OPD_CODE_teacherQwen3-8B_y_o_kl_reverse_full_vocab_clip0_vanilla_ms1_20260603-042150_LORA"

TOPK32_MODEL_PATH="${TOPK32_MODEL_PATH:-/data/data/jiangli/models/OPD_CODE/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_reverse_full_vocab_clip0_topk32_vanilla_ms1_20260603-042450/epoch1/ms1/batch00001/hf_merged}"
VANILLA_MODEL_PATH="${VANILLA_MODEL_PATH:-/data/data/jiangli/models/OPD_CODE/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_reverse_full_vocab_clip0_vanilla_ms1_20260603-042150/epoch1/ms1/batch00001/hf_merged}"
BASE_TOKENIZER_PATH="${BASE_TOKENIZER_PATH:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"

mkdir -p "$LOG_DIR" "$(dirname "$RESULTS_FILE")"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_one() {
    local model_name="$1"
    local model_path="$2"
    local output_dir="$GEN_OUTPUT_BASE_DIR/$model_name"

    if [ ! -d "$model_path" ]; then
        log "ERROR: model not found: $model_path"
        exit 1
    fi

    "$PYTHON_BIN" - "$model_path" "$BASE_TOKENIZER_PATH" <<'PY'
import json
import shutil
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
base_tokenizer_path = Path(sys.argv[2])
cfg = model_path / "tokenizer_config.json"
if not cfg.exists():
    raise SystemExit(f"tokenizer_config.json not found: {cfg}")

data = json.load(open(cfg))
extra = data.get("extra_special_tokens")
if isinstance(extra, list):
    backup = cfg.with_suffix(cfg.suffix + ".bak_extra_special_tokens_list")
    if not backup.exists():
        shutil.copy2(cfg, backup)
    data["extra_special_tokens"] = {token: token for token in extra}
    tmp = cfg.with_suffix(cfg.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(cfg)

for filename in ("vocab.json", "merges.txt"):
    target = model_path / filename
    if target.exists():
        continue
    source = base_tokenizer_path / filename
    if not source.exists():
        raise SystemExit(f"missing base tokenizer file: {source}")
    shutil.copy2(source, target)
PY

    log "START $model_name"
    log "  model_path=$model_path"
    log "  datasets=$DATASETS"
    log "  pass_k=$PASS_K"
    log "  output_dir=$output_dir"
    log "  results_file=$RESULTS_FILE"

    PYTHON_BIN="$PYTHON_BIN" \
    NGPUS_PER_NODE="$NGPUS_PER_NODE" \
    GEN_TP="$GEN_TP" \
    DATASETS="$DATASETS" \
    PASS_K="$PASS_K" \
    EVAL_BASE_MODEL_NAME="Qwen3-1.7B" \
    EVAL_MODEL_NAME="$model_name" \
    EVAL_OUTPUT_DIR="$output_dir" \
    EVAL_RESULTS_FILE="$RESULTS_FILE" \
    WRITE_RESULTS_CSV=true \
    GEN_OUTPUT_BASE_DIR="$GEN_OUTPUT_BASE_DIR" \
    RESULTS_BASE_DIR="$VERL_ROOT/results/OPD/code" \
    bash "$BENCHMARK_SH" "$model_path"

    log "DONE $model_name"
}

main() {
    cd "$VERL_ROOT"
    log "Logging to $LOG_FILE"
    log "Serial order: topk32 reverse -> vanilla reverse"
    run_one "$TOPK32_MODEL_NAME" "$TOPK32_MODEL_PATH"
    run_one "$VANILLA_MODEL_NAME" "$VANILLA_MODEL_PATH"
    log "All requested EvalPlus evals completed."
}

main "$@" 2>&1 | tee -a "$LOG_FILE"
