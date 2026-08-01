#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/conda/envs/verl/bin/python}"
SHARDS_ROOT="$REPO_ROOT/gen_results/eval/code/base/Qwen3-8B/livecodebench/shards"

SHARD_SPECS=(
    "shard_1_2023-05-07_2023-08-05|133"
    "shard_2_2023-08-06_2023-11-11|132"
    "shard_3_2023-11-12_2024-03-02|136"
    "shard_4_2024-03-03_2024-06-08|130"
    "shard_5_2024-06-09_2024-09-01|136"
    "shard_6_2024-09-01_2024-11-17|133"
    "shard_7_2024-11-17_2025-01-26|125"
    "shard_8_2025-01-27_2025-04-06|131"
)

echo "Waiting for all Base LCB shard evaluations"
while :; do
    ready=true
    for spec in "${SHARD_SPECS[@]}"; do
        shard_name="${spec%%|*}"
        expected="${spec##*|}"
        eval_file="$SHARDS_ROOT/$shard_name/livecodebench/runtime/output/Qwen3-235B-A22B/Scenario.codegeneration_16_0.6_eval_all.json"
        "$PYTHON_BIN" - "$eval_file" "$expected" <<'PY' 2>/dev/null || ready=false
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
with path.open() as handle:
    rows = json.load(handle)
raise SystemExit(0 if len(rows) == expected else 1)
PY
    done
    if [ "$ready" = true ]; then
        break
    fi
    sleep 30
done

cd "$REPO_ROOT"
metrics="$($PYTHON_BIN recipe/code_evaluation/extract_lcb_metrics.py \
    --root gen_results/eval/code/base/Qwen3-8B/livecodebench \
    --pass_k 16 \
    --aggregate_all \
    --expected_tasks 1055)"
printf '%s\n' "$metrics"

env \
    CUDA_VISIBLE_DEVICES= \
    PYTHON_BIN="$PYTHON_BIN" \
    DATASETS=livecodebench_v6 \
    LCB_AGGREGATE_ALL=true \
    LCB_EXPECTED_TASKS=1055 \
    bash "$SCRIPT_DIR/eval_qwen3_8b_base_code.sh"

echo "Base LCB aggregate written to results/base/code/Qwen3-8B_code_avg16_pass16.json"
