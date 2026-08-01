#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SHARD_INDEX="${SHARD_INDEX:?Set SHARD_INDEX to an integer from 1 through 8}"

case "$SHARD_INDEX" in
    1) START_DATE=2023-05-07; END_DATE=2023-08-05 ;;
    2) START_DATE=2023-08-06; END_DATE=2023-11-11 ;;
    3) START_DATE=2023-11-12; END_DATE=2024-03-02 ;;
    4) START_DATE=2024-03-03; END_DATE=2024-06-08 ;;
    # LiveCodeBench parses end_date as midnight at the start of that day. Use
    # the following day for shards whose boundary date contains timed contests;
    # the aggregate step de-duplicates any boundary-day overlap by question_id.
    5) START_DATE=2024-06-09; END_DATE=2024-09-01 ;;
    6) START_DATE=2024-09-01; END_DATE=2024-11-17 ;;
    7) START_DATE=2024-11-17; END_DATE=2025-01-26 ;;
    8) START_DATE=2025-01-27; END_DATE=2025-04-06 ;;
    *) echo "ERROR: SHARD_INDEX must be from 1 through 8, got: $SHARD_INDEX" >&2; exit 2 ;;
esac

SHARD_NAME="shard_${SHARD_INDEX}_${START_DATE}_${END_DATE}"
export DATASETS=livecodebench_v6
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/gen_results/eval/code/base/Qwen3-8B/livecodebench/shards/$SHARD_NAME}"
export RESULTS_FILE="${RESULTS_FILE:-/data2/tmp/qwen3_8b_base_lcb_${SHARD_NAME}.json}"
export LOG_DIR="${LOG_DIR:-/data2/tmp/qwen3_8b_base_lcb_shard_logs/$SHARD_NAME}"
export LCB_EXTRA_ARGS="--start_date $START_DATE --end_date $END_DATE ${LCB_EXTRA_ARGS:-}"

exec "$SCRIPT_DIR/eval_qwen3_8b_base_code.sh" "$@"
