#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/data2/.huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-4B-Instruct-2507}"
exec "$SCRIPT_DIR/_run_qwen3_taco_sft.sh" "$@"
