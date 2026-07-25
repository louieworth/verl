#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/data2/.huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-8B}"
exec "$SCRIPT_DIR/_run_qwen3_deepscaler_sft_7gpu.sh" "$@"
