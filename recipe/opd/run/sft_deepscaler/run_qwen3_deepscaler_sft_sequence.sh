#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SFT_SCRIPTS=(
    "$SCRIPT_DIR/qwen3_4b_instruct_deepscaler_sft.sh"
    "$SCRIPT_DIR/qwen3_8b_deepscaler_sft.sh"
)

echo "DeepScaleR SFT sequence"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Order: Qwen3-4B-Instruct-2507 train+eval -> Qwen3-8B train+eval"

for index in "${!SFT_SCRIPTS[@]}"; do
    script="${SFT_SCRIPTS[$index]}"
    number=$((index + 1))
    start_time="$(date +%s)"

    echo
    echo "[$number/${#SFT_SCRIPTS[@]}] START $(date -u '+%Y-%m-%dT%H:%M:%SZ') $script"
    bash "$script" "$@"
    elapsed=$(( $(date +%s) - start_time ))
    echo "[$number/${#SFT_SCRIPTS[@]}] DONE  $(date -u '+%Y-%m-%dT%H:%M:%SZ') elapsed=${elapsed}s"
done

echo
echo "All DeepScaleR SFT training and evaluation jobs completed."
