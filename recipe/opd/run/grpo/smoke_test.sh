#!/usr/bin/env bash
# CPU-only structural smoke test for the portable Qwen3-1.7B GRPO launchers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

launchers=(
    "$SCRIPT_DIR/qwen3_1_7b_math_grpo_8h100.sh"
    "$SCRIPT_DIR/qwen3_1_7b_code_grpo_8h100.sh"
)

bash -n "$SCRIPT_DIR/_run_qwen3_grpo_8h100.sh" "${launchers[@]}"

for launcher in "${launchers[@]}"; do
    output="$(TIMESTAMP=smoke GRPO_DRY_RUN=print bash "$launcher")"
    case "$launcher" in
        *_math_*)
            [[ "$output" == *"Task:              math"* ]]
            [[ "$output" == *"data/train_dataset/deepscaler/train_grpo.parquet"* ]]
            [[ "$output" == *"Qwen3-1.7B_math_grpo_smoke"* ]]
            [[ "$output" == *"Training time cap: 16200s"* ]]
            ;;
        *_code_*)
            [[ "$output" == *"Task:              code"* ]]
            [[ "$output" == *"data/train_dataset/taco/train_grpo.parquet"* ]]
            [[ "$output" == *"Qwen3-1.7B_code_grpo_smoke"* ]]
            [[ "$output" == *"Training time cap: 8106s"* ]]
            ;;
    esac
    [[ "$output" == *"Eval pass_k:       16"* ]]
    [[ "$output" == *"Eval after train:  true"* ]]
    [[ "$output" == *"avg16_pass16.json"* ]]
done

if rg -n '/(data2?|home|opt)/' "${launchers[@]}"; then
    echo "ERROR: a Qwen3-1.7B launcher contains a machine-specific path" >&2
    exit 1
fi

echo "Qwen3-1.7B GRPO smoke test: PASS (${#launchers[@]} launchers)"
