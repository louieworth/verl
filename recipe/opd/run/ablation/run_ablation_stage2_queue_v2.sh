#!/bin/bash
# =============================================================================
# Driver v2: run the 2 follow-up y_cor ablations (exp 6 + exp 7) serially,
# evaluate each (PASS_K=16), and aggregate scores into ours_ablation_stage_2.json.
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$SCRIPT_DIR")"
VERL_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
LOG_DIR="$VERL_ROOT/outputs/ablation_stage2_logs"
mkdir -p "$LOG_DIR"

# Honor any externally-set WANDB_MODE, but keep ablation runs offline by default.
export WANDB_MODE="${WANDB_MODE:-offline}"

EXPERIMENTS=(
    "exp6_forward_ycor_s1r0:forward_y_cor_fv_s1r0.sh"
    "exp7_forward_ycor_s1r0_s2c:forward_y_cor_fv_s1r0_s2c.sh"
)

cd "$VERL_ROOT"

for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%:*}"
    script="${entry##*:}"
    log="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "###################################################################"
    echo "# Starting ${name} (${script})"
    echo "# Log: ${log}"
    echo "# Time: $(date)"
    echo "###################################################################"

    if bash "$SCRIPT_DIR/${script}" 2>&1 | tee "$log"; then
        echo "###################################################################"
        echo "# ${name} COMPLETED at $(date)"
        echo "###################################################################"
    else
        echo "###################################################################"
        echo "# ${name} FAILED at $(date) — continuing to next"
        echo "###################################################################"
    fi

    python3 "$SCRIPT_DIR/_aggregate_ablation_stage2.py" || true
done

echo ""
echo "All ${#EXPERIMENTS[@]} follow-up ablation experiments processed."
echo "Aggregated results: results/Qwen3-8B/ours_ablation_stage_2.json"
