#!/bin/bash
# =============================================================================
# Driver: run the 5 stage-2 ablation experiments serially, evaluate each,
# and aggregate scores into results/Qwen3-8B/ours_ablation_stage_2.json.
#
# Each step trains one model and evaluates with PASS_K=16 (avg@16, pass@8,
# pass@16). Failures of one experiment do NOT stop the queue; the failure
# is logged and the next one starts.
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$SCRIPT_DIR")"
VERL_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
LOG_DIR="$VERL_ROOT/outputs/ablation_stage2_logs"
mkdir -p "$LOG_DIR"

# Honor any externally-set WANDB_MODE (offline/disabled) but default to
# online: ~/.netrc has a wandb API key, so wandb.init() should succeed.
export WANDB_MODE="${WANDB_MODE:-online}"

EXPERIMENTS=(
    "exp1_forward_ycor_s1r1:forward_y_cor_fv_s1r1.sh"
    "exp2_forward_yraw_r0:forward_y_raw_fv_r0.sh"
    "exp3_forward_yraw_r1:forward_y_raw_fv_r1.sh"
    "exp4_reverse_yraw_r0:reverse_y_raw_fv_r0.sh"
    "exp5_reverse_yraw_r1:reverse_y_raw_fv_r1.sh"
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

    # Update aggregation file regardless of success/failure
    python3 "$SCRIPT_DIR/_aggregate_ablation_stage2.py" || true
done

echo ""
echo "All ${#EXPERIMENTS[@]} ablation experiments processed."
echo "Aggregated results: results/Qwen3-8B/ours_ablation_stage_2.json"
