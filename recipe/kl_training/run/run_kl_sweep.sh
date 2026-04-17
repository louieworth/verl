#!/bin/bash
# =============================================================================
# Serial sweep over KL training variants
# =============================================================================
#
# Runs the listed wrappers one after another in the same shell. Each wrapper
# already:
#   - generates / resumes its own stage1/stage2 data
#   - skips epochs whose checkpoint marker exists
#   - runs eval after training
# so re-launching this sweep after a crash will pick up where it left off.
#
# Failures are logged but do NOT abort the sweep — a transient OOM in one
# variant should not waste the remaining variants' GPU time. The summary at
# the end shows exit codes; inspect each wrapper's logs for details.
#
# Excluded by user:
#   - benchmark_kl_model.sh          (eval helper, not a training run)
#   - run_forward_correction_kl_training.sh  (forward MC correction)
#
# Usage:
#   bash run_kl_sweep.sh
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "run_kl_training.sh"                    # defaults: reverse + monte_carlo + rewrite
    "run_reverse_rewrite_full_vocab.sh"     # reverse + full_vocab + rewrite
    "run_opsd_jsd_8b.sh"                    # jsd  + monte_carlo (OPSD 8B preset)
    "run_opsd_jsd_full_vocab_8b.sh"         # jsd  + full_vocab  (OPSD 8B preset)
    "run_forward_correction_full_vocab.sh"  # forward + full_vocab + correction
)

SWEEP_LOG_DIR="$SCRIPT_DIR/../sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"
SWEEP_RUN_TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="$SWEEP_LOG_DIR/sweep_${SWEEP_RUN_TS}.log"

echo "=========================================="
echo "KL Training Sweep — start $(date)"
echo "  Sweep log: $SWEEP_LOG"
echo "  Variants:  ${#SCRIPTS[@]}"
for s in "${SCRIPTS[@]}"; do echo "    - $s"; done
echo "=========================================="

declare -a STATUSES
declare -a DURATIONS

for idx in "${!SCRIPTS[@]}"; do
    name="${SCRIPTS[$idx]}"
    path="$SCRIPT_DIR/$name"
    n=$((idx + 1))
    total=${#SCRIPTS[@]}

    if [ ! -x "$path" ]; then
        echo "[$n/$total] SKIP $name — not executable / missing" | tee -a "$SWEEP_LOG"
        STATUSES[$idx]="MISSING"
        DURATIONS[$idx]="-"
        continue
    fi

    echo ""
    echo "==========================================" | tee -a "$SWEEP_LOG"
    echo "[$n/$total] START $name — $(date)"        | tee -a "$SWEEP_LOG"
    echo "==========================================" | tee -a "$SWEEP_LOG"

    start=$(date +%s)
    set +e
    bash "$path" 2>&1 | tee -a "$SWEEP_LOG"
    rc=${PIPESTATUS[0]}
    set -e
    end=$(date +%s)
    elapsed=$((end - start))
    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))

    if [ "$rc" -eq 0 ]; then
        STATUSES[$idx]="OK"
        echo "[$n/$total] OK    $name (${h}h${m}m)" | tee -a "$SWEEP_LOG"
    else
        STATUSES[$idx]="FAIL(rc=$rc)"
        echo "[$n/$total] FAIL  $name rc=$rc (${h}h${m}m) — continuing sweep" | tee -a "$SWEEP_LOG"
    fi
    DURATIONS[$idx]="${h}h${m}m"
done

echo ""
echo "==========================================" | tee -a "$SWEEP_LOG"
echo "Sweep finished — $(date)"                   | tee -a "$SWEEP_LOG"
echo "==========================================" | tee -a "$SWEEP_LOG"
printf "%-45s %-12s %s\n" "Script" "Duration" "Status" | tee -a "$SWEEP_LOG"
printf "%-45s %-12s %s\n" "------" "--------" "------" | tee -a "$SWEEP_LOG"
for idx in "${!SCRIPTS[@]}"; do
    printf "%-45s %-12s %s\n" "${SCRIPTS[$idx]}" "${DURATIONS[$idx]}" "${STATUSES[$idx]}" | tee -a "$SWEEP_LOG"
done
echo "" | tee -a "$SWEEP_LOG"
echo "Full sweep log: $SWEEP_LOG"
