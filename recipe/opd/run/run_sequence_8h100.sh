#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
SEQUENCE_DRY_RUN="${SEQUENCE_DRY_RUN:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"

usage() {
    cat <<'EOF'
Usage:
  bash recipe/opd/run/run_sequence_8h100.sh grpo
  bash recipe/opd/run/run_sequence_8h100.sh best_of_n
  bash recipe/opd/run/run_sequence_8h100.sh all

Modes:
  grpo       Run the four Qwen3 GRPO experiments.
  best_of_n  Run the four Qwen3 OPSD Best-of-N experiments.
  all        Run GRPO first, then OPSD Best-of-N (eight jobs total).

Environment:
  SEQUENCE_DRY_RUN=true    Resolve and validate every command without using GPUs.
  CONTINUE_ON_ERROR=true   Continue to later jobs after a failed experiment.
EOF
}

case "$MODE" in
    grpo|best_of_n|all) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "ERROR: mode must be grpo, best_of_n, or all; got: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac

case "$SEQUENCE_DRY_RUN" in
    true|false) ;;
    *) echo "ERROR: SEQUENCE_DRY_RUN must be true or false" >&2; exit 2 ;;
esac
case "$CONTINUE_ON_ERROR" in
    true|false) ;;
    *) echo "ERROR: CONTINUE_ON_ERROR must be true or false" >&2; exit 2 ;;
esac

GRPO_SCRIPTS=(
    recipe/opd/run/grpo/qwen3_4b_instruct_math_grpo_8h100.sh
    recipe/opd/run/grpo/qwen3_4b_instruct_code_grpo_8h100.sh
    recipe/opd/run/grpo/qwen3_8b_math_grpo_8h100.sh
    recipe/opd/run/grpo/qwen3_8b_code_grpo_8h100.sh
)

BEST_OF_N_SCRIPTS=(
    recipe/opd/run/opsd_best_of_n/qwen3_4b_instruct_math_best_of_n_8h100.sh
    recipe/opd/run/opsd_best_of_n/qwen3_4b_instruct_code_best_of_n_8h100.sh
    recipe/opd/run/opsd_best_of_n/qwen3_8b_math_best_of_n_8h100.sh
    recipe/opd/run/opsd_best_of_n/qwen3_8b_code_best_of_n_8h100.sh
)

METHODS=()
SCRIPTS=()
if [ "$MODE" = "grpo" ] || [ "$MODE" = "all" ]; then
    for script in "${GRPO_SCRIPTS[@]}"; do
        METHODS+=(grpo)
        SCRIPTS+=("$script")
    done
fi
if [ "$MODE" = "best_of_n" ] || [ "$MODE" = "all" ]; then
    for script in "${BEST_OF_N_SCRIPTS[@]}"; do
        METHODS+=(best_of_n)
        SCRIPTS+=("$script")
    done
fi

run_experiment() {
    local method="$1"
    local script="$2"

    if [ ! -f "$script" ]; then
        echo "ERROR: missing experiment launcher: $script" >&2
        return 127
    fi

    if [ "$SEQUENCE_DRY_RUN" = "true" ]; then
        if [ "$method" = "grpo" ]; then
            GRPO_DRY_RUN=true bash "$script"
        else
            OPSD_BEST_OF_N_DRY_RUN=true bash "$script"
        fi
    else
        bash "$script"
    fi
}

sequence_start="$(date +%s)"
total="${#SCRIPTS[@]}"
failures=()

echo "Sequence mode:      $MODE"
echo "Experiment count:   $total"
echo "Dry run:            $SEQUENCE_DRY_RUN"
echo "Continue on error:  $CONTINUE_ON_ERROR"

for index in "${!SCRIPTS[@]}"; do
    method="${METHODS[$index]}"
    script="${SCRIPTS[$index]}"
    number=$((index + 1))
    experiment_start="$(date +%s)"

    echo
    echo "[$number/$total] START $(date -u '+%Y-%m-%dT%H:%M:%SZ') [$method] $script"
    if run_experiment "$method" "$script"; then
        elapsed=$(( $(date +%s) - experiment_start ))
        echo "[$number/$total] DONE  $(date -u '+%Y-%m-%dT%H:%M:%SZ') elapsed=${elapsed}s"
    else
        status=$?
        elapsed=$(( $(date +%s) - experiment_start ))
        echo "[$number/$total] FAIL  $(date -u '+%Y-%m-%dT%H:%M:%SZ') status=$status elapsed=${elapsed}s" >&2
        failures+=("$script:$status")
        if [ "$CONTINUE_ON_ERROR" != "true" ]; then
            exit "$status"
        fi
    fi
done

sequence_elapsed=$(( $(date +%s) - sequence_start ))
if [ "${#failures[@]}" -gt 0 ]; then
    echo
    echo "Sequence completed with ${#failures[@]} failure(s) in ${sequence_elapsed}s:" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

echo
echo "Sequence completed successfully in ${sequence_elapsed}s."
