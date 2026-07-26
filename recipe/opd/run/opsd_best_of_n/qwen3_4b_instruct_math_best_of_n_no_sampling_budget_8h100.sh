#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK=math
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-4B-Instruct-2507}"
export BEST_OF_N="${BEST_OF_N:-4}"

# Generate all N candidates for every prompt. No wall-clock sampling deadline is
# passed to main_generation_server.
export SAMPLING_BUDGET_MODE=unlimited
unset SAMPLING_BUDGET_SECONDS

# Keep every artifact and semantic key separate from the historical
# budget-constrained Best-of-N runs.
export RUN_NAMESPACE=opsd_best_of_n_no_sampling_budget
export RUN_VARIANT_TAG=no_sampling_budget
export GEN_RESULTS_ROOT="${GEN_RESULTS_ROOT:-gen_results/$RUN_NAMESPACE/training/${MODEL_ALIAS}_math_best_of_${BEST_OF_N}}"
export MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-model/trained/$RUN_NAMESPACE/best_of_${BEST_OF_N}}"
export RESULTS_FILE="${RESULTS_FILE:-results/$RUN_NAMESPACE/math/${MODEL_ALIAS}_math_best_of_${BEST_OF_N}.json}"

exec "$SCRIPT_DIR/_run_qwen3_best_of_n_8h100.sh" "$@"
