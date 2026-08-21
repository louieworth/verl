#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)"
source "$OPD_ROOT/scripts_math/lib/run_matrix_common.sh"
run_experiment_matrix "$SCRIPT_ROOT" "$@"
