#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_ROOT/lib/run_matrix_common.sh"
run_experiment_matrix "$SCRIPT_ROOT" "$@"
