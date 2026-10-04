#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Only select the negative split; inherit the strict no-weight recipe and
# preserve the GPU mask assigned by Slurm.
export SINGLE_WISE_DPO_SAMPLE_VARIANT=negative_only

echo "Unweighted Prospect negative-only run: experiment=${SINGLE_WISE_DPO_EXPERIMENT_NAME:-auto}"
exec bash "${SCRIPT_DIR}/run_single_wise_click_hist_all_no_weight.sh" "$@"
