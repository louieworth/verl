#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Strict no-weight counterpart: retain the same Prospect sigmoid loss,
# optimizer, topology, data pipeline and evaluation; use alpha=lambda=1.
export SINGLE_WISE_DPO_SAMPLE_VARIANT="${SINGLE_WISE_DPO_SAMPLE_VARIANT:-all}"
export POINTWISE_DPO_LOSS_TYPE=prospect_dpo
export SINGLE_WISE_DPO_USE_FEEDBACK_WEIGHTS=false

echo "Unweighted Prospect run: sample_variant=${SINGLE_WISE_DPO_SAMPLE_VARIANT}, experiment=${SINGLE_WISE_DPO_EXPERIMENT_NAME:-auto}"
exec bash "${SCRIPT_DIR}/run_single_wise_click_hist_all.sh" "$@"
