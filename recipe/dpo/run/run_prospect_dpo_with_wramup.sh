#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Sampled Run-E variant: only parameters that differ from the shared preset.
export PENS_DPO_DATA_VARIANT="${PENS_DPO_DATA_VARIANT:-sampled}"
export SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS="${SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS:-4500}"

exec bash "${SCRIPT_DIR}/run_single_wise_dpo.sh" "$@"
