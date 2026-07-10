#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Ordered-part Run-E variant: only parameters that differ from the shared preset.
export PENS_DPO_DATA_VARIANT="${PENS_DPO_DATA_VARIANT:-ordered_part}"
export SINGLE_WISE_DPO_MICRO_BATCH_SIZE="${SINGLE_WISE_DPO_MICRO_BATCH_SIZE:-4}"
export SINGLE_WISE_DPO_ACTOR_LR="${SINGLE_WISE_DPO_ACTOR_LR:-2e-6}"
export SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS="${SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS:-4500}"

exec bash "${SCRIPT_DIR}/run_single_wise_dpo.sh" "data.shuffle=${SINGLE_WISE_DPO_SHUFFLE:-false}" "$@"
