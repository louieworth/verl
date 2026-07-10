#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Full-data variant: checkpoint selection/evaluation lives in the workflow below.
export PENS_DPO_DATA_VARIANT="${PENS_DPO_DATA_VARIANT:-full}"
export PENS_DPO_WORKFLOW="${PENS_DPO_WORKFLOW:-all_checkpoints}"
export SINGLE_WISE_DPO_MICRO_BATCH_SIZE="${SINGLE_WISE_DPO_MICRO_BATCH_SIZE:-7}"

exec bash "${SCRIPT_DIR}/run_single_wise_dpo.sh" "$@"
