#!/usr/bin/env bash
set -euxo pipefail

export SINGLE_WISE_DPO_INPUT_VARIANT="${SINGLE_WISE_DPO_INPUT_VARIANT:-click_hist}"
export SINGLE_WISE_DPO_SAMPLE_VARIANT="${SINGLE_WISE_DPO_SAMPLE_VARIANT:-all}"
# single_wise_dpo
export POINTWISE_DPO_LOSS_TYPE="${POINTWISE_DPO_LOSS_TYPE:-prospect_dpo}"

bash recipe/dpo/run/run_single_wise_dpo.sh "$@"
