#!/usr/bin/env bash
set -euxo pipefail

export SINGLE_WISE_DPO_INPUT_VARIANT="${SINGLE_WISE_DPO_INPUT_VARIANT:-click_hist}"
export SINGLE_WISE_DPO_SAMPLE_VARIANT="${SINGLE_WISE_DPO_SAMPLE_VARIANT:-negative_only}"

bash recipe/dpo/run/run_single_wise_dpo.sh "$@"
