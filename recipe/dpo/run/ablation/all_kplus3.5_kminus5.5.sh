#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# All-data feedback-weight ablation; inherit the main recipe's other settings.
export SINGLE_WISE_DPO_INPUT_VARIANT=click_hist
export SINGLE_WISE_DPO_SAMPLE_VARIANT=all
export POINTWISE_DPO_LOSS_TYPE=prospect_dpo
export SINGLE_WISE_DPO_USE_FEEDBACK_WEIGHTS=true
export SINGLE_WISE_DPO_POSITIVE_KAPPA=3.5
export SINGLE_WISE_DPO_NEGATIVE_KAPPA=5.5
export SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS=600

export SINGLE_WISE_DPO_RUN_TIMESTAMP="${SINGLE_WISE_DPO_RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export SINGLE_WISE_DPO_EXPERIMENT_NAME="${SINGLE_WISE_DPO_EXPERIMENT_NAME:-prospect_click_hist_all_with_weight_kplus3.5_kminus5.5_${SINGLE_WISE_DPO_RUN_TIMESTAMP}}"

echo "Prospect all weight ablation: kappa+=3.5, kappa-=5.5, experiment=${SINGLE_WISE_DPO_EXPERIMENT_NAME}"
exec bash "${SCRIPT_DIR}/../all.sh" "$@"
