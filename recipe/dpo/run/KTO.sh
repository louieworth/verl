#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Inherit the all-data recipe's model, optimizer, LoRA, batching, tokenization
# and checkpoint evaluation. KTO needs a fixed reference and mismatched KL
# sequences, so compute reference scores live rather than materializing pairs.
export POINTWISE_DPO_LOSS_TYPE=kto
export SINGLE_WISE_DPO_SAMPLE_VARIANT="${SINGLE_WISE_DPO_SAMPLE_VARIANT:-all}"
export SINGLE_WISE_DPO_REFERENCE_FREE=false
export SINGLE_WISE_DPO_AUTO_PRECOMPUTE_REFERENCE_LOGPS=false
# Finish after update 500 and its final checkpoint evaluation.
export SINGLE_WISE_DPO_TOTAL_TRAINING_STEPS=500
export KTO_DESIRABLE_WEIGHT="${KTO_DESIRABLE_WEIGHT:-1.0}"
export KTO_UNDESIRABLE_WEIGHT="${KTO_UNDESIRABLE_WEIGHT:-1.0}"

exec bash "${SCRIPT_DIR}/all.sh" "$@"
