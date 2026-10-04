#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Positive-only ablation of the current reference-free Prospect recipe.
# All optimization, model, checkpoint, and interleaved-eval defaults are
# inherited from run_single_wise_click_hist_all.sh. Only the train split
# and output namespace differ.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export SINGLE_WISE_DPO_N_GPUS_PER_NODE=4
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_NGPUS_PER_NODE=4

# SINGLE_WISE_DPO_RUN_TIMESTAMP=20260811_145245 bash recipe/dpo/run/run_single_wise_click_hist_positive_only.sh

export SINGLE_WISE_DPO_INPUT_VARIANT=click_hist
export SINGLE_WISE_DPO_SAMPLE_VARIANT=positive_only
export POINTWISE_DPO_LOSS_TYPE=prospect_dpo

export SINGLE_WISE_DPO_RUN_TIMESTAMP="${SINGLE_WISE_DPO_RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

echo "Prospect positive-only run: experiment=${SINGLE_WISE_DPO_EXPERIMENT_NAME:-auto}"
bash "${SCRIPT_DIR}/run_single_wise_click_hist_all.sh" "$@"
