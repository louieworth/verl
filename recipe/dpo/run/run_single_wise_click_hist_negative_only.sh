#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Negative-only ablation of the current reference-free Prospect recipe.
# All optimization, model, checkpoint, and interleaved-eval defaults are
# inherited from run_single_wise_click_hist_all.sh. Only the train split
# and output namespace differ. In particular, preserve Slurm's CUDA mask and
# inherit its topology from the same shared launcher as the all run.
export SINGLE_WISE_DPO_SAMPLE_VARIANT=negative_only

echo "Prospect negative-only run: experiment=${SINGLE_WISE_DPO_EXPERIMENT_NAME:-auto}"
exec bash "${SCRIPT_DIR}/run_single_wise_click_hist_all.sh" "$@"
