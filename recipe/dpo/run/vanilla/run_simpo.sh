#!/usr/bin/env bash
set -euxo pipefail

python3 -m recipe.dpo.main_dpo \
  --config-name=dpo_trainer \
  algorithm.dpo_loss_type=simpo \
  algorithm.reference_free=true \
  algorithm.dpo_beta=2.0 \
  algorithm.dpo_label_smoothing=0.0 \
  algorithm.simpo_gamma=0.5 \
  "$@"
