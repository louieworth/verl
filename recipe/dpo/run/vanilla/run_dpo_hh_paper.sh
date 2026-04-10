#!/usr/bin/env bash
set -euxo pipefail

TRAIN_FILE="${DPO_HH_TRAIN_FILE:-/data/data/jiangli/parquet/dpo_hh/train.parquet}"
VAL_FILE="${DPO_HH_VAL_FILE:-/data/data/jiangli/parquet/dpo_hh/val.parquet}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing HH train parquet: ${TRAIN_FILE}" >&2
  exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing HH val parquet: ${VAL_FILE}" >&2
  exit 1
fi

python3 -m recipe.dpo.main_dpo \
  --config-name=dpo_hh_paper \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  "$@"
