#!/usr/bin/env bash
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NEWS_FILE="${NEWS_FILE:-/data/data/jiangli/Microsoft-PeNS/PENS/news.tsv}"
TRAIN_FILE="${TRAIN_FILE:-/data/data/jiangli/Microsoft-PeNS/PENS/train_unique_balance.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/dpo_pens/data/pens_click_hist_runE_23pct_seed42}"
COMPRESSION="${COMPRESSION:-snappy}"
WRITE_BATCH_SIZE="${WRITE_BATCH_SIZE:-10000}"
MAX_INPUT_ROWS="${MAX_INPUT_ROWS:-}"

COMMON_ARGS=(
  --news-file "${NEWS_FILE}"
  --train-file "${TRAIN_FILE}"
  --output-root "${OUTPUT_ROOT}"
  --compression "${COMPRESSION}"
  --write-batch-size "${WRITE_BATCH_SIZE}"
)

if [[ -n "${MAX_INPUT_ROWS}" ]]; then
  COMMON_ARGS+=(--max-input-rows "${MAX_INPUT_ROWS}")
fi

_SITE_PATHS="$("${PYTHON_BIN}" -c "import sys; print(':'.join(p for p in sys.path if p))")"
PYTHONPATH="${REPO_ROOT}:${_SITE_PATHS}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m recipe.dpo.data.prepare_pens_singlewise_dpo "${COMMON_ARGS[@]}" --sample-mode positive_only
PYTHONPATH="${REPO_ROOT}:${_SITE_PATHS}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m recipe.dpo.data.prepare_pens_singlewise_dpo "${COMMON_ARGS[@]}" --sample-mode negative_only
