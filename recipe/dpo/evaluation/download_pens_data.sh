#!/usr/bin/env bash
set -euo pipefail

PENS_URL="${PENS_URL:-https://mind201910small.blob.core.windows.net/release/PENS.tar.gz}"
PENS_ROOT="${PENS_ROOT:-/data/data/jiangli/datasets/PENS}"
PENS_ARCHIVE="${PENS_ARCHIVE:-${PENS_ROOT}/PENS.tar.gz}"
PENS_EXTRACT="${PENS_EXTRACT:-true}"

mkdir -p "${PENS_ROOT}"

if [[ ! -f "${PENS_ARCHIVE}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L "${PENS_URL}" -o "${PENS_ARCHIVE}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${PENS_ARCHIVE}" "${PENS_URL}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
fi

if [[ "${PENS_EXTRACT}" == "true" ]]; then
  tar -xzf "${PENS_ARCHIVE}" -C "${PENS_ROOT}"
fi

echo "PENS archive: ${PENS_ARCHIVE}"
echo "PENS root: ${PENS_ROOT}"
echo "Official offline PENS evaluation does not require an extra judge model."

