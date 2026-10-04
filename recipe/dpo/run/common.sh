#!/usr/bin/env bash
# Shell bridge to the stdlib-only Python defaults; no Conda/site initialization.
_PENS_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PENS_DEFAULTS_PYTHON="${PENS_DEFAULTS_PYTHON:-/usr/bin/python3}"
export PENS_DEFAULTS_PYTHON

# Keep credentials out of bash -x output, including command substitutions.
_pens_env_xtrace=false
case "$-" in
  *x*) _pens_env_xtrace=true; set +x ;;
esac
if ! _pens_env_exports="$("${PENS_DEFAULTS_PYTHON}" -S "${_PENS_COMMON_DIR}/common.py" --shell-env)"; then
  if [[ "${_pens_env_xtrace}" == true ]]; then set -x; fi
  unset _pens_env_exports _pens_env_xtrace
  return 1
fi
eval "${_pens_env_exports}"
unset _pens_env_exports
if [[ "${_pens_env_xtrace}" == true ]]; then set -x; fi
unset _pens_env_xtrace

_pens_exports="$("${PENS_DEFAULTS_PYTHON}" -S "${_PENS_COMMON_DIR}/common.py" --shell-defaults)" || return
eval "${_pens_exports}"
unset _pens_exports

pens_init_run() {
  local kind="$1"
  shift
  local run_exports
  run_exports="$("${PENS_DEFAULTS_PYTHON}" -S "${_PENS_COMMON_DIR}/common.py" --shell-run "${kind}" -- "$@")" || return
  eval "${run_exports}"
}

pens_hydra_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\'/\\\'}"
  printf "'%s'" "${value}"
}
