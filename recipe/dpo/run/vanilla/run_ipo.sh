#!/usr/bin/env bash
set -euxo pipefail

python3 -m recipe.dpo.main_dpo --config-name=dpo_ipo "$@"
