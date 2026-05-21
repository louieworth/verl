#!/bin/bash
set -e
set -o pipefail

export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.06}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/reverse_y_o.sh" "$@"
