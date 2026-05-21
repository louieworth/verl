#!/bin/bash
set -e
set -o pipefail

export TOP_K="${TOP_K:-32}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/forward_y_o.sh" "$@"
