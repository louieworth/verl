#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/code/direct_skd_y_o_ms1_qwen3_1_7b_8b.sh"
bash "$SCRIPT_DIR/code/direct_skd_y_o_ms1_qwen3_4b_instruct_8b.sh"
# bash "$SCRIPT_DIR/math/direct_skd_y_o_ms1_qwen3_1_7b_8b.sh"
# bash "$SCRIPT_DIR/math/direct_skd_y_o_ms1_qwen3_4b_instruct_8b.sh"
