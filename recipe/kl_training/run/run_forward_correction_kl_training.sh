#!/bin/bash
# =============================================================================
# Forward KL + Correction (thin wrapper around run_kl_training.sh)
# =============================================================================
#
# This wrapper sets correction-mode defaults (forward KL, teacher uses
# initial_response + expert guidance, longer contexts) and forwards every
# other env var / CLI arg through to run_kl_training.sh. Any variable the
# caller exports will override the defaults below.
#
# Usage:
#   bash run_correction_kl_training.sh                 # use defaults
#   KL_TOKEN_CLIP=0.3 bash run_correction_kl_training.sh  # override clip
# =============================================================================

set -e
set -o pipefail

# KL Settings
export KL_TYPE="${KL_TYPE:-forward}"
export KL_METHOD="${KL_METHOD:-monte_carlo}"
export TEMPERATURE="${TEMPERATURE:-1}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.1}"
export USE_INITIAL_RESPONSE="${USE_INITIAL_RESPONSE:-true}"
export FORWARD_STAGE2_MODE="${FORWARD_STAGE2_MODE:-rewrite_all}"

# Training Settings (correction mode uses bigger effective batch)
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

# FSDP / Length (correction mode carries longer contexts)
export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20480}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-130000}"
export STAGE2_PROMPT_LENGTH="${STAGE2_PROMPT_LENGTH:-24576}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
