#!/bin/bash
# =============================================================================
# OPD recipe 1 — Forward KL + per-token clip, trained on y_o (teacher ≠ student).
# =============================================================================
#
# Applicable knobs (override via env):
#   MODEL_PATH              : student model (default Qwen/Qwen3-1.7B)
#   TEACHER_MODEL_PATH      : teacher model (default Qwen/Qwen3-8B, must differ from student)
#   Y_MODE                  : "y_o" (default) | "y_r"
#   MAX_PROMPT_LENGTH       : default depends on Y_MODE
#   MAX_RESPONSE_LENGTH     : default 16384
#   TEMPERATURE             : default 1.0
#   LEARNING_RATE           : default 5e-6
#   TOTAL_EPOCHS            : default 1
#   KL_TOKEN_CLIP           : default 0.06 (forward-only knob)
#   TEACHER_TRAINING_PROMPT : "vanilla" (auto for y_o) | "refine"
#
# Inert / not applicable:
#   TOP_K                   : reverse-KL only.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"

export DISTILL_MODE="opd"
export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export TOP_K=0

export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export GRADIENT_ACCUMULATION_STEPS=2

# OPD with Qwen3-8B teacher + full_vocab needs much smaller token budget than
# OPSD (teacher = student = 1.7B). On A100-40G, default 49152 OOMs in the
# teacher forward (logits cat ≈ 15 GiB on top of ~25 GiB teacher+student
# resident). 16384 keeps logits peak ≈ 4 GiB.
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-16384}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
