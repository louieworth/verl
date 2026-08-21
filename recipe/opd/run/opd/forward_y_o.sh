#!/bin/bash
# =============================================================================
# OPD recipe 1 — Forward KL + per-token clip, trained on y_o (teacher ≠ student).
# =============================================================================
#
# Applicable knobs (override via env):
#   MODEL_PATH              : student model (default Qwen/Qwen3-1.7B)
#   TEACHER_MODEL_PATH      : teacher model (default Qwen/Qwen3-14B, must differ from student)
#   Y_MODE                  : "y_o" (default) | "y_r"
#   MAX_PROMPT_LENGTH       : auto-derived teacher prompt budget
#   MAX_RESPONSE_LENGTH     : default 8192
#   TEMPERATURE             : default 1.0
#   TOTAL_EPOCHS            : default 1
#   MULTI_STEP              : default 40 offline on-policy updates
#   TRAIN_BATCH_SIZE        : inherited from run_kl_training.sh default; grad accumulation defaults to 16
#   KL_TOKEN_CLIP           : default 0.06 (forward-only knob)
#   KL_FULL_VOCAB_CHUNK_SIZE: optional full-vocab KL chunk override; trainer defaults to 512
#   TEACHER_TRAINING_PROMPT : "vanilla" (auto for y_o) | "refine"
#
# Inert / not applicable:
#   TOP_K                   : reverse-KL only.
# =============================================================================

set -e
set -o pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RESIDENT_STUDENT_ROLLOUT="${RESIDENT_STUDENT_ROLLOUT:-false}"

export DISTILL_MODE="opd"
export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="${Y_MODE:-y_o}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export TOP_K=0

export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MULTI_STEP="${MULTI_STEP:-40}"

# OPD with Qwen3-14B teacher + full_vocab needs much smaller token budget than
# OPSD (teacher = student = 1.7B). On A100-40G, default 49152 OOMs in the
# teacher forward (logits cat ≈ 15 GiB on top of ~25 GiB teacher+student
# resident). Keep this at least BASE_PROMPT_LENGTH + MAX_RESPONSE_LENGTH so
# the default 2048 + 8192 y_o sequence is not split below one full sample.
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-14336}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$SCRIPT_DIR")/run_kl_training.sh" "$@"
