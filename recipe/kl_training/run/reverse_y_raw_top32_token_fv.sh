#!/bin/bash
# =============================================================================
# Reverse KL + Rewrite + Full Vocab + Teacher Top-K Local Support
# Faithful to: Fu et al. 2026, "Revisiting On-Policy Distillation: Empirical
#              Failure Modes and Simple Fixes" (arXiv:2603.25562)
# =============================================================================
#
# Re-implements the paper's "teacher top-K local support matching" objective
# (Eqs. 6–8 in §3.1) on top of the existing reverse-KL + rewrite + full_vocab
# pipeline. The data path (stage1 student rollouts; rewrite-mode prompts) is
# the closest offline analogue to the paper's on-policy rollouts that this
# repo supports.
#
# Loss (paper Eq. 8 verbatim):
#   1. S(c_t) = TopK_q(c_t)                         # top-K under teacher q
#   2. π̂_θ(v) = π_θ(v) / Σ_{u∈S} π_θ(u)
#      q̂(v)   = q(v)   / Σ_{u∈S} q(u)
#   3. L = E_{c_t} Σ_{v∈S} π̂_θ(v) (log π̂_θ(v) − log q̂(v))   # reverse KL inside S
#
# Paper-faithful hyperparameters (Table A1, math single-task):
#   * Teacher top-K = 32   (TOP_K=32)
#   * Learning rate = 2e-6
#   * Warmup steps = 0
#   * Temperature  = 1
#   * Effective batch = 128 (their batch=128, mini-batch=64)
#       — On 8 GPUs with per-GPU bs=2 (memory-bound for full_vocab Qwen3-8B)
#         we use grad_accum=8 → effective batch = 2 * 8 GPUs * 8 = 128.
#   * Max prompt+response = 2048 + 16384 = 18432 tokens
#   * AdamW (verl default)
#   * No KL clip (the top-K renormalization already bounds per-position KL)
#
# Non-paper-specified knobs are inherited from
# run_reverse_rewrite_full_vocab.sh (FSDP, SP=1, MAX_TOKEN_LEN_PER_GPU=20480,
# LoRA r=64 α=128).
#
# Output discrimination — auto-derived from EXPERIMENT_TAG by run_kl_training.sh:
#   experiment tag    : kl_reverse_full_vocab_rewrite_clip0_topk32
#   results.json key  : Qwen3-8B_kl_reverse_full_vocab_rewrite_clip0_topk32_<DATE>_epoch1
#   model save dir    : /data/data/jiangli/models/Qwen3-8B_kl_reverse_full_vocab_rewrite_clip0_topk32_<DATE>/
#   wandb run name    : kl_reverse_full_vocab_rewrite_clip0_topk32_<DATE>_epoch1
#
# Usage:
#   bash run_reverse_rewrite_topk_full_vocab.sh                 # paper defaults (K=32)
#   TOP_K=16 bash run_reverse_rewrite_topk_full_vocab.sh        # paper ablation (K=16)
#   TOP_K=48 bash run_reverse_rewrite_topk_full_vocab.sh        # paper ablation (K=48)
# =============================================================================

set -e
set -o pipefail

# --- Paper-pinned hyperparameters (Table A1) ---
export KL_TYPE="${KL_TYPE:-reverse}"
export KL_METHOD="${KL_METHOD:-full_vocab}"
export TEMPERATURE="${TEMPERATURE:-1}"
# Paper's truncated objective renormalizes inside S; per-token KL is naturally
# bounded so they don't apply an additional clip. 0 disables.
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
# y_raw: stage1 student rollouts (closest offline analogue to the paper's on-policy rollouts);
# teacher prompt = π(·|x, y*) (no initial response).
export Y_MODE="${Y_MODE:-y_raw}"
export TOP_K="${TOP_K:-32}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# --- Effective batch = 128 to match paper, 2 * 8 GPUs * 8 accum = 128 ---
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

# --- LoRA matches paper's PEFT setup convention used in this repo ---
export USE_LORA="${USE_LORA:-true}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"

# --- FSDP / Length: paper uses 2048 prompt + 16384 response = 18432 tokens ---
export SP_SIZE="${SP_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-18432}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-20480}"

# OOM mitigation (must be set before python starts)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Diagnostic metrics (T1–T4) — same set as run_opsd_jsd_full_vocab_8b.sh
# so wandb dashboards line up across the OPD-family runs ---
export GRAD_COSINE_INTERVAL="${GRAD_COSINE_INTERVAL:-50}"
export CORRECTION_TOKEN_PHRASES="${CORRECTION_TOKEN_PHRASES:-Wait,Actually,However,Alternatively,Oops,Wrong,Error,Incorrect,Correction,Sorry,Hmm,Oh,Hold,Pause,Uh,Um}"
export LOG_DIFFICULTY_BUCKETS="${LOG_DIFFICULTY_BUCKETS:-true}"
export SCORE_STAGE1="${SCORE_STAGE1:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Reverse KL + Rewrite + Full Vocab + Top-K Teacher Local Support"
echo "(Fu et al. 2026, arXiv:2603.25562 — paper Eq. 8)"
echo "=========================================="
echo "  KL_TYPE=$KL_TYPE  KL_METHOD=$KL_METHOD  TOP_K=$TOP_K  CLIP=$KL_TOKEN_CLIP  TEMP=$TEMPERATURE"
echo "  Y_MODE=$Y_MODE  (PROMPT_MODE=rewrite, data=stage1 student rollouts)"
echo "  LR=$LEARNING_RATE  WARMUP=$WARMUP_RATIO  WD=$WEIGHT_DECAY"
echo "  BATCH=$TRAIN_BATCH_SIZE  GRAD_ACCUM=$GRADIENT_ACCUMULATION_STEPS  (effective batch = $((TRAIN_BATCH_SIZE * 8 * GRADIENT_ACCUMULATION_STEPS)))"
echo "  LoRA=$USE_LORA r=$LORA_RANK α=$LORA_ALPHA"
echo "  MAX_LEN=$MAX_LENGTH  MAX_TOKEN_LEN_PER_GPU=$MAX_TOKEN_LEN_PER_GPU  SP=$SP_SIZE"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  Loss:    truncated reverse KL on TEACHER's top-${TOP_K} support, with renormalization"
echo "  Metrics: T1(auto) T2(every $GRAD_COSINE_INTERVAL) T3 T4=$LOG_DIFFICULTY_BUCKETS"
echo "=========================================="

exec "$SCRIPT_DIR/run_kl_training.sh" "$@"
