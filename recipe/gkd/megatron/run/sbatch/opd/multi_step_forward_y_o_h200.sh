#!/bin/bash
#SBATCH --job-name=opd-ms-fwd-yo-h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --output=opd-ms-fwd-yo-h200-%j.out
#SBATCH --error=opd-ms-fwd-yo-h200-%j.err

# 1-node H200 "share" layout for y_o (8 GPUs on one H200 node):
#   GPU 0:       teacher Qwen3-8B vLLM (TP=1, ~30 GB)
#   GPU 1,2,3:   student rollout vLLM (DP=3)
#   GPU 4,5,6,7: student Megatron actor (DP=4, hybrid_engine=False)
#
# Same logical layout as the H100 2-node share script (1 teacher + 3 rollout
# shared, 4 actor isolated), just collapsed onto a single 8-GPU H200 node.

set -E
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/recipe/gkd/megatron/run/sbatch/opd" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/recipe/gkd/megatron/run/sbatch/opd"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export DISTILL_MODE="${DISTILL_MODE:-opd}"
export KL_TYPE="${KL_TYPE:-forward}"
export Y_MODE="${Y_MODE:-y_o}"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export TOP_K="${TOP_K:-0}"
export RUN_SCRIPT="${RUN_SCRIPT:-opd/forward_y_o.sh}"
export TEACHER_BACKEND="${TEACHER_BACKEND:-vllm_server}"
export N_LOGPROBS="${N_LOGPROBS:-full_vocab}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export STUDENT_ACTOR_MAX_TOKENS_PER_GPU="${STUDENT_ACTOR_MAX_TOKENS_PER_GPU:-49152}"
export TEACHER_MAX_PROMPT_LENGTH="${TEACHER_MAX_PROMPT_LENGTH:-$MAX_PROMPT_LENGTH}"
export TEACHER_SEQ_LEN="${TEACHER_SEQ_LEN:-$((TEACHER_MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-$TEACHER_SEQ_LEN}"
export TEACHER_N_SERVER_WORKERS="${TEACHER_N_SERVER_WORKERS:-1}"

export TEACHER_REQUEST_BATCH_SIZE="${TEACHER_REQUEST_BATCH_SIZE:-1}"

# Effective batch resolution, matching y_r:
#   EFFECTIVE_BATCH_SIZE = prompts per optimizer.step
#   TRAIN_BATCH_SIZE     = prompts per update_policy accumulation chunk
#   GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE / TRAIN_BATCH_SIZE
#
# Note: TRAIN_BATCH_SIZE is the user-facing "micro batch" for accumulation
# here. Megatron's MICRO_BATCH_SIZE remains an internal actor setting.
export EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-4}}"
export MAX_SAFE_TRAIN_BATCH="${MAX_SAFE_TRAIN_BATCH:-8}"
export DP_WORLD_SIZE="${DP_WORLD_SIZE:-4}"

if ! [[ "$EFFECTIVE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: EFFECTIVE_BATCH_SIZE must be a positive integer, got '$EFFECTIVE_BATCH_SIZE'" >&2
    exit 1
fi

if [ -n "${TRAIN_BATCH_SIZE:-}" ]; then
    if ! [[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TRAIN_BATCH_SIZE must be a positive integer, got '$TRAIN_BATCH_SIZE'" >&2
        exit 1
    fi
    if (( TRAIN_BATCH_SIZE % DP_WORLD_SIZE != 0 )); then
        echo "ERROR: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be a multiple of DP_WORLD_SIZE=$DP_WORLD_SIZE" >&2
        exit 1
    fi
    if (( EFFECTIVE_BATCH_SIZE % TRAIN_BATCH_SIZE != 0 )); then
        echo "ERROR: EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE not divisible by TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE" >&2
        exit 1
    fi
else
    TRAIN_BATCH_SIZE=0
    for candidate in $(seq "$MAX_SAFE_TRAIN_BATCH" -1 1); do
        if (( candidate % DP_WORLD_SIZE == 0 )) && (( EFFECTIVE_BATCH_SIZE % candidate == 0 )); then
            TRAIN_BATCH_SIZE=$candidate
            break
        fi
    done
    if (( TRAIN_BATCH_SIZE == 0 )); then
        echo "ERROR: no valid TRAIN_BATCH_SIZE in [1..$MAX_SAFE_TRAIN_BATCH] that is a multiple of DP_WORLD_SIZE=$DP_WORLD_SIZE and divides EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE" >&2
        exit 1
    fi
    export TRAIN_BATCH_SIZE
fi
export GRADIENT_ACCUMULATION_STEPS=$((EFFECTIVE_BATCH_SIZE / TRAIN_BATCH_SIZE))

echo "[OPD config] EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE -> TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE x GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"

export OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-multi_step}"
export SCHEDULER="${SCHEDULER:-one_step_off}"
export TEACHER_CLIENT_TIMEOUT_MS="${TEACHER_CLIENT_TIMEOUT_MS:-7200000}"
export TEACHER_CLIENT_RCVTIMEO_MS="${TEACHER_CLIENT_RCVTIMEO_MS:-$TEACHER_CLIENT_TIMEOUT_MS}"

export HYBRID_ENGINE=False

source "$SCRIPT_DIR/../_h200_1node_share_defaults.sh"
exec bash "$SCRIPT_DIR/../_launch_1node_share.sh" "$@"
