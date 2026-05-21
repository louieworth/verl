#!/bin/bash
# Shared H100 3-node streaming defaults for OPD y_o/y_r.
# Layout:
#   node 0-1: teacher, 4 replicas x TP=2 = 8 H100
#   node 2: rollout 2 H100 + actor 2 H100 under one Ray head

if [ -z "${SCRIPT_DIR:-}" ]; then
    echo "ERROR: SCRIPT_DIR must be set before sourcing _h100_3node_streaming_common.sh" >&2
    exit 1
fi

# Keep student and teacher prompt budgets separate:
#   student prompt budget = 4096
#   teacher prompt budget = 4096 + 8192 for y_r refine; 4096 for vanilla/y_o
#   teacher total budget  = 20480 for refine; 12288 for vanilla/y_o
# The external teacher server controls its vLLM scheduling peak with
# TEACHER_MAX_NUM_BATCHED_TOKENS.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
if [ -z "${TEACHER_MAX_PROMPT_LENGTH+x}" ]; then
    if [ "${TEACHER_TRAINING_PROMPT:-vanilla}" = "refine" ]; then
        export TEACHER_MAX_PROMPT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    else
        export TEACHER_MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH"
    fi
fi
# Full-vocab KL with 8k responses can OOM when dynamic batching packs
# multiple long samples onto one actor GPU. Keep this conservative by default.
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
export STUDENT_ACTOR_MAX_TOKENS_PER_GPU="${STUDENT_ACTOR_MAX_TOKENS_PER_GPU:-16384}"
export TEACHER_SEQ_LEN="${TEACHER_SEQ_LEN:-$((TEACHER_MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-$TEACHER_SEQ_LEN}"
export TEACHER_DISABLE_CUSTOM_ALL_REDUCE="${TEACHER_DISABLE_CUSTOM_ALL_REDUCE:-true}"

export TEACHER_NNODES="${TEACHER_NNODES:-2}"
export TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-4}"
export TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-2}"
export TEACHER_REPLICAS="${TEACHER_REPLICAS:-4}"
export TEACHER_N_SERVER_WORKERS="${TEACHER_N_SERVER_WORKERS:-4}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-2}"
export TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-2}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0,1}"
export TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-2,3}"
export RAY_HEAD_CUDA_VISIBLE_DEVICES="${RAY_HEAD_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_HEAD_GPUS_PER_NODE="${RAY_HEAD_GPUS_PER_NODE:-4}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.90}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
export VERL_VLLM_USE_SHM_WEIGHT_SYNC="${VERL_VLLM_USE_SHM_WEIGHT_SYNC:-1}"
# Multi-node teacher is launched as ZMQ workers that join one proxy; the data
# path remains ZMQ and the teacher nodes do not need to join Ray.
export TEACHER_RAY_MANAGED="${TEACHER_RAY_MANAGED:-false}"

# Decouple online rollout/teacher batch from actor full-vocab KL microbatch:
#   ROLLOUT_BATCH_SIZE = prompts generated/scored together by y_o/y_r
#   TRAIN_BATCH_SIZE   = actor update microbatch size
#   EFFECTIVE_BATCH_SIZE = prompts per optimizer.step
export EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-1024}"
export MAX_SAFE_TRAIN_BATCH="${MAX_SAFE_TRAIN_BATCH:-4}"
export DP_WORLD_SIZE="${DP_WORLD_SIZE:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
case "$ROLLOUT_BATCH_SIZE" in
    ""|*[!0-9]*) echo "ERROR: ROLLOUT_BATCH_SIZE must be a positive integer, got $ROLLOUT_BATCH_SIZE" >&2; exit 1 ;;
esac
if (( ROLLOUT_BATCH_SIZE < 1 )); then
    echo "ERROR: ROLLOUT_BATCH_SIZE must be positive, got $ROLLOUT_BATCH_SIZE" >&2
    exit 1
fi

if [ -n "${TRAIN_BATCH_SIZE:-}" ]; then
    if (( EFFECTIVE_BATCH_SIZE % TRAIN_BATCH_SIZE != 0 )); then
        echo "ERROR: EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE not divisible by TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE" >&2
        exit 1
    fi
    if (( TRAIN_BATCH_SIZE % DP_WORLD_SIZE != 0 )); then
        echo "ERROR: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be a multiple of DP_WORLD_SIZE=$DP_WORLD_SIZE" >&2
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
if (( ROLLOUT_BATCH_SIZE % TRAIN_BATCH_SIZE != 0 )); then
    echo "ERROR: ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE must be divisible by actor TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE" >&2
    exit 1
fi

# Rollout agent workers are a concurrency knob. The rollout path pads prompts to
# a multiple of this value, so validate against ROLLOUT_BATCH_SIZE.
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"
case "$AGENT_NUM_WORKERS" in
    ""|*[!0-9]*) echo "ERROR: AGENT_NUM_WORKERS must be a positive integer, got $AGENT_NUM_WORKERS" >&2; exit 1 ;;
esac
if (( AGENT_NUM_WORKERS < 1 )); then
    echo "ERROR: AGENT_NUM_WORKERS must be positive, got $AGENT_NUM_WORKERS" >&2
    exit 1
fi
if (( ROLLOUT_BATCH_SIZE % AGENT_NUM_WORKERS != 0 )); then
    echo "ERROR: ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE must be divisible by AGENT_NUM_WORKERS=$AGENT_NUM_WORKERS to avoid rollout padding duplicate prompts" >&2
    exit 1
fi

export OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-multi_step}"
export SCHEDULER="${SCHEDULER:-three_step_off}"
export TEACHER_INFLIGHT_SAMPLES="${TEACHER_INFLIGHT_SAMPLES:-128}"
export MAX_POLICY_LAG="${MAX_POLICY_LAG:-4}"

echo "[OPD config] EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE -> actor TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE x GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS, ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE, AGENT_NUM_WORKERS=$AGENT_NUM_WORKERS, SCHEDULER=$SCHEDULER"

# Full-vocab teacher requests are memory-bound. Keep the conservative default
# for three_step_off; bounded lag can opt into true teacher request batching.
if [ "$SCHEDULER" = "bounded_lag_y_r" ]; then
    export TEACHER_REQUEST_BATCH_SIZE="${TEACHER_REQUEST_BATCH_SIZE:-4}"
else
    export TEACHER_REQUEST_BATCH_SIZE="${TEACHER_REQUEST_BATCH_SIZE:-1}"
fi
export TEACHER_CLIENT_TIMEOUT_MS="${TEACHER_CLIENT_TIMEOUT_MS:-7200000}"
export TEACHER_CLIENT_RCVTIMEO_MS="${TEACHER_CLIENT_RCVTIMEO_MS:-$TEACHER_CLIENT_TIMEOUT_MS}"
export TEACHER_LOCAL_CHUNK_SIZE="${TEACHER_LOCAL_CHUNK_SIZE:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HYBRID_ENGINE=False

source "$SCRIPT_DIR/../_h100_3node_defaults.sh"
