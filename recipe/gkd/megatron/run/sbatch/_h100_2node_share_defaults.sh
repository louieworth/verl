#!/bin/bash
# Defaults for the 2-node "y_o share" layout:
#   node 0: teacher Qwen3-8B vLLM replicas on all 4 GPUs (TP=1 each)
#   node 1: student rollout vLLM on GPUs 0,1 + Megatron actor on GPUs 2,3
#
# Rationale: with 8k responses and full-vocab teacher logprobs, teacher
# knowledge is the measured bottleneck. Keep request_batch_size=1 for memory
# safety, but scale throughput with independent teacher replicas.

export SCHEDULER="${SCHEDULER:-two_step_off}"
export OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-multi_step}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
export STUDENT_ACTOR_MAX_TOKENS_PER_GPU="${STUDENT_ACTOR_MAX_TOKENS_PER_GPU:-49152}"
export GPU_TYPE="${GPU_TYPE:-h100}"
export RESOURCE_LAYOUT="two_node_teacher_train_share"
export TRAINING_NNODES="${TRAINING_NNODES:-1}"
export ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
export TEACHER_NNODES="${TEACHER_NNODES:-1}"

# Physical CUDA layout:
#   teacher node: GPU 0,1,2,3
#   student node: rollout GPU 0,1; actor GPU 2,3
export TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0,1}"
export TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-2,3}"

export TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-4}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-2}"
export TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-2}"

export TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
export TEACHER_REPLICAS="${TEACHER_REPLICAS:-4}"

# User-requested aggressive teacher KV cache fraction for the 4-replica run.
export TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.90}"

# Rollout and actor share the student node through separate Ray resource pools.
export ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-False}"
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"
export ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-20480}"

export KEEP_ALIVE_ON_FAILURE="${KEEP_ALIVE_ON_FAILURE:-true}"
