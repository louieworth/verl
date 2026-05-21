#!/bin/bash
# Defaults for the 3-node split layout:
#   node 0: teacher Qwen3-8B vLLM (4 H100)
#   node 1: student vLLM rollout (4 H100, vLLM-only, gpu_memory_utilization=0.85)
#   node 2: student Megatron training (4 H100, actor-only)
#
# This bypasses the hybrid_engine + sleep_mode IMA bug entirely by physically
# separating rollout from training. The OPD y_r entrypoint sets
# scheduler=three_step_off to overlap y_o rollout, y_r generation, and y_r
# scoring.

export SCHEDULER="${SCHEDULER:-one_step_off}"
export OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-multi_step}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
export STUDENT_ACTOR_MAX_TOKENS_PER_GPU="${STUDENT_ACTOR_MAX_TOKENS_PER_GPU:-49152}"
export GPU_TYPE="${GPU_TYPE:-h100}"
export RESOURCE_LAYOUT="three_node"
export TRAINING_NNODES="${TRAINING_NNODES:-1}"
export ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
export TEACHER_NNODES="${TEACHER_NNODES:-1}"

export TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAINING_CUDA_VISIBLE_DEVICES="${TRAINING_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-4}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-4}"
export TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-4}"

export TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}"

# Student rollout owns the GPU, so it can use a higher vLLM memory fraction
# than the old hybrid layout where rollout and Megatron shared the same GPU.
# In STANDALONE mode (hybrid_engine=False), verl never calls engine.sleep()
# (see vllm_async_server.py:637), so sleep_mode is effectively a no-op and
# the hybrid sleep IMA path cannot trigger. Keep sleep/cache disabled for the
# first 3-node runs until this path has enough clean coverage.
export ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-False}"
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"
export ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-40960}"

export KEEP_ALIVE_ON_FAILURE="${KEEP_ALIVE_ON_FAILURE:-true}"
