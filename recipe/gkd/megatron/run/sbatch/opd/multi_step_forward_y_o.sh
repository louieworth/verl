#!/bin/bash
#SBATCH --job-name=opd-ms-fwd-yo-3node
#SBATCH --nodes=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=48
#SBATCH --output=opd-ms-fwd-yo-3node-%j.out
#SBATCH --error=opd-ms-fwd-yo-3node-%j.err

# H100 y_o streaming layout: 12 GPUs total across 3 nodes.
#   node 0-1: teacher Qwen3-8B vLLM replicas, 4 replicas x TP=2 = 8 GPUs
#   node 2: student rollout vLLM on GPUs 0,1 + student Megatron actor on GPUs 2,3

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

source "$SCRIPT_DIR/_h100_3node_streaming_common.sh"
exec bash "$SCRIPT_DIR/../_launch_3node_split.sh" "$@"
