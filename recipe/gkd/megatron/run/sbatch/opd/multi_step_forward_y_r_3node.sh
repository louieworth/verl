#!/bin/bash
#SBATCH --job-name=opd-ms-fwd-yr-3node
#SBATCH --nodes=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=48
#SBATCH --output=opd-ms-fwd-yr-3node-%j.out
#SBATCH --error=opd-ms-fwd-yr-3node-%j.err

set -E
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/recipe/gkd/megatron/run/sbatch/opd" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/recipe/gkd/megatron/run/sbatch/opd"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export DISTILL_MODE="${DISTILL_MODE:-opd}"
export KL_TYPE="${KL_TYPE:-forward}"
export Y_MODE="${Y_MODE:-y_r}"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-refine}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
export TOP_K="${TOP_K:-0}"
export RUN_SCRIPT="${RUN_SCRIPT:-opd/forward_y_r.sh}"
export TEACHER_BACKEND="${TEACHER_BACKEND:-vllm_server}"
export N_LOGPROBS="${N_LOGPROBS:-full_vocab}"

source "$SCRIPT_DIR/_h100_3node_streaming_common.sh"
exec bash "$SCRIPT_DIR/../_launch_3node_split.sh" "$@"
