#!/bin/bash
# Launch the two target model evals concurrently inside job 303379.

set -Eeuo pipefail

cd /scratch/l/luli/src/verl

export RUN_ROOT="${RUN_ROOT:-/scratch/l/luli/jiangli/eval_diag/full_pass16_parallel/job${SLURM_JOB_ID:-303379}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_ROOT"

BAD_MODEL="/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/epoch1/ms1/batch00001/hf_merged"
YR_MODEL="/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms40_20260523-175757/final/hf_merged"

BAD_NODE="${BAD_NODE:-tg10603}"
YR_NODE="${YR_NODE:-tg11103}"

echo "RUN_ROOT=$RUN_ROOT"
echo "bad_node=$BAD_NODE"
echo "yr_node=$YR_NODE"

srun --jobid=303379 --overlap --nodes=1 --ntasks=1 --nodelist="$BAD_NODE" --gres=gpu:h100:4 \
    --export=ALL,RUN_ROOT="$RUN_ROOT",MODEL_PATH="$BAD_MODEL",MODEL_LABEL="ms1_y_o_no_lora_bad" \
    bash recipe/math_evaluation/run_single_existing_alloc_eval.sh &
bad_pid=$!

srun --jobid=303379 --overlap --nodes=1 --ntasks=1 --nodelist="$YR_NODE" --gres=gpu:h100:4 \
    --export=ALL,RUN_ROOT="$RUN_ROOT",MODEL_PATH="$YR_MODEL",MODEL_LABEL="ms40_y_r_no_lora_final" \
    bash recipe/math_evaluation/run_single_existing_alloc_eval.sh &
yr_pid=$!

bad_status=0
yr_status=0
wait "$bad_pid" || bad_status=$?
wait "$yr_pid" || yr_status=$?

echo "bad_status=$bad_status"
echo "yr_status=$yr_status"
echo "RUN_ROOT=$RUN_ROOT"

if [ "$bad_status" -ne 0 ] || [ "$yr_status" -ne 0 ]; then
    exit 1
fi
