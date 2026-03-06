#!/usr/bin/env bash
# export RAY_DEBUG_POST_MORTEM=1

# This script combines run_generation.sh and run_eval.sh
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}  # Default tensor parallel size to 2
PASS_K=${PASS_K:-1}

# Generation
# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
# Automatically generate output directory from MODEL_PATH
MODEL_NAME=$(basename "${MODEL_PATH}")
OUTPUT_DIR="gen_results/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

# 262144 / 1024 = 256k

OUTPUT_PATH="${OUTPUT_DIR}/deepsclar_step3.parquet"
EVAL_OUTPUT_PATH="evaluation_results/eval_results.json"

deepsclar_path=gen_results/DeepSeek-R1-Distill-Qwen-1.5B/deepsclar_step2_reward0_correction.parquet
train_files="['$deepsclar_path']"

echo "Running generation... Output will be saved to ${OUTPUT_PATH}"
python3 -m verl.trainer.main_generation_server \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.project_name="${project_name}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.prompt_length=16384 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${PASS_K} \
    data.train_files="$train_files" \
    data.prompt_key=prompt \
    +data.output_path="${OUTPUT_PATH}"
    # data.split='train[:20%]'

                                    

