#!/usr/bin/env bash
#
# Quick evaluation script for ablation study
# Assumes training is already completed
#

set -e

# Configuration
MODEL_NAME="Qwen3-1.7B"
HF_MERGED_DIR="results/${MODEL_NAME}/ablation_only_corrected/sft_checkpoint/hf_merged"
EVAL_OUTPUT_DIR="results/${MODEL_NAME}/ablation_only_corrected/evaluation"
RESULT_FILE="results/${MODEL_NAME}/ablation_only_corrected/results.json"
PASS_K=1
NGPUS_PER_NODE=4
NNODES=1
GEN_TP=1

EVAL_DATASETS_DIR="/data/data/jiangli/huggingface/datasets"

declare -A EVAL_DATASETS=(
    ["aime24"]="${EVAL_DATASETS_DIR}/aime24/aime24_test.parquet"
    ["aime25"]="${EVAL_DATASETS_DIR}/aime25/aime25_test.parquet"
    ["amc23"]="${EVAL_DATASETS_DIR}/amc23/amc23_test.parquet"
    ["math500"]="${EVAL_DATASETS_DIR}/math500/math500_test.parquet"
    ["hmmt25"]="${EVAL_DATASETS_DIR}/hmmt25/hmmt25_test.parquet"
    ["hmmt24"]="${EVAL_DATASETS_DIR}/hmmt24/hmmt24_test.parquet"
)

mkdir -p "${EVAL_OUTPUT_DIR}"
echo '{}' > "${RESULT_FILE}"

for DATASET_NAME in aime24 aime25 amc23 math500 hmmt25 hmmt24; do
    DATASET_PATH="${EVAL_DATASETS[$DATASET_NAME]}"

    if [ ! -f "${DATASET_PATH}" ]; then
        echo "Warning: ${DATASET_NAME} not found, skipping..."
        continue
    fi

    echo ""
    echo "[Evaluating on ${DATASET_NAME}]"

    GEN_OUTPUT="${EVAL_OUTPUT_DIR}/${DATASET_NAME}_pass${PASS_K}_generation.parquet"

    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="${NNODES}" \
        trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
        actor_rollout_ref.model.path="${HF_MERGED_DIR}" \
        actor_rollout_ref.model.trust_remote_code=true \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.prompt_length=4096 \
        actor_rollout_ref.rollout.response_length=32768 \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n="${PASS_K}" \
        data.train_files="['${DATASET_PATH}']" \
        data.prompt_key=prompt \
        +data.output_path="${GEN_OUTPUT}"

    python3 -m verl.trainer.main_eval \
        data.path="${GEN_OUTPUT}" \
        data.prompt_key=prompt \
        custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
        custom_reward_function.name=compute_score_data_source \
        +output_json_path="${RESULT_FILE}" \
        +model_name="${MODEL_NAME}_only_corrected" \
        +pass_k=${PASS_K}

    echo "${DATASET_NAME} evaluation completed!"
done

echo ""
echo "Evaluation completed! Results saved to: ${RESULT_FILE}"
