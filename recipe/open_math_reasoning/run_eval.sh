#!/usr/bin/env bash

OUTPUT_PATH='gen_results/DeepSeek-R1-Distill-Qwen-1.5B/deepsclar_step3.parquet'
EVAL_OUTPUT_PATH="evaluation_results/eval_results.json"
MODEL_NAME='DeepSeek-R1-Distill-Qwen-1.5B'
# Evaluation
python3 -m verl.trainer.main_eval \
    data.path="${OUTPUT_PATH}" \
    custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
    custom_reward_function.name=compute_score_data_source \
    +output_json_path="${EVAL_OUTPUT_PATH}" \
    +model_name="${MODEL_NAME}" \
    +pass_k=1