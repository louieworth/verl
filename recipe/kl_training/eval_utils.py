#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Evaluation utilities for KL Training

import os
import json
import subprocess
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def run_evaluation_on_dataset(
    model_path: str,
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    pass_k: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    ngpus: int = 4,
    output_json_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> float:
    """
    Run evaluation on a single dataset.

    Args:
        model_path: Path to the trained model
        dataset_name: Name of the dataset (e.g., "aime24")
        dataset_path: Path to the dataset file
        output_dir: Directory to save generation results
        pass_k: Number of samples per problem
        temperature: Sampling temperature
        top_p: Top-p sampling
        ngpus: Number of GPUs
        output_json_path: Path to save results.json (same format as run_full_pipeline_multi_epoch.sh)
        model_name: Model name to use as key in results.json

    Returns:
        Accuracy on the dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    gen_output = os.path.join(output_dir, f"{dataset_name}_pass{pass_k}_generation.parquet")

    logger.info(f"  Generating responses for {dataset_name} with pass@{pass_k}...")

    # Step 1: Generate responses
    gen_cmd = [
        "python3", "-m", "verl.trainer.main_generation_server",
        f"trainer.n_gpus_per_node={ngpus}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=true",
        f"actor_rollout_ref.rollout.temperature={temperature}",
        f"actor_rollout_ref.rollout.top_p={top_p}",
        "actor_rollout_ref.rollout.top_k=20",
        "actor_rollout_ref.rollout.prompt_length=4096",
        "actor_rollout_ref.rollout.response_length=38912",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.95",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={pass_k}",
        f"data.train_files=['{dataset_path}']",
        "data.prompt_key=prompt",
        f"+data.output_path={gen_output}",
    ]

    try:
        subprocess.run(gen_cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        logger.error(f"  Generation failed for {dataset_name}: {e}")
        return 0.0

    logger.info(f"  Computing scores for {dataset_name}...")

    # Step 2: Evaluate responses and save to results.json
    eval_cmd = [
        "python3", "-m", "verl.trainer.main_eval",
        f"data.path={gen_output}",
        "data.prompt_key=prompt",
        "custom_reward_function.path=recipe/open_math_reasoning/compute_score.py",
        "custom_reward_function.name=compute_score_data_source",
    ]

    # Add output_json_path and model_name if provided (same pattern as run_full_pipeline_multi_epoch.sh)
    if output_json_path and model_name:
        eval_cmd.extend([
            f"+output_json_path={output_json_path}",
            f"+model_name={model_name}",
            f"+pass_k={pass_k}",
        ])

    try:
        subprocess.run(eval_cmd, check=True, capture_output=False)
        # Read accuracy from results.json if saved
        if output_json_path and model_name and os.path.exists(output_json_path):
            with open(output_json_path, "r") as f:
                results = json.load(f)
            if model_name in results:
                key = f"{dataset_name}_pass{pass_k}_generation_pass_{pass_k}"
                if key in results[model_name]:
                    return results[model_name][key]
        return 0.0
    except subprocess.CalledProcessError as e:
        logger.error(f"  Evaluation failed for {dataset_name}: {e}")
        return 0.0


def parse_accuracy_from_output(output: str) -> float:
    """
    Parse accuracy from evaluation output.

    This is a placeholder - adjust based on actual output format.
    """
    # TODO: Implement actual parsing logic
    # Example: look for "Accuracy: 0.45" or similar in output
    import re
    match = re.search(r'accuracy[:\s]+([0-9.]+)', output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def save_eval_results(
    results: Dict[str, float],
    output_path: str,
    model_name: str,
    config: Dict = None,
):
    """
    Save evaluation results to JSON file in the same format as run_full_pipeline_multi_epoch.sh.

    Format:
    {
        "model_name_epoch1": {
            "aime24_pass1_generation_pass_1": 0.433,
            "aime25_pass1_generation_pass_1": 0.367,
            ...
        }
    }

    Args:
        results: Dictionary with dataset names as keys and accuracy as values
        output_path: Path to save the JSON file
        model_name: Name of the model (e.g., "Qwen3-1.7B_kl_reverse")
        config: Optional training configuration to include
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results if file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Format results to match the expected format
    formatted_results = {}
    for dataset, accuracy in results.items():
        # Format: "{dataset}_pass1_generation_pass_1"
        key = f"{dataset}_pass1_generation_pass_1"
        formatted_results[key] = accuracy

    # Add to all_results with model_name as key
    all_results[model_name] = formatted_results

    # Save back to file
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    logger.info(f"Evaluation results saved to: {output_path}")

    # Also save a detailed version with config if provided
    if config:
        detailed_path = output_path.replace(".json", "_detailed.json")
        detailed_output = {
            "model_name": model_name,
            "results": formatted_results,
            "config": config,
        }
        with open(detailed_path, "w") as f:
            json.dump(detailed_output, f, indent=4)
        logger.info(f"Detailed results saved to: {detailed_path}")

