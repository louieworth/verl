#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Stage 1 Data Preparation: Prepare data for initial response generation
# Input: DeepScaleR dataset with problem, answer, solution
# Output: Parquet file with prompts, expert_cot in extra_info

import os
import argparse
import datasets
from verl.utils.reward_score.math_reward import remove_boxed

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


def make_map_fn_stage1(question_key="problem", data_source="deepscaleR"):
    """
    Prepare data for Stage 1 generation.
    Preserves expert solution for Stage 2 correction.
    """
    def process_fn(example, idx):
        # Extract problem and answer
        question_raw = example[question_key]
        answer_raw = example['answer']

        # Store expert CoT for Stage 2
        extra_info = {}
        extra_info['expert_cot'] = example.get('solution', '')
        extra_info[question_key] = question_raw
        extra_info['index'] = idx

        # Store original answer in extra_info
        extra_info['answer'] = answer_raw

        # Construct prompt with instruction
        question = question_raw + " " + instruction_following

        # Extract ground truth answer
        try:
            solution = remove_boxed(answer_raw)
        except Exception:
            solution = answer_raw

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": extra_info,
        }
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="/data/data/jiangli/huggingface/datasets/DeepScaleR",
        help="Path to DeepScaleR dataset directory",
    )
    parser.add_argument(
        "--output_file",
        default="gen_results/stage1/deepscaleR_stage1.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--data_source",
        default="deepscaleR",
        help="Data source name",
    )
    parser.add_argument(
        "--question_key",
        default="problem",
        help="Key name for question in dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.input_path}")
    # Try load_from_disk first (for datasets saved with save_to_disk)
    # Fallback to load_dataset for HuggingFace hub datasets
    try:
        ds_loaded = datasets.load_from_disk(args.input_path)
        # If it's a DatasetDict, get the 'train' split
        if isinstance(ds_loaded, datasets.DatasetDict):
            ds_raw = ds_loaded['train']
        else:
            ds_raw = ds_loaded
    except (ValueError, FileNotFoundError):
        # Not a disk-saved dataset, try loading from HuggingFace hub
        ds_raw = datasets.load_dataset(args.input_path, split="train")

    # Limit samples for testing if max_samples is specified
    if args.max_samples is not None:
        print(f"Limiting to first {args.max_samples} samples for testing...")
        ds_raw = ds_raw.select(range(min(args.max_samples, len(ds_raw))))

    print(f"Processing {len(ds_raw)} examples...")
    ds_processed = ds_raw.map(
        make_map_fn_stage1(question_key=args.question_key, data_source=args.data_source),
        with_indices=True,
        remove_columns=[col for col in ds_raw.column_names if col not in ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']],
    )

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving to: {args.output_file}")
    ds_processed.to_parquet(args.output_file)
    print("Stage 1 data preparation completed!")
