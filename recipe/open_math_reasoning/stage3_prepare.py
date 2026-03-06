#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Stage 3 Data Preparation: Convert corrected responses to SFT format
# Input:
#   - Stage 2 corrected responses (original reward=0, now corrected)
#     Note: Whether the correction is correct or not, we use it for SFT
# Output: SFT dataset in message format (only corrected data)

import os
import argparse
import datasets


def to_sft_format(example):
    """
    Convert example to SFT message format.
    Uses the 'responses' field as the assistant message.
    """
    # Extract response
    responses_val = example.get("responses", [""])
    response_text = responses_val[0] if isinstance(responses_val, list) else responses_val

    # Extract prompt content
    prompt_content = example["prompt"][0]["content"]

    return {
        "messages": [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": response_text}
        ]
    }


def extract_prompt_from_correction_example(example):
    """
    For Stage 2 correction examples, the prompt is a complex template.
    We need to extract the original problem for SFT training.
    """
    extra_info = example.get("extra_info", {})

    # Get the original problem
    problem = extra_info.get("problem", "")

    # Reconstruct the simple prompt (same as Stage 1)
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    prompt_content = problem + " " + instruction_following

    return {
        "prompt": [{"role": "user", "content": prompt_content}],
        "responses": example.get("responses", [""]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2_corrected",
        required=True,
        help="Stage 2 corrected output parquet file (from generation on reward=0). "
             "Note: We use ALL corrected data regardless of whether correction is correct.",
    )
    parser.add_argument(
        "--output_file",
        default="gen_results/stage3/deepscaleR_stage3_sft.parquet",
        help="Output SFT dataset parquet file",
    )
    parser.add_argument(
        "--keep_correction_prompt",
        action="store_true",
        help="If set, keep the correction prompt (for training correction capability). "
             "Default: extract original problem for training problem-solving capability.",
    )
    args = parser.parse_args()

    # Load Stage 2 corrected dataset (original reward=0)
    print(f"Loading Stage 2 corrected dataset: {args.stage2_corrected}")
    ds_corrected = datasets.load_dataset("parquet", data_files=args.stage2_corrected, split="train")
    print(f"  - Samples: {len(ds_corrected)}")

    # Process corrected dataset
    print("\nProcessing corrected dataset...")
    if args.keep_correction_prompt:
        # Keep the correction prompt (for training correction capability)
        print("  Mode: Keeping correction prompt for SFT")
        ds_sft = ds_corrected.map(to_sft_format)
    else:
        # Extract original problem for SFT (for training problem-solving capability)
        print("  Mode: Extracting original problem for SFT")
        ds_sft = ds_corrected.map(extract_prompt_from_correction_example).map(to_sft_format)

    print(f"Final SFT dataset size: {len(ds_sft)}")
    print(f"  All samples from corrected dataset (original reward=0)")

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save SFT dataset
    print(f"\nSaving SFT dataset to: {args.output_file}")
    ds_sft.to_parquet(args.output_file)

    print("\nStage 3 data preparation completed!")
    print(f"SFT dataset ready for training with {len(ds_sft)} examples")
