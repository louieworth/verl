# Evaluation Datasets

This document lists all evaluation datasets used in the multi-epoch policy correction pipeline.

## Available Datasets

### American Invitational Mathematics Examination (AIME)
- **AIME 2024**: 30 problems
- **AIME 2025**: 30 problems
- Source: `math-ai/aime24`, `math-ai/aime25`
- Format: Numeric answers in `\boxed{}` notation

### American Mathematics Competition (AMC)
- **AMC 2023**: Multiple choice problems
- Source: `math-ai/amc23`

### MATH500
- 500 challenging math problems
- Source: `math-ai/math500`

### Harvard-MIT Mathematics Tournament (HMMT)
- **HMMT 2025**: 30 problems
- **HMMT 2024**: 30 problems
- **HMMT 2023**: 30 problems
- Source: `PraMamba/HMMT-202502`, `MathArena/hmmt_feb_2024`, `MathArena/hmmt_feb_2023`
- Format: Numeric answers in `\boxed{}` notation

### USA Mathematical Olympiad (USAMO)
- **USAMO 2025**: 6 problems (proof-based)
- **USAMO 2024**: 3 problems (proof-based)
- Source: `MathArena/usamo_2025`, `MathArena/usamo_2024`
- Note: These are proof-based problems with more complex solutions

## Dataset Preparation

All datasets can be prepared using the following scripts:

```bash
# AIME datasets
python3 recipe/open_math_reasoning/prepare_aime.py --local_dataset_path /path/to/datasets

# MATH500 and AMC23
python3 recipe/open_math_reasoning/prepare_math500.py --local_dataset_path /path/to/datasets

# HMMT datasets
python3 recipe/open_math_reasoning/prepare_hmmt.py --local_dataset_path /path/to/datasets

# USAMO datasets
python3 recipe/open_math_reasoning/prepare_usamo.py --local_dataset_path /path/to/datasets
```

## Dataset Format

All datasets follow a standard parquet format with the following columns:

- `data_source`: Source dataset name (e.g., "aime24", "hmmt25")
- `prompt`: Chat format prompt with instruction
- `ability`: Problem category ("math")
- `reward_model`: Dictionary containing:
  - `type`: Evaluation method ("rule")
  - `ground_truth`: Correct answer
  - `is_proof_based`: (Optional) Boolean flag for proof-based problems
- `extra_info`: Additional metadata (problem, year, points, etc.)

## Usage in Evaluation Pipeline

The `run_full_pipeline_multi_epoch.sh` script automatically evaluates on all available datasets during Stage 3 Evaluation. Results are saved to:

- **Test Results**: `results/{MODEL_NAME}/results.json` - Evaluation on independent test sets
- **Training Results**: `results/{MODEL_NAME}/training_results.json` - Stage 1 & 2 performance on training data

## Notes

- USAMO problems are proof-based and may require different evaluation methods
- All problems use the instruction: "Please reason step by step, and put your final answer within \boxed{}."
- pass@k evaluation can be configured using the `TEST_PASS_K_VALUES` environment variable
