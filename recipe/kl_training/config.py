#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Configuration for KL Divergence Training

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class KLTrainingConfig:
    """Configuration for KL Divergence Training."""

    # KL Settings
    kl_type: Literal["reverse", "forward"] = "reverse"
    kl_method: Literal["monte_carlo", "full_vocab"] = "monte_carlo"
    kl_coef: float = 1.0  # Pure KL divergence loss (no scaling)
    temperature: float = 1.0

    # Model Settings
    student_model_path: str = "Qwen/Qwen3-1.7B"
    teacher_model_path: str = ""  # Empty means same as student (memory efficient with LoRA)
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # Training Settings
    learning_rate: float = 2e-5  # Lower LR for pure KL loss (kl_coef=1.0)
    train_batch_size: int = 96
    gradient_accumulation_steps: int = 1
    total_epochs: int = 1
    max_length: int = 20480
    warmup_steps_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data Settings
    data_path: str = ""  # Path to stage1 generation results (contains expert_cot in extra_info)
    corrected_responses_path: str = ""  # Required for forward KL
    max_samples: Optional[int] = None
    use_initial_response: bool = False  # For reverse KL: Variant 1 (False) or Variant 2 (True)

    # Distributed Training
    local_rank: int = -1
    world_size: int = 1
    distributed_backend: str = "nccl"

    # Output Settings
    output_dir: str = "outputs/kl_training"
    model_save_dir: str = "/data/data/jiangli/models"  # Models saved here to avoid /home space
    gen_results_dir: str = ""  # Intermediate generation files (auto-generated if empty)
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 500
    save_merged_model: bool = True  # Merge LoRA adapters after training

    # Evaluation Settings
    eval_datasets: list = field(default_factory=lambda: ["aime24", "aime25", "math500"])
    run_eval_after_training: bool = False  # Set to True to run evaluation automatically
    eval_dataset_paths: dict = field(default_factory=lambda: {
        "aime24": "data/aime24.parquet",
        "aime25": "data/aime25.parquet",
        "math500": "data/math500.parquet",
        "hmmt24": "data/hmmt24.parquet",
        "hmmt25": "data/hmmt25.parquet",
        "amc23": "data/amc23.parquet",
    })

    # Optimization
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Wandb
    wandb_project: str = "verl-kl-training"
    wandb_run_name: str = ""

    # Adaptive KL coefficient (optional)
    use_adaptive_kl: bool = False
    target_kl: float = 0.01
    kl_coef_min: float = 0.01
    kl_coef_max: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if self.kl_type == "forward" and not self.corrected_responses_path:
            raise ValueError("Forward KL requires corrected_responses_path")

        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16")

        if not self.bf16 and not self.fp16:
            print("Warning: Neither bf16 nor fp16 is enabled. Training will use fp32.")


# ============================================================================
# Preset Configurations
# ============================================================================

def get_reverse_kl_monte_carlo_config(**kwargs) -> KLTrainingConfig:
    """Preset for Reverse KL + Monte Carlo (most efficient)."""
    config = KLTrainingConfig(
        kl_type="reverse",
        kl_method="monte_carlo",
        kl_coef=1.0,  # Pure KL
        temperature=1.0,
        learning_rate=2e-6,  # Lower LR for pure KL
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_reverse_kl_full_vocab_config(**kwargs) -> KLTrainingConfig:
    """Preset for Reverse KL + Full Vocabulary (more accurate)."""
    config = KLTrainingConfig(
        kl_type="reverse",
        kl_method="full_vocab",
        kl_coef=1.0,  # Pure KL
        temperature=1.0,
        learning_rate=1e-6,  # Even lower LR for full vocab (memory intensive)
        gradient_checkpointing=True,  # Required for memory
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_forward_kl_monte_carlo_config(**kwargs) -> KLTrainingConfig:
    """Preset for Forward KL + Monte Carlo (distribution covering)."""
    config = KLTrainingConfig(
        kl_type="forward",
        kl_method="monte_carlo",
        kl_coef=1.0,  # Pure KL
        temperature=1.0,
        learning_rate=2e-6,  # Lower LR for pure KL
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def get_forward_kl_full_vocab_config(**kwargs) -> KLTrainingConfig:
    """Preset for Forward KL + Full Vocabulary (precise distribution matching)."""
    config = KLTrainingConfig(
        kl_type="forward",
        kl_method="full_vocab",
        kl_coef=1.0,  # Pure KL
        temperature=1.0,
        learning_rate=1e-6,  # Even lower LR for full vocab
        gradient_checkpointing=True,
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config
