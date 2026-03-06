#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# KL Divergence Trainer for Math Reasoning

import os
import json
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
import logging
from typing import Dict, Optional, List

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .config import KLTrainingConfig
from .data_utils import create_kl_dataloader
from .kl_utils import compute_kl_divergence
from .eval_utils import save_eval_results, run_evaluation_on_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KLTrainer:
    """
    Trainer for Token-Level KL Divergence Training.

    Supports 4 variants:
    1. Reverse KL + Monte Carlo
    2. Reverse KL + Full Vocabulary
    3. Forward KL + Monte Carlo
    4. Forward KL + Full Vocabulary
    """

    def __init__(self, config: KLTrainingConfig):
        self.config = config
        self.global_step = 0
        self.epoch = 0

        # Setup distributed training
        self._setup_distributed()

        # Initialize wandb (only on main process)
        if self.config.local_rank in [-1, 0] and HAS_WANDB:
            self._init_wandb()

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Load models
        self.student_model = self._load_student_model()
        self.teacher_model = self._load_teacher_model()

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = None  # Will be set after dataloader is created

        # Setup dataloader
        self.train_dataloader = self._create_dataloader()

        # Setup scheduler (needs total steps from dataloader)
        self.scheduler = self._setup_scheduler()

        logger.info(f"KL Trainer initialized:")
        logger.info(f"  KL Type: {config.kl_type}")
        logger.info(f"  KL Method: {config.kl_method}")
        logger.info(f"  Student Model: {config.student_model_path}")
        logger.info(f"  Teacher Model: {config.teacher_model_path or 'Same as student'}")
        logger.info(f"  Use LoRA: {config.use_lora}")
        logger.info(f"  Model Save Dir: {config.model_save_dir}")
        logger.info(f"  Gen Results Dir: {config.gen_results_dir}")
        if HAS_WANDB and self.config.local_rank in [-1, 0]:
            logger.info(f"  Wandb Project: {config.wandb_project}")
            logger.info(f"  Wandb Run Name: {config.wandb_run_name}")

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not HAS_WANDB:
            logger.warning("wandb not installed, skipping wandb initialization")
            return

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config={
                "kl_type": self.config.kl_type,
                "kl_method": self.config.kl_method,
                "kl_coef": self.config.kl_coef,
                "temperature": self.config.temperature,
                "use_initial_response": self.config.use_initial_response,
                "model_path": self.config.student_model_path,
                "use_lora": self.config.use_lora,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "train_batch_size": self.config.train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "total_epochs": self.config.total_epochs,
                "max_length": self.config.max_length,
                "warmup_ratio": self.config.warmup_steps_ratio,
                "weight_decay": self.config.weight_decay,
                "max_grad_norm": self.config.max_grad_norm,
            },
        )
        logger.info(f"Wandb initialized: {wandb.run.url}")

    def _setup_distributed(self):
        """Setup distributed training."""
        if self.config.local_rank != -1:
            torch.cuda.set_device(self.config.local_rank)
            dist.init_process_group(backend=self.config.distributed_backend)
            self.config.world_size = dist.get_world_size()
            logger.info(f"Distributed training: rank {self.config.local_rank}/{self.config.world_size}")
        else:
            self.config.world_size = 1
            logger.info("Single GPU training")

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer."""
        logger.info(f"Loading tokenizer from: {self.config.student_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_student_model(self) -> nn.Module:
        """Load student model (trainable)."""
        logger.info(f"Loading student model from: {self.config.student_model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map=None,
            trust_remote_code=True,
        )

        # Apply LoRA if enabled
        if self.config.use_lora:
            logger.info(f"Applying LoRA: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Move to device
        if self.config.local_rank != -1:
            model = model.to(self.config.local_rank)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
            )
        else:
            model = model.cuda()

        return model

    def _load_teacher_model(self) -> nn.Module:
        """
        Load teacher model (reference model).

        Design choice:
        - If teacher_model_path is provided: Load a separate teacher model
        - If teacher_model_path is empty AND student uses LoRA:
          Teacher shares the base model with student (memory efficient)
        - If teacher_model_path is empty AND student doesn't use LoRA:
          Load the same model separately (fallback)
        """
        config = self.config

        # If a separate teacher path is provided, load it
        if config.teacher_model_path:
            logger.info(f"Loading separate teacher model from: {config.teacher_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                config.teacher_model_path,
                torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
                device_map=None,
                trust_remote_code=True,
            )

            # Move to device
            if config.local_rank != -1:
                model = model.to(config.local_rank)
            else:
                model = model.cuda()

            model.eval()
            return model

        # If student uses LoRA, teacher can share the base model
        if config.use_lora:
            logger.info("Teacher model shares base model with student (memory efficient)")
            # Get the base model from the LoRA-wrapped student
            if hasattr(self.student_model, 'module'):
                # Unwrap DDP
                base_model = self.student_model.module.get_base_model()
            else:
                base_model = self.student_model.get_base_model()

            base_model.eval()
            return base_model

        # Fallback: load the same model separately (not ideal but works)
        logger.warning(
            "Loading teacher model separately from same path as student. "
            "This uses extra memory. Consider using LoRA or providing a separate teacher_model_path."
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.student_model_path,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map=None,
            trust_remote_code=True,
        )

        # Move to device
        if config.local_rank != -1:
            model = model.to(config.local_rank)
        else:
            model = model.cuda()

        model.eval()
        return model

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        # Get trainable parameters
        if self.config.use_lora:
            # Only optimize LoRA parameters
            if hasattr(self.student_model, 'module'):
                trainable_params = [p for p in self.student_model.module.parameters() if p.requires_grad]
            else:
                trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        else:
            trainable_params = self.student_model.parameters()

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return optimizer

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
        """Setup learning rate scheduler."""
        if self.train_dataloader is None:
            return None

        total_steps = len(self.train_dataloader) * self.config.total_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_steps_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return scheduler

    def _create_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return create_kl_dataloader(
            data_path=self.config.data_path,
            tokenizer=self.tokenizer,
            kl_type=self.config.kl_type,
            batch_size=self.config.train_batch_size,
            max_length=self.config.max_length,
            max_samples=self.config.max_samples,
            corrected_responses_path=self.config.corrected_responses_path,
            use_initial_response=self.config.use_initial_response,
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute KL divergence loss and additional metrics.

        Key insight: We compute KL divergence ONLY over the response part,
        not the prompt part. The labels tensor has -100 for prompt tokens.

        Returns:
            Dictionary with 'loss' and other metrics including perplexity
        """
        # Move batch to device
        device = self.config.local_rank if self.config.local_rank != -1 else 0
        student_input_ids = batch["student_input_ids"].to(device)
        student_attention_mask = batch["student_attention_mask"].to(device)
        student_labels = batch["student_labels"].to(device)
        student_prompt_len = batch["student_prompt_len"]

        teacher_input_ids = batch["teacher_input_ids"].to(device)
        teacher_attention_mask = batch["teacher_attention_mask"].to(device)
        teacher_labels = batch["teacher_labels"].to(device)
        teacher_prompt_len = batch["teacher_prompt_len"]

        # Forward pass: Student
        student_outputs = self.student_model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            use_cache=False,
        )

        # Forward pass: Teacher (no gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                use_cache=False,
            )

        # Create response masks (labels != -100)
        # For student: shift by 1 because logits[i] predicts token[i+1]
        student_response_mask = (student_labels[:, 1:] != -100).float()
        teacher_response_mask = (teacher_labels[:, 1:] != -100).float()

        # Compute KL divergence
        if self.config.kl_method == "monte_carlo":
            # Monte Carlo: need log probabilities
            # Logits shape: [batch, seq_len, vocab]
            # We need logits[:, :-1, :] to predict tokens[:, 1:]
            student_logprobs_full = torch.log_softmax(student_outputs.logits[:, :-1, :], dim=-1)
            teacher_logprobs_full = torch.log_softmax(teacher_outputs.logits[:, :-1, :], dim=-1)

            # Get log probs of actual response tokens
            # Use labels[:, 1:] as the target tokens (shifted by 1)
            student_labels_shifted = student_labels[:, 1:].clamp(min=0)  # Replace -100 with 0 for gather
            teacher_labels_shifted = teacher_labels[:, 1:].clamp(min=0)

            student_logprobs = torch.gather(
                student_logprobs_full,
                dim=-1,
                index=student_labels_shifted.unsqueeze(-1)
            ).squeeze(-1)
            teacher_logprobs = torch.gather(
                teacher_logprobs_full,
                dim=-1,
                index=teacher_labels_shifted.unsqueeze(-1)
            ).squeeze(-1)

            kl_loss = compute_kl_divergence(
                student_logprobs,
                teacher_logprobs,
                student_response_mask,  # Only compute over response
                kl_type=self.config.kl_type,
                kl_method="monte_carlo",
                is_logprobs=True,
            )

            # Compute perplexity over response tokens
            student_nll = -student_logprobs * student_response_mask
            student_ppl = torch.exp(student_nll.sum() / student_response_mask.sum().clamp(min=1))

            teacher_nll = -teacher_logprobs * teacher_response_mask
            teacher_ppl = torch.exp(teacher_nll.sum() / teacher_response_mask.sum().clamp(min=1))

        else:
            # Full vocabulary: use logits directly
            # Shift logits to align with labels
            student_logits_shifted = student_outputs.logits[:, :-1, :]
            teacher_logits_shifted = teacher_outputs.logits[:, :-1, :]

            kl_loss = compute_kl_divergence(
                student_logits_shifted,
                teacher_logits_shifted,
                student_response_mask,
                kl_type=self.config.kl_type,
                kl_method="full_vocab",
                is_logprobs=False,
                temperature=self.config.temperature,
            )

            # Compute perplexity using response tokens
            student_logprobs_full = torch.log_softmax(student_logits_shifted, dim=-1)
            teacher_logprobs_full = torch.log_softmax(teacher_logits_shifted, dim=-1)

            student_labels_shifted = student_labels[:, 1:].clamp(min=0)
            teacher_labels_shifted = teacher_labels[:, 1:].clamp(min=0)

            student_token_logprobs = torch.gather(
                student_logprobs_full, dim=-1, index=student_labels_shifted.unsqueeze(-1)
            ).squeeze(-1)
            teacher_token_logprobs = torch.gather(
                teacher_logprobs_full, dim=-1, index=teacher_labels_shifted.unsqueeze(-1)
            ).squeeze(-1)

            student_nll = -student_token_logprobs * student_response_mask
            student_ppl = torch.exp(student_nll.sum() / student_response_mask.sum().clamp(min=1))

            teacher_nll = -teacher_token_logprobs * teacher_response_mask
            teacher_ppl = torch.exp(teacher_nll.sum() / teacher_response_mask.sum().clamp(min=1))

        # Loss = KL loss
        loss = kl_loss

        return {
            "loss": loss,
            "kl_loss": kl_loss.detach(),
            "student_perplexity": student_ppl.detach(),
            "teacher_perplexity": teacher_ppl.detach(),
        }

    def train_epoch(self):
        """Train for one epoch."""
        self.student_model.train()
        self.teacher_model.eval()

        total_loss = 0
        total_kl_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            disable=self.config.local_rank not in [-1, 0],
        )

        for step, batch in enumerate(progress_bar):
            # Compute loss
            loss_dict = self.compute_loss(batch)
            loss = loss_dict["loss"]

            # Backward
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            # Update
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Compute gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.config.max_grad_norm,
                )

                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate

                    # Log to wandb
                    if HAS_WANDB and self.config.local_rank in [-1, 0]:
                        wandb.log({
                            "train/loss": loss_dict['loss'].item(),
                            "train/kl_loss": loss_dict['kl_loss'].item(),
                            "train/student_perplexity": loss_dict['student_perplexity'].item(),
                            "train/teacher_perplexity": loss_dict['teacher_perplexity'].item(),
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "train/learning_rate": lr,
                            "train/epoch": self.epoch,
                            "train/global_step": self.global_step,
                        })

                    # Log to console
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={loss_dict['loss'].item():.4f}, "
                        f"kl_loss={loss_dict['kl_loss'].item():.4f}, "
                        f"student_ppl={loss_dict['student_perplexity'].item():.2f}, "
                        f"teacher_ppl={loss_dict['teacher_perplexity'].item():.2f}, "
                        f"grad_norm={grad_norm:.4f}, "
                        f"lr={lr:.2e}"
                    )

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

            # Accumulate metrics
            total_loss += loss_dict["loss"].item()
            total_kl_loss += loss_dict["kl_loss"].item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "kl": f"{total_kl_loss / num_batches:.4f}",
            })

        return {
            "loss": total_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
        }

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.total_epochs}")
        logger.info(f"Batch size: {self.config.train_batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Total steps: {len(self.train_dataloader) * self.config.total_epochs // self.config.gradient_accumulation_steps}")

        for epoch in range(self.config.total_epochs):
            self.epoch = epoch
            metrics = self.train_epoch()

            # Log epoch metrics to wandb
            if HAS_WANDB and self.config.local_rank in [-1, 0]:
                wandb.log({
                    "epoch/total_loss": metrics["loss"],
                    "epoch/total_kl_loss": metrics["kl_loss"],
                    "epoch/epoch": epoch,
                })

            logger.info(f"Epoch {epoch} completed: {metrics}")

        logger.info("Training completed!")

        # Save final checkpoint
        self.save_checkpoint(final=True)

        # Merge LoRA adapters if needed
        if self.config.use_lora and self.config.save_merged_model:
            self.merge_lora_model()

        # Run evaluation if enabled
        if getattr(self.config, 'run_eval_after_training', False):
            self.run_evaluation()

        # Finish wandb run
        if HAS_WANDB and self.config.local_rank in [-1, 0]:
            wandb.finish()

    def run_evaluation(self):
        """Run evaluation on specified datasets and save results."""
        if self.config.local_rank not in [-1, 0]:
            return

        logger.info("Running evaluation...")

        # Determine which model to evaluate
        if self.config.use_lora and self.config.save_merged_model:
            eval_model_path = os.path.join(self.config.model_save_dir, "hf_merged")
        else:
            eval_model_path = os.path.join(self.config.model_save_dir, "final")

        logger.info(f"Evaluating model: {eval_model_path}")

        # Get evaluation datasets
        eval_datasets = getattr(self.config, 'eval_datasets', [])
        if not eval_datasets:
            eval_datasets = ["aime24", "aime25", "math500"]

        # Get dataset paths from config
        dataset_paths = getattr(self.config, 'eval_dataset_paths', {
            "aime24": "data/aime24.parquet",
            "aime25": "data/aime25.parquet",
            "math500": "data/math500.parquet",
            "hmmt24": "data/hmmt24.parquet",
            "hmmt25": "data/hmmt25.parquet",
            "amc23": "data/amc23.parquet",
        })

        # Generate model name for results (format: MODEL_NAME_kl_TYPE_METHOD)
        base_model_name = self.config.student_model_path.split('/')[-1]
        model_name = f"{base_model_name}_kl_{self.config.kl_type}_{self.config.kl_method}"

        # Create evaluation output directory (gen_results/evaluate/)
        # Use gen_results_dir if set, otherwise use output_dir
        gen_results_dir = getattr(self.config, 'gen_results_dir', '') or self.config.output_dir
        eval_output_dir = os.path.join(gen_results_dir, "evaluate")
        os.makedirs(eval_output_dir, exist_ok=True)

        # Results file path (same format as run_full_pipeline_multi_epoch.sh)
        verl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        results_base_dir = os.path.join(verl_root, "results", base_model_name)
        results_file = os.path.join(results_base_dir, "results.json")

        # Run evaluation on each dataset
        eval_results = {}
        for dataset in eval_datasets:
            if dataset not in dataset_paths:
                logger.warning(f"Dataset {dataset} not found in dataset_paths, skipping...")
                continue

            dataset_path = dataset_paths[dataset]
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset file not found: {dataset_path}, skipping...")
                continue

            logger.info(f"Evaluating on {dataset}...")

            try:
                accuracy = run_evaluation_on_dataset(
                    model_path=eval_model_path,
                    dataset_name=dataset,
                    dataset_path=dataset_path,
                    output_dir=eval_output_dir,
                    pass_k=1,
                    temperature=0.6,
                    top_p=0.95,
                    ngpus=self.config.world_size,
                    output_json_path=results_file,
                    model_name=model_name,
                )
                eval_results[dataset] = accuracy
                logger.info(f"  {dataset}: {accuracy:.2%}")
            except Exception as e:
                logger.error(f"  Failed to evaluate {dataset}: {e}")
                eval_results[dataset] = 0.0

        # Results are already saved by run_evaluation_on_dataset to results_file
        logger.info(f"Evaluation completed. Results saved to: {results_file}")
        logger.info(f"Evaluation generation files saved to: {eval_output_dir}")

        # Log eval results to wandb
        if HAS_WANDB and self.config.local_rank in [-1, 0]:
            wandb.log({f"eval/{dataset}": acc for dataset, acc in eval_results.items()})

    def merge_lora_model(self):
        """Merge LoRA adapters into base model."""
        if self.config.local_rank not in [-1, 0]:
            return

        logger.info("Merging LoRA adapters into base model...")

        # Get the final checkpoint path
        final_checkpoint = os.path.join(self.config.model_save_dir, "final")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="cpu",  # Load to CPU for merging
            trust_remote_code=True,
        )

        # Load LoRA model
        model = PeftModel.from_pretrained(base_model, final_checkpoint)

        # Merge and unload
        merged_model = model.merge_and_unload()

        # Save merged model
        merged_dir = os.path.join(self.config.model_save_dir, "hf_merged")
        os.makedirs(merged_dir, exist_ok=True)

        merged_model.save_pretrained(merged_dir)
        self.tokenizer.save_pretrained(merged_dir)

        logger.info(f"Merged model saved to: {merged_dir}")

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if self.config.local_rank not in [-1, 0]:
            return

        # Save to model_save_dir instead of output_dir
        if final:
            save_dir = os.path.join(self.config.model_save_dir, "final")
        else:
            save_dir = os.path.join(self.config.model_save_dir, f"checkpoint-{self.global_step}")

        os.makedirs(save_dir, exist_ok=True)

        # Save student model
        if self.config.use_lora:
            # Save LoRA adapters
            if hasattr(self.student_model, 'module'):
                self.student_model.module.save_pretrained(save_dir)
            else:
                self.student_model.save_pretrained(save_dir)
        else:
            # Save full model
            if hasattr(self.student_model, 'module'):
                self.student_model.module.save_pretrained(save_dir)
            else:
                self.student_model.save_pretrained(save_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)

        logger.info(f"Checkpoint saved to: {save_dir}")
