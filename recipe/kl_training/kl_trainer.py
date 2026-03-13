#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# KL divergence trainer backed by verl FSDP TrainingWorker engines.

import glob
import logging
import math
import os
import subprocess
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from verl.trainer.config.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

from .config import KLTrainingConfig
from .data_utils import create_kl_dataloader
from .eval_utils import run_evaluation_suite
from .kl_utils import compute_kl_divergence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KLTrainer:
    """Token-level KL training using verl's sharded FSDP engines."""

    def __init__(self, config: KLTrainingConfig):
        self.config = config
        self.global_step = 0
        self.epoch = 0

        self.local_rank, self.rank, self.world_size = initialize_global_process_group()
        self.config.local_rank = self.local_rank
        self.config.world_size = self.world_size

        self.tokenizer = self._load_tokenizer()
        self.train_dataloader = self._create_dataloader()
        self.total_training_steps = (
            math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps) * self.config.total_epochs
        )

        self.student_worker = self._build_student_worker()
        self.teacher_worker = self._build_teacher_worker()
        self.student_engine = self.student_worker.engine
        self.teacher_engine = self.teacher_worker.engine

        self.is_logging = (
            self.student_engine.is_mp_src_rank_with_outputs() and self.student_engine.get_data_parallel_rank() == 0
        )

        if self.is_logging and HAS_WANDB:
            self._init_wandb()

        logger.info("KL Trainer initialized with verl FSDP backend")
        logger.info("  KL Type: %s", self.config.kl_type)
        logger.info("  KL Method: %s", self.config.kl_method)
        logger.info("  Student Model: %s", self.config.student_model_path)
        logger.info("  Teacher Model: %s", self.config.teacher_model_path or self.config.student_model_path)
        logger.info("  World Size: %s", self.world_size)
        logger.info("  FSDP Strategy: %s", self.config.fsdp_strategy)
        logger.info("  FSDP Size: %s", self.config.fsdp_size)
        logger.info("  Max Token Len/GPU: %s", self.config.max_token_len_per_gpu)
        logger.info("  Per-GPU Batch Size: %s", self.config.train_batch_size)
        logger.info("  Grad Accum Steps: %s", self.config.gradient_accumulation_steps)

    def close(self):
        if HAS_WANDB and self.is_logging:
            wandb.finish()
        destroy_global_process_group()

    def _init_wandb(self):
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config={
                "kl_type": self.config.kl_type,
                "kl_method": self.config.kl_method,
                "temperature": self.config.temperature,
                "use_initial_response": self.config.use_initial_response,
                "student_model_path": self.config.student_model_path,
                "teacher_model_path": self.config.teacher_model_path or self.config.student_model_path,
                "use_lora": self.config.use_lora,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "train_batch_size_per_gpu": self.config.train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "total_epochs": self.config.total_epochs,
                "max_length": self.config.max_length,
                "max_token_len_per_gpu": self.config.max_token_len_per_gpu,
                "fsdp_strategy": self.config.fsdp_strategy,
                "fsdp_size": self.config.fsdp_size,
            },
        )
        logger.info("Wandb initialized: %s", wandb.run.url)

    def _load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.config.student_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _create_dataloader(self):
        return create_kl_dataloader(
            data_path=self.config.data_path,
            tokenizer=self.tokenizer,
            kl_type=self.config.kl_type,
            batch_size=self.config.train_batch_size,
            max_length=self.config.max_length,
            max_samples=self.config.max_samples,
            corrected_responses_path=self.config.corrected_responses_path,
            use_initial_response=self.config.use_initial_response,
            num_workers=self.config.num_workers,
            local_rank=self.rank,
            world_size=self.world_size,
        )

    def _build_model_config(self, model_path: str, *, trainable: bool) -> HFModelConfig:
        return HFModelConfig(
            path=model_path,
            tokenizer_path=model_path,
            trust_remote_code=True,
            load_tokenizer=False,
            enable_gradient_checkpointing=trainable and self.config.gradient_checkpointing,
            use_remove_padding=self.config.use_remove_padding,
            lora_rank=self.config.lora_rank if trainable and self.config.use_lora else 0,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
        )

    def _build_engine_config(self, *, forward_only: bool) -> FSDPEngineConfig:
        return FSDPEngineConfig(
            strategy=self.config.fsdp_strategy,
            fsdp_size=self.config.fsdp_size,
            ulysses_sequence_parallel_size=self.config.ulysses_sequence_parallel_size,
            forward_only=forward_only,
            use_dynamic_bsz=True,
            max_token_len_per_gpu=self.config.max_token_len_per_gpu,
            infer_max_token_len_per_gpu=self.config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=None,
            infer_micro_batch_size_per_gpu=None,
            use_remove_padding=self.config.use_remove_padding,
            use_torch_compile=self.config.use_torch_compile,
            param_offload=self.config.param_offload,
            optimizer_offload=self.config.optimizer_offload,
            offload_policy=self.config.offload_policy,
            dtype="bfloat16" if self.config.bf16 else "float16",
        )

    def _build_optimizer_config(self) -> FSDPOptimizerConfig:
        return FSDPOptimizerConfig(
            lr=self.config.learning_rate,
            lr_warmup_steps_ratio=self.config.warmup_steps_ratio,
            total_training_steps=self.total_training_steps,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            clip_grad=self.config.max_grad_norm,
            min_lr_ratio=self.config.min_lr_ratio,
            lr_scheduler_type="cosine",
        )

    def _build_worker(self, model_path: str, *, trainable: bool) -> TrainingWorker:
        checkpoint_config = CheckpointConfig(
            save_contents=["model", "optimizer", "extra"],
            load_contents=["model", "optimizer", "extra"],
        )
        worker_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self._build_model_config(model_path, trainable=trainable),
            engine_config=self._build_engine_config(forward_only=not trainable),
            optimizer_config=self._build_optimizer_config(),
            checkpoint_config=checkpoint_config,
        )
        worker = TrainingWorker(config=worker_config)
        worker.reset()
        return worker

    def _build_student_worker(self) -> TrainingWorker:
        return self._build_worker(self.config.student_model_path, trainable=True)

    def _build_teacher_worker(self) -> TrainingWorker:
        teacher_model_path = self.config.teacher_model_path or self.config.student_model_path
        return self._build_worker(teacher_model_path, trainable=False)

    def _common_meta(self, *, return_logits: bool) -> dict[str, Any]:
        return {
            "pad_mode": DatasetPadMode.NO_PADDING,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_remove_padding": self.config.use_remove_padding,
            "use_dynamic_bsz": True,
            "max_token_len_per_gpu": self.config.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": None,
            "calculate_entropy": False,
            "return_logits": return_logits,
        }

    def _make_teacher_batch(self, batch: dict[str, torch.Tensor]) -> TensorDict:
        batch_size = batch["teacher_input_ids"].size(0)
        return tu.get_tensordict(
            tensor_dict={
                "input_ids": batch["teacher_input_ids"],
                "position_ids": batch["teacher_position_ids"],
                "loss_mask": batch["teacher_loss_mask"],
                "temperature": torch.full((batch_size,), self.config.temperature, dtype=torch.float32),
            },
            non_tensor_dict=self._common_meta(return_logits=self.config.kl_method == "full_vocab"),
        )

    def _make_student_batch(self, batch: dict[str, torch.Tensor], teacher_model_output: dict[str, torch.Tensor]) -> TensorDict:
        batch_size = batch["student_input_ids"].size(0)
        tensor_dict = {
            "input_ids": batch["student_input_ids"],
            "position_ids": batch["student_position_ids"],
            "loss_mask": batch["student_loss_mask"],
            "temperature": torch.full((batch_size,), self.config.temperature, dtype=torch.float32),
            "teacher_log_probs": teacher_model_output["log_probs"],
            "teacher_loss_mask": batch["teacher_loss_mask"],
        }
        if self.config.kl_method == "full_vocab":
            tensor_dict["teacher_logits"] = teacher_model_output["logits"]

        meta = self._common_meta(return_logits=self.config.kl_method == "full_vocab")
        meta["grad_accum_steps"] = self.config.gradient_accumulation_steps
        return tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=meta)

    @staticmethod
    def _masked_nested_to_padded(tensor: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensor_rows = list(tensor.unbind())
        mask_rows = list(mask.unbind())
        trailing_shape = tensor_rows[0].shape[1:]

        selected_rows = []
        max_len = 0
        for tensor_row, mask_row in zip(tensor_rows, mask_rows, strict=True):
            selected = tensor_row[mask_row.to(torch.bool)]
            selected_rows.append(selected)
            max_len = max(max_len, selected.shape[0])

        if max_len == 0:
            raise ValueError("No response tokens remain in the current micro-batch.")

        padded = tensor_rows[0].new_zeros((len(selected_rows), max_len, *trailing_shape))
        padded_mask = torch.zeros((len(selected_rows), max_len), dtype=torch.float32, device=padded.device)

        for idx, selected in enumerate(selected_rows):
            if selected.shape[0] == 0:
                continue
            padded[idx, : selected.shape[0]] = selected
            padded_mask[idx, : selected.shape[0]] = 1.0

        if padded.dim() == 3 and not trailing_shape:
            padded = padded.squeeze(-1)

        return padded, padded_mask

    @staticmethod
    def _all_reduce_in_place(tensors: list[torch.Tensor], dp_group):
        if dp_group is None or dist.get_world_size(group=dp_group) == 1:
            return
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_group)

    def _compute_kl_loss(self, model_output: dict[str, torch.Tensor], data: TensorDict, dp_group=None):
        student_log_probs, mask = self._masked_nested_to_padded(model_output["log_probs"], data["loss_mask"])
        teacher_log_probs, teacher_mask = self._masked_nested_to_padded(data["teacher_log_probs"], data["teacher_loss_mask"])

        student_counts = mask.sum(dim=1)
        teacher_counts = teacher_mask.sum(dim=1)
        if not torch.equal(student_counts, teacher_counts):
            raise ValueError("Student and teacher response lengths diverged inside the KL loss.")

        if self.config.kl_method == "full_vocab":
            student_logits, _ = self._masked_nested_to_padded(model_output["logits"], data["loss_mask"])
            teacher_logits, _ = self._masked_nested_to_padded(data["teacher_logits"], data["teacher_loss_mask"])
            kl_per_position = compute_kl_divergence(
                student_logits,
                teacher_logits,
                mask,
                kl_type=self.config.kl_type,
                kl_method="full_vocab",
                is_logprobs=False,
                reduction="none",
                temperature=self.config.temperature,
            )
        else:
            kl_per_position = compute_kl_divergence(
                student_log_probs,
                teacher_log_probs,
                mask,
                kl_type=self.config.kl_type,
                kl_method="monte_carlo",
                is_logprobs=True,
                reduction="none",
            )

        kl_num = (kl_per_position * mask).sum()
        student_nll_num = (-student_log_probs * mask).sum()
        teacher_nll_num = (-teacher_log_probs * mask).sum()
        response_tokens = mask.sum()

        batch_num_tokens = float(data["batch_num_tokens"])
        dp_size = float(data["dp_size"])
        kl_loss = kl_num / max(batch_num_tokens, 1.0) * dp_size
        loss = kl_loss / float(tu.get_non_tensor_data(data, "grad_accum_steps", 1))

        kl_num_metric = kl_num.detach().clone()
        student_nll_num_metric = student_nll_num.detach().clone()
        teacher_nll_num_metric = teacher_nll_num.detach().clone()
        response_tokens_metric = response_tokens.detach().clone()
        self._all_reduce_in_place(
            [kl_num_metric, student_nll_num_metric, teacher_nll_num_metric, response_tokens_metric],
            dp_group,
        )

        metrics = {
            "kl_num": kl_num_metric.float().item(),
            "student_nll_num": student_nll_num_metric.float().item(),
            "teacher_nll_num": teacher_nll_num_metric.float().item(),
            "response_tokens": response_tokens_metric.float().item(),
        }
        return loss, metrics

    @staticmethod
    def _summarize_step_output(output: dict[str, Any]) -> dict[str, float]:
        metrics = output.get("metrics", {})
        kl_num = float(sum(metrics.get("kl_num", [])))
        student_nll_num = float(sum(metrics.get("student_nll_num", [])))
        teacher_nll_num = float(sum(metrics.get("teacher_nll_num", [])))
        response_tokens = float(sum(metrics.get("response_tokens", [])))
        response_tokens = max(response_tokens, 1.0)

        return {
            "kl_loss": kl_num / response_tokens,
            "student_perplexity": math.exp(student_nll_num / response_tokens),
            "teacher_perplexity": math.exp(teacher_nll_num / response_tokens),
            "response_tokens": response_tokens,
        }

    def _save_checkpoint(self):
        ckpt_dir = os.path.join(self.config.model_save_dir, f"global_step_{self.global_step}")
        self.student_engine.save_checkpoint(
            local_path=ckpt_dir,
            global_step=self.global_step,
            max_ckpt_to_keep=None,
        )
        logger.info("Checkpoint saved to %s", ckpt_dir)

    def _find_latest_checkpoint(self) -> str:
        ckpt_dirs = glob.glob(os.path.join(self.config.model_save_dir, "global_step_*"))
        if not ckpt_dirs:
            raise FileNotFoundError(f"No FSDP checkpoints found under {self.config.model_save_dir}")
        return max(ckpt_dirs, key=lambda path: int(path.rsplit("_", 1)[-1]))

    def _broadcast_rank0_error(self, error_message: str | None) -> str | None:
        error_holder = [error_message]
        dist.broadcast_object_list(error_holder, src=0)
        return error_holder[0]

    def _export_hf_model(self):
        latest_ckpt = self._find_latest_checkpoint()
        merged_dir = os.path.join(self.config.model_save_dir, "hf_merged")
        hf_assets_source = self.config.student_model_path
        rank0_exception = None
        rank0_error = None

        if self.rank == 0:
            try:
                os.makedirs(merged_dir, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-m",
                    "verl.model_merger",
                    "merge",
                    "--backend",
                    "fsdp",
                    "--local_dir",
                    latest_ckpt,
                    "--hf_model_config_path",
                    hf_assets_source,
                    "--target_dir",
                    merged_dir,
                ]
                if hf_assets_source:
                    logger.info("Using HF assets from %s for merged export", hf_assets_source)
                cmd.append("--trust-remote-code")
                logger.info("Merging FSDP checkpoint %s -> %s", latest_ckpt, merged_dir)
                subprocess.run(cmd, check=True)
            except Exception as e:
                rank0_exception = e
                rank0_error = f"{type(e).__name__}: {e}"
                logger.exception("Failed to export checkpoint %s to %s", latest_ckpt, merged_dir)

        exported_error = self._broadcast_rank0_error(rank0_error)
        if exported_error is not None:
            raise RuntimeError(f"FSDP checkpoint export failed on rank 0: {exported_error}") from rank0_exception

    def train_epoch(self) -> dict[str, float]:
        sampler = getattr(self.train_dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.epoch)

        running_kl = 0.0
        running_student_ppl = 0.0
        running_teacher_ppl = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.config.total_epochs}",
            disable=not self.is_logging,
        )

        self.student_engine.optimizer_zero_grad()
        grad_norm = 0.0
        lr = self.config.learning_rate

        with self.student_engine.train_mode():
            for batch_idx, batch in enumerate(progress_bar):
                teacher_batch = self._make_teacher_batch(batch)
                with self.teacher_engine.eval_mode():
                    teacher_output = self.teacher_engine.forward_backward_batch(
                        teacher_batch,
                        loss_function=None,
                        forward_only=True,
                    )

                student_batch = self._make_student_batch(batch, teacher_output["model_output"])

                start_time = time.time()
                train_output = self.student_engine.forward_backward_batch(
                    student_batch,
                    loss_function=self._compute_kl_loss,
                    forward_only=False,
                )
                step_time = time.time() - start_time

                step_metrics = self._summarize_step_output(train_output)
                running_kl += step_metrics["kl_loss"]
                running_student_ppl += step_metrics["student_perplexity"]
                running_teacher_ppl += step_metrics["teacher_perplexity"]
                num_batches += 1

                should_step = (
                    (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                    or batch_idx + 1 == len(self.train_dataloader)
                )

                if should_step:
                    grad_norm = self.student_engine.optimizer_step()
                    self.student_engine.optimizer_zero_grad()
                    lr = self.student_engine.lr_scheduler_step()
                    self.global_step += 1

                    if self.is_logging and HAS_WANDB:
                        wandb.log(
                            {
                                "train/kl_loss": step_metrics["kl_loss"],
                                "train/student_perplexity": step_metrics["student_perplexity"],
                                "train/teacher_perplexity": step_metrics["teacher_perplexity"],
                                "train/response_tokens": step_metrics["response_tokens"],
                                "train/grad_norm": grad_norm,
                                "train/learning_rate": lr,
                                "train/step_time_sec": step_time,
                                "train/epoch": self.epoch,
                                "train/global_step": self.global_step,
                            }
                        )

                    if self.global_step % self.config.logging_steps == 0 and self.is_logging:
                        logger.info(
                            "step=%s kl=%.4f student_ppl=%.2f teacher_ppl=%.2f grad_norm=%.4f lr=%.2e",
                            self.global_step,
                            step_metrics["kl_loss"],
                            step_metrics["student_perplexity"],
                            step_metrics["teacher_perplexity"],
                            grad_norm,
                            lr,
                        )

                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

                progress_bar.set_postfix(
                    {
                        "kl": f"{running_kl / num_batches:.4f}",
                        "student_ppl": f"{running_student_ppl / num_batches:.2f}",
                        "teacher_ppl": f"{running_teacher_ppl / num_batches:.2f}",
                    }
                )

                del teacher_output
                del train_output

        return {
            "kl_loss": running_kl / max(num_batches, 1),
            "student_perplexity": running_student_ppl / max(num_batches, 1),
            "teacher_perplexity": running_teacher_ppl / max(num_batches, 1),
        }

    def train(self):
        logger.info("Starting KL training")
        logger.info("Total epochs: %s", self.config.total_epochs)
        logger.info("Total optimizer steps: %s", self.total_training_steps)

        for epoch in range(self.config.total_epochs):
            self.epoch = epoch
            epoch_metrics = self.train_epoch()
            if self.is_logging:
                logger.info("Epoch %s metrics: %s", epoch + 1, epoch_metrics)
                if HAS_WANDB:
                    wandb.log(
                        {
                            "epoch/kl_loss": epoch_metrics["kl_loss"],
                            "epoch/student_perplexity": epoch_metrics["student_perplexity"],
                            "epoch/teacher_perplexity": epoch_metrics["teacher_perplexity"],
                            "epoch/index": epoch + 1,
                        }
                    )

        if self.global_step == 0:
            raise RuntimeError("Training finished without any optimizer step.")

        self._save_checkpoint()

        if self.config.save_merged_model or self.config.run_eval_after_training:
            self._export_hf_model()

        if self.config.run_eval_after_training:
            self.run_evaluation()

    def run_evaluation(self):
        if not self.is_logging:
            return

        eval_model_path = os.path.join(self.config.model_save_dir, "hf_merged")
        if not os.path.isdir(eval_model_path):
            raise FileNotFoundError(
                f"Evaluation requires a merged HuggingFace checkpoint at {eval_model_path}."
            )

        eval_datasets = self.config.eval_datasets or ["aime24", "aime25", "math500"]
        dataset_paths = self.config.eval_dataset_paths

        base_model_name = self.config.base_model_name or self.config.student_model_path.split("/")[-1]
        eval_tag = f"kl_{self.config.kl_type}_{self.config.kl_method}"
        model_name = f"{base_model_name}_{eval_tag}_epoch{self.config.epoch_index}"

        gen_results_dir = self.config.gen_results_dir or self.config.output_dir
        eval_output_dir = os.path.join(gen_results_dir, f"evaluate_{eval_tag}")
        os.makedirs(eval_output_dir, exist_ok=True)

        verl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        results_base_dir = os.path.join(verl_root, "results", base_model_name)
        results_file = os.path.join(results_base_dir, "results.json")

        selected_dataset_paths = {dataset: dataset_paths.get(dataset) for dataset in eval_datasets}
        eval_results = run_evaluation_suite(
            eval_model_path,
            selected_dataset_paths,
            eval_output_dir,
            pass_k=1,
            temperature=0.6,
            top_p=0.95,
            nnodes=self.config.nnodes,
            n_gpus_per_node=self.config.n_gpus_per_node,
            tensor_model_parallel_size=1,
            output_json_path=results_file,
            model_name=model_name,
        )

        for dataset, accuracy in eval_results.items():
            logger.info("Evaluation %s: %.2f%%", dataset, accuracy * 100)

        if not eval_results:
            raise RuntimeError(
                "Automatic evaluation was requested, but no valid evaluation datasets were found. "
                f"Checked datasets={eval_datasets} under {self.config.eval_datasets_dir}."
            )

        if HAS_WANDB and eval_results:
            wandb.log({f"eval/{dataset}": acc for dataset, acc in eval_results.items()})
