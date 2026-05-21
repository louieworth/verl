# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
# Copyright 2025 Individual Contributor: Brilliant Hanabi, funrunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import faulthandler
import logging
import os
import signal
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

try:
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
except Exception:
    pass
from codetiming import Timer
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import DictConfig, OmegaConf
from torch import nn

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.flops_counter import FlopsCounter
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron_utils import get_model_config
from verl.utils.profiler import (
    DistProfiler,
    GPUMemoryLogger,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import gather_timing
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.torch_functional import use_original_torch_compile
from verl.workers.megatron_workers import ActorRolloutRefWorker

try:
    from .megatron_distill_losses import build_vocab_parallel_distill_loss
except ImportError:
    from megatron_distill_losses import build_vocab_parallel_distill_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _patch_transformers_tokenizer_compat():
    """Provide tokenizer property expected by vLLM 0.11 on transformers 5."""
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)


class TensorBuffer:
    def __init__(self, memory_alloc, dtype):
        self.device = get_device_id()
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        self.capacity = memory_alloc // dtype_size
        self.dtype = dtype
        self.tensor = torch.empty(self.capacity, dtype=self.dtype, device=self.device)
        self.keys = []
        self.shapes = []

    @property
    def size(self):
        return sum(shape.numel() for shape in self.shapes)

    def clear(self):
        self.keys.clear()
        self.shapes.clear()
        self.tensor = torch.empty(self.capacity, dtype=self.dtype, device=self.device)

    def append(self, key, shape, weight=None):
        if weight is not None:
            self.tensor[self.size : self.size + shape.numel()] = weight.view(-1)
        self.keys.append(key)
        self.shapes.append(shape)

    def to_tensors(self):
        tensors = []
        start = 0
        for key_, shape_ in zip(self.keys, self.shapes, strict=False):
            tensors.append((key_, self.tensor[start : start + shape_.numel()].view(shape_)))
            start += shape_.numel()
        return tensors


def record_time(func):
    def wrapper(*args, **kwargs):
        tik = time.time()
        func(*args, **kwargs)
        tok = time.time()
        return tok - tik

    return wrapper


class OnPolicyDistillActor:
    """
    Responsible purely for the training step (forward-backward + optimizer).
    """

    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
                pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
                more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation.
                Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
                It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        self.config = config
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.prof = None
        # Cross-call gradient accumulation counter. See update_policy().
        self.gradient_accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
        self._grad_accum_iter = 0
        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )

        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

        # Build distill loss operator (selectable by config)
        loss_cfg = self.config.get("distill_loss", None)
        self.distill_loss_op = build_vocab_parallel_distill_loss(loss_cfg).cuda()
        self.distill_loss_name = str(loss_cfg.get("name", "kl")).lower() if loss_cfg is not None else "kl"
        self.distill_loss_beta = float(loss_cfg.get("beta", 0.5)) if loss_cfg is not None else 0.5
        self.distill_loss_temperature = float(loss_cfg.get("temperature", 1.0)) if loss_cfg is not None else 1.0
        self.kl_token_clip = float(loss_cfg.get("kl_token_clip", 0.0)) if loss_cfg is not None else 0.0
        self.distill_top_k = int(loss_cfg.get("top_k", 0)) if loss_cfg is not None else 0
        self.local_teacher_model_path = str(loss_cfg.get("local_teacher_model_path", "") or "") if loss_cfg else ""
        self.local_teacher_chunk_size = int(loss_cfg.get("local_teacher_chunk_size", 128)) if loss_cfg else 128
        self.local_teacher_prefill_chunk_size = (
            int(loss_cfg.get("local_teacher_prefill_chunk_size", 512)) if loss_cfg else 512
        )
        self.local_teacher_attn_implementation = (
            str(loss_cfg.get("local_teacher_attn_implementation", "flash_attention_2")) if loss_cfg else "flash_attention_2"
        )
        self.use_local_teacher = bool(self.local_teacher_model_path)
        setattr(config, "verl_skip_float16_output_conversion", self.use_local_teacher)
        self.local_teacher_model = None
        if self.use_local_teacher and mpu.is_pipeline_last_stage():
            from transformers import AutoModelForCausalLM

            device = torch.device("cuda", get_device_id())
            self.local_teacher_model = AutoModelForCausalLM.from_pretrained(
                self.local_teacher_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=self.local_teacher_attn_implementation,
            ).eval().to(device)
            self.local_teacher_model.requires_grad_(False)
            if hasattr(self.local_teacher_model, "config"):
                self.local_teacher_model.config.use_cache = False

    def _advance_local_teacher_cache(
        self,
        row_input_ids: torch.Tensor,
        row_attention_mask: torch.Tensor,
        processed: int,
        target: int,
        past_key_values,
    ):
        prefill_chunk_size = max(1, int(self.local_teacher_prefill_chunk_size))
        while processed < target:
            segment_end = min(processed + prefill_chunk_size, target)
            segment_input_ids = row_input_ids[:, processed:segment_end]
            segment_attention_mask = row_attention_mask[:, :segment_end]
            with torch.inference_mode():
                teacher_output = self.local_teacher_model(
                    input_ids=segment_input_ids,
                    attention_mask=segment_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            past_key_values = teacher_output.past_key_values
            processed = segment_end
            del teacher_output, segment_input_ids, segment_attention_mask
            torch.cuda.empty_cache()
        return processed, past_key_values

    def _dense_full_vocab_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        losses = []
        chunk_size = max(1, int(self.local_teacher_chunk_size))
        beta = float(self.distill_loss_beta)
        for start in range(0, student_logits.size(0), chunk_size):
            end = min(start + chunk_size, student_logits.size(0))
            s_slice = student_logits[start:end]
            t_slice = teacher_logits[start:end]
            if t_slice.device != s_slice.device:
                t_slice = t_slice.to(s_slice.device, non_blocking=True)
            if self.distill_loss_temperature != 1.0:
                s_slice = s_slice / self.distill_loss_temperature
                t_slice = t_slice / self.distill_loss_temperature
            s_logp = F.log_softmax(s_slice.float(), dim=-1)
            with torch.no_grad():
                if self.distill_loss_temperature == 1.0:
                    t_logp = t_slice.float()
                else:
                    t_logp = F.log_softmax(t_slice.float(), dim=-1)

            if self.distill_loss_name == "rkl":
                s_prob = s_logp.exp()
                loss = (s_prob * (s_logp - t_logp)).sum(dim=-1)
            elif self.distill_loss_name == "jsd":
                if beta == 0:
                    t_prob = t_logp.exp()
                    loss = (t_prob * (t_logp - s_logp)).sum(dim=-1)
                elif beta == 1:
                    s_prob = s_logp.exp()
                    loss = (s_prob * (s_logp - t_logp)).sum(dim=-1)
                else:
                    beta_t = torch.tensor(beta, dtype=s_logp.dtype, device=s_logp.device)
                    mixture_logp = torch.logsumexp(
                        torch.stack([s_logp + torch.log1p(-beta_t), t_logp + torch.log(beta_t)]),
                        dim=0,
                    )
                    t_prob = t_logp.exp()
                    s_prob = s_logp.exp()
                    loss = beta * (t_prob * (t_logp - mixture_logp)).sum(dim=-1) + (1 - beta) * (
                        s_prob * (s_logp - mixture_logp)
                    ).sum(dim=-1)
            else:
                t_prob = t_logp.exp()
                loss = (t_prob * (t_logp - s_logp)).sum(dim=-1)
            losses.append(loss)
        return torch.cat(losses, dim=0) if losses else student_logits.new_zeros(0, dtype=torch.float32)

    @staticmethod
    def _align_dense_teacher_vocab(student_logits: torch.Tensor, teacher_logps: torch.Tensor) -> torch.Tensor:
        student_vocab = student_logits.size(-1)
        teacher_vocab = teacher_logps.size(-1)
        if teacher_vocab == student_vocab:
            return teacher_logps
        if teacher_vocab > student_vocab:
            aligned = teacher_logps[..., :student_vocab].float()
            return aligned - torch.logsumexp(aligned, dim=-1, keepdim=True)
        raise RuntimeError(
            "Dense full-vocab remote teacher loss requires actor TP=1 or matching vocab partition: "
            f"student logits last dim={student_vocab}, teacher logps last dim={teacher_vocab}"
        )

    @staticmethod
    def _is_compact_teacher_payload(payload) -> bool:
        return isinstance(payload, np.ndarray) and payload.dtype == object

    @staticmethod
    def _teacher_payload_row_to_tensor(row, *, is_logps: bool) -> torch.Tensor:
        if isinstance(row, torch.Tensor):
            tensor = row
        else:
            tensor = torch.from_numpy(np.asarray(row))
        if is_logps and tensor.dtype == torch.int16:
            tensor = tensor.view(torch.bfloat16)
        return tensor

    @classmethod
    def _compact_teacher_payload_for_rows(
        cls,
        payload,
        row_indices,
        expected_counts,
        *,
        is_logps: bool,
        name: str,
    ) -> torch.Tensor:
        rows = []
        empty_template = None
        for row_idx, expected in zip(row_indices, expected_counts, strict=True):
            row_tensor = cls._teacher_payload_row_to_tensor(payload[int(row_idx)], is_logps=is_logps)
            if empty_template is None and row_tensor.ndim >= 2:
                empty_template = row_tensor
            expected = int(expected)
            if expected <= 0:
                continue
            if row_tensor.size(0) < expected:
                raise RuntimeError(
                    f"{name} compact payload shorter than actor loss mask for row {int(row_idx)}: "
                    f"payload_rows={row_tensor.size(0)}, expected={expected}"
                )
            rows.append(row_tensor[:expected])

        if rows:
            return torch.cat(rows, dim=0)
        if empty_template is not None and empty_template.ndim >= 2:
            return empty_template.new_empty((0, *empty_template.shape[1:]))
        dtype = torch.bfloat16 if is_logps else torch.int32
        return torch.empty((0, 0), dtype=dtype)

    def _local_teacher_loss(
        self,
        student_logits: torch.Tensor,
        student_calc_kl_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        calc_kl_mask_full: torch.Tensor,
    ) -> torch.Tensor:
        n_loss_tokens = int(calc_kl_mask_full.sum().item())
        if self.local_teacher_model is None:
            return student_logits.new_zeros(n_loss_tokens, dtype=torch.float32)

        device = student_logits.device
        flat_student_logits = student_logits.reshape(-1, student_logits.size(-1))
        flat_student_mask = student_calc_kl_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if flat_student_mask.numel() != flat_student_logits.size(0):
            raise RuntimeError(
                f"Local teacher KL packed mask/logits shape mismatch: mask has {flat_student_mask.numel()} "
                f"positions, logits have {flat_student_logits.size(0)} positions"
            )

        packed_loss_positions = flat_student_mask.nonzero(as_tuple=False).flatten()
        if packed_loss_positions.numel() != n_loss_tokens:
            raise RuntimeError(
                f"Local teacher KL packed/full mask mismatch: packed has {packed_loss_positions.numel()} "
                f"loss tokens, full mask has {n_loss_tokens}"
            )

        chunk_size = max(1, int(self.local_teacher_chunk_size))
        losses = []
        packed_cursor = 0
        torch.cuda.empty_cache()
        from transformers import DynamicCache

        for row in range(input_ids.size(0)):
            row_mask = calc_kl_mask_full[row].to(device=device, dtype=torch.bool)
            token_positions = row_mask.nonzero(as_tuple=False).flatten()
            if token_positions.numel() == 0:
                continue

            row_input_ids = input_ids[row : row + 1].to(device, non_blocking=True)
            row_attention_mask = attention_mask[row : row + 1].to(device, non_blocking=True)
            # Keep exact full-context teacher logits while moving past KV layers off GPU between decoder layers.
            teacher_cache = DynamicCache(
                config=self.local_teacher_model.config,
                offloading=True,
                offload_only_non_sliding=False,
            )
            processed = 0
            for start in range(0, token_positions.numel(), chunk_size):
                pos_chunk = token_positions[start : start + chunk_size]
                packed_pos_chunk = packed_loss_positions[packed_cursor : packed_cursor + pos_chunk.numel()]
                span_start = int(pos_chunk[0].item())
                span_end = int(pos_chunk[-1].item())

                processed, teacher_cache = self._advance_local_teacher_cache(
                    row_input_ids,
                    row_attention_mask,
                    processed,
                    span_start,
                    teacher_cache,
                )
                segment_end = span_end + 1
                segment_input_ids = row_input_ids[:, processed:segment_end]
                segment_attention_mask = row_attention_mask[:, :segment_end]
                keep_offsets = pos_chunk - processed
                with torch.inference_mode():
                    teacher_output = self.local_teacher_model(
                        input_ids=segment_input_ids,
                        attention_mask=segment_attention_mask,
                        past_key_values=teacher_cache,
                        use_cache=True,
                        logits_to_keep=segment_input_ids.size(1),
                    )
                    teacher_logits = teacher_output.logits[0, keep_offsets]
                teacher_cache = teacher_output.past_key_values
                processed = segment_end
                student_slice = flat_student_logits.index_select(0, packed_pos_chunk)
                losses.append(self._dense_full_vocab_loss(student_slice, teacher_logits))
                packed_cursor += pos_chunk.numel()
                del teacher_output, teacher_logits, segment_input_ids, segment_attention_mask, student_slice
                torch.cuda.empty_cache()
            del teacher_cache, row_input_ids, row_attention_mask
            torch.cuda.empty_cache()

        consumed_loss_tokens = sum(loss.numel() for loss in losses)
        if consumed_loss_tokens != n_loss_tokens or packed_cursor != packed_loss_positions.numel():
            raise RuntimeError(
                f"Local teacher KL alignment mismatch: produced {consumed_loss_tokens} losses, "
                f"consumed {packed_cursor} packed positions, expected {n_loss_tokens}"
            )
        return torch.cat(losses, dim=0) if losses else student_logits.new_zeros(0, dtype=torch.float32)

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    def forward_backward_batch(
        self,
        data: DataProto,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        # broadcast_dict_tensor(
        #     data.batch,
        #     src=mpu.get_pipeline_model_parallel_last_rank(),
        #     group=mpu.get_pipeline_model_parallel_group(),
        # )
        # split into micro-batches
        batch = data.batch.clone()
        batch["attention_mask"] = batch["attention_mask"].to(bool)

        def attach_distill_loss_fields(mb):
            responses = mb["responses"]
            response_length = responses.size(1)
            if "distill_loss_mask" in mb:
                calc_kl_mask = mb["distill_loss_mask"].to(torch.bool).clone()
            else:
                calc_kl_mask = mb["attention_mask"].clone()
                calc_kl_mask[:, : (-response_length - 1)] = False
            mb["calc_kl_mask"] = calc_kl_mask
            mb["kl_losses"] = torch.zeros_like(calc_kl_mask, dtype=torch.float32)
            return calc_kl_mask

        indices = None
        compact_teacher_payloads = []
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {len(micro_batches)} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            micro_batches = [mb.clone() for mb in micro_batches]
            # total_seqlen = max_token_len
            if mpu.is_pipeline_last_stage():
                for mb in micro_batches:
                    attach_distill_loss_fields(mb)
                if not self.use_local_teacher:
                    has_teacher_indices = "teacher_topk_indices" in data.non_tensor_batch
                    logps_payload = data.non_tensor_batch["teacher_topk_logps"]
                    compact_payload = self._is_compact_teacher_payload(logps_payload)
                    if compact_payload:
                        indices_payload = (
                            data.non_tensor_batch["teacher_topk_indices"] if has_teacher_indices else None
                        )
                        for i, mb in enumerate(micro_batches):
                            expected_counts = mb["calc_kl_mask"].sum(dim=1).detach().cpu().tolist()
                            payload = {
                                "counts": expected_counts,
                                "logps": self._compact_teacher_payload_for_rows(
                                    logps_payload,
                                    indices[i],
                                    expected_counts,
                                    is_logps=True,
                                    name="teacher_topk_logps",
                                ).pin_memory(),
                            }
                            if has_teacher_indices:
                                payload["indices"] = self._compact_teacher_payload_for_rows(
                                    indices_payload,
                                    indices[i],
                                    expected_counts,
                                    is_logps=False,
                                    name="teacher_topk_indices",
                                ).pin_memory()
                            compact_teacher_payloads.append(payload)
                    else:
                        # logps were packed as int16 (bf16 bits re-viewed) on the
                        # trainer side to bypass numpy's lack of bf16 support.
                        _logps_np = logps_payload
                        if _logps_np.dtype == np.int16:
                            teacher_topk_logps_tensor = torch.from_numpy(_logps_np).view(torch.bfloat16)
                        else:
                            teacher_topk_logps_tensor = torch.tensor(_logps_np)
                        teacher_topk_indices_tensor = (
                            torch.tensor(data.non_tensor_batch["teacher_topk_indices"]) if has_teacher_indices else None
                        )
                        for i, partition in enumerate(indices):
                            curr_logp_micro_batch, curr_idx_micro_batch = [], []
                            for idx in partition:
                                curr_logp_micro_batch.append(teacher_topk_logps_tensor[idx : idx + 1])
                                if has_teacher_indices:
                                    curr_idx_micro_batch.append(teacher_topk_indices_tensor[idx : idx + 1])
                            micro_batches[i]["teacher_topk_logps"] = torch.cat(curr_logp_micro_batch).pin_memory()
                            if has_teacher_indices:
                                micro_batches[i]["teacher_topk_indices"] = torch.cat(curr_idx_micro_batch).pin_memory()
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = [mb.clone() for mb in batch.split(micro_batch_size)]
            # seq_len = micro_batches[0]["input_ids"].shape[1]
            # total_seqlen = micro_batch_size * seq_len
            if mpu.is_pipeline_last_stage():
                for mb in micro_batches:
                    attach_distill_loss_fields(mb)
                if not self.use_local_teacher:
                    has_teacher_indices = "teacher_topk_indices" in data.non_tensor_batch
                    logps_payload = data.non_tensor_batch["teacher_topk_logps"]
                    compact_payload = self._is_compact_teacher_payload(logps_payload)
                    if compact_payload:
                        indices_payload = (
                            data.non_tensor_batch["teacher_topk_indices"] if has_teacher_indices else None
                        )
                        row_start = 0
                        for mb in micro_batches:
                            row_end = row_start + mb["input_ids"].size(0)
                            row_indices = list(range(row_start, row_end))
                            expected_counts = mb["calc_kl_mask"].sum(dim=1).detach().cpu().tolist()
                            payload = {
                                "counts": expected_counts,
                                "logps": self._compact_teacher_payload_for_rows(
                                    logps_payload,
                                    row_indices,
                                    expected_counts,
                                    is_logps=True,
                                    name="teacher_topk_logps",
                                ).pin_memory(),
                            }
                            if has_teacher_indices:
                                payload["indices"] = self._compact_teacher_payload_for_rows(
                                    indices_payload,
                                    row_indices,
                                    expected_counts,
                                    is_logps=False,
                                    name="teacher_topk_indices",
                                ).pin_memory()
                            compact_teacher_payloads.append(payload)
                            row_start = row_end
                    else:
                        _logps_np = logps_payload
                        if _logps_np.dtype == np.int16:
                            teacher_topk_logps_tensor = torch.from_numpy(_logps_np).view(torch.bfloat16)
                        else:
                            teacher_topk_logps_tensor = torch.tensor(_logps_np)
                        teacher_topk_logps = torch.tensor_split(teacher_topk_logps_tensor, len(micro_batches), dim=0)
                        if has_teacher_indices:
                            teacher_topk_indices_tensor = torch.tensor(data.non_tensor_batch["teacher_topk_indices"])
                            teacher_topk_indices = torch.tensor_split(
                                teacher_topk_indices_tensor, len(micro_batches), dim=0
                            )
                        for i, mb in enumerate(micro_batches):
                            mb["teacher_topk_logps"] = teacher_topk_logps[i].pin_memory()
                            if has_teacher_indices:
                                mb["teacher_topk_indices"] = teacher_topk_indices[i].pin_memory()

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            metrics = {}

            ret_entropy = None
            stats = {}
            kl_losses = output["kl_losses"]
            calc_kl_mask = output["calc_kl_mask"]
            # inf_cnt = masked_kl_lossed.isinf().sum().item()
            # nan_cnt = masked_kl_lossed.isnan().sum().item()
            # total_cnt = masked_kl_lossed.nelement()
            # print(f"rank: {rank}, kl_loss inf_cnt/nan_cnt/total_cnt: {inf_cnt} / {nan_cnt} /{total_cnt}")
            masked_kl_lossed = kl_losses[calc_kl_mask]
            if masked_kl_lossed.numel() == 0:
                mean_kl_loss = kl_losses.sum() * 0.0
            else:
                mean_kl_loss = masked_kl_lossed.mean()
            stats.update({"actor/kl_loss": mean_kl_loss.detach().item()})

            # Gradient accumulation: scale loss by 1/N so accumulated grads
            # over N update_policy calls match the magnitude of a single true
            # batch of N×TRAIN_BATCH_SIZE prompts (no need to adjust LR).
            # Matches recipe/opd/kl_trainer.py:608 semantics.
            accum_steps = getattr(self, "gradient_accumulation_steps", 1)
            if accum_steps > 1:
                mean_kl_loss = mean_kl_loss / accum_steps

            append_to_dict(metrics, stats)
            return mean_kl_loss, [metrics, ret_entropy]

        compact_payload_cursor = 0

        def forward_step(batch_iter, model):
            nonlocal compact_payload_cursor
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)

            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)
            compact_teacher_topk_logps = None
            compact_teacher_topk_indices = None
            compact_teacher_loss_counts = None
            if mpu.is_pipeline_last_stage() and not self.use_local_teacher and compact_teacher_payloads:
                if compact_payload_cursor >= len(compact_teacher_payloads):
                    raise RuntimeError(
                        "Ran out of compact teacher payloads for Megatron microbatches: "
                        f"cursor={compact_payload_cursor}, total={len(compact_teacher_payloads)}"
                    )
                compact_payload = compact_teacher_payloads[compact_payload_cursor]
                compact_payload_cursor += 1
                compact_teacher_topk_logps = compact_payload["logps"]
                compact_teacher_loss_counts = compact_payload["counts"]
                if "indices" in compact_payload:
                    device = get_device_id()
                    compact_teacher_topk_logps = compact_teacher_topk_logps.to(device, non_blocking=True)
                    compact_teacher_topk_indices = compact_payload["indices"].to(device, non_blocking=True)

            def logits_processor(
                logits,
                calc_kl_mask,
                kl_losses,
                teacher_topk_logps=None,
                teacher_topk_indices=None,
            ):
                assert logits.shape[:2] == calc_kl_mask.shape[:2]

                if self.use_local_teacher:
                    per_token_loss = self._local_teacher_loss(
                        logits,
                        calc_kl_mask,
                        input_ids,
                        attention_mask,
                        batch["calc_kl_mask"],
                    )
                    if self.kl_token_clip > 0:
                        per_token_loss = per_token_loss.clamp(max=self.kl_token_clip)
                    kl_losses[calc_kl_mask] = per_token_loss
                    return {"kl_losses": kl_losses, "calc_kl_mask": calc_kl_mask}

                if teacher_topk_logps is None and compact_teacher_topk_logps is not None:
                    teacher_topk_logps = compact_teacher_topk_logps
                    teacher_topk_indices = compact_teacher_topk_indices

                assert teacher_topk_logps is not None

                masked_logits = logits[calc_kl_mask]
                if masked_logits.size(0) == 0:
                    return {"kl_losses": kl_losses, "calc_kl_mask": calc_kl_mask}
                if teacher_topk_indices is None:
                    if teacher_topk_logps.dim() == 2:
                        masked_teacher_topk_logps = teacher_topk_logps
                    else:
                        assert logits.shape[:2] == teacher_topk_logps.shape[:2]
                        if teacher_topk_logps.device.type == "cpu" and calc_kl_mask.device.type != "cpu":
                            teacher_mask = calc_kl_mask.detach().to("cpu", non_blocking=True)
                        else:
                            teacher_mask = calc_kl_mask
                        masked_teacher_topk_logps = teacher_topk_logps[teacher_mask]
                    if masked_teacher_topk_logps.size(0) != masked_logits.size(0):
                        raise RuntimeError(
                            "Dense full-vocab remote teacher payload does not match actor loss mask: "
                            f"teacher_tokens={masked_teacher_topk_logps.size(0)}, "
                            f"actor_tokens={masked_logits.size(0)}"
                        )
                    masked_teacher_topk_logps = self._align_dense_teacher_vocab(masked_logits, masked_teacher_topk_logps)
                    if teacher_topk_logps.dim() == 2 and compact_teacher_loss_counts is not None:
                        row_losses = []
                        offset = 0
                        for count in compact_teacher_loss_counts:
                            count = int(count)
                            if count > 0:
                                end = offset + count
                                row_losses.append(
                                    self._dense_full_vocab_loss(
                                        masked_logits[offset:end],
                                        masked_teacher_topk_logps[offset:end],
                                    )
                                )
                                offset = end
                        if offset != masked_logits.size(0):
                            raise RuntimeError(
                                "Compact full-vocab teacher row counts do not sum to actor loss tokens: "
                                f"row_count_sum={offset}, actor_tokens={masked_logits.size(0)}"
                            )
                        per_token_loss = (
                            torch.cat(row_losses, dim=0)
                            if row_losses
                            else masked_logits.new_zeros(0, dtype=torch.float32)
                        )
                    else:
                        per_token_loss = self._dense_full_vocab_loss(masked_logits, masked_teacher_topk_logps)
                    if self.kl_token_clip > 0:
                        per_token_loss = per_token_loss.clamp(max=self.kl_token_clip)
                    kl_losses[calc_kl_mask] = per_token_loss
                    return {"kl_losses": kl_losses, "calc_kl_mask": calc_kl_mask}

                if self.distill_loss_temperature != 1.0:
                    masked_logits = masked_logits / self.distill_loss_temperature
                if teacher_topk_logps.dim() == 2:
                    masked_teacher_topk_logps = teacher_topk_logps
                    masked_teacher_topk_indices = teacher_topk_indices
                else:
                    assert logits.shape[:2] == teacher_topk_logps.shape[:2]
                    assert logits.shape[:2] == teacher_topk_indices.shape[:2]
                    masked_teacher_topk_logps = teacher_topk_logps[calc_kl_mask]
                    masked_teacher_topk_indices = teacher_topk_indices[calc_kl_mask]
                if masked_teacher_topk_logps.size(0) != masked_logits.size(0):
                    raise RuntimeError(
                        "Teacher top-k payload does not match actor loss mask: "
                        f"teacher_tokens={masked_teacher_topk_logps.size(0)}, actor_tokens={masked_logits.size(0)}"
                    )
                if self.distill_top_k > 0:
                    masked_teacher_topk_logps = masked_teacher_topk_logps[..., : self.distill_top_k]
                    masked_teacher_topk_indices = masked_teacher_topk_indices[..., : self.distill_top_k]

                if teacher_topk_logps.dim() == 2 and compact_teacher_loss_counts is not None:
                    row_losses = []
                    offset = 0
                    for count in compact_teacher_loss_counts:
                        count = int(count)
                        if count > 0:
                            end = offset + count
                            row_losses.append(
                                self.distill_loss_op(
                                    masked_logits[offset:end],
                                    masked_teacher_topk_logps[offset:end],
                                    masked_teacher_topk_indices[offset:end],
                                )
                            )
                            offset = end
                    if offset != masked_logits.size(0):
                        raise RuntimeError(
                            "Compact top-k teacher row counts do not sum to actor loss tokens: "
                            f"row_count_sum={offset}, actor_tokens={masked_logits.size(0)}"
                        )
                    per_token_loss = (
                        torch.cat(row_losses, dim=0)
                        if row_losses
                        else masked_logits.new_zeros(0, dtype=torch.float32)
                    )
                else:
                    per_token_loss = self.distill_loss_op(
                        masked_logits, masked_teacher_topk_logps, masked_teacher_topk_indices
                    )
                if self.kl_token_clip > 0:
                    per_token_loss = per_token_loss.clamp(max=self.kl_token_clip)
                kl_losses[calc_kl_mask] = per_token_loss
                return {"kl_losses": kl_losses, "calc_kl_mask": calc_kl_mask}

            if mpu.is_pipeline_last_stage():
                logits_processor_args = {
                    "calc_kl_mask": batch["calc_kl_mask"],
                    "kl_losses": batch["kl_losses"],
                }
                if not self.use_local_teacher:
                    device = get_device_id()
                    if compact_teacher_topk_logps is not None:
                        pass
                    elif "teacher_topk_indices" in batch:
                        logits_processor_args["teacher_topk_logps"] = batch["teacher_topk_logps"].to(
                            device, non_blocking=True
                        )
                        logits_processor_args["teacher_topk_indices"] = batch["teacher_topk_indices"].to(
                            device, non_blocking=True
                        )
                    else:
                        logits_processor_args["teacher_topk_logps"] = batch["teacher_topk_logps"]
            else:
                logits_processor_args = None

            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
            )

            return output, loss_func

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=-1,  # no use when variable_seq_lengths was set
            micro_batch_size=-1,  # no use when variable_seq_lengths was set
            forward_only=False,
        )
        if mpu.is_pipeline_last_stage() and compact_teacher_payloads and compact_payload_cursor != len(compact_teacher_payloads):
            raise RuntimeError(
                "Not all compact teacher payloads were consumed by Megatron microbatches: "
                f"consumed={compact_payload_cursor}, total={len(compact_teacher_payloads)}"
            )

        # loss_reduces contains the stats returned from loss_func

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, data: DataProto) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        # self.prof.start()
        data.to(get_device_id())

        # Cross-call gradient accumulation:
        # - zero_grad only on FIRST iter of each accumulation group
        # - optimizer.step only on LAST iter of each accumulation group
        # - intervening iters just accumulate gradients in .grad buffers
        accum_steps = self.gradient_accumulation_steps
        is_first_accum = (self._grad_accum_iter == 0)
        is_last_accum = (self._grad_accum_iter == accum_steps - 1)

        if is_first_accum:
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

        micro_batch_size = self.config.micro_batch_size
        max_token_len = None
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.max_token_len * self.config.megatron.context_parallel_size

        metric_micro_batch = self.forward_backward_batch(
            data,
            use_dynamic_bsz=self.config.use_dynamic_bsz,
            micro_batch_size=micro_batch_size,
            max_token_len=max_token_len,
        )

        metric_micro_batch = metric_micro_batch["output"]
        for metric in metric_micro_batch:
            # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
            append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

        if is_last_accum:
            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data_metric = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data_metric)
            if not update_successful:
                raise NotImplementedError
        else:
            # Still accumulating — don't step optimizer, don't update LR.
            append_to_dict(metrics, {"actor/grad_norm": 0.0})

        # Track accumulation progress so update_actor can gate lr_scheduler.step.
        metrics["actor/grad_accum_iter"] = self._grad_accum_iter
        metrics["actor/optimizer_stepped"] = 1 if is_last_accum else 0
        self._grad_accum_iter = (self._grad_accum_iter + 1) % accum_steps
        # self.prof.step()
        # add empty cache after each compute
        # self.prof.stop_and_save()
        # self.prof.stop_trace()
        get_torch_device().empty_cache()
        return metrics


class MegatronOnPolicyDistillActorWorker(ActorRolloutRefWorker):
    """
    Megatron actor/rollout worker for OPD.

    The actor side owns the trainable Megatron model and optimizer. The rollout
    side follows the current official async vLLM server path and receives
    weights through ``update_weights`` before real-time generation.
    """

    def __init__(self, config: DictConfig, role: str):
        is_struct = OmegaConf.is_struct(config) or False
        OmegaConf.set_struct(config, False)
        if "router_replay" not in config.actor or not hasattr(config.actor.router_replay, "mode"):
            config.actor.router_replay = {
                "_target_": "verl.workers.config.RouterReplayConfig",
                "mode": "disabled",
                "record_file": None,
                "replay_file": None,
            }
        OmegaConf.set_struct(config, is_struct)

        super().__init__(config, role)
        assert self._is_actor, "OPD worker must include the actor role."

    def _get_actor_params_generator(self):
        assert self._is_actor
        if self.bridge is not None:
            generator = self.bridge.export_weights(self.actor.actor_module)
        else:
            try:
                from .megatron_utils import per_tensor_generator
            except ImportError:
                from megatron_utils import per_tensor_generator

            from verl.models.mcore import get_mcore_weight_converter

            layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            generator = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                weight_converter,
                self.tf_config,
                layer_name_mapping,
            )
        return generator

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(
            self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True
        )
        override_ddp_config = OmegaConf.to_container(
            self.config.actor.megatron.get("override_ddp_config", OmegaConf.create()), resolve=True
        )

        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        optim_config = self.config.actor.optim
        (
            self.actor_module,
            self.actor_optimizer,
            self.actor_optimizer_scheduler,
            self.actor_model_config,
            self.actor_optim_config,
        ) = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )

        self.actor = OnPolicyDistillActor(
            config=self.config.actor,
            model_config=self.actor_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )
        log_gpu_memory_usage("After OnPolicyDistillActor init", logger=logger)

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None
        if not self.config.actor.megatron.get("use_mbridge", False):
            from verl.models.mcore import get_mcore_weight_converter

            self.weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)

        if self._is_rollout:
            if self.config.rollout.name == "vllm":
                _patch_transformers_tokenizer_compat()
            with use_original_torch_compile():
                self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            log_gpu_memory_usage("After async rollout adapter init", logger=logger)

        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.actor.checkpoint,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            role="actor",
            model=self.actor_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.actor_optimizer,
            optimizer_scheduler=self.actor_optimizer_scheduler,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
        )
        get_torch_device().empty_cache()
        log_gpu_memory_usage("Actor/rollout init_model finished", logger=logger)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        assert self._is_actor

        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)

        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        # Step LR scheduler only when optimizer.step() actually ran this iter
        # (gradient accumulation: skip on non-final accum micro-iter).
        if metrics.get("actor/optimizer_stepped", 1):
            self.actor_optimizer_scheduler.step(1)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        assert self._is_actor and self._is_rollout
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert self._is_actor and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None
        import torch.distributed as _td
        _rank = _td.get_rank() if _td.is_initialized() else -1
        print(f"[OPD][actor rank={_rank}] sync_rollout_weights begin (n_weights={len(self._weights_info)})", flush=True)

        params_generator = self._get_actor_params_generator()

        from ray.util.collective import collective

        update_weights_bucket_mb = self.config.rollout.get(
            "update_weights_bucket_megabytes",
            self.config.rollout.checkpoint_engine.get("update_weights_bucket_megabytes", 512),
        )
        update_weights_bucket_bytes = int(update_weights_bucket_mb) << 20
        tensor_buffer = TensorBuffer(update_weights_bucket_bytes, self.param_dtype)

        for key, shape, dtype in self._weights_info:
            weight_key, weight = next(params_generator)
            assert key == weight_key
            assert shape == weight.size()
            try:
                assert dtype == weight.dtype
            except AssertionError:
                if not key.endswith("e_score_correction_bias"):
                    raise
                # weight = weight.to(dtype)

            if shape.numel() > tensor_buffer.capacity:
                collective.broadcast(weight, src_rank=0, group_name="actor_rollout")
            else:
                if tensor_buffer.size + shape.numel() > tensor_buffer.capacity:
                    collective.broadcast(tensor_buffer.tensor, src_rank=0, group_name="actor_rollout")
                    tensor_buffer.clear()
                tensor_buffer.append(key, shape, weight)
        if tensor_buffer.size > 0:
            collective.broadcast(tensor_buffer.tensor, src_rank=0, group_name="actor_rollout")
            tensor_buffer.clear()
        print(f"[OPD][actor rank={_rank}] sync_rollout_weights done", flush=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info

        params_generator = self._get_actor_params_generator()
        ret = []
        for key, tensor in params_generator:
            ret.append((key, tensor.size(), tensor.dtype))

        self._weights_info = ret
        return ret


class MegatronOnPolicyDistillRolloutWorker(ActorRolloutRefWorker):
    """
    Rollout-only worker: owns the inference engine (vLLM/SGlang, or Megatron forward) and generates sequences.
    """

    def __init__(self, config: DictConfig, role: str):
        # Ensure we run as rollout-only worker
        # is_struct = OmegaConf.is_struct(config) or False
        # OmegaConf.set_struct(config, False)
        # # Set a safe minimal rollout micro-batch size if not provided by config
        # if OmegaConf.select(config, "actor.ppo_mini_batch_size") is None:
        #     config.actor.ppo_mini_batch_size = 2
        # if OmegaConf.select(config, "rollout.n") is None:
        #     config.rollout.n = 1
        # OmegaConf.set_struct(config, is_struct)
        import datetime

        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.device import (
            get_nccl_backend,
            get_torch_device,
        )
        from verl.utils.distributed import set_numa_affinity
        from verl.utils.fs import copy_to_local
        from verl.utils.model import get_generation_config
        from verl.utils.profiler import DistProfilerExtension, ProfilerConfig
        from verl.workers.megatron_workers import MegatronWorker

        MegatronWorker.__init__(self)
        self.config = config
        self.local_path = copy_to_local(self.config.model.path)

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

        self.role = role
        assert self.role == "rollout"

        self._is_actor = False
        self._is_rollout = True
        self._is_ref = False

        # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
        # This is for extendability in AsyncRL cases
        omega_profiler_config = config.rollout.get("profiler", {})

        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # self._build_rollout will use this variable
        self.bridge = "none"
        self.generation_config = get_generation_config(self.local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Build the actor module only for inference + rollout engine; no optimizer/updates.
        """
        from verl.utils.torch_dtypes import PrecisionType

        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init rollout model", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        if self.config.rollout.name == "vllm":
            _patch_transformers_tokenizer_compat()
        if self.config.rollout.get("mode", "async") != "async":
            raise ValueError("Megatron GKD rollout uses the official async server path; set rollout.mode=async.")
        self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        self.rollout_device_mesh = self.rollout.device_mesh
        log_gpu_memory_usage("After rollout init", logger=logger)
        get_torch_device().empty_cache()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        """
        Asynchronous-friendly rollout. When called via Ray with blocking=False,
        returns immediately with a future. The actual method execution generates
        sequences and optionally fetches teacher knowledge, and returns DataProto.
        """
        assert self._is_rollout and not self._is_actor
        prompts.batch = prompts.batch.to(get_device_name())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        # No context switching here; rollout-only worker always in rollout mode.

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = gather_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        # clear kv cache
        get_torch_device().empty_cache()

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    def async_generate_sequences(self, *args, **kwargs):
        return self.generate_sequences(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def sync_rollout_weights(self, global_steps: int = None):
        from ray.util.collective import collective

        assert self._is_rollout and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None
        import torch.distributed as _td
        _rank = _td.get_rank() if _td.is_initialized() else -1
        print(f"[OPD][rollout rank={_rank}] sync_rollout_weights begin (n_weights={len(self._weights_info)}, step={global_steps})", flush=True)
        if getattr(self.rollout, "skip_weight_sync", False):
            print(f"[OPD][rollout rank={_rank}] skip_weight_sync=True; returning", flush=True)
            return
        rollout_name = self.config.rollout.name
        if rollout_name not in ("vllm", "sglang"):
            raise NotImplementedError(f"Unknown rollout name: {rollout_name}")

        update_weights_bucket_mb = self.config.rollout.get(
            "update_weights_bucket_megabytes",
            self.config.rollout.checkpoint_engine.get("update_weights_bucket_megabytes", 512),
        )
        update_weights_bucket_bytes = int(update_weights_bucket_mb) << 20
        tensor_buffer = TensorBuffer(update_weights_bucket_bytes, self.param_dtype)

        def all_weights_generator():
            # Mirrors the actor-side broadcast schedule in
            # MegatronOnPolicyDistillActorWorker.sync_rollout_weights so each
            # recv matches one send.
            for key, shape, dtype in self._weights_info:
                # MoE e_score_correction_bias may carry a different dtype than
                # the buffer; the actor side tolerates this. Standalone tensors
                # are broadcast in their native dtype, so we accept any dtype
                # for keys we receive solo and only enforce equality for
                # bucket-packed tensors.
                if shape.numel() > tensor_buffer.capacity:
                    tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
                    collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")
                    yield key, tensor
                else:
                    if tensor_buffer.size + shape.numel() > tensor_buffer.capacity:
                        collective.broadcast(tensor_buffer.tensor, src_rank=0, group_name="actor_rollout")
                        for k, t in tensor_buffer.to_tensors():
                            yield k, t
                        tensor_buffer.clear()
                    tensor_buffer.append(key, shape)
            if tensor_buffer.size > 0:
                collective.broadcast(tensor_buffer.tensor, src_rank=0, group_name="actor_rollout")
                for k, t in tensor_buffer.to_tensors():
                    yield k, t
                tensor_buffer.clear()

        # In standalone (non-hybrid) mode, self.rollout is the vLLM
        # ServerAdapter (HTTP wrapper). Push received weights into the
        # colocated vLLM worker subprocesses via ServerAdapter.update_weights,
        # which uses ZMQ-IPC on the local node (rollout worker and vLLM
        # worker share the same GPU UUID).
        print(f"[OPD][rollout rank={_rank}] calling ServerAdapter.update_weights", flush=True)
        await self.rollout.update_weights(all_weights_generator(), global_steps=global_steps)
        print(f"[OPD][rollout rank={_rank}] sync_rollout_weights done", flush=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info
