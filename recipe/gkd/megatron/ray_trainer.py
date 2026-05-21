# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import faulthandler
import math
import queue
import signal
import sys
import threading
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler

# Register SIGUSR1 → dump all thread stacks (for debugging 3-node deadlocks).
try:
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
except Exception:
    pass
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils.debug import marked_timer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger

try:
    from .teacher import TeacherClient
    from .teacher_utils import get_teacher_knowledge
except ImportError:
    try:
        from recipe.gkd.megatron.teacher import TeacherClient
        from recipe.gkd.megatron.teacher_utils import get_teacher_knowledge
    except ImportError:
        from teacher import TeacherClient
        from teacher_utils import get_teacher_knowledge

WorkerType = type[Worker]


class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output, prompt_batch=None):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.prompt_batch = prompt_batch
        self.teacher_batch_output = None

    def set_teacher_batch_output(self, teacher_batch_output):
        """Set the teacher batch output for this generation batch.

        Args:
            teacher_batch_output: The teacher model's output (DataProtoFuture or raw output)
                to be associated with this generation batch. This will be used for
                distillation or guidance during training.
        """
        self.teacher_batch_output = teacher_batch_output

    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (batch, gen_batch_result)
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
            self.gen_batch_output = gen_batch_result

        if self.teacher_batch_output is None:
            return self.epoch, self.batch, self.gen_batch_output

        if hasattr(self.teacher_batch_output, "get"):
            try:
                teacher_batch_result = self.teacher_batch_output.get()
            except Exception as e:
                teacher_batch_result = None
                print(f"{e}")
        else:
            teacher_batch_result = self.teacher_batch_output

        return self.epoch, self.batch, self.gen_batch_output, teacher_batch_result


class OnPolicyDistillTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config
        self._apply_opd_loss_config()

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        if not self.hybrid_engine:
            print("[OPD] hybrid_engine=False — using standalone actor/rollout split path (experimental)")
        # Rollout weights are synchronized after actor optimizer.step() actually
        # runs. One exception is vLLM load_format=dummy: rollout starts without
        # real model weights, so it needs exactly one initial broadcast before
        # the first generation. After that, optimizer_stepped gates all syncs.
        rollout_load_format = str(
            OmegaConf.select(config, "actor_rollout_ref.rollout.load_format", default="")
        ).lower()
        self._pending_rollout_sync = rollout_load_format == "dummy"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.use_critic = False

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.teacher_config = self.config.actor_rollout_ref.teacher
        self.n_server_workers = self.teacher_config.n_server_workers
        self.opd_config = self.config.get("opd", {})
        self.opd_kl_method = str(self.opd_config.get("kl_method", "full_vocab")).lower()
        self.opd_y_mode = str(self.opd_config.get("y_mode", "y_o")).lower()
        if self.opd_y_mode in ("y_raw", "raw"):
            self.opd_y_mode = "y_o"
        if self.opd_y_mode in ("y_cor", "y_correction", "correction"):
            self.opd_y_mode = "y_r"
        if self.opd_y_mode not in ("y_o", "y_r"):
            raise ValueError(f"opd.y_mode must be y_o or y_r, got {self.opd_y_mode!r}")
        teacher_max_prompt_length = self.opd_config.get("teacher_max_prompt_length", None)
        if teacher_max_prompt_length in (None, "", "null", "None"):
            self.teacher_max_prompt_length = None
        else:
            self.teacher_max_prompt_length = int(teacher_max_prompt_length)
        teacher_backend = str(self.teacher_config.get("backend", "auto")).lower()
        if teacher_backend in ("auto", "", "null", "none"):
            teacher_backend = "vllm_server"
        if teacher_backend not in ("vllm_server", "local_hf"):
            raise ValueError(
                "actor_rollout_ref.teacher.backend must be auto, vllm_server, or local_hf, "
                f"got {teacher_backend!r}"
            )
        self.teacher_backend = teacher_backend
        if teacher_backend == "vllm_server":
            self.teacher_client = TeacherClient(
                self.teacher_config.server_ip,
                self.teacher_config.server_port,
                n_server_workers=self.n_server_workers,
                temperature=self.teacher_config.get("temperature", self.opd_config.get("temperature", 1.0)),
                max_seq_len=self.teacher_config.get("max_seq_len", None),
                recv_timeout_ms=self.teacher_config.get("client_timeout_ms", None),
                request_batch_size=self.teacher_config.get("request_batch_size", None),
            )
        else:
            self.teacher_client = None
            if self.opd_y_mode != "y_o":
                raise ValueError("actor_rollout_ref.teacher.backend=local_hf currently supports only opd.y_mode=y_o")
            teacher_model_path = self.teacher_config.get("model_path", None)
            if not teacher_model_path:
                raise ValueError("actor_rollout_ref.teacher.model_path is required for backend=local_hf")
            with open_dict(self.config):
                actor_loss_cfg = self.config.actor_rollout_ref.actor.distill_loss
                actor_loss_cfg.local_teacher_model_path = teacher_model_path
                actor_loss_cfg.local_teacher_chunk_size = int(self.teacher_config.get("local_chunk_size", 128))
                actor_loss_cfg.local_teacher_prefill_chunk_size = int(
                    self.teacher_config.get("local_prefill_chunk_size", 512)
                )
                actor_loss_cfg.local_teacher_attn_implementation = str(
                    self.teacher_config.get("local_attn_implementation", "flash_attention_2")
                )

        self.params_dtype = PrecisionType.to_dtype("bfloat16")

    def _apply_opd_loss_config(self):
        opd_cfg = self.config.get("opd", {})
        kl_type = str(opd_cfg.get("kl_type", "")).lower()
        if not kl_type:
            return
        name_map = {"forward": "kl", "reverse": "rkl", "jsd": "jsd"}
        if kl_type not in name_map:
            raise ValueError(f"opd.kl_type must be forward, reverse, or jsd, got {kl_type!r}")
        with open_dict(self.config):
            actor_cfg = self.config.actor_rollout_ref.actor
            if "distill_loss" not in actor_cfg or actor_cfg.distill_loss is None:
                actor_cfg.distill_loss = {}
            actor_cfg.distill_loss.name = name_map[kl_type]
            actor_cfg.distill_loss.beta = float(opd_cfg.get("beta", 0.0))
            actor_cfg.distill_loss.temperature = float(opd_cfg.get("temperature", 1.0))
            actor_cfg.distill_loss.kl_token_clip = float(opd_cfg.get("kl_token_clip", 0.0))
            actor_cfg.distill_loss.top_k = int(opd_cfg.get("top_k", 0))
            actor_cfg.distill_loss.kl_method = str(opd_cfg.get("kl_method", "full_vocab")).lower()

    def _get_optimization_mode(self) -> str:
        optimization_mode = str(self.config.trainer.get("optimization_mode", "multi_step")).lower()
        if optimization_mode not in {"multi_step", "one_step"}:
            raise ValueError(
                "trainer.optimization_mode must be 'multi_step' or 'one_step', "
                f"got {optimization_mode!r}"
            )
        return optimization_mode

    def _resolve_scheduler_type(self) -> str:
        optimization_mode = self._get_optimization_mode()
        raw_scheduler = self.config.trainer.get("scheduler", "auto")
        scheduler_type = "auto" if raw_scheduler is None else str(raw_scheduler).lower()

        if scheduler_type in {"auto", "", "none", "null"}:
            if optimization_mode == "one_step":
                scheduler_type = "one_step"
            elif getattr(self, "opd_y_mode", "y_o") == "y_r":
                scheduler_type = "three_step_off"
            else:
                scheduler_type = "one_step_off"

        valid_schedulers = {"one_step", "one_step_off", "two_step_off", "three_step_off", "bounded_lag_y_r"}
        if scheduler_type not in valid_schedulers:
            raise TypeError(
                "trainer.scheduler must be auto, one_step, one_step_off, two_step_off, three_step_off, "
                "or bounded_lag_y_r, "
                f"got {raw_scheduler!r}"
            )

        if optimization_mode == "one_step" and scheduler_type != "one_step":
            raise ValueError(
                "trainer.optimization_mode=one_step requires trainer.scheduler=auto or one_step. "
                f"Got trainer.scheduler={raw_scheduler!r}, which would not run one full-dataset update."
            )

        with open_dict(self.config):
            self.config.trainer.scheduler = scheduler_type
        if str(raw_scheduler).lower() != scheduler_type:
            print(
                "Resolved trainer.scheduler="
                f"{scheduler_type} from trainer.scheduler={raw_scheduler!r} "
                f"and trainer.optimization_mode={optimization_mode}"
            )
        return scheduler_type

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_sampler

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        optimization_mode = self._get_optimization_mode()
        one_step_mode = optimization_mode == "one_step"
        train_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        if one_step_mode:
            train_batch_size = len(self.train_dataset)
            if train_batch_size < 1:
                raise ValueError("one_step optimization requires a non-empty train dataset")
            print(f"One-step optimization enabled: using full-dataset batch_size={train_batch_size}")

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            drop_last=not one_step_mode,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        if self.val_dataset:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

            print(
                f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
                f"{len(self.val_dataloader)}"
            )
        else:
            print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(self.config.trainer.total_training_steps, total_training_steps)

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        rollout_mode = self.config.actor_rollout_ref.rollout.get("mode", "async")
        if rollout_mode != "async":
            raise ValueError(
                "Megatron OPD follows the current official async rollout path; "
                f"actor_rollout_ref.rollout.mode must be async, got {rollout_mode!r}."
            )

        self.resource_pool_manager.create_resource_pool()

        # Build Ray classes per pool
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            actor_rollout_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            resource_pool_to_cls[actor_rollout_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            # Split layout: actor on actor_pool, rollout on rollout_pool.
            actor_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
            rollout_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
            actor_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Actor],
                config=self.config.actor_rollout_ref,
                role="actor",
            )
            rollout_cls_ = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Rollout],
                config=self.config.actor_rollout_ref,
                role="rollout",
            )
            resource_pool_to_cls[actor_pool][str(Role.Actor)] = actor_cls
            resource_pool_to_cls[rollout_pool][str(Role.Rollout)] = rollout_cls_

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            time.sleep(20)  # avoid port conflict

        if self.hybrid_engine:
            self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
            self.actor_wg = self.actor_rollout_wg
            self.actor_rollout_wg.init_model()
            agent_loop_wg = self.actor_rollout_wg
            agent_loop_pool = actor_rollout_pool
        else:
            # Split layout: separate actor and rollout worker groups.
            self.actor_wg = all_wg[str(Role.Actor)]
            self.rollout_wg = all_wg[str(Role.Rollout)]
            self.actor_rollout_wg = self.actor_wg  # backward-compat alias for training paths
            self.actor_wg.init_model()
            self.rollout_wg.init_model()
            agent_loop_wg = self.rollout_wg
            agent_loop_pool = rollout_pool

        from verl.experimental.agent_loop import AgentLoopManager

        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=agent_loop_wg,
            rollout_resource_pool=agent_loop_pool,
            reward_loop_worker_handles=None,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )
        self.checkpoint_manager.sleep_replicas()

        if not self.hybrid_engine:
            # Non-hybrid: set up NCCL collective group for actor → rollout
            # weight broadcast. Trainer-side workers (actor) broadcast each
            # bucket; rollout-side workers receive and forward into vLLM via
            # ServerAdapter.update_weights (ZMQ-IPC on the local node).
            from ray.util.collective import collective

            from verl.utils.device import get_nccl_backend

            weights_info = self.actor_wg.get_actor_weights_info()[0]
            self.rollout_wg.set_actor_weights_info(weights_info)

            actor_workers = list(self.actor_wg.workers)
            rollout_workers = list(self.rollout_wg.workers)
            sync_workers = actor_workers + rollout_workers
            n_workers = len(sync_workers)
            print(
                f"[OPD] creating collective group 'actor_rollout' with "
                f"{len(actor_workers)} actor + {len(rollout_workers)} rollout = {n_workers} workers",
                flush=True,
            )
            collective.create_collective_group(
                sync_workers,
                n_workers,
                list(range(n_workers)),
                backend=get_nccl_backend(),
                group_name="actor_rollout",
            )
            print("[OPD] actor_rollout collective group created", flush=True)

    def sync_rollout_weights(self):
        # Gradient-accumulation aware sync: actor weights only change when
        # optimizer.step() actually runs (every accum_steps iterations). Skip
        # the broadcast when nothing has changed since the last sync.
        # Mathematically correct because all rollouts within one accumulation
        # group should use the same on-policy snapshot.
        if not getattr(self, "_pending_rollout_sync", False):
            return False
        self._pending_rollout_sync = False

        if self.hybrid_engine:
            # Hybrid: trainer.update_weights uses naive CheckpointEngine — push
            # via colocated ZMQ-IPC to vLLM.
            self.checkpoint_manager.update_weights(self.global_steps)
            return True

        # Standalone (3-node split): trainer-side broadcasts the actor weights
        # over the `actor_rollout` NCCL group; rollout-side receives each
        # bucket and forwards into vLLM workers via ServerAdapter.update_weights
        # (ZMQ-IPC on the rollout node). Each wg.sync_rollout_weights() returns
        # a list[ObjectRef] (one per worker), so flatten before ray.get.
        import ray as _ray

        print(f"[OPD][trainer] sync_rollout_weights step={self.global_steps} firing actor", flush=True)
        actor_futs = self.actor_wg.sync_rollout_weights()
        print(f"[OPD][trainer] sync_rollout_weights actor dispatched (type={type(actor_futs).__name__})", flush=True)
        rollout_futs = self.rollout_wg.sync_rollout_weights(global_steps=self.global_steps)
        print(f"[OPD][trainer] sync_rollout_weights rollout dispatched (type={type(rollout_futs).__name__})", flush=True)
        if not isinstance(actor_futs, list):
            actor_futs = [actor_futs]
        if not isinstance(rollout_futs, list):
            rollout_futs = [rollout_futs]
        print(f"[OPD][trainer] sync_rollout_weights waiting on {len(actor_futs)} actor + {len(rollout_futs)} rollout futures", flush=True)
        _ray.get(actor_futs + rollout_futs)
        print(f"[OPD][trainer] sync_rollout_weights step={self.global_steps} done", flush=True)
        return True

    def sync_rollout_weights_if_pending(self, timing=None):
        if not getattr(self, "_pending_rollout_sync", False):
            return False
        if timing is None:
            return self.sync_rollout_weights()
        with marked_timer("sync_rollout_weights", timing):
            return self.sync_rollout_weights()

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b) if a and b else max(a, b, 1)

    def _one_step_dispatch_multiple(self) -> int:
        actor_world = int(self.config.trainer.nnodes) * int(self.config.trainer.n_gpus_per_node)
        actor_mp = (
            int(self.config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size)
            * int(self.config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size)
            * int(self.config.actor_rollout_ref.actor.megatron.get("context_parallel_size", 1))
        )
        actor_dp = max(1, actor_world // max(actor_mp, 1))

        rollout_world = actor_world
        rollout_mp = (
            int(self.config.actor_rollout_ref.rollout.tensor_model_parallel_size)
            * int(self.config.actor_rollout_ref.rollout.get("pipeline_model_parallel_size", 1))
        )
        rollout_dp = max(1, rollout_world // max(rollout_mp, 1))
        return self._lcm(actor_dp, rollout_dp)

    def _pad_batch_dict_for_one_step(self, batch_dict: dict) -> dict:
        if str(self.config.trainer.get("optimization_mode", "multi_step")).lower() != "one_step":
            return batch_dict

        batch_size = None
        for value in batch_dict.values():
            if isinstance(value, torch.Tensor):
                batch_size = int(value.size(0))
                break
            if isinstance(value, np.ndarray):
                batch_size = int(value.shape[0])
                break
        if batch_size is None:
            return batch_dict

        multiple = self._one_step_dispatch_multiple()
        pad_size = (multiple - (batch_size % multiple)) % multiple
        if pad_size == 0:
            padded = dict(batch_dict)
            padded["is_padded_sample"] = torch.zeros(batch_size, dtype=torch.bool)
            return padded

        padded = {}
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor) and value.size(0) == batch_size:
                repeat_shape = [pad_size, *([1] * (value.dim() - 1))]
                padded[key] = torch.cat([value, value[:1].repeat(*repeat_shape)], dim=0)
            elif isinstance(value, np.ndarray) and value.shape[0] == batch_size:
                padded[key] = np.concatenate([value, np.repeat(value[:1], pad_size, axis=0)], axis=0)
            else:
                padded[key] = value
        padded["is_padded_sample"] = torch.cat(
            [torch.zeros(batch_size, dtype=torch.bool), torch.ones(pad_size, dtype=torch.bool)],
            dim=0,
        )
        print(
            f"One-step batch padded from {batch_size} to {batch_size + pad_size} "
            f"for actor/rollout dispatch multiple={multiple}; padded samples have zero loss."
        )
        return padded

    @staticmethod
    def _apply_sample_padding_loss_mask(batch: DataProto):
        if batch.batch is None or "is_padded_sample" not in batch.batch.keys():
            return
        padded_rows = batch.batch["is_padded_sample"].to(torch.bool)
        if not bool(padded_rows.any()):
            return
        if "distill_loss_mask" not in batch.batch.keys():
            responses = batch.batch["responses"]
            response_length = responses.size(1)
            calc_kl_mask = batch.batch["attention_mask"].to(torch.bool).clone()
            calc_kl_mask[:, : (-response_length - 1)] = False
            batch.batch["distill_loss_mask"] = calc_kl_mask
        batch.batch["distill_loss_mask"][padded_rows] = False

    def _actor_update_batch_size(self) -> int | None:
        value = OmegaConf.select(self.config, "trainer.actor_update_batch_size", default=None)
        if value in (None, "", "null", "None", 0, "0"):
            return None
        value = int(value)
        if value < 1:
            raise ValueError(f"trainer.actor_update_batch_size must be positive, got {value}")
        return value

    @staticmethod
    def _slice_actor_meta_info(meta_info: dict, start: int, end: int) -> dict:
        sliced = dict(meta_info)
        token_nums = sliced.get("global_token_num", None)
        if isinstance(token_nums, torch.Tensor):
            sliced["global_token_num"] = token_nums[start:end].tolist()
        elif isinstance(token_nums, np.ndarray):
            sliced["global_token_num"] = token_nums[start:end].tolist()
        elif isinstance(token_nums, list):
            sliced["global_token_num"] = token_nums[start:end]
        return sliced

    def _iter_actor_update_batches(self, batch: DataProto, actor_batch_size: int | None):
        if actor_batch_size is None or actor_batch_size >= len(batch):
            yield 0, len(batch), batch
            return
        for start in range(0, len(batch), actor_batch_size):
            end = min(start + actor_batch_size, len(batch))
            micro_batch = batch[start:end]
            micro_batch.meta_info = self._slice_actor_meta_info(batch.meta_info, start, end)
            yield start, end, micro_batch

    @staticmethod
    def _merge_actor_update_metrics(metrics_list: list[dict]) -> dict:
        if len(metrics_list) == 1:
            merged = dict(metrics_list[0])
            merged["actor/update_micro_batches"] = 1
            return merged

        merged = {}
        keys = set().union(*(metrics.keys() for metrics in metrics_list))
        last_value_keys = {"actor/grad_accum_iter", "actor/grad_norm", "actor/lr"}
        max_value_keys = {
            "actor/optimizer_stepped",
            "perf/max_memory_allocated_gb",
            "perf/max_memory_reserved_gb",
            "perf/cpu_memory_used_gb",
        }
        for key in keys:
            values = [metrics[key] for metrics in metrics_list if key in metrics]
            if not values:
                continue
            if key in last_value_keys:
                merged[key] = values[-1]
            elif key in max_value_keys:
                merged[key] = max(values)
            else:
                try:
                    merged[key] = sum(values) / len(values)
                except TypeError:
                    merged[key] = values[-1]
        merged["actor/update_micro_batches"] = len(metrics_list)
        return merged

    def _update_actor_with_microbatches(self, batch: DataProto, timing_raw: dict) -> dict:
        actor_batch_size = self._actor_update_batch_size()
        actor_metrics = []
        micro_times = []
        for micro_idx, (start, end, micro_batch) in enumerate(
            self._iter_actor_update_batches(batch, actor_batch_size)
        ):
            if actor_batch_size is not None and len(batch) > actor_batch_size:
                print(
                    f"[OPD][trainer] update_actor microbatch {micro_idx} rows={start}:{end} "
                    f"of {len(batch)}",
                    flush=True,
                )
            tik = time.time()
            actor_output = self.actor_wg.update_actor(micro_batch)
            micro_times.append(time.time() - tik)
            actor_metrics.append(reduce_metrics(actor_output.meta_info["metrics"]))

        if len(micro_times) > 1:
            timing_raw["update_actor_microbatch_mean"] = float(np.mean(micro_times))
            timing_raw["update_actor_microbatch_max"] = float(np.max(micro_times))
        return self._merge_actor_update_metrics(actor_metrics)

    def _prepare_rollout_generation_batch(self, epoch, batch_dict):
        batch = DataProto.from_single_dict(batch_dict)
        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        for key in ("raw_prompt", "opd_raw_prompt"):
            if key in gen_batch.non_tensor_batch and key not in batch.non_tensor_batch:
                batch.non_tensor_batch[key] = gen_batch.non_tensor_batch[key]
        if "extra_info" in batch.non_tensor_batch and "extra_info" not in gen_batch.non_tensor_batch:
            gen_batch.non_tensor_batch["extra_info"] = batch.non_tensor_batch["extra_info"]
        # Keep the tokenized prompt around for y_r mode. The rollout worker
        # needs raw_prompt in gen_batch, while the later y_r remap needs the
        # exact padded student prompt ids to rebuild x+y_r for actor training.
        prompt_batch = gen_batch.select(deepcopy=True)
        gen_batch.meta_info["global_steps"] = self.global_steps
        return batch, gen_batch, prompt_batch

    def _async_gen_next_batch(self, epoch, batch_dict, sync_before_generation=True, sync_timing=None):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        batch, gen_batch, prompt_batch = self._prepare_rollout_generation_batch(epoch, batch_dict)
        # sync weights from actor to rollout
        if sync_before_generation:
            self.sync_rollout_weights_if_pending(sync_timing)
        rollout_pad_size = 0
        agent_workers = int(self.config.actor_rollout_ref.rollout.agent.get("num_workers", 1))
        if agent_workers > 1:
            gen_batch, rollout_pad_size = pad_dataproto_to_divisor(gen_batch, agent_workers)

        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
        self.checkpoint_manager.sleep_replicas()
        gen_batch_output = unpad_dataproto(gen_batch_output, rollout_pad_size)
        return GenerationBatchFuture(epoch, batch, gen_batch_output, prompt_batch=prompt_batch)

    def _gen_single_sample_on_rollout_worker(self, epoch: int, batch_dict: dict, worker_index: int):
        batch, gen_batch, prompt_batch = self._prepare_rollout_generation_batch(epoch, batch_dict)
        if len(gen_batch) != 1:
            raise ValueError(f"single-sample rollout expected batch size 1, got {len(gen_batch)}")
        workers = getattr(self.async_rollout_manager, "agent_loop_workers", None)
        if not workers:
            future = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
            _, batch, gen_batch_output = future.get()
            return batch, future.prompt_batch, gen_batch_output, {}

        worker = workers[worker_index % len(workers)]
        gen_batch_output = ray.get(worker.generate_sequences.remote(gen_batch))
        rollout_timing = {}
        metrics = gen_batch_output.meta_info.pop("metrics", None)
        if metrics is not None:
            try:
                rollout_timing = self.async_rollout_manager._performance_metrics([metrics], gen_batch_output)
            except Exception:
                rollout_timing = {}
        gen_batch_output.meta_info = {"timing": rollout_timing}
        return batch, prompt_batch, gen_batch_output, rollout_timing

    def _apply_teacher_chat_template(self, content: str) -> list[int]:
        from verl.utils.tokenizer import normalize_token_ids

        token_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )
        return normalize_token_ids(token_ids)

    def _truncate_teacher_initial_response(self, content: str, max_length: int) -> str:
        init_marker = "**Your Initial Solution:**"
        end_marker = "\n\n**Instructions:**"
        init_start = content.find(init_marker)
        init_end = content.find(end_marker, init_start + len(init_marker)) if init_start >= 0 else -1
        if init_start < 0 or init_end <= init_start:
            return content

        block_start = init_start + len(init_marker)
        block_text = content[block_start:init_end]
        block_ids = self.tokenizer.encode(block_text, add_special_tokens=False)
        if not block_ids:
            return content

        prefix = content[:block_start]
        suffix = content[init_end:]
        notice = "\n...[truncated initial solution]...\n"

        best = None
        lo, hi = 0, len(block_ids)
        while lo <= hi:
            mid = (lo + hi) // 2
            kept = self.tokenizer.decode(block_ids[:mid], skip_special_tokens=True)
            candidate = prefix + kept + notice + suffix
            if len(self._apply_teacher_chat_template(candidate)) <= max_length:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best if best is not None else content

    def _tokenize_chat_prompt(self, content: str) -> list[int]:
        max_length = self.teacher_max_prompt_length
        token_ids = self._apply_teacher_chat_template(content)
        if max_length is None or len(token_ids) <= max_length:
            return token_ids

        truncated_content = self._truncate_teacher_initial_response(content, max_length)
        token_ids = self._apply_teacher_chat_template(truncated_content)
        if len(token_ids) <= max_length:
            return token_ids

        print(
            f"[OPD] teacher prompt length {len(token_ids)} exceeds "
            f"teacher_max_prompt_length={max_length}; applying final right truncation.",
            flush=True,
        )
        return token_ids[:max_length]

    def _extra_info_at(self, batch: DataProto, idx: int) -> dict:
        extra_infos = batch.non_tensor_batch.get("extra_info")
        if extra_infos is None:
            return {}
        value = extra_infos[idx]
        return value if isinstance(value, dict) else {}

    def _non_tensor_value_at(self, batch: DataProto, key: str, idx: int):
        values = batch.non_tensor_batch.get(key)
        if values is None:
            return None
        try:
            return values[idx]
        except Exception:
            return values

    def _strip_instruction_following(self, text: str) -> str:
        instruction = str(
            self.opd_config.get(
                "instruction_following",
                "Please reason step by step, and put your final answer within \\boxed{}.",
            )
        ).strip()
        text = str(text).strip()
        if instruction and text.endswith(instruction):
            return text[: -len(instruction)].rstrip()
        return text

    def _raw_problem_at(self, batch: DataProto, idx: int) -> str:
        extra_info = self._extra_info_at(batch, idx)
        for key in ("problem", "question"):
            problem = extra_info.get(key)
            if problem:
                return self._strip_instruction_following(str(problem))
        for key in ("problem", "question"):
            problem = self._non_tensor_value_at(batch, key, idx)
            if problem:
                return self._strip_instruction_following(str(problem))
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raw_prompts = batch.non_tensor_batch.get("opd_raw_prompt")
        if raw_prompts is not None:
            messages = raw_prompts[idx]
            if messages and isinstance(messages, list):
                content = messages[0].get("content", "")
                if isinstance(content, str):
                    return self._strip_instruction_following(content)
        return ""

    def _expert_solution_at(self, batch: DataProto, idx: int) -> str:
        extra_info = self._extra_info_at(batch, idx)
        # Training-time y* is the cleaned full solution/COT when available.
        # If solution is missing, fall back to the final answer field by design.
        solution = extra_info.get("solution")
        if solution:
            return str(solution)
        solution = self._non_tensor_value_at(batch, "solution", idx)
        if solution:
            return str(solution)
        answer = extra_info.get("answer")
        if answer:
            return str(answer)
        answer = self._non_tensor_value_at(batch, "answer", idx)
        if answer:
            return str(answer)
        return ""

    def _decode_response(self, response_ids: torch.Tensor) -> str:
        pad_id = self.tokenizer.pad_token_id
        ids = response_ids.detach().cpu()
        if pad_id is not None:
            ids = ids[ids != pad_id]
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)

    def _build_teacher_prompt_content(
        self,
        *,
        problem: str,
        expert_solution: str,
        initial_response: str,
        use_initial_response: bool,
    ) -> str:
        distill_mode = str(self.opd_config.get("distill_mode", "opd")).lower()
        instruction = self.opd_config.get(
            "instruction_following",
            "Please reason step by step, and put your final answer within \\boxed{}.",
        )
        if distill_mode == "opd":
            if use_initial_response:
                prompt = f"""Your task is to rewrite your mathematical solution.

**Problem:**
{problem}

**Your Initial Solution:**
{initial_response}

**Instructions:**
1. Preserve the overall structure and reasoning path of your original solution
2. Identify and fix errors in computation or logic
3. Keep correct intermediate steps and meaningful work
4. Output ONLY the rewritten solution
"""
            else:
                prompt = problem
        elif distill_mode == "opsd":
            if not str(expert_solution).strip():
                raise ValueError("opd.distill_mode=opsd requires solution or answer for training y*")
            if use_initial_response:
                prompt = f"""Your task is to rewrite your mathematical solution using the reference solution as guidance.

**Problem:**
{problem}

**Reference Solution:**
{expert_solution}

**Your Initial Solution:**
{initial_response}

**Instructions:**
1. Review the reference solution to understand the target reasoning and method
2. Rewrite your solution so it is consistent with the reference solution
3. Keep useful parts of your original structure and style when appropriate
4. Output ONLY the rewritten solution
"""
            else:
                prompt = f"""{problem}

Here is a reference solution:
{expert_solution}

After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:
"""
        else:
            raise ValueError(f"opd.distill_mode must be opd or opsd, got {distill_mode!r}")
        return prompt.strip() + " " + instruction

    def _submit_teacher_requests(
        self,
        prompt_token_ids: list[list[int]],
        *,
        max_tokens: int,
        only_response: bool,
        temperature: float,
        logprob_row_indices: list[list[int]] | None = None,
        request_chunk_size: int | None = None,
    ):
        batch_size = len(prompt_token_ids)
        if request_chunk_size is None:
            n_chunks = max(1, min(self.n_server_workers, batch_size))
            chunk_size = (batch_size + n_chunks - 1) // n_chunks
        else:
            chunk_size = max(1, int(request_chunk_size))
        futures = []
        for start in range(0, batch_size, chunk_size):
            chunk = prompt_token_ids[start : start + chunk_size]
            chunk_row_indices = (
                logprob_row_indices[start : start + chunk_size] if logprob_row_indices is not None else None
            )
            request_kwargs = {}
            if chunk_row_indices is not None:
                request_kwargs["logprob_row_indices"] = chunk_row_indices
            futures.append(
                self.teacher_client.submit(
                    chunk,
                    max_tokens=max_tokens,
                    only_response=only_response,
                    temperature=temperature,
                    **request_kwargs,
                )
            )
        return futures

    @staticmethod
    def _collect_teacher_futures(futures):
        responses, topk_logps, topk_indices = [], [], []
        for future in futures:
            try:
                resp, logps, indices = future.result()
            except Exception as e:
                raise RuntimeError(f"Teacher request failed: {e}") from e
            responses.extend(resp)
            topk_logps.extend(logps)
            topk_indices.extend(indices)
        return responses, topk_logps, topk_indices

    def _bounded_lag_teacher_request_chunk_size(self) -> int:
        teacher_config = getattr(self, "teacher_config", None)
        if teacher_config is None:
            teacher_config = OmegaConf.select(self.config, "actor_rollout_ref.teacher", default={})
        value = teacher_config.get("request_batch_size", None)
        if value in (None, "", "null", "None", 0, "0"):
            actor_batch_size = self._actor_update_batch_size()
            value = actor_batch_size if actor_batch_size is not None else self.config.data.train_batch_size
        return max(1, int(value))

    def _build_yr_generation_prompts(self, batch: DataProto, y_o_output: DataProto) -> list[list[int]]:
        teacher_prompt = str(self.opd_config.get("teacher_training_prompt", "refine")).lower()
        if teacher_prompt not in ("refine", "vanilla"):
            raise ValueError("opd.teacher_training_prompt must be refine or vanilla")
        use_initial_response = teacher_prompt == "refine"
        prompts = []
        for i, response_ids in enumerate(y_o_output.batch["responses"]):
            y_o_text = self._decode_response(response_ids)
            content = self._build_teacher_prompt_content(
                problem=self._raw_problem_at(batch, i),
                expert_solution=self._expert_solution_at(batch, i),
                initial_response=y_o_text,
                use_initial_response=use_initial_response,
            )
            prompts.append(self._tokenize_chat_prompt(content))
        return prompts

    def _build_yr_scoring_prompts(
        self,
        batch: DataProto,
        y_o_output: DataProto,
        y_r_responses: list[torch.Tensor],
        *,
        use_initial_response: bool,
    ) -> tuple[list[list[int]], list[int], list[int]]:
        prompts, prompt_lens, response_lens = [], [], []
        for i, (y_o_ids, y_r_ids) in enumerate(zip(y_o_output.batch["responses"], y_r_responses, strict=True)):
            y_o_text = self._decode_response(y_o_ids)
            content = self._build_teacher_prompt_content(
                problem=self._raw_problem_at(batch, i),
                expert_solution=self._expert_solution_at(batch, i),
                initial_response=y_o_text,
                use_initial_response=use_initial_response,
            )
            prompt_ids = self._tokenize_chat_prompt(content)
            response_ids = y_r_ids.detach().cpu().to(torch.int64).tolist()
            prompts.append(prompt_ids + response_ids)
            prompt_lens.append(len(prompt_ids))
            response_lens.append(len(response_ids))
        return prompts, prompt_lens, response_lens

    @staticmethod
    def _slice_scored_response_topk(topk_tensors, prompt_lens: list[int], response_lens: list[int]):
        if topk_tensors and topk_tensors[0] is None:
            return [None for _ in response_lens]
        sliced = []
        for tensor, prompt_len, response_len in zip(topk_tensors, prompt_lens, response_lens, strict=True):
            start = max(prompt_len - 1, 0)
            end = start + response_len
            if tensor.size(0) >= end:
                sliced.append(tensor[start:end])
            elif tensor.size(0) >= response_len:
                sliced.append(tensor[-response_len:])
            else:
                raise RuntimeError(
                    f"Teacher scoring returned too few top-k rows: got {tensor.size(0)}, need {response_len}"
                )
        return sliced

    @staticmethod
    def _scored_response_row_indices(prompt_lens: list[int], response_lens: list[int]) -> list[list[int]]:
        row_indices = []
        for prompt_len, response_len in zip(prompt_lens, response_lens, strict=True):
            start = max(prompt_len - 1, 0)
            row_indices.append(list(range(start, start + response_len)))
        return row_indices

    @staticmethod
    def _teacher_tensor_to_numpy_payload(tensor: torch.Tensor):
        tensor = tensor.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.int16).numpy()
        return tensor.numpy()

    def _build_yr_actor_batches(
        self,
        prompt_batch: DataProto,
        y_r_responses: list[torch.Tensor],
        teacher_topk_logps: list[torch.Tensor],
        teacher_topk_indices: list[torch.Tensor],
        timing: dict,
    ) -> tuple[DataProto, DataProto]:
        prompt_ids_padded = prompt_batch.batch["input_ids"].detach().cpu()
        prompt_mask_padded = prompt_batch.batch["attention_mask"].detach().cpu().to(torch.bool)
        batch_size = prompt_ids_padded.size(0)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        prompt_rows = [prompt_ids_padded[i][prompt_mask_padded[i]].to(torch.long) for i in range(batch_size)]
        response_rows = [resp.detach().cpu().to(torch.long) for resp in y_r_responses]
        response_lens = [int(row.numel()) for row in response_rows]
        if any(length <= 0 for length in response_lens):
            raise RuntimeError("y_r teacher generation produced an empty response")

        prompt_width = max(int(row.numel()) for row in prompt_rows)
        response_width = max(response_lens)
        seq_len = prompt_width + response_width
        has_indices = bool(teacher_topk_indices) and teacher_topk_indices[0] is not None

        input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        responses = torch.full((batch_size, response_width), pad_id, dtype=torch.long)
        distill_loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        topk_logps_payload = np.empty((batch_size,), dtype=object)
        topk_indices_payload = np.empty((batch_size,), dtype=object) if has_indices else None
        teacher_loss_lens = torch.zeros(batch_size, dtype=torch.int32)

        for i, (prompt_row, response_row, logps, indices) in enumerate(
            zip(prompt_rows, response_rows, teacher_topk_logps, teacher_topk_indices, strict=True)
        ):
            prompt_len = int(prompt_row.numel())
            response_len = int(response_row.numel())
            prompt_start = prompt_width - prompt_len
            response_start = prompt_width

            input_ids[i, prompt_start:prompt_width] = prompt_row
            attention_mask[i, prompt_start:prompt_width] = True
            position_ids[i, prompt_start:prompt_width] = torch.arange(prompt_len, dtype=torch.long)

            input_ids[i, response_start : response_start + response_len] = response_row
            attention_mask[i, response_start : response_start + response_len] = True
            position_ids[i, response_start : response_start + response_width] = (
                torch.arange(response_width, dtype=torch.long) + prompt_len
            )
            responses[i, :response_len] = response_row

            loss_start = response_start - 1
            loss_end = loss_start + response_len
            distill_loss_mask[i, loss_start:loss_end] = True
            if logps.size(0) < response_len:
                raise RuntimeError(
                    f"Teacher y_r logprobs shorter than response: row={i}, "
                    f"teacher_rows={logps.size(0)}, response_len={response_len}"
                )
            selected_logps = logps[:response_len].to(torch.bfloat16)
            teacher_loss_lens[i] = selected_logps.size(0)
            topk_logps_payload[i] = self._teacher_tensor_to_numpy_payload(selected_logps)
            if has_indices:
                if indices.size(0) < response_len:
                    raise RuntimeError(
                        f"Teacher y_r indices shorter than response: row={i}, "
                        f"teacher_rows={indices.size(0)}, response_len={response_len}"
                    )
                topk_indices_payload[i] = self._teacher_tensor_to_numpy_payload(indices[:response_len].to(torch.int32))

        gen_output = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "distill_loss_mask": distill_loss_mask,
            },
            meta_info={"timing": timing},
        )
        teacher_output = DataProto.from_single_dict(
            data={
                "real_seq_lens": torch.tensor(response_lens, dtype=torch.int32),
                "teacher_loss_lens": teacher_loss_lens,
            }
        )
        # Per-sample compact payload: each object row is [response_loss_tokens_i, topk].
        # This avoids materializing [batch, prompt+response, vocab] for y_r full-vocab KL.
        non_tensor = {"teacher_topk_logps": topk_logps_payload}
        if has_indices:
            non_tensor["teacher_topk_indices"] = topk_indices_payload
        teacher_output.non_tensor_batch.update(non_tensor)
        teacher_output.meta_info["timing"] = {}
        return gen_output, teacher_output

    @staticmethod
    def _teacher_failed_result(future: GenerationBatchFuture, error: Exception):
        print(f"{error}")
        return future.epoch, future.batch, future.gen_batch_output, None

    def _async_generate_y_r(self, future: GenerationBatchFuture, *, request_chunk_size: int | None = None):
        _, _, y_o_output = future.get()
        temperature = float(self.opd_config.get("temperature", self.teacher_config.get("temperature", 1.0)))
        max_tokens = int(self.config.data.max_response_length)
        teacher_prompt_ids = self._build_yr_generation_prompts(future.batch, y_o_output)
        tik = time.time()
        gen_futures = self._submit_teacher_requests(
            teacher_prompt_ids,
            max_tokens=max_tokens,
            only_response=True,
            temperature=temperature,
            request_chunk_size=request_chunk_size,
        )
        cache = {}

        def handle_generation():
            if "result" in cache:
                return cache["result"]
            responses, gen_logps, gen_indices = self._collect_teacher_futures(gen_futures)
            cache["result"] = SimpleNamespace(
                source_future=future,
                y_o_output=y_o_output,
                responses=responses,
                gen_logps=gen_logps,
                gen_indices=gen_indices,
                temperature=temperature,
                started_at=tik,
                generated_at=time.time(),
            )
            return cache["result"]

        return SimpleNamespace(get=handle_generation, source_future=future)

    def _async_score_y_r_generation(
        self,
        y_r_generation_future,
        *,
        return_full_result=False,
        request_chunk_size: int | None = None,
    ):
        try:
            y_r_generation = y_r_generation_future.get()
        except Exception as e:
            source_future = y_r_generation_future.source_future
            if return_full_result:
                return SimpleNamespace(get=lambda: self._teacher_failed_result(source_future, e))

            def handle_failed_generation():
                print(f"{e}")
                return None

            return SimpleNamespace(get=handle_failed_generation)

        source_future = y_r_generation.source_future
        y_o_output = y_r_generation.y_o_output
        responses = y_r_generation.responses
        teacher_prompt = str(self.opd_config.get("teacher_training_prompt", "refine")).lower()
        if teacher_prompt not in ("refine", "vanilla"):
            raise ValueError("opd.teacher_training_prompt must be refine or vanilla")

        score_futures = None
        prompt_lens = None
        response_lens = None
        score_started_at = None
        if teacher_prompt == "vanilla":
            score_prompt_ids, prompt_lens, response_lens = self._build_yr_scoring_prompts(
                source_future.batch,
                y_o_output,
                responses,
                use_initial_response=False,
            )
            score_row_indices = self._scored_response_row_indices(prompt_lens, response_lens)
            score_started_at = time.time()
            score_futures = self._submit_teacher_requests(
                score_prompt_ids,
                max_tokens=1,
                only_response=False,
                temperature=y_r_generation.temperature,
                logprob_row_indices=score_row_indices,
                request_chunk_size=request_chunk_size,
            )

        cache = {}

        def handle_score():
            if "result" in cache:
                return cache["result"]
            try:
                if score_futures is None:
                    teacher_logps = y_r_generation.gen_logps
                    teacher_indices = y_r_generation.gen_indices
                    timing = {"generate_y_r_and_teacher_topk": y_r_generation.generated_at - y_r_generation.started_at}
                else:
                    _, score_logps, score_indices = self._collect_teacher_futures(score_futures)
                    if all(logps.size(0) == response_len for logps, response_len in zip(score_logps, response_lens)):
                        teacher_logps = score_logps
                        teacher_indices = score_indices
                    else:
                        teacher_logps = self._slice_scored_response_topk(score_logps, prompt_lens, response_lens)
                        teacher_indices = self._slice_scored_response_topk(score_indices, prompt_lens, response_lens)
                    timing = {
                        "generate_y_r": y_r_generation.generated_at - y_r_generation.started_at,
                        "score_y_r": time.time() - score_started_at,
                    }

                gen_output, teacher_output = self._build_yr_actor_batches(
                    source_future.prompt_batch,
                    responses,
                    teacher_logps,
                    teacher_indices,
                    timing,
                )
                source_future.gen_batch_output = gen_output
                if return_full_result:
                    cache["result"] = source_future.epoch, source_future.batch, gen_output, teacher_output
                else:
                    cache["result"] = teacher_output
            except Exception as e:
                if return_full_result:
                    cache["result"] = self._teacher_failed_result(source_future, e)
                else:
                    print(f"{e}")
                    cache["result"] = None
            return cache["result"]

        return SimpleNamespace(get=handle_score, source_future=source_future)

    def _async_get_yr_teacher_knowledge(self, future: GenerationBatchFuture, y_o_output: DataProto = None):
        del y_o_output
        y_r_generation_future = self._async_generate_y_r(future)
        return self._async_score_y_r_generation(y_r_generation_future)

    def _async_get_teacher_knowledge(self, future: GenerationBatchFuture):
        """Asynchronously obtain teacher model knowledge for generated sequences.

        This method retrieves generated sequences from the future object, adds response length metadata,
        and asynchronously queries the teacher model for knowledge distillation. The teacher model's output
        is set in the future object for subsequent processing.

        Args:
            future (GenerationBatchFuture): Future object containing generated sequences and metadata

        Returns:
            GenerationBatchFuture: The same future object with teacher knowledge set

        Raises:
            RuntimeError: If teacher client initialization fails or knowledge retrieval fails
        """
        _, _, gen_batch_output = future.get()
        if self.opd_y_mode == "y_r":
            future.set_teacher_batch_output(self._async_get_yr_teacher_knowledge(future, gen_batch_output))
            return future

        gen_batch_output.meta_info["response_length"] = self.config.data.max_response_length

        if self.teacher_backend == "local_hf":
            batch_size = gen_batch_output.batch["input_ids"].size(0)

            def handle_local_teacher():
                return DataProto.from_single_dict(
                    data={"real_seq_lens": torch.zeros(batch_size, dtype=torch.int32)},
                    meta_info={"timing": {"get_teacher_knowledge": 0.0}},
                )

            future.set_teacher_batch_output(SimpleNamespace(get=handle_local_teacher))
            return future

        future.set_teacher_batch_output(
            get_teacher_knowledge(gen_batch_output, self.teacher_client, self.n_server_workers, is_async=True)
        )
        return future

    def one_step_off_scheduler(self, continuous_iterator):
        """One-step-off scheduler implementation (version 1) for GKD training with improved pipeline.

        This scheduler optimizes the training pipeline by:
        1. Overlapping rollout weight synchronization with teacher knowledge processing
        2. Maintaining consistent timing measurement across iterations
        3. Reducing idle time between generation and knowledge distillation phases

        The scheduler maintains the following timing metrics:
        - sync_rollout_weights: Time taken to synchronize rollout weights
        - wait_prev_gen: Time waiting for previous generation to complete
        - wait_prev_teacher: Time waiting for teacher knowledge to be ready

        Args:
            continuous_iterator: Iterator providing (epoch, batch_dict) tuples for training

        Yields:
            tuple: Contains (batch, gen_batch_output, teacher_batch_output, timing_metrics)
                - batch: Original input batch data
                - gen_batch_output: Generated sequences from main model
                - teacher_batch_output: Knowledge distillation from teacher model
                - timing_metrics: Dictionary of timing measurements
        """
        timing = {}
        for i, (epoch, batch_dict) in enumerate(continuous_iterator):
            if i == 0:
                # sync weights and start first async rollout
                fut = self._async_gen_next_batch(epoch, batch_dict, sync_timing=timing)
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)
                # no yield here, so we will continue to the next loop and enter `else` block
            elif i == 1:
                # we don't need to sync weights here because we have not trained the actor yet
                # start second async rollout. If rollout replicas sleep/free cache
                # after generation, sync here also wakes the engine before reuse.
                sync_second_rollout = bool(
                    OmegaConf.select(self.config, "actor_rollout_ref.rollout.free_cache_engine", default=False)
                )
                fut = self._async_gen_next_batch(
                    epoch,
                    batch_dict,
                    sync_before_generation=sync_second_rollout,
                    sync_timing=timing,
                )
                # wait for generating teacher knowledge finish
                # and get previous result including rollout and teacher knowledge
                with marked_timer("wait_prev_teacher", timing):
                    prev_result = prev_fut.get()
                yield *prev_result, timing

                # start next step from here
                timing = {}
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)
            else:
                # sync weights and start next async rollout
                fut = self._async_gen_next_batch(epoch, batch_dict, sync_timing=timing)
                # wait for generating teacher knowledge finish
                # and get previous result including rollout and teacher knowledge
                with marked_timer("wait_prev_teacher", timing):
                    prev_result = prev_fut.get()
                yield *prev_result, timing

                # start next step from here
                timing = {}
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)

        # for last step
        with marked_timer("wait_prev_teacher", timing):
            prev_result = prev_fut.get()
        yield *prev_result, timing

    def two_step_off_scheduler(self, continuous_iterator):
        """Two-step-off scheduler implementation for GKD training with optimized pipeline.

        This scheduler implements a double-buffered pipeline that overlaps:
        1. Sequence generation with teacher knowledge distillation
        2. Weight synchronization with previous batch processing

        Key features:
        - Maintains two parallel processing streams (current and previous batches)
        - Overlaps computation and communication where possible
        - Provides consistent timing metrics across iterations

        Pipeline stages:
        1. Initialization: Start first generation without teacher processing
        2. Steady state: Alternate between processing teacher knowledge and starting new generation
        3. Final state: Process last batch of teacher knowledge

        Timing metrics collected:
        - sync_rollout_weights: Time for weight synchronization between actor and rollout workers
        - wait_prev_prev_teacher: Time waiting for teacher knowledge from two batches ago
        - wait_prev_gen: Time waiting for previous generation to complete

        Args:
            continuous_iterator: Iterator providing (epoch, batch_dict) tuples for training

        Yields:
            tuple: Contains (batch, gen_batch_output, teacher_batch_output, timing_metrics)
                - batch: Original input batch data
                - gen_batch_output: Generated sequences from main model
                - teacher_batch_output: Knowledge distillation from teacher model
                - timing_metrics: Dictionary of timing measurements
        """
        timing = {}
        for i, (epoch, batch_dict) in enumerate(continuous_iterator):
            if i == 0:
                rollout_future = self._async_gen_next_batch(epoch, batch_dict, sync_timing=timing)
                continue
            elif i == 1:
                teacher_future = self._async_get_teacher_knowledge(rollout_future)
                rollout_future = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                continue
            elif i == 2:
                with marked_timer("wait_prev_prev_teacher", timing):
                    result = teacher_future.get()
                with marked_timer("wait_prev_gen", timing):
                    teacher_future = self._async_get_teacher_knowledge(rollout_future)
                rollout_future = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                yield *result, timing
                timing = {}
            else:
                with marked_timer("wait_prev_prev_teacher", timing):
                    result = teacher_future.get()
                with marked_timer("wait_prev_gen", timing):
                    teacher_future = self._async_get_teacher_knowledge(rollout_future)
                rollout_future = self._async_gen_next_batch(epoch, batch_dict, sync_timing=timing)
                yield *result, timing
                timing = {}

        # for second to last step
        with marked_timer("wait_prev_prev_teacher", timing):
            result = teacher_future.get()
        with marked_timer("wait_prev_gen", timing):
            teacher_future = self._async_get_teacher_knowledge(rollout_future)
        yield *result, timing

        # for last step
        with marked_timer("wait_prev_prev_teacher", timing):
            result = teacher_future.get()
        yield *result, timing

    def three_step_off_scheduler(self, continuous_iterator):
        """Sample-streaming three-stage scheduler for y_o/y_r.

        `data.train_batch_size` is the rollout window. Samples in that window
        are dispatched independently to rollout workers, immediately enter
        teacher y_o scoring or y_r generation/scoring when y_o is ready, then
        are packed into actor-sized microbatches for gradient accumulation.
        """
        actor_batch_size, max_inflight_samples, effective_batch, request_chunk_size = (
            self._three_step_streaming_config()
        )
        rollout_batch_size = int(self.config.data.train_batch_size)
        ready_queue: queue.Queue = queue.Queue()
        inflight_items: dict[int, SimpleNamespace] = {}
        ready_samples: list[tuple[SimpleNamespace, SimpleNamespace, float]] = []
        inflight_samples = 0
        next_item_id = 0
        next_window_id = 0
        iterator_exhausted = False
        active_policy_version = int(getattr(self, "_policy_version", 0))
        submitted_samples_this_policy = 0
        yielded_samples_this_policy = 0

        print(
            "[OPD][three_step_off] sample streaming enabled "
            f"rollout_batch_size={rollout_batch_size} actor_batch_size={actor_batch_size} "
            f"effective_batch={effective_batch} max_inflight_samples={max_inflight_samples} "
            f"teacher_request_chunk_size={request_chunk_size}",
            flush=True,
        )

        def refresh_policy_window():
            nonlocal active_policy_version, submitted_samples_this_policy, yielded_samples_this_policy
            current_policy_version = int(getattr(self, "_policy_version", 0))
            if current_policy_version == active_policy_version:
                return
            if inflight_items or ready_samples:
                raise RuntimeError(
                    "three_step_off observed an optimizer step while old-policy samples are still queued: "
                    f"old_policy_version={active_policy_version} train_policy_version={current_policy_version} "
                    f"inflight_samples={len(inflight_items)} ready_samples={len(ready_samples)}. Increase "
                    "EFFECTIVE_BATCH_SIZE or reduce ROLLOUT_BATCH_SIZE to keep the rollout window on-policy."
                )
            if submitted_samples_this_policy != yielded_samples_this_policy:
                raise RuntimeError(
                    "three_step_off policy window accounting mismatch before policy advance: "
                    f"submitted={submitted_samples_this_policy} yielded={yielded_samples_this_policy}"
                )
            active_policy_version = current_policy_version
            submitted_samples_this_policy = 0
            yielded_samples_this_policy = 0

        def can_submit_more():
            refresh_policy_window()
            if iterator_exhausted:
                return False
            if submitted_samples_this_policy + rollout_batch_size > effective_batch:
                return False
            if inflight_samples > 0 and inflight_samples + rollout_batch_size > max_inflight_samples:
                return False
            return True

        def submit_window():
            nonlocal inflight_samples, iterator_exhausted, next_item_id, next_window_id
            nonlocal submitted_samples_this_policy
            try:
                epoch, batch_dict = next(continuous_iterator)
            except StopIteration:
                iterator_exhausted = True
                return False

            batch_size = self._batch_dict_size(batch_dict)
            if batch_size % actor_batch_size != 0:
                raise ValueError(
                    f"three_step_off sample streaming requires rollout batch size {batch_size} to be divisible by "
                    f"actor batch size {actor_batch_size}"
                )
            if submitted_samples_this_policy + batch_size > effective_batch:
                raise ValueError(
                    "three_step_off sample streaming would cross an optimizer-step boundary with one rollout window: "
                    f"submitted_this_policy={submitted_samples_this_policy} batch_size={batch_size} "
                    f"effective_batch={effective_batch}"
                )

            sync_timing = {}
            self.sync_rollout_weights_if_pending(sync_timing)
            window_id = next_window_id
            next_window_id += 1
            policy_version = active_policy_version
            submitted_samples_this_policy += batch_size

            for sample_index in range(batch_size):
                sample_batch_dict = self._slice_batch_dict(batch_dict, sample_index, sample_index + 1, batch_size=batch_size)
                item = SimpleNamespace(
                    item_id=next_item_id,
                    window_id=window_id,
                    sample_index=sample_index,
                    chunk_index=sample_index // actor_batch_size,
                    batch_size=1,
                    rollout_policy_version=policy_version,
                    submitted_at=time.time(),
                    timing=dict(sync_timing) if sample_index == 0 else {},
                )
                worker_index = next_item_id
                next_item_id += 1
                inflight_items[item.item_id] = item
                inflight_samples += 1
                thread = threading.Thread(
                    target=self._run_three_step_streaming_sample,
                    args=(item, epoch, sample_batch_dict, ready_queue, request_chunk_size, worker_index),
                    daemon=True,
                )
                thread.start()
            return True

        def build_actor_pack():
            nonlocal yielded_samples_this_policy
            if len(ready_samples) < actor_batch_size:
                return None
            packed_records = ready_samples[:actor_batch_size]
            del ready_samples[:actor_batch_size]
            yielded_samples_this_policy += actor_batch_size
            result = self._pack_three_step_streaming_samples(
                packed_records,
                actor_batch_size=actor_batch_size,
                inflight_samples=inflight_samples,
                ready_queue_size=ready_queue.qsize(),
                submitted_samples_this_policy=submitted_samples_this_policy,
                yielded_samples_this_policy=yielded_samples_this_policy,
            )
            return result

        while can_submit_more():
            submit_window()

        while inflight_items or ready_samples or not iterator_exhausted:
            while can_submit_more():
                submit_window()

            packed = build_actor_pack()
            if packed is not None:
                yield packed
                continue

            if not inflight_items:
                refresh_policy_window()
                if iterator_exhausted:
                    if ready_samples:
                        raise RuntimeError(
                            f"three_step_off ended with {len(ready_samples)} leftover ready samples, "
                            f"actor_batch_size={actor_batch_size}"
                        )
                    return
                if submitted_samples_this_policy >= effective_batch:
                    raise RuntimeError(
                        "three_step_off is waiting for an optimizer step before submitting more rollout work, "
                        "but policy_version did not advance. Check EFFECTIVE_BATCH_SIZE/TRAIN_BATCH_SIZE and "
                        "actor/optimizer_stepped metrics."
                    )
                continue

            wait_tik = time.time()
            try:
                item, sample_result, error = ready_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            wait_ready = time.time() - wait_tik

            inflight_items.pop(item.item_id, None)
            inflight_samples -= item.batch_size
            if error is not None:
                raise RuntimeError(
                    f"three_step_off sample window={item.window_id} sample={item.sample_index} failed"
                ) from error
            if sample_result is None:
                raise RuntimeError(
                    f"three_step_off sample window={item.window_id} sample={item.sample_index} returned no result"
                )

            train_policy_version = int(getattr(self, "_policy_version", 0))
            policy_lag = train_policy_version - int(item.rollout_policy_version)
            if policy_lag != 0:
                raise RuntimeError(
                    "three_step_off sample streaming is configured as on-policy but consumed a stale sample: "
                    f"window={item.window_id} sample={item.sample_index} "
                    f"rollout_policy_version={item.rollout_policy_version} "
                    f"train_policy_version={train_policy_version} policy_lag={policy_lag}. "
                    "Increase EFFECTIVE_BATCH_SIZE so one optimizer step covers the full rollout window."
                )

            item.ready_at = time.time()
            item.wait_ready = wait_ready
            item.train_policy_version = train_policy_version
            item.policy_lag = policy_lag
            ready_samples.append((item, sample_result, wait_ready))

    @staticmethod
    def _batch_dict_size(batch_dict: dict) -> int:
        for value in batch_dict.values():
            if isinstance(value, torch.Tensor):
                return int(value.size(0))
            if isinstance(value, np.ndarray):
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        return 1

    @staticmethod
    def _slice_batch_dict(batch_dict: dict, start: int, end: int, *, batch_size: int | None = None) -> dict:
        if batch_size is None:
            batch_size = OnPolicyDistillTrainer._batch_dict_size(batch_dict)
        sliced = {}
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor) and value.size(0) == batch_size:
                sliced[key] = value[start:end]
            elif isinstance(value, np.ndarray) and value.shape[0] == batch_size:
                sliced[key] = value[start:end]
            elif isinstance(value, list) and len(value) == batch_size:
                sliced[key] = value[start:end]
            elif isinstance(value, tuple) and len(value) == batch_size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _three_step_streaming_config(self) -> tuple[int, int, int, int]:
        actor_batch_size = self._actor_update_batch_size() or int(self.config.data.train_batch_size)
        rollout_batch_size = int(self.config.data.train_batch_size)
        if rollout_batch_size % actor_batch_size != 0:
            raise ValueError(
                f"three_step_off streaming requires data.train_batch_size={rollout_batch_size} "
                f"to be divisible by trainer.actor_update_batch_size={actor_batch_size}"
            )
        gradient_accumulation_steps = max(
            1,
            int(OmegaConf.select(self.config, "actor_rollout_ref.actor.gradient_accumulation_steps", default=1)),
        )
        effective_batch = actor_batch_size * gradient_accumulation_steps
        if rollout_batch_size > actor_batch_size and effective_batch % rollout_batch_size != 0:
            raise ValueError(
                "three_step_off streaming keeps samples on-policy, so EFFECTIVE_BATCH_SIZE must be a multiple "
                "of ROLLOUT_BATCH_SIZE when ROLLOUT_BATCH_SIZE is larger than the actor batch. Got "
                f"effective_batch={effective_batch}, rollout_batch_size={rollout_batch_size}, "
                f"actor_batch_size={actor_batch_size}."
            )
        max_inflight_samples = int(OmegaConf.select(self.config, "trainer.teacher_inflight_samples", default=0) or 0)
        if max_inflight_samples <= 0:
            max_inflight_samples = rollout_batch_size
        max_inflight_samples = max(max_inflight_samples, rollout_batch_size)
        request_chunk_size = self._bounded_lag_teacher_request_chunk_size()
        return max(1, actor_batch_size), max(1, max_inflight_samples), max(1, effective_batch), request_chunk_size

    @staticmethod
    def _merge_stream_timing(sample_records: list[tuple[SimpleNamespace, SimpleNamespace, float]]) -> dict:
        values: dict[str, list[float]] = {}
        wait_values = []
        e2e_values = []
        for item, sample, wait_ready in sample_records:
            wait_values.append(float(wait_ready))
            if hasattr(item, "completed_at"):
                e2e_values.append(float(item.completed_at - item.submitted_at))
            for key, value in getattr(sample, "timing", {}).items():
                if isinstance(value, (int, float, np.number)):
                    values.setdefault(key, []).append(float(value))
        timing = {}
        for key, vals in values.items():
            timing[key] = max(vals)
            if len(vals) > 1:
                timing[f"{key}_mean"] = float(np.mean(vals))
                timing[f"{key}_min"] = float(np.min(vals))
                timing[f"{key}_max"] = float(np.max(vals))
        if wait_values:
            timing["stream_wait_ready_sample"] = max(wait_values)
            timing["stream_wait_ready_sample_mean"] = float(np.mean(wait_values))
        if e2e_values:
            timing["stream_sample_e2e"] = max(e2e_values)
            timing["stream_sample_e2e_mean"] = float(np.mean(e2e_values))
        return timing

    def _pack_three_step_streaming_samples(
        self,
        sample_records: list[tuple[SimpleNamespace, SimpleNamespace, float]],
        *,
        actor_batch_size: int,
        inflight_samples: int,
        ready_queue_size: int,
        submitted_samples_this_policy: int,
        yielded_samples_this_policy: int,
    ):
        items = [record[0] for record in sample_records]
        samples = [record[1] for record in sample_records]
        epoch = samples[0].epoch
        batch = DataProto.concat([sample.batch for sample in samples])
        timing = self._merge_stream_timing(sample_records)

        if self.opd_y_mode == "y_r":
            prompt_batch = DataProto.concat([sample.prompt_batch for sample in samples])
            responses = [sample.response for sample in samples]
            teacher_logps = [sample.teacher_logps for sample in samples]
            teacher_indices = [sample.teacher_indices for sample in samples]
            gen_batch_output, teacher_batch_output = self._build_yr_actor_batches(
                prompt_batch,
                responses,
                teacher_logps,
                teacher_indices,
                {},
            )
        else:
            gen_outputs = [sample.gen_batch_output for sample in samples]
            teacher_outputs = [sample.teacher_batch_output for sample in samples]
            # Per-sample outputs carry distinct timing metadata; the scheduler
            # returns merged timing separately, so clear meta_info before concat.
            for output in gen_outputs + teacher_outputs:
                output.meta_info = {}
            gen_batch_output = DataProto.concat(gen_outputs)
            teacher_batch_output = DataProto.concat(teacher_outputs)
            gen_batch_output.meta_info["response_length"] = self.config.data.max_response_length

        gen_batch_output.meta_info["timing"] = {}
        teacher_batch_output.meta_info["timing"] = {}
        train_policy_version = int(getattr(self, "_policy_version", 0))
        policy_lags = [int(item.policy_lag) for item in items]
        teacher_metrics = dict(teacher_batch_output.meta_info.get("metrics", {}))
        teacher_metrics.update(
            {
                "three_step_stream/sample_level": 1,
                "three_step_stream/packed_samples": actor_batch_size,
                "three_step_stream/window_id_min": min(int(item.window_id) for item in items),
                "three_step_stream/window_id_max": max(int(item.window_id) for item in items),
                "three_step_stream/sample_index_min": min(int(item.sample_index) for item in items),
                "three_step_stream/sample_index_max": max(int(item.sample_index) for item in items),
                "three_step_stream/rollout_policy_version": int(items[0].rollout_policy_version),
                "three_step_stream/train_policy_version": train_policy_version,
                "three_step_stream/policy_lag": max(policy_lags),
                "three_step_stream/inflight_samples": inflight_samples,
                "three_step_stream/ready_queue_size": ready_queue_size,
                "three_step_stream/submitted_samples_this_policy": submitted_samples_this_policy,
                "three_step_stream/yielded_samples_this_policy": yielded_samples_this_policy,
            }
        )
        teacher_batch_output.meta_info["metrics"] = teacher_metrics
        print(
            "[OPD][three_step_off] yield sample-stream actor batch "
            f"mode={self.opd_y_mode} samples={[(item.window_id, item.sample_index) for item in items]} "
            f"rollout_policy_version={items[0].rollout_policy_version} "
            f"train_policy_version={train_policy_version} inflight_samples={inflight_samples} "
            f"ready_queue={ready_queue_size}",
            flush=True,
        )
        return epoch, batch, gen_batch_output, teacher_batch_output, timing

    def _score_y_o_raw_sample(
        self,
        epoch: int,
        batch: DataProto,
        prompt_batch: DataProto,
        y_o_output: DataProto,
    ) -> SimpleNamespace:
        future = GenerationBatchFuture(epoch, batch, y_o_output, prompt_batch=prompt_batch)
        future = self._async_get_teacher_knowledge(future)
        result = future.get()
        if len(result) != 4:
            raise RuntimeError("y_o teacher scoring did not return teacher output")
        result_epoch, result_batch, gen_batch_output, teacher_batch_output = result
        if teacher_batch_output is None:
            raise RuntimeError("y_o teacher scoring returned no teacher output")

        teacher_timing = teacher_batch_output.meta_info.get("timing", {})
        timing = {}
        for key, value in teacher_timing.items():
            if isinstance(value, (int, float, np.number)):
                timing[f"teacher_y_o/{key}"] = float(value)

        return SimpleNamespace(
            epoch=result_epoch,
            batch=result_batch,
            prompt_batch=prompt_batch,
            gen_batch_output=gen_batch_output,
            teacher_batch_output=teacher_batch_output,
            timing=timing,
        )

    def _score_y_r_raw_sample(
        self,
        epoch: int,
        batch: DataProto,
        prompt_batch: DataProto,
        y_o_output: DataProto,
        request_chunk_size: int,
    ) -> SimpleNamespace:
        temperature = float(self.opd_config.get("temperature", self.teacher_config.get("temperature", 1.0)))
        max_tokens = int(self.config.data.max_response_length)
        teacher_prompt_ids = self._build_yr_generation_prompts(batch, y_o_output)
        gen_started_at = time.time()
        gen_futures = self._submit_teacher_requests(
            teacher_prompt_ids,
            max_tokens=max_tokens,
            only_response=True,
            temperature=temperature,
            request_chunk_size=1,
        )
        responses, gen_logps, gen_indices = self._collect_teacher_futures(gen_futures)
        gen_finished_at = time.time()

        teacher_prompt = str(self.opd_config.get("teacher_training_prompt", "refine")).lower()
        if teacher_prompt not in ("refine", "vanilla"):
            raise ValueError("opd.teacher_training_prompt must be refine or vanilla")

        if teacher_prompt == "vanilla":
            score_prompt_ids, prompt_lens, response_lens = self._build_yr_scoring_prompts(
                batch,
                y_o_output,
                responses,
                use_initial_response=False,
            )
            score_row_indices = self._scored_response_row_indices(prompt_lens, response_lens)
            score_started_at = time.time()
            score_futures = self._submit_teacher_requests(
                score_prompt_ids,
                max_tokens=1,
                only_response=False,
                temperature=temperature,
                logprob_row_indices=score_row_indices,
                request_chunk_size=1,
            )
            _, score_logps, score_indices = self._collect_teacher_futures(score_futures)
            if all(logps.size(0) == response_len for logps, response_len in zip(score_logps, response_lens)):
                teacher_logps = score_logps
                teacher_indices = score_indices
            else:
                teacher_logps = self._slice_scored_response_topk(score_logps, prompt_lens, response_lens)
                teacher_indices = self._slice_scored_response_topk(score_indices, prompt_lens, response_lens)
            timing = {
                "generate_y_r": gen_finished_at - gen_started_at,
                "score_y_r": time.time() - score_started_at,
            }
        else:
            teacher_logps = gen_logps
            teacher_indices = gen_indices
            timing = {"generate_y_r_and_teacher_topk": gen_finished_at - gen_started_at}

        return SimpleNamespace(
            epoch=epoch,
            batch=batch,
            prompt_batch=prompt_batch,
            response=responses[0],
            teacher_logps=teacher_logps[0],
            teacher_indices=teacher_indices[0] if teacher_indices else None,
            timing=timing,
        )

    def _run_three_step_streaming_sample(
        self,
        item: SimpleNamespace,
        epoch: int,
        batch_dict: dict,
        ready_queue: queue.Queue,
        request_chunk_size: int,
        rollout_worker_index: int,
    ):
        timing = item.timing
        try:
            with marked_timer("stream_generate_y_o", timing):
                batch, prompt_batch, y_o_output, rollout_timing = self._gen_single_sample_on_rollout_worker(
                    epoch,
                    batch_dict,
                    rollout_worker_index,
                )
            for key, value in rollout_timing.items():
                if isinstance(value, (int, float, np.number)):
                    timing[f"stream_y_o/{key}"] = float(value)

            if self.opd_y_mode == "y_r":
                with marked_timer("stream_teacher_y_r", timing):
                    sample = self._score_y_r_raw_sample(
                        epoch,
                        batch,
                        prompt_batch,
                        y_o_output,
                        request_chunk_size,
                    )
            else:
                with marked_timer("stream_teacher_y_o", timing):
                    sample = self._score_y_o_raw_sample(
                        epoch,
                        batch,
                        prompt_batch,
                        y_o_output,
                    )
            timing.update(sample.timing)
            sample.timing = dict(timing)
            item.completed_at = time.time()
            ready_queue.put((item, sample, None))
        except Exception as exc:
            item.completed_at = time.time()
            ready_queue.put((item, None, exc))

    def _bounded_lag_config(self) -> tuple[int, int, int]:
        actor_batch_size = self._actor_update_batch_size() or int(self.config.data.train_batch_size)
        max_inflight_samples = int(OmegaConf.select(self.config, "trainer.teacher_inflight_samples", default=0) or 0)
        if max_inflight_samples <= 0:
            max_inflight_samples = max(actor_batch_size, int(self.config.data.train_batch_size))
        max_policy_lag = int(OmegaConf.select(self.config, "trainer.max_policy_lag", default=0) or 0)
        if max_policy_lag < 0:
            max_policy_lag = 0
        effective_batch = actor_batch_size * max(
            1, int(OmegaConf.select(self.config, "actor_rollout_ref.actor.gradient_accumulation_steps", default=1))
        )
        return max(1, max_inflight_samples), max_policy_lag, max(1, effective_batch)

    def _run_bounded_lag_item(
        self,
        item: SimpleNamespace,
        epoch: int,
        batch_dict: dict,
        ready_queue: queue.Queue,
        request_chunk_size: int,
    ):
        timing = item.timing
        try:
            with marked_timer("bounded_generate_y_o", timing):
                rollout_future = self._async_gen_next_batch(
                    epoch,
                    batch_dict,
                    sync_before_generation=False,
                )

            if self.opd_y_mode == "y_r":
                with marked_timer("bounded_submit_y_r", timing):
                    y_r_generation_future = self._async_generate_y_r(
                        rollout_future,
                        request_chunk_size=request_chunk_size,
                    )
                with marked_timer("bounded_wait_y_r_score", timing):
                    y_r_score_future = self._async_score_y_r_generation(
                        y_r_generation_future,
                        return_full_result=True,
                        request_chunk_size=request_chunk_size,
                    )
                    result = y_r_score_future.get()
            else:
                with marked_timer("bounded_wait_y_o_score", timing):
                    y_o_score_future = self._async_get_teacher_knowledge(rollout_future)
                    result = y_o_score_future.get()

            item.completed_at = time.time()
            ready_queue.put((item, result, None))
        except Exception as exc:
            item.completed_at = time.time()
            ready_queue.put((item, None, exc))

    def bounded_lag_y_r_scheduler(self, continuous_iterator):
        """Bounded-lag streaming scheduler for y_o/y_r.

        This keeps many y_o -> teacher jobs in flight and yields whichever
        batch becomes ready first. Rollout samples carry the policy version used
        to generate y_o, while actor updates log the policy version that consumed
        them. `trainer.max_policy_lag` is used as backpressure on how many
        samples from one policy version may be queued at once.
        """
        max_inflight_samples, max_policy_lag, effective_batch = self._bounded_lag_config()
        request_chunk_size = self._bounded_lag_teacher_request_chunk_size()
        per_version_sample_cap = max(1, (max_policy_lag + 1) * effective_batch)
        ready_queue: queue.Queue = queue.Queue()
        inflight_items: dict[int, SimpleNamespace] = {}
        inflight_samples = 0
        inflight_samples_by_version: dict[int, int] = {}
        next_item_id = 0
        iterator_exhausted = False

        print(
            "[OPD][bounded_lag_y_r] enabled "
            f"max_inflight_samples={max_inflight_samples} max_policy_lag={max_policy_lag} "
            f"effective_batch={effective_batch} per_version_sample_cap={per_version_sample_cap} "
            f"teacher_request_chunk_size={request_chunk_size}",
            flush=True,
        )

        def can_submit_more():
            if iterator_exhausted:
                return False
            policy_version = int(getattr(self, "_policy_version", 0))
            version_samples = inflight_samples_by_version.get(policy_version, 0)
            return inflight_samples < max_inflight_samples and version_samples < per_version_sample_cap

        def submit_one():
            nonlocal inflight_samples, iterator_exhausted, next_item_id
            try:
                epoch, batch_dict = next(continuous_iterator)
            except StopIteration:
                iterator_exhausted = True
                return False

            policy_version = int(getattr(self, "_policy_version", 0))
            batch_size = self._batch_dict_size(batch_dict)
            timing = {}
            item = SimpleNamespace(
                item_id=next_item_id,
                epoch=epoch,
                batch_size=batch_size,
                rollout_policy_version=policy_version,
                submitted_at=time.time(),
                timing=timing,
            )
            next_item_id += 1
            self.sync_rollout_weights_if_pending(timing)
            inflight_items[item.item_id] = item
            inflight_samples += batch_size
            inflight_samples_by_version[policy_version] = (
                inflight_samples_by_version.get(policy_version, 0) + batch_size
            )
            thread = threading.Thread(
                target=self._run_bounded_lag_item,
                args=(item, epoch, batch_dict, ready_queue, request_chunk_size),
                daemon=True,
            )
            thread.start()
            return True

        while can_submit_more():
            submit_one()

        while inflight_items or not iterator_exhausted:
            while can_submit_more():
                submit_one()

            wait_tik = time.time()
            try:
                item, result, error = ready_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            wait_ready = time.time() - wait_tik

            inflight_items.pop(item.item_id, None)
            inflight_samples -= item.batch_size
            rollout_version_samples = inflight_samples_by_version.get(item.rollout_policy_version, 0)
            rollout_version_samples -= item.batch_size
            if rollout_version_samples > 0:
                inflight_samples_by_version[item.rollout_policy_version] = rollout_version_samples
            else:
                inflight_samples_by_version.pop(item.rollout_policy_version, None)

            if error is not None:
                raise RuntimeError(f"bounded_lag_y_r item {item.item_id} failed") from error
            if result is None:
                raise RuntimeError(f"bounded_lag_y_r item {item.item_id} returned no result")

            train_policy_version = int(getattr(self, "_policy_version", 0))
            policy_lag = train_policy_version - int(item.rollout_policy_version)
            if policy_lag > max_policy_lag:
                print(
                    "[OPD][bounded_lag_y_r] drop stale item "
                    f"item={item.item_id} rollout_policy_version={item.rollout_policy_version} "
                    f"train_policy_version={train_policy_version} policy_lag={policy_lag} "
                    f"max_policy_lag={max_policy_lag}",
                    flush=True,
                )
                continue

            timing = dict(item.timing)
            timing["bounded_wait_ready_actor_batch"] = wait_ready
            timing["bounded_e2e"] = item.completed_at - item.submitted_at
            epoch, batch, gen_batch_output, teacher_batch_output = result
            teacher_metrics = dict(teacher_batch_output.meta_info.get("metrics", {}))
            teacher_metrics.update(
                {
                    "bounded_lag/item_id": item.item_id,
                    "bounded_lag/rollout_policy_version": item.rollout_policy_version,
                    "bounded_lag/train_policy_version": train_policy_version,
                    "bounded_lag/policy_lag": policy_lag,
                    "bounded_lag/inflight_samples": inflight_samples,
                    "bounded_lag/ready_queue_size": ready_queue.qsize(),
                }
            )
            teacher_batch_output.meta_info["metrics"] = teacher_metrics
            print(
                "[OPD][bounded_lag_y_r] yield "
                f"item={item.item_id} batch={item.batch_size} "
                f"rollout_policy_version={item.rollout_policy_version} "
                f"train_policy_version={train_policy_version} policy_lag={policy_lag} "
                f"inflight_samples={inflight_samples} ready_queue={ready_queue.qsize()}",
                flush=True,
            )
            yield epoch, batch, gen_batch_output, teacher_batch_output, timing

    def one_step_scheduler(self, continuous_iterator):
        """Synchronous full-batch scheduler for OPD one-step optimization.

        This keeps the official actor/rollout/teacher/update interfaces, but
        does not pipeline across batches. With trainer.optimization_mode=one_step
        the dataloader contains exactly one full-dataset batch, so the yielded
        item performs: sync weights -> rollout y_o -> teacher signal -> actor
        update.
        """
        for epoch, batch_dict in continuous_iterator:
            batch_dict = self._pad_batch_dict_for_one_step(batch_dict)
            timing = {}
            future = self._async_gen_next_batch(epoch, batch_dict, sync_timing=timing)
            with marked_timer("wait_teacher", timing):
                future = self._async_get_teacher_knowledge(future)
                result = future.get()
            yield *result, timing

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        scheduler_type = self._resolve_scheduler_type()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._policy_version = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        max_steps_duration = 0

        # Pre-warm: submit the first rollout
        continuous_iterator = self._create_continuous_iterator()

        if scheduler_type == "one_step":
            scheduler = self.one_step_scheduler(continuous_iterator)
        elif scheduler_type == "one_step_off":
            scheduler = self.one_step_off_scheduler(continuous_iterator)
        elif scheduler_type == "two_step_off":
            scheduler = self.two_step_off_scheduler(continuous_iterator)
        elif scheduler_type == "three_step_off":
            scheduler = self.three_step_off_scheduler(continuous_iterator)
        elif scheduler_type == "bounded_lag_y_r":
            scheduler = self.bounded_lag_y_r_scheduler(continuous_iterator)
        else:
            raise TypeError(f"unrecognized scheduler type: {scheduler_type}")

        # Main loop
        while True:
            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )
            if do_profile:
                self.async_rollout_manager.start_profile(global_step=self.global_steps)
                self.actor_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                _, batch, gen_batch_output, teacher_batch_output, schedule_timing = next(scheduler)
                if teacher_batch_output is None:
                    raise RuntimeError("Teacher knowledge request failed; aborting instead of skipping the batch.")

                timing_raw.update(schedule_timing)

                gen_timing = gen_batch_output.meta_info.pop("timing", {})
                for k, v in gen_timing.items():
                    if isinstance(v, list):
                        array_v = np.array(v)
                        timing_raw[k + "_mean"] = array_v.mean().item()
                        timing_raw[k + "_min"] = array_v.min().item()
                        timing_raw[k + "_max"] = array_v.max().item()
                        timing_raw[k] = array_v.max().item()
                    else:
                        timing_raw[k] = v

                teacher_metrics = teacher_batch_output.meta_info.pop("metrics", {})
                if teacher_metrics:
                    metrics.update(teacher_metrics)
                timing_raw.update(teacher_batch_output.meta_info.pop("timing"))

                # Compute statistics of generated response lengths distribution
                response_lens_tensor = (
                    (gen_batch_output.batch["responses"] != self.tokenizer.pad_token_id).sum(dim=-1)
                )
                if batch.batch is not None and "is_padded_sample" in batch.batch.keys():
                    valid_rows = ~batch.batch["is_padded_sample"].to(torch.bool)
                    response_lens_tensor = response_lens_tensor[valid_rows]
                response_lens = response_lens_tensor.tolist()
                if not response_lens:
                    response_lens = [0]
                metrics.update(
                    {
                        "response_seq_len/average": sum(response_lens) / len(response_lens),
                        "response_seq_len/max": max(response_lens),
                        "response_seq_len/min": min(response_lens),
                        "response_seq_len/max_count": response_lens.count(max(response_lens)),
                        "response_seq_len/min_count": response_lens.count(min(response_lens)),
                    }
                )

                # Merge generated outputs back
                batch = batch.union(gen_batch_output)

                # Debug print
                one_attention_mask = batch.batch["attention_mask"][0].to(torch.bool)
                one_sentence = batch.batch["input_ids"][0]
                print("INFO:", "generate text done.")
                print("DEBUG:", self.tokenizer.decode(one_sentence[one_attention_mask].tolist()))

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                batch = batch.union(teacher_batch_output)
                self._apply_sample_padding_loss_mask(batch)

                # # update actor
                # with marked_timer("send_teacher_knowledge", timing_raw, color="red"):
                #     self.actor_wg.send_teacher_knowledge(teacher_batch_output)

                # update actor
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output_metrics = self._update_actor_with_microbatches(batch, timing_raw)

                print("INFO:", "update actor done.")
                metrics.update(actor_output_metrics)

                # Mark rollout sync pending only when optimizer.step actually
                # ran this iteration (skipped during gradient accumulation).
                # Saves ~135s of useless NCCL broadcast per non-stepping iter.
                if actor_output_metrics.get("actor/optimizer_stepped", 1):
                    self._pending_rollout_sync = True
                    self._policy_version += 1

                # save model
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            # Metrics and bookkeeping
            steps_duration = timing_raw["step"]
            max_steps_duration = max(max_steps_duration, steps_duration)
            # training metrics
            metrics["training/global_step"] = self.global_steps
            # collect metrics
            # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if do_profile:
                self.async_rollout_manager.stop_profile()
                self.actor_wg.stop_profile()

            if is_last_step:
                progress_bar.close()
                return
