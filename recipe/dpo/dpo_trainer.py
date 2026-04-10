# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import time
from collections import defaultdict
from pprint import pprint
from typing import Optional

import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from recipe.dpo.batching import (
    build_dpo_update_proto,
    build_pair_log_prob_batch,
    build_prospect_dpo_update_proto,
    build_point_log_prob_batch,
    build_single_wise_dpo_update_proto,
    compute_sequence_log_probs,
)
from recipe.dpo.core_algos import (
    compute_dpo_loss,
    compute_prospect_dpo_loss,
    compute_single_wise_dpo_loss,
    is_pointwise_dpo_loss,
    is_prospect_dpo_loss,
    is_single_wise_dpo_loss,
    requires_reference_model,
    use_average_sequence_log_probs,
)
from verl.trainer.main_ppo import create_rl_sampler
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.tracking import Tracking


class RayDPOTrainer:
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.loss_type = self.config.algorithm.get("dpo_loss_type", "sigmoid")
        self.prospect_dpo_enabled = is_prospect_dpo_loss(self.loss_type)
        self.single_wise_dpo_enabled = is_single_wise_dpo_loss(self.loss_type)
        self.pointwise_dpo_enabled = is_pointwise_dpo_loss(self.loss_type)
        if self.pointwise_dpo_enabled and self.config.algorithm.get("reference_free", False):
            raise ValueError("Point-wise offline DPO variants require algorithm.reference_free=false")
        self.requires_reference_model = requires_reference_model(
            self.loss_type,
            self.config.algorithm.get("reference_free", False),
        )
        self.pointwise_precomputed_reference_logps = (
            self.pointwise_dpo_enabled and self._datasets_have_precomputed_point_reference_logps(train_dataset, val_dataset)
        )
        self.use_reference_policy = self.requires_reference_model and not self.pointwise_precomputed_reference_logps
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        self.device_name = self.config.trainer.device

        self.actor_wg = None
        self.ref_policy_wg = None
        self.global_steps = 0
        self.max_steps_duration = 0.0

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    @staticmethod
    def _dataset_has_precomputed_point_reference_logps(dataset: Optional[Dataset]) -> bool:
        return bool(dataset is not None and hasattr(dataset, "has_reference_logps") and dataset.has_reference_logps())

    def _datasets_have_precomputed_point_reference_logps(
        self,
        train_dataset: Optional[Dataset],
        val_dataset: Optional[Dataset],
    ) -> bool:
        if not self.pointwise_dpo_enabled:
            return False
        if not self._dataset_has_precomputed_point_reference_logps(train_dataset):
            return False
        if val_dataset is not None and not self._dataset_has_precomputed_point_reference_logps(val_dataset):
            return False
        return True

    def _use_average_log_prob(self) -> bool:
        return use_average_sequence_log_probs(self.loss_type)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        if train_dataset is None:
            raise ValueError("train_dataset must be provided for RayDPOTrainer")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 0),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataloader = None
        if self.val_dataset is not None:
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.data.get("val_batch_size", self.config.data.train_batch_size),
                num_workers=self.config.data.get("dataloader_num_workers", 0),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
            )

        if len(self.train_dataloader) < 1:
            raise ValueError("Train dataloader is empty for offline DPO")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as exc:
            print(f"Warning: failed to propagate total_training_steps into actor config: {exc}")

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role=str(Role.Actor),
        )
        resource_pool_to_cls[actor_pool][str(Role.Actor)] = actor_cls

        if self.use_reference_policy and not self.ref_in_actor:
            ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            resource_pool_to_cls[ref_pool][str(Role.RefPolicy)] = ref_cls

        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        for resource_pool, class_dict in resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        self.actor_wg = all_wg[str(Role.Actor)]
        self.actor_wg.init_model()

        if self.use_reference_policy:
            if self.ref_in_actor:
                self.ref_policy_wg = self.actor_wg
            else:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()

    def _save_checkpoint(self):
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        self.actor_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=self.config.trainer.get("max_actor_ckpt_to_keep", None),
        )

        os.makedirs(local_global_step_folder, exist_ok=True)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, "data.pt"))
        with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
            if global_step_folder is None:
                print("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        else:
            raise ValueError(f"Unsupported resume_mode: {self.config.trainer.resume_mode}")

        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Load from checkpoint folder: {global_step_folder}")

        self.actor_wg.load_checkpoint(
            os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        dataloader_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_path):
            self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))

    def _compute_reference_log_probs(self, batch: DataProto) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.use_reference_policy:
            return None, None

        log_prob_batch, response_mask = build_pair_log_prob_batch(batch)
        ref_output = self.ref_policy_wg.compute_ref_log_prob(log_prob_batch)
        seq_logps = compute_sequence_log_probs(
            ref_output.batch["ref_log_prob"],
            response_mask,
            average_log_prob=self._use_average_log_prob(),
        )
        batch_size = batch.batch["chosen_input_ids"].shape[0]
        return seq_logps[:batch_size], seq_logps[batch_size:]

    def _compute_point_reference_log_probs(self, batch: DataProto) -> torch.Tensor:
        if not self.use_reference_policy:
            raise ValueError("Point-wise DPO requires a reference policy")

        log_prob_batch, response_mask = build_point_log_prob_batch(batch)
        ref_output = self.ref_policy_wg.compute_ref_log_prob(log_prob_batch)
        return compute_sequence_log_probs(
            ref_output.batch["ref_log_prob"],
            response_mask,
            average_log_prob=False,
        )

    def _get_point_reference_log_probs(self, batch: DataProto) -> torch.Tensor:
        if "reference_logps" in batch.batch:
            return batch.batch["reference_logps"].float()
        return self._compute_point_reference_log_probs(batch)

    def _compute_policy_log_probs(self, batch: DataProto) -> tuple[torch.Tensor, torch.Tensor]:
        log_prob_batch, response_mask = build_pair_log_prob_batch(batch)
        policy_output = self.actor_wg.compute_log_prob(log_prob_batch)
        seq_logps = compute_sequence_log_probs(
            policy_output.batch["old_log_probs"],
            response_mask,
            average_log_prob=self._use_average_log_prob(),
        )
        batch_size = batch.batch["chosen_input_ids"].shape[0]
        return seq_logps[:batch_size], seq_logps[batch_size:]

    def _compute_point_policy_log_probs(self, batch: DataProto) -> torch.Tensor:
        log_prob_batch, response_mask = build_point_log_prob_batch(batch)
        policy_output = self.actor_wg.compute_log_prob(log_prob_batch)
        return compute_sequence_log_probs(
            policy_output.batch["old_log_probs"],
            response_mask,
            average_log_prob=False,
        )

    def _validate(self) -> dict[str, float]:
        if self.val_dataloader is None:
            return {}

        metrics: dict[str, list[float]] = defaultdict(list)
        for batch_dict in self.val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            if self.pointwise_dpo_enabled:
                reference_logps = self._get_point_reference_log_probs(batch)
                policy_logps = self._compute_point_policy_log_probs(batch)
                if self.prospect_dpo_enabled:
                    loss, stats = compute_prospect_dpo_loss(
                        policy_logps=policy_logps,
                        reference_logps=reference_logps,
                        labels=batch.batch["label"],
                        beta=self.config.algorithm.dpo_beta,
                        s_dwell=batch.batch["s_dwell"],
                        p_ctr=batch.batch["p_ctr"],
                        alpha_tau=self.config.algorithm.get("prospect_dpo_alpha_tau", 0.2),
                        alpha_k=self.config.algorithm.get("prospect_dpo_alpha_k", 10.0),
                        lambda_max=self.config.algorithm.get("prospect_dpo_lambda_max", 2.0),
                        lambda_gamma=self.config.algorithm.get("prospect_dpo_lambda_gamma", 2.0),
                    )
                    metrics["val/prospect_dpo_loss_pos"].append(stats["positive_loss"].item())
                    metrics["val/prospect_dpo_loss_neg"].append(stats["negative_loss"].item())
                    metrics["val/prospect_dpo_alpha"].append(stats["alpha"].item())
                    metrics["val/prospect_dpo_lambda"].append(stats["lambda"].item())
                    metrics["val/prospect_dpo_positive_fraction"].append(stats["positive_fraction"].item())
                else:
                    loss, stats = compute_single_wise_dpo_loss(
                        policy_logps=policy_logps,
                        reference_logps=reference_logps,
                        labels=batch.batch["label"],
                        beta=self.config.algorithm.dpo_beta,
                    )
                    metrics["val/single_wise_dpo_loss_pos"].append(stats["positive_loss"].item())
                    metrics["val/single_wise_dpo_loss_neg"].append(stats["negative_loss"].item())
                    metrics["val/single_wise_dpo_positive_fraction"].append(stats["positive_fraction"].item())
            else:
                ref_chosen, ref_rejected = self._compute_reference_log_probs(batch)
                policy_chosen, policy_rejected = self._compute_policy_log_probs(batch)
                loss, stats = compute_dpo_loss(
                    policy_chosen_logps=policy_chosen,
                    policy_rejected_logps=policy_rejected,
                    reference_chosen_logps=ref_chosen,
                    reference_rejected_logps=ref_rejected,
                    beta=self.config.algorithm.dpo_beta,
                    label_smoothing=self.config.algorithm.get("dpo_label_smoothing", 0.0),
                    loss_type=self.loss_type,
                    reference_free=self.config.algorithm.get("reference_free", False),
                    simpo_gamma=self.config.algorithm.get("simpo_gamma", 0.5),
                )
            metrics["val/dpo_loss"].append(loss.item())
            metrics["val/dpo_accuracy"].append(stats["accuracy"].item())
            metrics["val/dpo_margin"].append(stats["margin"].item())

        return {key: sum(values) / len(values) for key, values in metrics.items() if values}

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self._load_checkpoint()

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        steps_per_epoch = max(len(self.train_dataloader), 1)
        current_epoch = self.global_steps // steps_per_epoch
        save_freq_steps = self.config.trainer.get("save_freq", -1)
        save_freq_epochs = self.config.trainer.get("save_freq_epochs", -1)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            epoch_start_step = epoch * steps_per_epoch
            remaining_training_steps = max(self.total_training_steps - epoch_start_step, 0)
            epoch_total_steps = min(steps_per_epoch, remaining_training_steps)
            if epoch_total_steps <= 0:
                break

            resumed_epoch_steps = 0
            if epoch == current_epoch:
                resumed_epoch_steps = min(max(self.global_steps - epoch_start_step, 0), epoch_total_steps)

            progress_bar = tqdm(
                total=epoch_total_steps,
                initial=resumed_epoch_steps,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            )
            for batch_idx, batch_dict in enumerate(self.train_dataloader, start=1):
                if self.global_steps >= self.total_training_steps:
                    break
                
                step_start = time.perf_counter()
                batch = DataProto.from_single_dict(batch_dict)
                if self.pointwise_dpo_enabled:
                    reference_logps = self._get_point_reference_log_probs(batch)
                    if self.prospect_dpo_enabled:
                        dpo_update_batch = build_prospect_dpo_update_proto(
                            batch=batch,
                            beta=self.config.algorithm.dpo_beta,
                            alpha_tau=self.config.algorithm.get("prospect_dpo_alpha_tau", 0.2),
                            alpha_k=self.config.algorithm.get("prospect_dpo_alpha_k", 10.0),
                            lambda_max=self.config.algorithm.get("prospect_dpo_lambda_max", 2.0),
                            lambda_gamma=self.config.algorithm.get("prospect_dpo_lambda_gamma", 2.0),
                            reference_logps=reference_logps,
                        )
                    else:
                        dpo_update_batch = build_single_wise_dpo_update_proto(
                            batch=batch,
                            beta=self.config.algorithm.dpo_beta,
                            reference_logps=reference_logps,
                        )
                else:
                    ref_chosen, ref_rejected = self._compute_reference_log_probs(batch)
                    dpo_update_batch = build_dpo_update_proto(
                        batch=batch,
                        beta=self.config.algorithm.dpo_beta,
                        loss_type=self.loss_type,
                        label_smoothing=self.config.algorithm.get("dpo_label_smoothing", 0.0),
                        reference_free=self.config.algorithm.get("reference_free", False),
                        simpo_gamma=self.config.algorithm.get("simpo_gamma", 0.5),
                        reference_chosen_logps=ref_chosen,
                        reference_rejected_logps=ref_rejected,
                    )

                actor_output = self.actor_wg.update_actor_dpo(dpo_update_batch)
                metrics = reduce_metrics(actor_output.meta_info["metrics"])
                step_duration = time.perf_counter() - step_start
                metrics["timing/step"] = step_duration

                self.global_steps += 1
                self.max_steps_duration = max(self.max_steps_duration, step_duration)
                is_last_step = self.global_steps >= self.total_training_steps
                is_epoch_end = batch_idx == len(self.train_dataloader) or is_last_step
                log_freq = max(int(self.config.trainer.get("log_freq", 1)), 1)
                if self.global_steps % log_freq == 0 or is_last_step:
                    logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)

                should_save_by_step = save_freq_steps > 0 and self.global_steps % save_freq_steps == 0
                should_save_by_epoch = (
                    save_freq_epochs > 0 and is_epoch_end and (epoch + 1) % save_freq_epochs == 0
                )
                if should_save_by_step or should_save_by_epoch:
                    self._save_checkpoint()

                test_freq = self.config.trainer.get("test_freq", -1)
                if test_freq > 0 and self.global_steps % test_freq == 0:
                    val_metrics = self._validate()
                    if val_metrics:
                        logger.log(data=val_metrics, step=self.global_steps)

                if should_save_ckpt_esi(
                    self.max_steps_duration,
                    redundant_time=self.config.trainer.get("esi_redundant_time", 0),
                ):
                    self._save_checkpoint()

            if self.global_steps >= self.total_training_steps:
                progress_bar.close()
                break

            progress_bar.close()

        final_ckpt_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        should_save_final = self.config.trainer.get("save_final_checkpoint", True)
        if should_save_final and self.global_steps > 0 and not os.path.exists(final_ckpt_dir):
            self._save_checkpoint()
