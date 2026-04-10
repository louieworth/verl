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
import socket

import hydra
import ray

from recipe.dpo.core_algos import is_pointwise_dpo_loss, is_prospect_dpo_loss, requires_reference_model
from recipe.dpo.collate import pointwise_dynamic_prompt_collate_fn
from recipe.dpo.dpo_trainer import RayDPOTrainer
from recipe.dpo.pointwise_dataset import PointwiseDPODataset
from recipe.dpo.reference_logps_materializer import maybe_materialize_pointwise_reference_logps
from recipe.dpo.recipe_worker import RecipeDPOWorker
from recipe.dpo.sampler import LengthBucketSampler
from recipe.dpo.singlewise_dataset import SingleWiseDPODataset
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import create_rl_sampler, run_ppo
from verl.trainer.ppo.utils import Role
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.dataset import DPOPairDataset


class DPOTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.single_controller.ray import ResourcePoolManager, RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("Offline DPO currently supports only FSDP legacy workers")

        loss_type = config.algorithm.get("dpo_loss_type", "sigmoid")
        if is_pointwise_dpo_loss(loss_type) and config.algorithm.get("reference_free", False):
            raise ValueError("Point-wise offline DPO variants require algorithm.reference_free=false")

        if is_pointwise_dpo_loss(loss_type) and requires_reference_model(
            loss_type,
            config.algorithm.get("reference_free", False),
        ):
            materialized_train_files, materialized_val_files = maybe_materialize_pointwise_reference_logps(config)
            config.data.train_files = materialized_train_files
            if materialized_val_files is not None:
                config.data.val_files = materialized_val_files

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        if is_prospect_dpo_loss(loss_type):
            dataset_cls = PointwiseDPODataset
        elif is_pointwise_dpo_loss(loss_type):
            dataset_cls = SingleWiseDPODataset
        else:
            dataset_cls = DPOPairDataset
        collate_fn = pointwise_dynamic_prompt_collate_fn if is_pointwise_dpo_loss(loss_type) else default_collate_fn
        train_dataset = dataset_cls(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            config=config.data,
            processor=processor,
            max_samples=config.data.get("train_max_samples", -1),
        )

        val_dataset = None
        if config.data.get("val_files", None):
            val_dataset = dataset_cls(
                data_files=config.data.val_files,
                tokenizer=tokenizer,
                config=config.data,
                processor=processor,
                max_samples=config.data.get("val_max_samples", -1),
            )

        pointwise_precomputed_reference_logps = (
            is_pointwise_dpo_loss(loss_type)
            and hasattr(train_dataset, "has_reference_logps")
            and train_dataset.has_reference_logps()
            and (val_dataset is None or (hasattr(val_dataset, "has_reference_logps") and val_dataset.has_reference_logps()))
        )
        if pointwise_precomputed_reference_logps:
            print("Detected precomputed point-wise reference_logps in parquet. RefPolicy workers will be skipped.")

        use_reference_policy = requires_reference_model(
            loss_type,
            config.algorithm.get("reference_free", False),
        ) and not pointwise_precomputed_reference_logps

        role_worker_mapping = {
            Role.Actor: ray.remote(RecipeDPOWorker),
        }
        global_pool_id = "global_pool"
        mapping = {Role.Actor: global_pool_id}
        if use_reference_policy:
            role_worker_mapping[Role.RefPolicy] = ray.remote(RecipeDPOWorker)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
            mapping=mapping,
        )

        validate_config(
            config=config,
            use_reference_policy=use_reference_policy,
            use_critic=False,
        )

        if is_pointwise_dpo_loss(loss_type) and config.data.get("use_length_bucket_sampler", False):
            train_sampler = LengthBucketSampler(
                train_dataset,
                batch_size=config.data.train_batch_size,
                shuffle=config.data.get("shuffle", True),
                seed=config.data.get("seed"),
                bucket_size_multiplier=config.data.get("length_bucket_size_multiplier", 50),
            )
        else:
            train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayDPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DPOTaskRunner))


if __name__ == "__main__":
    main()
