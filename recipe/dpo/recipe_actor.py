import torch

from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.dp_actor import logger as dp_actor_logger

from recipe.dpo.core_algos import (
    compute_prospect_dpo_loss,
    compute_single_wise_dpo_loss,
    is_prospect_dpo_loss,
    is_single_wise_dpo_loss,
)


class RecipeDPOActor(DataParallelPPOActor):
    @GPUMemoryLogger(role="dp actor", logger=dp_actor_logger)
    def update_policy_dpo(self, data: DataProto) -> dict[str, float]:
        loss_type = data.meta_info.get("dpo_loss_type", "sigmoid")
        if not (is_prospect_dpo_loss(loss_type) or is_single_wise_dpo_loss(loss_type)):
            return super().update_policy_dpo(data=data)

        self.actor_module.train()

        if is_prospect_dpo_loss(loss_type):
            return self._update_policy_prospect_dpo(data)
        return self._update_policy_single_wise_dpo(data)

    def _compute_point_policy_logps(self, inputs: dict[str, torch.Tensor], response_mask: torch.Tensor) -> torch.Tensor:
        # Reuse the shared actor forward so point-wise offline DPO benefits from
        # the same response-only logprob path and remove-padding optimization.
        outputs = self._forward_micro_batch(inputs, temperature=1.0, calculate_entropy=False)
        token_logps = outputs["log_probs"]
        return (token_logps * response_mask.to(token_logps.dtype)).sum(dim=-1)

    def _iter_pointwise_micro_batches(self, data: DataProto) -> list[DataProto]:
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, _ = prepare_dynamic_batch(data, max_token_len=max_token_len)
            return micro_batches

        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        if micro_batch_size is None:
            raise ValueError("actor.ppo_micro_batch_size_per_gpu must be set when dynamic batching is disabled")
        return data.split(micro_batch_size)

    def _accumulate_weighted_metrics(
        self,
        aggregated_metrics: dict[str, list[float]],
        micro_metrics: dict[str, float],
        weight: float,
    ) -> None:
        for key, value in micro_metrics.items():
            append_to_dict(aggregated_metrics, {key: float(value) * weight})

    def _update_policy_prospect_dpo(self, data: DataProto) -> dict[str, float]:
        required_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "label",
            "s_dwell",
            "p_ctr",
            "reference_logps",
        ]
        missing_keys = [key for key in required_keys if key not in data.batch]
        if missing_keys:
            raise KeyError(f"Missing required Prospect-DPO batch keys: {missing_keys}")

        beta = data.meta_info.get("dpo_beta", 0.1)
        alpha_tau = data.meta_info.get("prospect_dpo_alpha_tau", 0.2)
        alpha_k = data.meta_info.get("prospect_dpo_alpha_k", 10.0)
        lambda_max = data.meta_info.get("prospect_dpo_lambda_max", 2.0)
        lambda_gamma = data.meta_info.get("prospect_dpo_lambda_gamma", 2.0)

        batch_size = data.batch["input_ids"].shape[0]
        if batch_size == 0:
            return {"actor/dpo_loss": 0.0, "actor/dpo_accuracy": 0.0, "actor/grad_norm": 0.0}

        micro_batches = self._iter_pointwise_micro_batches(data)
        self.actor_optimizer.zero_grad()

        aggregated_metrics: dict[str, list[float]] = {}
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            inputs = {
                "input_ids": micro_batch.batch["input_ids"],
                "attention_mask": micro_batch.batch["attention_mask"],
                "position_ids": micro_batch.batch["position_ids"],
                "responses": micro_batch.batch["responses"],
                "response_mask": micro_batch.batch["response_mask"],
            }
            sample_labels = micro_batch.batch["label"]
            s_dwell = micro_batch.batch["s_dwell"]
            p_ctr = micro_batch.batch["p_ctr"]
            micro_ref_logps = micro_batch.batch["reference_logps"]
            loss_weight = sample_labels.shape[0] / batch_size

            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                policy_logps = self._compute_point_policy_logps(inputs, inputs["response_mask"])
                loss, stats = compute_prospect_dpo_loss(
                    policy_logps=policy_logps,
                    reference_logps=micro_ref_logps,
                    labels=sample_labels,
                    beta=beta,
                    s_dwell=s_dwell,
                    p_ctr=p_ctr,
                    alpha_tau=alpha_tau,
                    alpha_k=alpha_k,
                    lambda_max=lambda_max,
                    lambda_gamma=lambda_gamma,
                )
                scaled_loss = loss * loss_weight

            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            micro_metrics = {
                "actor/dpo_loss": loss.detach().item(),
                "actor/dpo_accuracy": stats["accuracy"].detach().item(),
                "actor/dpo_margin": stats["margin"].detach().item(),
                "actor/prospect_dpo_loss_pos": stats["positive_loss"].detach().item(),
                "actor/prospect_dpo_loss_neg": stats["negative_loss"].detach().item(),
                "actor/prospect_dpo_reward": stats["reward"].detach().item(),
                "actor/prospect_dpo_reward_pos": stats["positive_reward"].detach().item(),
                "actor/prospect_dpo_reward_neg": stats["negative_reward"].detach().item(),
                "actor/prospect_dpo_alpha": stats["alpha"].detach().item(),
                "actor/prospect_dpo_lambda": stats["lambda"].detach().item(),
                "actor/prospect_dpo_positive_fraction": stats["positive_fraction"].detach().item(),
                "actor/policy_logps": policy_logps.detach().mean().item(),
                "actor/reference_logps": micro_ref_logps.detach().mean().item(),
            }
            self._accumulate_weighted_metrics(aggregated_metrics, micro_metrics, loss_weight)

        grad_norm = self._optimizer_step()
        self.actor_optimizer.zero_grad()

        final_metrics = {key: sum(values) for key, values in aggregated_metrics.items() if values}
        final_metrics["actor/grad_norm"] = grad_norm.detach().item()
        return final_metrics

    def _update_policy_single_wise_dpo(self, data: DataProto) -> dict[str, float]:

        required_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "label",
            "reference_logps",
        ]
        missing_keys = [key for key in required_keys if key not in data.batch]
        if missing_keys:
            raise KeyError(f"Missing required single-wise DPO batch keys: {missing_keys}")

        beta = data.meta_info.get("dpo_beta", 0.1)

        batch_size = data.batch["input_ids"].shape[0]
        if batch_size == 0:
            return {"actor/dpo_loss": 0.0, "actor/dpo_accuracy": 0.0, "actor/grad_norm": 0.0}

        micro_batches = self._iter_pointwise_micro_batches(data)
        self.actor_optimizer.zero_grad()

        aggregated_metrics: dict[str, list[float]] = {}
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            inputs = {
                "input_ids": micro_batch.batch["input_ids"],
                "attention_mask": micro_batch.batch["attention_mask"],
                "position_ids": micro_batch.batch["position_ids"],
                "responses": micro_batch.batch["responses"],
                "response_mask": micro_batch.batch["response_mask"],
            }
            sample_labels = micro_batch.batch["label"]
            micro_ref_logps = micro_batch.batch["reference_logps"]
            loss_weight = sample_labels.shape[0] / batch_size

            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                policy_logps = self._compute_point_policy_logps(inputs, inputs["response_mask"])
                loss, stats = compute_single_wise_dpo_loss(
                    policy_logps=policy_logps,
                    reference_logps=micro_ref_logps,
                    labels=sample_labels,
                    beta=beta,
                )
                scaled_loss = loss * loss_weight

            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            micro_metrics = {
                "actor/dpo_loss": loss.detach().item(),
                "actor/dpo_accuracy": stats["accuracy"].detach().item(),
                "actor/dpo_margin": stats["margin"].detach().item(),
                "actor/single_wise_dpo_loss_pos": stats["positive_loss"].detach().item(),
                "actor/single_wise_dpo_loss_neg": stats["negative_loss"].detach().item(),
                "actor/single_wise_dpo_reward": stats["reward"].detach().item(),
                "actor/single_wise_dpo_reward_pos": stats["positive_reward"].detach().item(),
                "actor/single_wise_dpo_reward_neg": stats["negative_reward"].detach().item(),
                "actor/single_wise_dpo_positive_fraction": stats["positive_fraction"].detach().item(),
                "actor/policy_logps": policy_logps.detach().mean().item(),
                "actor/reference_logps": micro_ref_logps.detach().mean().item(),
            }
            self._accumulate_weighted_metrics(aggregated_metrics, micro_metrics, loss_weight)

        grad_norm = self._optimizer_step()
        self.actor_optimizer.zero_grad()

        final_metrics = {key: sum(values) for key, values in aggregated_metrics.items() if values}
        final_metrics["actor/grad_norm"] = grad_norm.detach().item()
        return final_metrics
