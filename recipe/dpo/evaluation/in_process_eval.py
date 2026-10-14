"""PENS evaluation with persistent training workers and in-memory vLLM weights."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


def configure_in_process_eval(config):
    """Use a hybrid vLLM rollout without changing offline training batches."""
    if not config.trainer.get("in_process_eval", False):
        return
    from omegaconf import OmegaConf, open_dict

    evaluation = json.loads(config.trainer.eval_config_json)
    options = json.loads(config.trainer.in_process_eval_options_json)
    generation = evaluation["generation"]
    if "actor_rollout_ref" in config:
        rollout = config.actor_rollout_ref.rollout
    else:
        # The YAML includes concrete Hydra targets for nested configs too.
        rollout = OmegaConf.load(Path(__file__).resolve().parents[3] / "verl/trainer/config/rollout/rollout.yaml")
    with open_dict(rollout):
        rollout.name = "vllm"
        rollout.mode = "async"
        # Load the original base model once. Every eval synchronizes the live
        # training weights (or LoRA adapter); no merged checkpoint is needed.
        rollout.load_format = "auto"
        rollout.free_cache_engine = True
        rollout.enable_sleep_mode = True
        rollout.n = 1  # Eval pass_k must not multiply the offline update batch.
        rollout.n_gpus_per_node = config.trainer.n_gpus_per_node
        for key in (
            "top_p",
            "top_k",
            "prompt_length",
            "response_length",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
        ):
            rollout[key] = generation[key]
        # Legacy DPO workers also use rollout.temperature for policy/reference
        # log probabilities. Keep it unchanged; eval sampling is per request.
        rollout.tensor_model_parallel_size = generation["tensor_parallel_size"]
        for key, value in options["rollout_overrides"].items():
            rollout[key] = value
        if "vllm" not in rollout.engine_kwargs:
            rollout.engine_kwargs.vllm = {}
        rollout.engine_kwargs.vllm.update(options["engine_kwargs"])
    if "actor_rollout_ref" not in config:
        with open_dict(config):
            config.rollout = rollout


class PensInProcessEvaluator:
    def __init__(self, config, tokenizer, worker_group):
        from verl.checkpoint_engine import CheckpointEngineManager
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.rollout.replica import get_rollout_replica_class

        self.config = config
        self.tokenizer = tokenizer
        self.worker_group = worker_group
        self.evaluation = json.loads(config.trainer.eval_config_json)
        self.options = json.loads(config.trainer.in_process_eval_options_json)
        self.interval = int(self.evaluation["interval_steps"])
        self.state_file = Path(self.options["state_file"])
        self.output_dir = Path(self.options["output_dir"])
        self.generation = self.evaluation["generation"]
        if self.interval <= 0:
            raise ValueError("Evaluation interval must be positive")
        if self.generation["pass_k"] < 1:
            raise ValueError("PASS_K must be positive")
        if self.generation["temperature"] == 0 and self.generation["pass_k"] != 1:
            raise ValueError("Greedy evaluation requires PASS_K=1")
        if self.options["request_concurrency"] < 1:
            raise ValueError("GEN_REQUEST_CONCURRENCY must be positive")
        for key in ("prompt_file", "reference_file"):
            if not Path(self.evaluation[key]).is_file():
                raise FileNotFoundError(f"Missing evaluation input: {self.evaluation[key]}")

        if "actor_rollout_ref" in config:
            rollout_config = config.actor_rollout_ref.rollout
            model_config = config.actor_rollout_ref.model
        else:
            rollout_config = config.rollout
            model_config = config.model
        self.rollout_config = omega_conf_to_dataclass(rollout_config)
        self.model_config = model_config
        infer_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        if worker_group.world_size % infer_world_size:
            raise ValueError("Training GPU count must be divisible by evaluation parallelism")
        replica_cls = get_rollout_replica_class("vllm")
        self.replicas = [
            replica_cls(rank, self.rollout_config, model_config, gpus_per_node=config.trainer.n_gpus_per_node)
            for rank in range(worker_group.world_size // infer_world_size)
        ]
        self.weight_manager = CheckpointEngineManager(
            self.rollout_config.checkpoint_engine, worker_group, self.replicas
        )
        self.dataset = None
        self.prompt_ids = None
        try:
            asyncio.run(self._initialize())
            self.weight_manager.sleep_replicas()
        except BaseException:
            self.close()
            raise

    async def _initialize(self):
        await asyncio.gather(*[replica.init_hybrid(self.worker_group) for replica in self.replicas])

    def should_evaluate(self, step, *, final=False):
        from recipe.dpo.evaluation.manage_pens_interleaved_checkpoint import load_json_object

        if step <= 0 or (step % self.interval and not final):
            return False
        return int(load_json_object(self.state_file).get("last_evaluated_step", 0)) < step

    def _load_prompts(self):
        from recipe.dpo.evaluation.score_pens_predictions import load_table

        self.dataset = load_table(Path(self.evaluation["prompt_file"]))
        if self.dataset.empty:
            raise ValueError("PENS evaluation prompts are empty")
        thinking = self.generation["thinking"].lower()
        if thinking not in {"true", "false", "skip"}:
            raise ValueError(f"Invalid evaluation thinking mode: {thinking}")
        template_kwargs = {} if thinking == "skip" else {"enable_thinking": thinking == "true"}
        self.prompt_ids = []
        for messages in self.dataset["prompt"]:
            if hasattr(messages, "tolist"):
                messages = messages.tolist()
            self.prompt_ids.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, **template_kwargs
                )
            )

    async def _generate(self, step):
        from tqdm import tqdm

        if self.dataset is None:
            self._load_prompts()
        n_samples = self.generation["pass_k"]
        responses = [[""] * n_samples for _ in self.prompt_ids]
        sampling = {
            "temperature": self.generation["temperature"],
            "top_p": self.generation["top_p"],
            "top_k": self.generation["top_k"],
            "max_tokens": self.generation["response_length"],
            "repetition_penalty": self.generation["repetition_penalty"],
        }
        semaphores = [asyncio.Semaphore(self.options["request_concurrency"]) for _ in self.replicas]

        async def request(replica_index, row, sample, progress):
            prompt = self.prompt_ids[row]
            # Match standalone chat completion's context limit. Keep rejected
            # rows as empty responses so scoring preserves sample alignment.
            if len(prompt) + sampling["max_tokens"] <= self.generation["max_model_len"]:
                async with semaphores[replica_index]:
                    output = await self.replicas[replica_index].server_handle.generate.remote(
                        request_id=f"pens-{step}-{row}-{sample}",
                        prompt_ids=prompt,
                        sampling_params=dict(sampling),
                    )
                    # The token API selects the current in-memory LoRA adapter.
                    responses[row][sample] = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            progress.update(1)

        width, extra = divmod(len(self.prompt_ids), len(self.replicas))
        offset = 0
        tasks = []
        with tqdm(total=len(self.prompt_ids) * n_samples, desc=f"PENS eval step {step}", unit="sample") as progress:
            for index in range(len(self.replicas)):
                end = offset + width + (index < extra)
                tasks.extend(
                    request(index, row, sample, progress) for row in range(offset, end) for sample in range(n_samples)
                )
                offset = end
            await asyncio.gather(*tasks)
        return responses

    def evaluate(self, step, tracking):
        from recipe.dpo.evaluation.manage_pens_interleaved_checkpoint import main as manage_checkpoint
        from recipe.dpo.evaluation.run_pens_personalized_eval import evaluate_saved_generations
        from recipe.dpo.evaluation.wandb_utils import define_rouge_metrics, rouge_wandb_values

        started = time.perf_counter()
        print(f"PENS eval at step {step}: pause updates and synchronize weights from training worker memory")
        try:
            self.weight_manager.update_weights(global_steps=step)
            responses = asyncio.run(self._generate(step))
        finally:
            # Training tensors and optimizer remain in the same workers, on
            # CPU when offloaded. The next update uses them directly.
            self.weight_manager.sleep_replicas()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_key = f"{self.config.trainer.experiment_name}_step_{step}"
        raw_file = self.output_dir / f"{model_key}.parquet"
        self.dataset["responses"] = responses
        self.dataset.to_parquet(raw_file, index=False)
        checkpoint_dir = Path(self.config.trainer.default_local_dir)
        model_path = str(checkpoint_dir / f"global_step_{step}")
        args = argparse.Namespace(
            workdir=Path.cwd(),
            model_path=model_path,
            model_key=model_key,
            test_file=Path(self.evaluation["reference_file"]),
            results_gen_dir=self.output_dir,
            raw_file=raw_file,
            checkpoint_state_file=self.state_file,
            result_json_file=None,
            response_index=self.options["response_index"],
            backend="rouge",
            align_by_order=self.options["align_by_order"],
            thinking=self.generation["thinking"],
            date_tag=f"step_{step}",
        )
        result_key, payload = evaluate_saved_generations(args, log_to_wandb=False, record_checkpoint_state=False)
        payload["evaluation_config"] = self.evaluation
        payload["weights_source"] = "training_worker_memory"
        from recipe.dpo.evaluation.manage_pens_interleaved_checkpoint import record_evaluation_result

        record_evaluation_result(self.state_file, result_key, payload)
        metrics = rouge_wandb_values(payload)
        metrics["eval/duration_seconds"] = time.perf_counter() - started
        wandb_logger = tracking.logger.get("wandb")
        run = getattr(wandb_logger, "run", None)
        if run is not None:
            define_rouge_metrics(run)
        tracking.log(metrics, step=step)
        if run is not None:
            run.summary.update(
                {
                    "eval/latest_result_key": result_key,
                    "eval/latest_model_path": model_path,
                    "eval/latest_generation_file": str(raw_file),
                    "eval/latest_evaluation": payload,
                }
            )
        argv = [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--state-file",
            str(self.state_file),
            "--result-key",
            result_key,
            "--step",
            str(step),
        ]
        for name, value in self.options["thresholds"].items():
            argv.extend([f"--{name}", str(value)])
        if self.options["keep_best_checkpoint"]:
            argv.append("--keep-best-checkpoint")
        state = manage_checkpoint(argv)
        print(f"PENS eval finished at step {step}; continue training with existing model and optimizer")
        return bool(state["should_stop_trial"])

    def close(self):
        """Stop inference servers without killing their training workers."""
        import ray

        async def shutdown():
            handles = [server for replica in self.replicas for server in replica.servers]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[server.shutdown.remote() for server in handles], return_exceptions=True),
                    timeout=120,
                )
            finally:
                for server in handles:
                    ray.kill(server, no_restart=True)

        asyncio.run(shutdown())
