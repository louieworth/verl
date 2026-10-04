"""Record external PeNS evaluation metrics in the training W&B run."""

from __future__ import annotations

import os
import re
from pathlib import Path


ROUGE_FIELDS = {
    "rouge_1_f1": "ROUGE-1",
    "rouge_2_f1": "ROUGE-2",
    "rouge_l_f1": "ROUGE-L",
}


def rouge_wandb_values(payload, *, namespace="eval"):
    """Expose one metric per ROUGE score, on the 0--100 scale."""
    return {f"{namespace}/{name}": 100 * payload[field] for field, name in ROUGE_FIELDS.items()}


def define_rouge_metrics(run, *, namespace="eval"):
    axis = f"{namespace}/global_step"
    run.define_metric(axis, hidden=True, summary="none", overwrite=True)
    for name in ROUGE_FIELDS.values():
        run.define_metric(f"{namespace}/{name}", step_metric=axis, hidden=False, summary="max", overwrite=True)


def evaluation_config():
    from recipe.dpo.run.common import evaluation_config as shared_evaluation_config

    return shared_evaluation_config(os.environ.get("PENS_RUN_KIND", "dpo"))


def log_eval_to_wandb(payload, *, result_key, date_tag, model_path):
    from recipe.dpo.run.common import default_paths, load_env_file

    load_env_file()
    eval_config = payload.get("evaluation_config") or evaluation_config()
    if os.environ.get("PENS_WANDB_ENABLED", "false").lower() != "true":
        return
    import wandb

    match = re.fullmatch(r"step_(\d+)", date_tag) or re.search(r"global_step_(\d+)", model_path)
    step = int(match.group(1)) if match else 0
    directory = Path(
        os.environ.get("WANDB_DIR")
        or Path(default_paths()["PENS_WANDB_ROOT"]) / (os.environ.get("PENS_EXPERIMENT_NAME") or "eval")
    )
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=os.environ.get("PENS_WANDB_PROJECT", "PENS"),
        name=os.environ.get("PENS_EXPERIMENT_NAME"),
        entity=os.environ.get("WANDB_ENTITY"),
        id=os.environ.get("WANDB_RUN_ID"),
        resume=os.environ.get("WANDB_RESUME", "allow"),
        mode=default_paths()["WANDB_MODE"],
        dir=str(directory),
        job_type="train-and-eval",
    )
    try:
        namespace = "eval"
        define_rouge_metrics(run, namespace=namespace)
        run.config.update({"eval": eval_config}, allow_val_change=True)
        metrics = rouge_wandb_values(payload, namespace=namespace)
        metrics[f"{namespace}/global_step"] = step
        # Let W&B increment its history index: train and eval can share an
        # optimizer step, including the boundary where a process resumes.
        run.log(metrics)
        run.summary.update(
            {
                f"{namespace}/latest_result_key": result_key,
                f"{namespace}/latest_model_path": model_path,
                f"{namespace}/latest_generation_file": payload["raw_generation_file"],
                f"{namespace}/latest_evaluation": payload,
            }
        )
    finally:
        run.finish()
