#!/usr/bin/env python3
"""Shared W&B metadata for canonical OPD/OPSD experiments.

The recipe dispatcher owns the experiment contract and exposes it through
environment variables. This module lives in the installed ``verl`` package so
the core tracking backend never depends on an unshipped ``recipe`` module.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


_CONFIG_ENV_FIELDS = {
    "OPD_TASK": "task",
    "OPD_FAMILY": "family",
    "OPD_VARIANT": "variant",
    "OPD_MODEL_SIZE": "model_size",
    "MODEL_ALIAS": "model",
    "TEACHER_MODEL": "teacher_model",
    "DISTILL_MODE": "distill_mode",
    "KL_TYPE": "kl_type",
    "KL_METHOD": "kl_method",
    "Y_MODE": "y_mode",
    "TEACHER_TRAINING_PROMPT": "teacher_prompt",
    "TEACHER_ENABLE_THINKING": "teacher_enable_thinking",
    "TEACHER_THINKING_NAME_TAG": "teacher_thinking_name_tag",
    "TEACHER_SUPERVISION_RENDER_MODE": "teacher_supervision_render_mode",
    "TEACHER_ROLLOUT_RENDER_MODE": "teacher_rollout_render_mode",
    "TEACHER_PROMPT_RENDER_CONTRACT": "teacher_prompt_render_contract",
    "Y_O_ROLLOUT_MODE": "rollout_mode",
    "TOP_K": "teacher_loss_top_k",
    "KL_TOKEN_CLIP": "kl_token_clip",
    "BETA": "beta",
    "MULTI_STEP": "requested_policy_steps",
    "GLOBAL_PROMPT_BATCH_SIZE": "global_prompt_batch_size",
    "BASE_PROMPT_LENGTH": "train_prompt_length",
    "MAX_RESPONSE_LENGTH": "train_response_length",
    "EVAL_PROMPT_LENGTH": "eval_prompt_length",
    "EVAL_RESPONSE_LENGTH": "eval_response_length",
    "PASS_K": "eval_pass_k",
    "EVAL_DATASETS": "eval_datasets",
    "EVAL_FRACTIONS": "eval_fractions",
    "LEARNING_RATE": "learning_rate",
    "USE_LORA": "use_lora",
    "LORA_RANK": "lora_rank",
    "LORA_ALPHA": "lora_alpha",
    "MICRO_BATCH_SIZE_PER_GPU": "micro_batch_size_per_gpu",
    "USE_DYNAMIC_BSZ": "use_dynamic_batching",
    "ROLLOUT_N": "rollout_n",
    "ROLLOUT_TEMPERATURE": "rollout_temperature",
    "ROLLOUT_TOP_P": "rollout_top_p",
    "EVAL_TEMPERATURE": "eval_temperature",
    "EVAL_TOP_P": "eval_top_p",
    "CODE_EVAL_TEMPERATURE": "code_eval_temperature",
    "CODE_EVAL_TOP_P": "code_eval_top_p",
    "SKD_GAMMA": "skd_gamma",
    "SKD_ACCEPT_TOP_K": "skd_accept_top_k",
    "SKD_ACCEPT_TOP_P": "skd_accept_top_p",
    "SKD_TEACHER_TEMPERATURE": "skd_teacher_temperature",
    "MODEL_ARTIFACT_POLICY": "model_artifact_policy",
    "PIPELINE_EPHEMERAL_MODELS": "pipeline_ephemeral_models",
    "PIPELINE_DEFER_MILESTONE_EVALS": "pipeline_defer_milestone_evals",
    "RESIDENT_STUDENT_ROLLOUT": "resident_student_rollout",
    "SEED": "seed",
}

_OPD_TEACHER_THINKING_ENV_FIELDS = {
    "TEACHER_ENABLE_THINKING",
    "TEACHER_THINKING_NAME_TAG",
    "TEACHER_SUPERVISION_RENDER_MODE",
    "TEACHER_ROLLOUT_RENDER_MODE",
    "TEACHER_PROMPT_RENDER_CONTRACT",
}


def _typed_value(value: str) -> Any:
    """Convert simple shell scalars while preserving model names and lists."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value
    return parsed if isinstance(parsed, (int, float)) else value


def wandb_init_metadata_from_env() -> dict[str, Any]:
    """Build optional ``wandb.init`` grouping fields without W&B tags."""
    metadata: dict[str, Any] = {}
    if group := os.environ.get("WANDB_GROUP", "").strip():
        metadata["group"] = group
    if job_type := os.environ.get("WANDB_JOB_TYPE", "").strip():
        metadata["job_type"] = job_type
    return metadata


def wandb_shared_run_enabled() -> bool:
    """Return whether a pipeline-level W&B primary owns the run state."""
    return os.environ.get("WANDB_SHARED_RUN", "").strip().lower() in {"1", "true", "yes"}


def wandb_process_settings(
    wandb_client: Any,
    *,
    primary: bool = False,
    label: str,
    **settings_kwargs: Any,
) -> Any | None:
    """Build W&B settings for one writer in a pipeline-owned shared run."""
    if wandb_shared_run_enabled():
        settings_kwargs.update(
            mode="shared",
            x_primary=primary,
            x_update_finish_state=primary,
            x_label=f"{label}-{os.getpid()}",
        )
    if not settings_kwargs:
        return None
    return wandb_client.Settings(**settings_kwargs)


def opd_experiment_config_from_env() -> dict[str, Any]:
    """Return the compact, queryable canonical experiment configuration."""
    result: dict[str, Any] = {}
    family = os.environ.get("OPD_FAMILY", os.environ.get("DISTILL_MODE", "")).strip().lower()
    for env_name, config_name in _CONFIG_ENV_FIELDS.items():
        if env_name in _OPD_TEACHER_THINKING_ENV_FIELDS and family != "opd":
            continue
        if (raw_value := os.environ.get(env_name, "").strip()) != "":
            result[config_name] = _typed_value(raw_value)
    return result


def merge_opd_wandb_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Attach canonical metadata without discarding trainer-native config."""
    merged = dict(config or {})
    if metadata := opd_experiment_config_from_env():
        merged["opd_experiment"] = metadata
    return merged
