"""Shared PeNS launch paths and experiment defaults; safe to run with Python -S."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/scratch/l/luli/data/pens")
RUNS_ROOT = Path("/scratch/l/luli/results/pens")
HF_HOME = Path("/scratch/l/luli/hf")
MODEL_SNAPSHOT = "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
PYTHON_BIN = Path("/scratch/l/luli/conda/envs/verl/bin/python")


def load_env_file(path=None, env=None):
    """Load dotenv assignments without executing shell code or replacing exports."""
    env = os.environ if env is None else env
    path = Path(path or env.get("PENS_ENV_FILE") or Path(env.get("PENS_REPO_ROOT") or REPO_ROOT) / ".env")
    if not path.is_file():
        return {}
    loaded = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            raise ValueError(f"Invalid dotenv assignment at {path}:{number}")
        key, raw_value = match.groups()
        if raw_value.startswith(("'", '"')):
            try:
                values = shlex.split(raw_value, comments=True)
                if len(values) != 1:
                    raise ValueError
                value = values[0]
            except ValueError:
                raise ValueError(f"Invalid dotenv value at {path}:{number}") from None
        else:
            value = re.split(r"\s+#", raw_value, maxsplit=1)[0].rstrip()
        if key not in env:
            env[key] = value
            loaded[key] = value
    return loaded


def default_paths(env=None):
    env = os.environ if env is None else env

    def value(key, default):
        return str(env.get(key) or default)

    repo = Path(value("PENS_REPO_ROOT", REPO_ROOT))
    data = Path(value("PENS_DATA_ROOT", DATA_ROOT))
    runs = Path(value("PENS_RUNS_ROOT", RUNS_ROOT))
    hf = Path(value("HF_HOME", HF_HOME))
    full_data = data / "pens_click_hist_runE_23pct_seed42"
    prompt = value("PENS_EVAL_PROMPT_FILE", data / "pens_eval/eval_final.parquet")
    tmp = value("TMPDIR", env.get("SLURM_TMPDIR") or "/tmp")
    user = env.get("USER") or "pens"
    return {
        "PENS_REPO_ROOT": str(repo),
        "PENS_ENV_FILE": value("PENS_ENV_FILE", repo / ".env"),
        "PENS_DATA_ROOT": str(data),
        "PENS_RUNS_ROOT": str(runs),
        "PENS_DPO_DATA_ROOT": value("PENS_DPO_DATA_ROOT", full_data / "sampled_313600_seed42"),
        "PENS_SFT_DATA_ROOT": value("PENS_SFT_DATA_ROOT", full_data),
        "PENS_MODEL_DIR": value("PENS_MODEL_DIR", hf / "hub" / MODEL_SNAPSHOT),
        "PENS_PYTHON_BIN": value("PENS_PYTHON_BIN", PYTHON_BIN),
        "PENS_CKPT_ROOT": value("PENS_CKPT_ROOT", runs / "models"),
        "PENS_EVAL_PROMPT_FILE": prompt,
        "PENS_EVAL_TEST_FILE": value("PENS_EVAL_TEST_FILE", prompt),
        "PENS_OUTPUT_ROOT": value("PENS_OUTPUT_ROOT", runs / "generations"),
        "PENS_WANDB_ROOT": value("PENS_WANDB_ROOT", repo / "wandb"),
        "PENS_TOTAL_TRAINING_STEPS": value("PENS_TOTAL_TRAINING_STEPS", "500"),
        "PENS_EVAL_INTERVAL": value("PENS_EVAL_INTERVAL", "50"),
        "PENS_LR": value("PENS_LR", "5e-6"),
        "PENS_PROJECT_NAME": value("PENS_PROJECT_NAME", env.get("WANDB_PROJECT") or "PENS"),
        "PENS_LOGGER": value("PENS_LOGGER", "[console,wandb]"),
        "WANDB_MODE": value("WANDB_MODE", "offline"),
        "HF_HOME": str(hf),
        "TMPDIR": tmp,
        "RAY_TMPDIR": value("RAY_TMPDIR", Path(tmp) / f"ray_{user}"),
        "TORCHINDUCTOR_CACHE_DIR": value("TORCHINDUCTOR_CACHE_DIR", Path(tmp) / f"torchinductor_{user}"),
        "TRITON_CACHE_DIR": value("TRITON_CACHE_DIR", Path(tmp) / f"triton_{user}"),
        "VLLM_CACHE_ROOT": value("VLLM_CACHE_ROOT", Path(tmp) / f"vllm_{user}"),
    }


def evaluation_config(kind="dpo", env=None):
    env = os.environ if env is None else env
    paths = default_paths(env)
    prefix = "SINGLE_WISE_DPO" if kind == "dpo" else "SFT"
    generation_defaults = {
        "temperature": ("GEN_TEMPERATURE", 0.0),
        "top_p": ("GEN_TOP_P", 0.8),
        "top_k": ("GEN_TOP_K", 20),
        "prompt_length": ("GEN_PROMPT_LENGTH", 8000),
        "response_length": ("GEN_RESPONSE_LENGTH", 128),
        "max_model_len": ("GEN_MAX_MODEL_LEN", 8300),
        "max_num_seqs": ("GEN_MAX_NUM_SEQS", 32),
        "max_num_batched_tokens": ("GEN_MAX_NUM_BATCHED_TOKENS", 4096),
        "repetition_penalty": ("GEN_REPETITION_PENALTY", 1.1),
        "gpu_memory_utilization": ("GEN_GPU_MEMORY_UTILIZATION", 0.9),
        "pass_k": ("PASS_K", 1),
    }
    generation = {name: type(default)(env.get(key) or default) for name, (key, default) in generation_defaults.items()}
    generation["tensor_parallel_size"] = int(env.get("GEN_TP") or env.get(f"{prefix}_CHECKPOINT_EVAL_GEN_TP") or 1)
    generation["thinking"] = (
        env.get("VLLM_ENABLE_THINKING") or env.get(f"{prefix}_CHECKPOINT_EVAL_THINKING_MODE") or "false"
    )
    return {
        "interval_steps": int(env.get(f"{prefix}_CHECKPOINT_EVAL_INTERVAL") or paths["PENS_EVAL_INTERVAL"]),
        "prompt_file": env.get("PROMPT_FILE")
        or env.get(f"{prefix}_CHECKPOINT_EVAL_PROMPT_FILE")
        or paths["PENS_EVAL_PROMPT_FILE"],
        "reference_file": env.get("TEST_FILE")
        or env.get(f"{prefix}_CHECKPOINT_EVAL_TEST_FILE")
        or paths["PENS_EVAL_TEST_FILE"],
        "nnodes": int(env.get("NNODES") or env.get(f"{prefix}_CHECKPOINT_EVAL_NNODES") or 1),
        "gpus_per_node": int(
            env.get("NGPUS_PER_NODE")
            or env.get(f"{prefix}_CHECKPOINT_EVAL_NGPUS_PER_NODE")
            or env.get(f"{prefix}_N_GPUS_PER_NODE")
            or 8
        ),
        "generation": generation,
    }


def run_defaults(kind, env=None, overrides=()):
    env = dict(os.environ if env is None else env)
    paths = default_paths(env)
    prefix = "SINGLE_WISE_DPO" if kind == "dpo" else "SFT"
    aliases = {
        "trainer.project_name": "PROJECT_NAME",
        "trainer.experiment_name": "EXPERIMENT_NAME",
        "trainer.logger": "LOGGER",
        "trainer.total_training_steps": "TOTAL_TRAINING_STEPS",
        "trainer.default_local_dir": "CKPT_DIR" if kind == "dpo" else "DEFAULT_LOCAL_DIR",
        "data.train_batch_size": "TRAIN_BATCH_SIZE",
        "data.seed": "SEED",
        "trainer.seed": "SEED",
        "actor_rollout_ref.actor.optim.lr": "ACTOR_LR",
        "optim.lr": "LR",
        "algorithm.dpo_beta": "BETA",
        "algorithm.prospect_dpo_positive_kappa": "POSITIVE_KAPPA",
        "algorithm.prospect_dpo_negative_kappa": "NEGATIVE_KAPPA",
        "algorithm.prospect_dpo_use_feedback_weights": "USE_FEEDBACK_WEIGHTS",
    }
    for arg in overrides:
        key, separator, val = arg.lstrip("+").partition("=")
        if separator and key in aliases:
            env[f"{prefix}_{aliases[key]}"] = val.strip("'\"")

    def value(key, default):
        return str(env.get(f"{prefix}_{key}") or default)

    eval_env = dict(env)
    for key, suffix in {
        "GEN_TP": "CHECKPOINT_EVAL_GEN_TP",
        "VLLM_ENABLE_THINKING": "CHECKPOINT_EVAL_THINKING_MODE",
        "NNODES": "CHECKPOINT_EVAL_NNODES",
        "NGPUS_PER_NODE": "CHECKPOINT_EVAL_NGPUS_PER_NODE",
    }.items():
        if env.get(f"{prefix}_{suffix}"):
            eval_env[key] = env[f"{prefix}_{suffix}"]
    eval_config = evaluation_config(kind, eval_env)
    timestamp = value("RUN_TIMESTAMP", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    if kind == "dpo":
        loss = env.get("POINTWISE_DPO_LOSS_TYPE") or value("LOSS_TYPE", "prospect_dpo")
        sample = value("SAMPLE_VARIANT", "all")
        sample = {"positive_only": "positive", "negative_only": "negative"}.get(sample, sample)
        family = loss.removesuffix("_dpo")
        use_feedback_weights = value("USE_FEEDBACK_WEIGHTS", "true").lower() in {"true", "1", "yes", "y", "on"}
        weight = "with_weight" if loss == "prospect_dpo" and use_feedback_weights else "without_weight"
        label = f"{family}_{value('INPUT_VARIANT', 'click_hist')}_{sample}_{weight}"
    else:
        label = "sft_click_hist_positive"
    experiment = value("EXPERIMENT_NAME", f"{label}_{timestamp}")
    if experiment in {".", ".."} or "/" in experiment or "\\" in experiment:
        raise ValueError("Experiment name must be a single directory name")
    ckpt_key = "CKPT_DIR" if kind == "dpo" else "DEFAULT_LOCAL_DIR"
    ckpt = value(ckpt_key, Path(value("CKPT_ROOT", paths["PENS_CKPT_ROOT"])) / experiment)
    generated = value("CHECKPOINT_EVAL_GEN_DIR", Path(paths["PENS_OUTPUT_ROOT"]) / experiment)
    project = value("PROJECT_NAME", paths["PENS_PROJECT_NAME"])
    run_id = env.get("WANDB_RUN_ID") or hashlib.sha256(f"{project}:{experiment}".encode()).hexdigest()[:16]
    logger = value("LOGGER", paths["PENS_LOGGER"])
    lr_key = "ACTOR_LR" if kind == "dpo" else "LR"
    return {
        f"{prefix}_{lr_key}": value(lr_key, paths["PENS_LR"]),
        f"{prefix}_RUN_TIMESTAMP": timestamp,
        f"{prefix}_TOTAL_TRAINING_STEPS": value("TOTAL_TRAINING_STEPS", paths["PENS_TOTAL_TRAINING_STEPS"]),
        f"{prefix}_EXPERIMENT_NAME": experiment,
        f"{prefix}_{ckpt_key}": ckpt,
        f"{prefix}_PROJECT_NAME": project,
        f"{prefix}_LOGGER": logger,
        f"{prefix}_CHECKPOINT_EVAL_GEN_DIR": generated,
        f"{prefix}_CHECKPOINT_EVAL_STATE_FILE": value(
            "CHECKPOINT_EVAL_STATE_FILE", Path(ckpt) / "interleaved_eval_state.json"
        ),
        "PENS_EXPERIMENT_NAME": experiment,
        "PENS_RUN_KIND": kind,
        "PENS_EVAL_CONFIG_JSON": json.dumps(eval_config, sort_keys=True),
        "PENS_WANDB_PROJECT": project,
        "PENS_WANDB_ENABLED": "true" if "wandb" in logger else "false",
        "WANDB_MODE": paths["WANDB_MODE"],
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": env.get("WANDB_RESUME") or "allow",
        "WANDB_DIR": env.get("WANDB_DIR") or str(Path(paths["PENS_WANDB_ROOT"]) / experiment),
    }


def shell_exports(values):
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in values.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-env", action="store_true")
    parser.add_argument("--shell-defaults", action="store_true")
    parser.add_argument("--shell-run", choices=["dpo", "sft"])
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    loaded = load_env_file()
    if args.shell_env:
        print(shell_exports(loaded))
        return
    print(shell_exports(run_defaults(args.shell_run, overrides=args.overrides) if args.shell_run else default_paths()))


if __name__ == "__main__":
    main()
