#!/usr/bin/env python3
"""Post-process and aggregate ablation results.

For each ablation model name pattern:
  1. If gen parquets exist under gen_results/{full_name}/evaluate/{ds}_pass16_generation.parquet
     and results.json lacks avg_pass1/pass8 keys for the dataset, run
     compute_pass_at_k_from_gen.py to compute and write them in.
  2. Copy the (now-rich) entry into results/Qwen3-8B/ours_ablation_stage_2.json.

Idempotent: re-running is safe; existing entries get overwritten with current scores.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path("/data/verl")
SRC = REPO / "results/Qwen3-8B/results.json"
DST = REPO / "results/Qwen3-8B/ours_ablation_stage_2.json"
GEN_BASE = REPO / "gen_results"
COMPUTE_SCRIPT = REPO / "recipe/math_evaluation/compute_pass_at_k_from_gen.py"

# (substring identifying the model name, evaluator-tag suffix used in eval gen dirs)
PATTERNS = [
    "kl_forward_full_vocab_y_cor_clip0_reward1",                   # exp 1
    "kl_jsd_full_vocab_y_raw_clip0_beta0_reward0",                 # exp 2
    "kl_jsd_full_vocab_y_raw_clip0_beta0_reward1",                 # exp 3
    "kl_reverse_full_vocab_y_raw_clip0_reward0",                   # exp 4
    "kl_reverse_full_vocab_y_raw_clip0_reward1",                   # exp 5
    # ".._reward0only_2" matches reward0only_<DATE> but NOT
    # reward0only_filtered_<DATE> (the char after "_" is "f", not "2").
    "kl_forward_full_vocab_y_cor_clip0_reward0only_2",             # exp 6
    "kl_forward_full_vocab_y_cor_clip0_reward0only_filtered",      # exp 7
]
DATASETS = ["aime24", "aime25", "hmmt25", "beyondaime", "amobench"]


def latest_match(src: dict, pat: str) -> str | None:
    matches = sorted(k for k in src if pat in k)
    return matches[-1] if matches else None


def needs_enrichment(entry: dict) -> bool:
    """True if any dataset has pass16 but missing avg_pass1/pass8."""
    for ds in DATASETS:
        if f"{ds}_pass16_generation_pass_16" in entry and (
            f"{ds}_avg_pass1_generation_pass_16" not in entry
            or f"{ds}_pass8_generation_pass_16" not in entry
        ):
            return True
    return False


def gen_dir_for(model_name: str) -> Path:
    """benchmark_kl_model.sh strips the trailing _epoch{N} suffix when
    building the gen_results path. So Qwen3-8B_..._reward1_20260504_epoch1
    → gen_results/Qwen3-8B_..._reward1_20260504/evaluate/."""
    full = model_name
    if "_epoch" in full:
        full = full[: full.rfind("_epoch")]
    return GEN_BASE / full / "evaluate"


def enrich(model_name: str) -> None:
    gen_dir = gen_dir_for(model_name)
    have_any = any((gen_dir / f"{ds}_pass16_generation.parquet").exists() for ds in DATASETS)
    if not have_any:
        print(f"  [enrich] no pass16 gen parquets at {gen_dir}; skip")
        return
    print(f"  [enrich] computing avg_pass1/pass8/pass16 for {model_name}")
    subprocess.run(
        [
            "python3",
            str(COMPUTE_SCRIPT),
            "--gen_dir", str(gen_dir),
            "--results_file", str(SRC),
            "--model_name", model_name,
            "--datasets", ",".join(DATASETS),
        ],
        check=False,
    )


def main() -> None:
    if not SRC.exists():
        print(f"[aggregate] source missing: {SRC}")
        return
    src = json.loads(SRC.read_text())

    # 1. Enrichment pass — fill in avg_pass1 / pass8 from gen parquets
    for pat in PATTERNS:
        key = latest_match(src, pat)
        if not key:
            continue
        if needs_enrichment(src[key]):
            enrich(key)

    # Re-read after enrichment
    src = json.loads(SRC.read_text())
    dst = json.loads(DST.read_text()) if DST.exists() else {}

    # 2. Copy enriched entries
    updated: list[str] = []
    for pat in PATTERNS:
        key = latest_match(src, pat)
        if not key:
            continue
        dst[key] = src[key]
        updated.append(key)

    DST.write_text(json.dumps(dst, indent=2))
    print(f"[aggregate] wrote {len(updated)} entries -> {DST}")
    for k in updated:
        print(f"  - {k}")


if __name__ == "__main__":
    main()
