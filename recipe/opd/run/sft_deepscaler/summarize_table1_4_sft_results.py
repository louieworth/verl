#!/usr/bin/env python3
"""Validate and render the missing Table 1-4 Base/SFT metrics as Markdown."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
SFT_LEARNING_RATE = os.environ.get("LEARNING_RATE", "1e-7")


@dataclass(frozen=True)
class ResultSpec:
    label: str
    affected_tables: str
    result_file: Path
    datasets: tuple[str, ...]
    metric_style: str


SPECS = (
    ResultSpec(
        "Qwen3-1.7B +SFT (math)",
        "1, 2",
        REPO_ROOT
        / f"results/SFT/math/Qwen3-1.7B_solution_cot_only_lr{SFT_LEARNING_RATE}/results.json",
        ("aime24", "aime25", "hmmt25", "beyondaime", "amobench"),
        "math",
    ),
    ResultSpec(
        "Qwen3-1.7B +SFT (code)",
        "1, 2",
        REPO_ROOT
        / f"results/SFT/code/Qwen3-1.7B_taco_solution_only_max8192_lr{SFT_LEARNING_RATE}/results.json",
        ("humaneval_plus", "mbpp_plus", "livecodebench_v6"),
        "code",
    ),
    ResultSpec(
        "Qwen3-4B-Instruct-2507 +SFT (code)",
        "1, 2, 3, 4",
        REPO_ROOT
        / f"results/SFT/code/Qwen3-4B-Instruct-2507_taco_solution_only_max8192_lr{SFT_LEARNING_RATE}/results.json",
        ("humaneval_plus", "mbpp_plus", "livecodebench_v6"),
        "code",
    ),
    ResultSpec(
        "Qwen3-8B +SFT (code)",
        "3, 4",
        REPO_ROOT
        / f"results/SFT/code/Qwen3-8B_taco_solution_only_max8192_lr{SFT_LEARNING_RATE}/results.json",
        ("humaneval_plus", "mbpp_plus", "livecodebench_v6"),
        "code",
    ),
    ResultSpec(
        "Qwen3-8B Base (code)",
        "3, 4",
        REPO_ROOT / "results/base/code/Qwen3-8B_code_avg16_pass16.json",
        ("humaneval_plus", "mbpp_plus", "livecodebench_v6"),
        "code",
    ),
)


DISPLAY_NAMES = {
    "aime24": "AIME24",
    "aime25": "AIME25",
    "hmmt25": "HMMT25",
    "beyondaime": "BeyondAIME",
    "amobench": "AMOBench",
    "humaneval_plus": "HumanEval+",
    "mbpp_plus": "MBPP+",
    "livecodebench_v6": "LiveCodeBench v6",
}


def load_single_entry(path: Path) -> tuple[str, dict]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing result file: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON result file: {path}: {exc}") from exc
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"expected exactly one model entry in {path}, got {len(data) if isinstance(data, dict) else type(data).__name__}")
    model_name, entry = next(iter(data.items()))
    if not isinstance(entry, dict):
        raise ValueError(f"model entry is not an object in {path}: {model_name}")
    model_path = Path(str(entry.get("model_path") or ""))
    if not (model_path / "config.json").is_file():
        raise ValueError(f"result model_path is not a readable checkpoint: {model_path} ({path})")
    return model_name, entry


def metric_keys(style: str, dataset: str) -> tuple[str, str]:
    if style == "math":
        return (
            f"{dataset}_avg_pass1_generation_pass_16",
            f"{dataset}_pass16_generation_pass_16",
        )
    if style == "code":
        return f"{dataset}_avg16", f"{dataset}_pass16"
    raise ValueError(f"unknown metric style: {style}")


def require_probability(entry: dict, key: str, path: Path) -> float:
    if key not in entry:
        raise ValueError(f"missing metric {key!r} in {path}")
    try:
        value = float(entry[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric {key!r} is not numeric in {path}: {entry[key]!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"metric {key!r} is outside [0, 1] in {path}: {value}")
    return value


def render_rows() -> Iterable[str]:
    for spec in SPECS:
        model_name, entry = load_single_entry(spec.result_file)
        for dataset in spec.datasets:
            avg_key, pass_key = metric_keys(spec.metric_style, dataset)
            avg = require_probability(entry, avg_key, spec.result_file)
            passed = require_probability(entry, pass_key, spec.result_file)
            relative_result = spec.result_file.relative_to(REPO_ROOT)
            yield (
                f"| {spec.label} | {spec.affected_tables} | {DISPLAY_NAMES[dataset]} | "
                f"{100 * avg:.1f} | {100 * passed:.1f} | `{relative_result}` |"
            )
        if not model_name.strip():
            raise ValueError(f"empty model name in {spec.result_file}")


def main() -> None:
    rows = list(render_rows())
    expected_rows = sum(len(spec.datasets) for spec in SPECS)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} benchmark rows, got {len(rows)}")
    print("## 已完成的缺失 Base/SFT 结果")
    print()
    print("以下数值均来自同一批 `K=16` completions；表中按百分数保留一位小数。")
    print()
    print("| 实验 | Affected tables | Benchmark | Avg@16 | Pass@16 | Result artifact |")
    print("| --- | --- | --- | ---: | ---: | --- |")
    print("\n".join(rows))
    print()
    print(f"验收通过：{len(rows)} 个 benchmark 行、{2 * len(rows)} 个唯一指标值完整。")


if __name__ == "__main__":
    main()
