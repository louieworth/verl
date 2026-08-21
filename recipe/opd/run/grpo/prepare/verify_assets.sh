#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-all}"
case "$TARGET" in
    all|models|train-data|eval-data) ;;
    *)
        echo "Usage: bash recipe/opd/run/grpo/prepare/verify_assets.sh [all|models|train-data|eval-data]" >&2
        exit 2
        ;;
esac

failed=0

check_file() {
    local path="$1"
    if [ -s "$path" ]; then
        printf 'OK      %s\n' "$path"
    else
        printf 'MISSING %s\n' "$path" >&2
        failed=1
    fi
}

check_line_count() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(wc -l < "$path" | tr -d ' ')"
    if [ "$actual" != "$expected" ]; then
        echo "ERROR: $path expected $expected records, found $actual" >&2
        exit 1
    fi
    printf 'RECORDS %s: %s\n' "$path" "$expected"
}

check_models() {
    check_file model/base/Qwen3-1.7B-Base/config.json
    check_file model/base/Qwen3-4B-Base/config.json
    check_file model/base/Qwen3-8B-Base/config.json
    check_file model/teacher/Qwen3-14B/config.json
    mkdir -p model/trained
    printf 'OK      %s\n' model/trained
}

check_train_data() {
    check_file data/train_dataset/openthoughts_math_30k_opsd/train_grpo.parquet
    check_file data/train_dataset/openthoughts_math_30k_opsd/train_sft.parquet
    check_file data/train_dataset/openthoughts_math_30k_opsd/manifest.json
    check_file data/train_dataset/taco/canonical/train_grpo.parquet
    check_file data/train_dataset/taco/canonical/train_distill.parquet
    check_file data/train_dataset/taco/canonical/train_sft.parquet
    check_file data/train_dataset/taco/canonical/manifest.json
    if [ "$failed" -eq 0 ]; then
        python3 - <<'PY'
import pyarrow.parquet as pq

import json

required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
for path in (
    "data/train_dataset/openthoughts_math_30k_opsd/train_grpo.parquet",
    "data/train_dataset/taco/canonical/train_grpo.parquet",
    "data/train_dataset/taco/canonical/train_distill.parquet",
):
    parquet = pq.ParquetFile(path)
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise SystemExit(f"{path}: missing GRPO columns {sorted(missing)}")
    if parquet.metadata.num_rows == 0:
        raise SystemExit(f"{path}: dataset is empty")
    reward_model = parquet.read_row_group(0, columns=["reward_model"]).slice(0, 1).to_pylist()[0]["reward_model"]
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        raise SystemExit(f"{path}: reward_model.ground_truth is missing")
    if parquet.metadata.num_rows % 512:
        raise SystemExit(f"{path}: row count is not padded to global batch 512")
    print(f"SCHEMA  {path}: {parquet.metadata.num_rows} rows")

distill_path = "data/train_dataset/taco/canonical/train_distill.parquet"
for row_group in range(pq.ParquetFile(distill_path).num_row_groups):
    table = pq.ParquetFile(distill_path).read_row_group(row_group, columns=["reward_model"])
    for row in table.to_pylist():
        reward = row["reward_model"] or {}
        if reward.get("style") != "distill_only" or reward.get("ground_truth"):
            raise SystemExit(f"{distill_path}: contains TACO execution tests; compact contract violated")
print(f"COMPACT {distill_path}: TACO execution tests are externalized")

grpo_path = "data/train_dataset/taco/canonical/train_grpo.parquet"
for row_group in range(pq.ParquetFile(grpo_path).num_row_groups):
    table = pq.ParquetFile(grpo_path).read_row_group(row_group, columns=["reward_model"])
    for row in table.to_pylist():
        reward = row["reward_model"] or {}
        if reward.get("style") != "rule":
            raise SystemExit(f"{grpo_path}: expected rule reward")
        try:
            test_cases = json.loads(reward.get("ground_truth") or "")
        except (TypeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{grpo_path}: invalid reward test JSON") from exc
        inputs = test_cases.get("inputs")
        outputs = test_cases.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise SystemExit(f"{grpo_path}: reward tests must be input/output lists")
        if not 1 <= len(inputs) <= 15 or len(inputs) != len(outputs):
            raise SystemExit(f"{grpo_path}: expected 1..15 aligned reward tests")
print(f"DEEPCODER {grpo_path}: binary reward over at most 15 longest-input tests")

for path in (
    "data/train_dataset/openthoughts_math_30k_opsd/train_sft.parquet",
    "data/train_dataset/taco/canonical/train_sft.parquet",
):
    parquet = pq.ParquetFile(path)
    if {"prompt", "response"} - set(parquet.schema_arrow.names):
        raise SystemExit(f"{path}: missing Base-completion SFT columns")
    print(f"SFT     {path}: {parquet.metadata.num_rows} rows")

expected_manifests = {
    "data/train_dataset/openthoughts_math_30k_opsd/manifest.json": (
        "siyanzhao/Openthoughts_math_30k_opsd",
        "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62",
    ),
    "data/train_dataset/taco/canonical/manifest.json": (
        "BAAI/TACO",
        "d593ed0a2becbbc952230bb89be09189bf1056dc",
    ),
}
for path, (repo_id, revision) in expected_manifests.items():
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("repo_id") != repo_id or manifest.get("revision") != revision:
        raise SystemExit(f"{path}: source revision is not pinned to {repo_id}@{revision}")
    if manifest.get("prompt_contract_version") != "plain_base_completion_v3":
        raise SystemExit(f"{path}: stale or unknown prompt contract")
    if manifest.get("completion_separator") != "\n":
        raise SystemExit(f"{path}: canonical completion separator must be one newline")
    expected_code_validation = "python_ast_v1" if manifest.get("task") == "code" else "not_applicable"
    if manifest.get("code_solution_validation") != expected_code_validation:
        raise SystemExit(f"{path}: stale code-solution validation contract")
    if manifest.get("prompt_coverage", 0) < 0.95:
        raise SystemExit(f"{path}: prompt coverage below 95%")
    if manifest.get("max_prompt_length") != 2048 or manifest.get("max_response_length") != 16384:
        raise SystemExit(f"{path}: canonical token caps are not 2048/16384")
    if manifest.get("task") == "code":
        if manifest.get("code_distill_artifact_version") != "taco_tests_externalized_v1":
            raise SystemExit(f"{path}: stale compact TACO artifact contract")
        expected_reward_contract = {
            "code_grpo_reward_contract_version": "deepcoder_binary_15_longest_v1",
            "code_grpo_reward_type": "binary_all_selected_tests",
            "code_grpo_test_selection": "longest_input_chars_desc_then_source_index",
            "code_grpo_max_test_cases": 15,
        }
        for key, expected in expected_reward_contract.items():
            if manifest.get(key) != expected:
                raise SystemExit(f"{path}: {key}={manifest.get(key)!r}, expected {expected!r}")
        artifacts = manifest.get("artifacts") or {}
        if set(artifacts) != {"grpo", "sft", "distill"}:
            raise SystemExit(f"{path}: incomplete code artifact manifest")
        import hashlib
        from pathlib import Path
        for name, artifact in artifacts.items():
            artifact_path = Path(artifact["path"])
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if artifact_path.stat().st_size != artifact.get("bytes") or digest != artifact.get("sha256"):
                raise SystemExit(f"{path}: {name} artifact size/hash mismatch")
    print(f"MANIFEST {path}: coverage={manifest['prompt_coverage']:.2%}")
PY
    fi
}

check_eval_data() {
    for name in aime25 aime26 hmmt26 amobench; do
        check_file "data/eval_dataset/math/$name/${name}_test.parquet"
    done
    check_file data/eval_dataset/code/evalplus/HumanEvalPlus-v0.1.10.jsonl
    check_file data/eval_dataset/code/evalplus/MbppPlus-v0.2.0.jsonl
    check_file data/eval_dataset/code/LiveCodeBench/lcb_runner/runner/main.py
    for index in 1 2 3 4 5 6; do
        file_name="test${index}.jsonl"
        [ "$index" = "1" ] && file_name="test.jsonl"
        check_file "data/eval_dataset/code/livecodebench/code_generation_lite/$file_name"
    done
    if [ "$failed" -eq 0 ]; then
        python3 - <<'PY'
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

pins_path = Path("recipe/opd/run/grpo/prepare/eval_asset_pins.json")
with pins_path.open() as stream:
    pins = json.load(stream)
if pins.get("schema_version") != 1 or not isinstance(pins.get("assets"), dict):
    raise SystemExit(f"invalid eval asset pin manifest: {pins_path}")
for raw_path, metadata in pins["assets"].items():
    path = Path(raw_path)
    if not path.is_file():
        raise SystemExit(f"pinned eval asset is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata.get("sha256"):
        raise SystemExit(
            f"{path}: sha256 mismatch; expected {metadata.get('sha256')}, found {digest}. "
            "Re-run preparation with FORCE_DOWNLOAD=true and audit upstream changes."
        )
    print(f"PINNED  {path}: {metadata['source']}@{metadata['revision']}")

math_rows = {
    "aime25": 30,
    "aime26": 30,
    "hmmt26": 33,
    "amobench": 39,
}
required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
for name, expected_rows in math_rows.items():
    path = f"data/eval_dataset/math/{name}/{name}_test.parquet"
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise SystemExit(f"{path}: expected {expected_rows} rows, found {parquet.metadata.num_rows}")
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    print(f"SCHEMA  {path}: {expected_rows} rows")
PY
        check_line_count data/eval_dataset/code/evalplus/HumanEvalPlus-v0.1.10.jsonl 164
        check_line_count data/eval_dataset/code/evalplus/MbppPlus-v0.2.0.jsonl 378
        check_line_count data/eval_dataset/code/livecodebench/code_generation_lite/test.jsonl 400
        check_line_count data/eval_dataset/code/livecodebench/code_generation_lite/test2.jsonl 111
        check_line_count data/eval_dataset/code/livecodebench/code_generation_lite/test3.jsonl 101
        check_line_count data/eval_dataset/code/livecodebench/code_generation_lite/test4.jsonl 101
        check_line_count data/eval_dataset/code/livecodebench/code_generation_lite/test5.jsonl 167
        check_line_count data/eval_dataset/code/livecodebench/code_generation_lite/test6.jsonl 175
    fi
}

if [ "$TARGET" = "all" ] || [ "$TARGET" = "models" ]; then
    check_models
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "train-data" ]; then
    check_train_data
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "eval-data" ]; then
    check_eval_data
fi

if [ "$failed" -ne 0 ]; then
    echo "Asset verification failed." >&2
    exit 1
fi
echo "Asset verification passed: $TARGET"
