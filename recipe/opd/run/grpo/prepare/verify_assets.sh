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
    check_file model/base/Qwen3-4B-Instruct-2507/config.json
    check_file model/base/Qwen3-8B/config.json
    mkdir -p model/trained
    printf 'OK      %s\n' model/trained
}

check_train_data() {
    check_file data/train_dataset/deepscaler/train_grpo.parquet
    check_file data/train_dataset/taco/train_grpo.parquet
    check_file data/train_dataset/taco/train_grpo_expert_cot.parquet
    if [ "$failed" -eq 0 ]; then
        python3 - <<'PY'
import pyarrow.parquet as pq

required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
for path in (
    "data/train_dataset/deepscaler/train_grpo.parquet",
    "data/train_dataset/taco/train_grpo.parquet",
    "data/train_dataset/taco/train_grpo_expert_cot.parquet",
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
    print(f"SCHEMA  {path}: {parquet.metadata.num_rows} rows")

expert_path = "data/train_dataset/taco/train_grpo_expert_cot.parquet"
for index, row in enumerate(pq.read_table(expert_path, columns=["extra_info"]).to_pylist()):
    expert_cot = str((row["extra_info"] or {}).get("expert_cot") or "").strip()
    if not expert_cot:
        raise SystemExit(f"{expert_path}: empty extra_info.expert_cot at row {index}")
print(f"EXPERT  {expert_path}: every row has non-empty extra_info.expert_cot")
PY
    fi
}

check_eval_data() {
    for name in aime24 aime25 hmmt25 beyondaime amobench; do
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
import pyarrow.parquet as pq

math_rows = {
    "aime24": 30,
    "aime25": 30,
    "hmmt25": 30,
    "beyondaime": 100,
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
