#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-all}"
case "$TARGET" in
    all|train-data|eval-data) ;;
    *)
        echo "Usage: bash recipe/opd/run/grpo/prepare/prepare_data.sh [all|train-data|eval-data]" >&2
        exit 2
        ;;
esac

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-data/download_cache/huggingface}"
mkdir -p data/train_dataset data/eval_dataset "$HF_HOME"

download_snapshot() {
    local repo_id="$1"
    local destination="$2"
    shift 2
    python3 - "$repo_id" "$destination" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, destination, *patterns = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=destination,
    allow_patterns=patterns or None,
)
PY
}

prepare_train_data() {
    local math_raw="data/train_dataset/deepscaler/raw"
    local math_output="data/train_dataset/deepscaler/train_grpo.parquet"
    local code_raw="data/train_dataset/taco/raw"
    local code_output="data/train_dataset/taco/train_grpo.parquet"

    if [ ! -s "$math_output" ] || [ "${FORCE_DOWNLOAD:-false}" = "true" ]; then
        mkdir -p "$math_raw"
        download_snapshot \
            "${MATH_TRAIN_REPO:-agentica-org/DeepScaleR-Preview-Dataset}" \
            "$math_raw" \
            "*.json" "*.jsonl" "*.parquet" "README.md"
        math_overwrite=()
        [ "${FORCE_DOWNLOAD:-false}" = "true" ] && math_overwrite=(--overwrite)
        python3 recipe/opd/run/grpo/prepare/prepare_grpo_train_data.py \
            --task math \
            --input "$math_raw" \
            --output "$math_output" \
            "${math_overwrite[@]}"
    else
        echo "Math GRPO train parquet already exists, skipping: $math_output"
    fi

    if [ ! -s "$code_output" ] || [ "${FORCE_DOWNLOAD:-false}" = "true" ]; then
        mkdir -p "$code_raw"
        download_snapshot \
            "${CODE_TRAIN_REPO:-BAAI/TACO}" \
            "$code_raw" \
            "ALL/train-*.parquet" "README.md"
        code_overwrite=()
        [ "${FORCE_DOWNLOAD:-false}" = "true" ] && code_overwrite=(--overwrite)
        python3 recipe/opd/run/grpo/prepare/prepare_grpo_train_data.py \
            --task code \
            --input "$code_raw" \
            --output "$code_output" \
            "${code_overwrite[@]}"
    else
        echo "Code GRPO train parquet already exists, skipping: $code_output"
    fi
}

copy_lcb_runtime() {
    local source_dir="external/LiveCodeBench"
    local destination="data/eval_dataset/code/LiveCodeBench"
    if [ ! -f "$source_dir/lcb_runner/runner/main.py" ]; then
        echo "ERROR: repository is missing $source_dir/lcb_runner/runner/main.py" >&2
        exit 1
    fi
    mkdir -p "$destination"
    cp -a "$source_dir/lcb_runner" "$destination/"
    for file_name in pyproject.toml LICENSE; do
        [ ! -f "$source_dir/$file_name" ] || cp -a "$source_dir/$file_name" "$destination/$file_name"
    done
}

prepare_eval_data() {
    local math_dir="data/eval_dataset/math"
    local code_dir="data/eval_dataset/code"
    mkdir -p "$math_dir" "$code_dir"

    local math_ready=true dataset_name
    for dataset_name in aime24 aime25 hmmt25 beyondaime amobench; do
        [ -s "$math_dir/$dataset_name/${dataset_name}_test.parquet" ] || math_ready=false
    done
    if [ "$math_ready" = "true" ] && [ "${FORCE_DOWNLOAD:-false}" != "true" ]; then
        echo "Math evaluation datasets already exist, skipping downloads."
    else
        export HF_HOME="$math_dir/huggingface_cache"
        python3 recipe/math_evaluation/datasets/prepare_aime.py \
            --local_save_dir "$math_dir"
        python3 recipe/math_evaluation/datasets/prepare_hmmt.py \
            --local_dataset_path "$math_dir"
        additional_args=()
        [ "${FORCE_DOWNLOAD:-false}" = "true" ] && additional_args=(--overwrite)
        python3 recipe/math_evaluation/datasets/prepare_additional_eval_datasets.py \
            --datasets beyondaime,amobench \
            --local_save_dir "$math_dir" \
            "${additional_args[@]}"
    fi

    python3 recipe/opd/run/grpo/prepare/download_evalplus_data.py \
        --output-dir "$code_dir/evalplus"
    copy_lcb_runtime

    local lcb_data_dir="$code_dir/livecodebench/code_generation_lite"
    if [ -n "${LCB_SOURCE_DIR:-}" ]; then
        mkdir -p "$lcb_data_dir"
        cp -a "$LCB_SOURCE_DIR/." "$lcb_data_dir/"
    fi
    export HF_HOME="$code_dir/huggingface_cache"
    python3 recipe/opd/run/grpo/prepare/prepare_lcb_release_v6.py \
        --output-dir "$lcb_data_dir"
}

if [ "$TARGET" = "all" ] || [ "$TARGET" = "train-data" ]; then
    prepare_train_data
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "eval-data" ]; then
    prepare_eval_data
fi

echo "Dataset preparation completed for: $TARGET"
