#!/usr/bin/env bash
# Restore immutable canonical TACO parquets from Git chunks and prepare the
# machine-local code evaluation runtime/assets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$OPD_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

if [ -x /data2/conda/envs/verl/bin/python ]; then
    PYTHON_BIN="${PYTHON_BIN:-/data2/conda/envs/verl/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHON_BIN PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EVALPLUS_VERSION="${EVALPLUS_VERSION:-0.3.1}"
VLLM_VERSION="${VLLM_VERSION:-0.12.0}"
RUNTIME_ENV="$SCRIPT_DIR/runtime.env"
TRAIN_BUNDLE_DIR="${TRAIN_BUNDLE_DIR:-$REPO_ROOT/data/train_dataset/taco/bundle}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$REPO_ROOT/data/train_dataset/taco/canonical}"

usage() {
    cat >&2 <<'EOF'
Usage: bash recipe/opd/script_code/prepare_code.sh [all|train|bundle|restore-train|eval|verify]

  all     restore/verify Git-bundled TACO parquets, then prepare/verify code eval
          datasets/runtime (recommended for each fresh EC2/Docker worker)
  train   build the canonical TACO parquets; use FORCE_DOWNLOAD=true to rebuild
  bundle  split the verified canonical parquets into <=40MB Git-safe chunks
  restore-train
          atomically restore the exact canonical parquets from Git chunks
  eval    install/verify the code eval runtime and download pinned eval datasets
  verify  offline verification of both restored train and prepared eval assets

Training parquets live under data/train_dataset/taco/canonical. `all` restores
them byte-for-byte from data/train_dataset/taco/bundle; it never
downloads or reprocesses TACO. Run `bundle` only after intentionally rebuilding
and verifying the canonical artifacts on a preparation machine. `all` may use
pip to install missing Python/code-evaluation dependencies; set
CODE_PREPARE_INSTALL_RUNTIME=false to require a preinstalled environment.
EOF
}

die() {
    echo "prepare_code.sh: $*" >&2
    exit 2
}

python_core_ready() {
    "$PYTHON_BIN" -c 'import datasets, huggingface_hub, pyarrow, transformers' >/dev/null 2>&1
}

prepare_python_core() {
    if python_core_ready; then
        return 0
    fi
    case "${CODE_PREPARE_INSTALL_RUNTIME:-true}" in
        true|1|yes)
            echo "[runtime] Installing code data/evaluation Python dependencies..."
            "$PYTHON_BIN" -m pip install \
                "datasets" \
                "huggingface-hub" \
                "pyarrow>=19.0.0" \
                "transformers"
            ;;
        *)
            die "Python is missing datasets/huggingface_hub/pyarrow/transformers and runtime installation is disabled"
            ;;
    esac
    python_core_ready || die "code data/evaluation Python dependency installation failed"
}

restore_code_train() {
    "$PYTHON_BIN" recipe/opd/script_code/materialize_train_bundle.py restore \
        --bundle-dir "$TRAIN_BUNDLE_DIR" \
        --output-dir "$TRAIN_OUTPUT_DIR"
}

bundle_code_train() {
    "$PYTHON_BIN" recipe/opd/script_code/materialize_train_bundle.py pack \
        --source-dir "$TRAIN_OUTPUT_DIR" \
        --bundle-dir "$TRAIN_BUNDLE_DIR" \
        --chunk-bytes "${TRAIN_BUNDLE_CHUNK_BYTES:-40000000}" \
        --overwrite
}

verify_code_train() {
    "$PYTHON_BIN" - <<'PY'
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

root = Path("data/train_dataset/taco/canonical")
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(
        f"missing restored TACO manifest: {manifest_path}\n"
        "Run prepare_code.sh restore-train (or all)."
    )
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "repo_id": "BAAI/TACO",
    "revision": "d593ed0a2becbbc952230bb89be09189bf1056dc",
    "prompt_contract_version": "plain_base_completion_v3",
    "code_distill_artifact_version": "taco_tests_externalized_v1",
    "code_grpo_reward_contract_version": "deepcoder_binary_15_longest_v1",
    "code_grpo_reward_type": "binary_all_selected_tests",
    "code_grpo_test_selection": "longest_input_chars_desc_then_source_index",
    "code_grpo_max_test_cases": 15,
    "max_prompt_length": 2048,
    "max_response_length": 16384,
    "pad_to_multiple": 512,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"{manifest_path}: {key}={manifest.get(key)!r}, expected {value!r}")

artifacts = manifest.get("artifacts") or {}
if set(artifacts) != {"grpo", "sft", "distill"}:
    raise SystemExit(f"{manifest_path}: expected grpo/sft/distill artifact records")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

tables = {}
for name, metadata in artifacts.items():
    path = Path(metadata["path"])
    if not path.is_file():
        raise SystemExit(f"missing restored TACO artifact: {path}")
    if path.stat().st_size != metadata.get("bytes") or sha256(path) != metadata.get("sha256"):
        raise SystemExit(f"TACO artifact does not match manifest size/hash: {path}")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != metadata.get("rows") or parquet.metadata.num_rows % 512:
        raise SystemExit(f"invalid row count for {path}: {parquet.metadata.num_rows}")
    codecs = {
        parquet.metadata.row_group(group).column(column).compression
        for group in range(parquet.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    }
    if codecs != {"ZSTD"}:
        raise SystemExit(f"{path}: expected only ZSTD parquet columns, found {sorted(codecs)}")
    tables[name] = parquet

required_rl = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
for name in ("grpo", "distill"):
    missing = required_rl - set(tables[name].schema_arrow.names)
    if missing:
        raise SystemExit(f"{name} TACO artifact is missing columns: {sorted(missing)}")
if {"prompt", "response"} - set(tables["sft"].schema_arrow.names):
    raise SystemExit("SFT TACO artifact is missing prompt/response")

for group in range(tables["grpo"].num_row_groups):
    rewards = tables["grpo"].read_row_group(group, columns=["reward_model"]).to_pylist()
    for row in rewards:
        reward = row["reward_model"] or {}
        if reward.get("style") != "rule":
            raise SystemExit("TACO GRPO parquet must use rule rewards")
        try:
            test_cases = json.loads(reward.get("ground_truth") or "")
        except (TypeError, json.JSONDecodeError) as exc:
            raise SystemExit("TACO GRPO parquet contains invalid test JSON") from exc
        inputs = test_cases.get("inputs")
        outputs = test_cases.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise SystemExit("TACO GRPO reward tests must contain input/output lists")
        if not 1 <= len(inputs) <= 15 or len(inputs) != len(outputs):
            raise SystemExit("TACO GRPO reward suite must contain 1..15 aligned tests")

for group in range(tables["distill"].num_row_groups):
    rewards = tables["distill"].read_row_group(group, columns=["reward_model"]).to_pylist()
    for row in rewards:
        reward = row["reward_model"] or {}
        if reward.get("style") != "distill_only" or reward.get("ground_truth"):
            raise SystemExit("compact TACO distill parquet contains execution-test payloads")

sizes = ", ".join(
    f"{name}={Path(metadata['path']).stat().st_size / (1024**2):.1f}MiB"
    for name, metadata in sorted(artifacts.items())
)
print(f"TACO training artifacts verified: {sizes}")
PY
}

runtime_ready() {
    (
        cd "$REPO_ROOT/external/LiveCodeBench"
        PYTHONPATH="$REPO_ROOT/external/LiveCodeBench:$REPO_ROOT:${PYTHONPATH:-}" \
            "$PYTHON_BIN" - "$EVALPLUS_VERSION" "$VLLM_VERSION" <<'PY'
import importlib.metadata
import sys

expected_evalplus, expected_vllm = sys.argv[1:]
if importlib.metadata.version("evalplus") != expected_evalplus:
    raise SystemExit(1)
if importlib.metadata.version("vllm") != expected_vllm:
    raise SystemExit(1)
import evalplus  # noqa: F401
import lcb_runner.runner.main  # noqa: F401
import vllm  # noqa: F401
PY
    ) >/dev/null 2>&1
}

prepare_runtime() {
    if ! runtime_ready; then
        case "${CODE_PREPARE_INSTALL_RUNTIME:-true}" in
            true|1|yes)
                echo "[runtime] Installing pinned EvalPlus and repository LiveCodeBench runtime..."
                "$PYTHON_BIN" -m pip install \
                    --no-build-isolation \
                    "evalplus[vllm]==$EVALPLUS_VERSION" \
                    "vllm==$VLLM_VERSION" \
                    "$REPO_ROOT/external/LiveCodeBench"
                ;;
            *)
                die "code eval runtime is incomplete and CODE_PREPARE_INSTALL_RUNTIME is disabled"
                ;;
        esac
    fi
    runtime_ready || die "code eval runtime validation failed after installation"

    local python_path
    python_path="$(command -v "$PYTHON_BIN")"
    cat > "$RUNTIME_ENV" <<EOF
# Generated by recipe/opd/script_code/prepare_code.sh; do not commit.
export PYTHON_BIN='$python_path'
export EVALPLUS_VERSION='$EVALPLUS_VERSION'
export VLLM_VERSION='$VLLM_VERSION'
export CODE_EVAL_DATA_ROOT='$REPO_ROOT/data/eval_dataset/code'
export HUMANEVAL_OVERRIDE_PATH='$REPO_ROOT/data/eval_dataset/code/evalplus/HumanEvalPlus-v0.1.10.jsonl'
export MBPP_OVERRIDE_PATH='$REPO_ROOT/data/eval_dataset/code/evalplus/MbppPlus-v0.2.0.jsonl'
export LCB_REPO='$REPO_ROOT/data/eval_dataset/code/LiveCodeBench'
export LCB_CODEGEN_LITE_DIR='$REPO_ROOT/data/eval_dataset/code/livecodebench/code_generation_lite'
export PYTHONPATH='$REPO_ROOT/external/LiveCodeBench:$REPO_ROOT':"\${PYTHONPATH:-}"
EOF
    echo "Code eval runtime ready: $RUNTIME_ENV"
}

prepare_eval() {
    prepare_runtime
    bash recipe/opd/run/grpo/prepare/prepare_data.sh code-eval-data
}

verify_code_eval() {
    [ -f "$RUNTIME_ENV" ] || die "missing runtime.env; run prepare_code.sh eval"
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV"
    runtime_ready || die "pinned EvalPlus/LiveCodeBench runtime is not importable"
    "$PYTHON_BIN" - <<'PY'
import hashlib
import json
from pathlib import Path

pins = json.loads(Path("recipe/opd/run/grpo/prepare/eval_asset_pins.json").read_text())
code_assets = {
    Path(path): metadata
    for path, metadata in pins["assets"].items()
    if path.startswith("data/eval_dataset/code/")
}
if len(code_assets) != 8:
    raise SystemExit(f"expected 8 pinned code eval data files, found {len(code_assets)}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path, metadata in code_assets.items():
    if not path.is_file() or sha256(path) != metadata["sha256"]:
        raise SystemExit(f"missing or hash-mismatched code eval asset: {path}")

runtime = Path("data/eval_dataset/code/LiveCodeBench/lcb_runner/runner/main.py")
if not runtime.is_file():
    raise SystemExit(f"missing LiveCodeBench runtime copy: {runtime}")
print("Code eval assets verified: HumanEval+ 164, MBPP+ 378, LiveCodeBench v6 1055")
PY
    CODE_GRPO_REWARD_CONTRACT=deepcoder_binary_15_longest_v1 \
    CODE_GRPO_MAX_TEST_CASES=15 \
    CODE_GRPO_EXEC_TIMEOUT_SECONDS=10 \
        "$PYTHON_BIN" recipe/opd/script_code/verify_code_runtime.py \
            --evalplus-version "$EVALPLUS_VERSION" \
            --vllm-version "$VLLM_VERSION" \
            --lcb-repo "$LCB_REPO"
}

print_ready_summary() {
    cat <<EOF

Code bootstrap READY
  train/grpo:    $TRAIN_OUTPUT_DIR/train_grpo.parquet
  train/sft:     $TRAIN_OUTPUT_DIR/train_sft.parquet
  train/distill: $TRAIN_OUTPUT_DIR/train_distill.parquet
  eval:          HumanEval+ 164 / MBPP+ 378 / LiveCodeBench v6 1055
  runtime:       EvalPlus $EVALPLUS_VERSION / vLLM $VLLM_VERSION / LiveCodeBench / local execution PASS
  environment:   $RUNTIME_ENV
EOF
}

mode="${1:-all}"
[ "$#" -le 1 ] || { usage; exit 2; }
case "$mode" in
    -h|--help) usage; exit 0 ;;
esac
prepare_python_core
case "$mode" in
    all)
        restore_code_train
        verify_code_train
        prepare_eval
        verify_code_eval
        print_ready_summary
        ;;
    train)
        train_args=()
        [ "${FORCE_DOWNLOAD:-false}" = true ] && train_args+=(--overwrite)
        bash recipe/opd/script_code/prepare_data.sh "${train_args[@]}"
        verify_code_train
        ;;
    bundle)
        verify_code_train
        bundle_code_train
        ;;
    restore-train)
        restore_code_train
        verify_code_train
        ;;
    eval)
        prepare_eval
        verify_code_eval
        ;;
    verify)
        verify_code_train
        verify_code_eval
        ;;
    *) usage; exit 2 ;;
esac

echo "Code preparation complete (scope=$mode)."
