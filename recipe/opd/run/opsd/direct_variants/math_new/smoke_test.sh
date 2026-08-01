#!/usr/bin/env bash
# CPU-only structural smoke test for every math_new model/variant wrapper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$VERL_ROOT"

mapfile -t wrappers < <(
    find "$SCRIPT_DIR" \
        -mindepth 2 \
        -maxdepth 2 \
        -type f \
        -name '*.sh' \
        -print |
        sort
)

expected_variants=(
    forward_kl.sh
    forward_kl_clip.sh
    forward_kl_y_r.sh
    reverse_kl.sh
    reverse_kl_topk.sh
)
for model_dir in qwen3-4b-instruct qwen3-8b; do
    for variant in "${expected_variants[@]}"; do
        if [ ! -x "$SCRIPT_DIR/$model_dir/$variant" ]; then
            echo "ERROR: missing executable wrapper: $model_dir/$variant" >&2
            exit 1
        fi
    done
done

if [ ! -x "$SCRIPT_DIR/qwen3-8b/skd.sh" ]; then
    echo "ERROR: missing executable wrapper: qwen3-8b/skd.sh" >&2
    exit 1
fi

if [ "${#wrappers[@]}" -ne 11 ]; then
    echo "ERROR: expected 11 model/variant wrappers, found ${#wrappers[@]}" >&2
    printf '  %s\n' "${wrappers[@]}" >&2
    exit 1
fi

bash -n "$SCRIPT_DIR/_run_math_direct_variant.sh" "${wrappers[@]}"

python3 - "$VERL_ROOT" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
train = root / "data/train_dataset/deepscaler/train_grpo.parquet"
required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
table = pq.read_table(train)
missing = required - set(table.column_names)
if missing:
    raise SystemExit(f"{train} is missing columns: {sorted(missing)}")
if table.num_rows == 0:
    raise SystemExit(f"{train} contains no rows")
for index, row in enumerate(table.select(["prompt", "extra_info"]).to_pylist()):
    if not row["prompt"]:
        raise SystemExit(f"{train} row {index} has an empty prompt")
    extra = row["extra_info"] or {}
    if not str(extra.get("expert_cot", "")).strip():
        raise SystemExit(f"{train} row {index} has no expert_cot for OPSD")

for name in ("aime24", "aime25", "hmmt25", "beyondaime", "amobench"):
    path = root / f"data/eval_dataset/math/{name}/{name}_test.parquet"
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0:
        raise SystemExit(f"{path} contains no rows")
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")

print(f"data schema: OK ({table.num_rows} training rows)")
PY

for wrapper in "${wrappers[@]}"; do
    bash "$wrapper" --dry-run >/dev/null
done

skd_preflight="$(RUN_DATE=smoke bash "$SCRIPT_DIR/qwen3-8b/skd.sh" --dry-run)"
[[ "$skd_preflight" == *"run name:                 y_o_skd_kl_forward_full_vocab_clip0_vanilla_ms1_smoke"* ]]
[[ "$skd_preflight" == *"evaluation:               after_train=true, avg@16 / pass@16"* ]]

stale_root="../outside"
poisoned_preflight="$(
    HF_HOME="$stale_root/huggingface" \
    TRAIN_DATA_PATH="$stale_root/train.parquet" \
    PRECOMPUTED_STAGE1_PROMPTS_PATH="$stale_root/prompts.parquet" \
    EVAL_DATASETS_DIR="$stale_root/eval" \
    MODEL_SAVE_DIR="$stale_root/model" \
    PIPELINE_ARCHIVE_MODEL_ROOT="$stale_root/archive" \
    PIPELINE_TEMP_MODEL_DIR="$stale_root/tmp" \
    GEN_RESULTS_ROOT="$stale_root/gen" \
    OUTPUT_DIR="$stale_root/output" \
    RESULTS_FILE="$stale_root/results.json" \
    DATA_PATH="$stale_root/data.parquet" \
    EVAL_OUTPUT_DIR="$stale_root/eval-output" \
    EVAL_RESULTS_FILE="$stale_root/eval-results.json" \
    TOKENIZER_PATH="$stale_root/tokenizer" \
    bash "$SCRIPT_DIR/qwen3-8b/forward_kl.sh" --dry-run
)"
if [[ "$poisoned_preflight" == *"$stale_root/"* ]]; then
    echo "ERROR: a stale machine-wide path leaked into the resolved config" >&2
    printf '%s\n' "$poisoned_preflight" >&2
    exit 1
fi

if rg -n '/(data2?|home|opt)/|DeepScaleR-Cleaned' "$SCRIPT_DIR" \
    --glob '*.sh' \
    --glob '!smoke_test.sh' \
    --glob '*.md'; then
    echo "ERROR: math_new contains a machine-specific absolute path" >&2
    exit 1
fi

echo "math_new smoke test: PASS (${#wrappers[@]} wrappers)"
