#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VERL_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python3}

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    shift
fi

MODEL_PATHS=("$@")
if [ ${#MODEL_PATHS[@]} -eq 0 ] && [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATHS=("$MODEL_PATH")
fi
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "ERROR: specify at least one model path" >&2
    exit 1
fi
is_hf_repo_id() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

for mp in "${MODEL_PATHS[@]}"; do
    if [ ! -d "$mp" ] && ! is_hf_repo_id "$mp"; then
        echo "ERROR: model must be an existing local directory or a Hugging Face repo ID: $mp" >&2
        exit 1
    fi
done

DATASETS_TO_TEST=${DATASETS:-"humaneval_plus mbpp_plus livecodebench_v6"}
DATASETS_TO_TEST=${DATASETS_TO_TEST//,/ }
METRICS_DATASETS=${EVAL_ALL_DATASETS:-${EVAL_DATASETS:-$DATASETS_TO_TEST}}
METRICS_DATASETS=${METRICS_DATASETS//,/ }
PASS_K=${PASS_K:-16}
export PASS_K
TEMPERATURE=${CODE_EVAL_TEMPERATURE:-0.6}
TOP_P=${CODE_EVAL_TOP_P:-0.95}
EVAL_SEED=${CODE_EVAL_SEED:-42}
MAX_PROMPT_TOKENS=${CODE_EVAL_MAX_PROMPT_TOKENS:-2048}
MAX_TOKENS=${CODE_EVAL_MAX_RESPONSE_TOKENS:-${CODE_EVAL_MAX_TOKENS:-16384}}
MAX_MODEL_LEN=${CODE_EVAL_MAX_MODEL_LEN:-$((MAX_PROMPT_TOKENS + MAX_TOKENS))}
CODE_EVAL_MAX_NUM_SEQS=${CODE_EVAL_MAX_NUM_SEQS:-128}
CODE_EVAL_LCB_ENFORCE_EAGER=${CODE_EVAL_LCB_ENFORCE_EAGER:-false}
EVAL_SIGNATURE="${CODE_EVAL_SIGNATURE:-n${PASS_K}_t${TEMPERATURE}_p${TOP_P}_prompt${MAX_PROMPT_TOKENS}_response${MAX_TOKENS}_seed${EVAL_SEED}_base}"
export CODE_EVAL_SIGNATURE="$EVAL_SIGNATURE"
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
GEN_TP=${GEN_TP:-$NGPUS_PER_NODE}
if ! [[ "$NGPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "$GEN_TP" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NGPUS_PER_NODE and GEN_TP must be positive integers" >&2
    exit 1
fi
if [ "$GEN_TP" -gt "$NGPUS_PER_NODE" ] || [ $((NGPUS_PER_NODE % GEN_TP)) -ne 0 ]; then
    echo "ERROR: GEN_TP=$GEN_TP must divide NGPUS_PER_NODE=$NGPUS_PER_NODE" >&2
    exit 1
fi
CODE_EVAL_DATA_ROOT=${CODE_EVAL_DATA_ROOT:-${EVAL_DATASETS_DIR:-data/eval_dataset/code}}
if [[ "$CODE_EVAL_DATA_ROOT" != /* ]]; then
    CODE_EVAL_DATA_ROOT="$VERL_ROOT/$CODE_EVAL_DATA_ROOT"
fi
LCB_REPO=${LCB_REPO:-$CODE_EVAL_DATA_ROOT/LiveCodeBench}
LCB_CODEGEN_LITE_DIR=${LCB_CODEGEN_LITE_DIR:-$CODE_EVAL_DATA_ROOT/livecodebench/code_generation_lite}
HUMANEVAL_OVERRIDE_PATH=${HUMANEVAL_OVERRIDE_PATH:-$CODE_EVAL_DATA_ROOT/evalplus/HumanEvalPlus-v0.1.10.jsonl}
MBPP_OVERRIDE_PATH=${MBPP_OVERRIDE_PATH:-$CODE_EVAL_DATA_ROOT/evalplus/MbppPlus-v0.2.0.jsonl}
[[ "$LCB_REPO" == /* ]] || LCB_REPO="$VERL_ROOT/$LCB_REPO"
[[ "$LCB_CODEGEN_LITE_DIR" == /* ]] || LCB_CODEGEN_LITE_DIR="$VERL_ROOT/$LCB_CODEGEN_LITE_DIR"
[[ "$HUMANEVAL_OVERRIDE_PATH" == /* ]] || HUMANEVAL_OVERRIDE_PATH="$VERL_ROOT/$HUMANEVAL_OVERRIDE_PATH"
[[ "$MBPP_OVERRIDE_PATH" == /* ]] || MBPP_OVERRIDE_PATH="$VERL_ROOT/$MBPP_OVERRIDE_PATH"
export HUMANEVAL_OVERRIDE_PATH MBPP_OVERRIDE_PATH LCB_CODEGEN_LITE_DIR
RESULTS_BASE_DIR=${RESULTS_BASE_DIR:-results}
GEN_OUTPUT_BASE_DIR=${GEN_OUTPUT_BASE_DIR:-gen_results/code_eval}
WRITE_RESULTS_CSV=${WRITE_RESULTS_CSV:-true}
case " $DATASETS_TO_TEST " in
    *" humaneval_plus "*|*" humaneval+ "*)
        [ -s "$HUMANEVAL_OVERRIDE_PATH" ] || {
            echo "ERROR: local HumanEval+ dataset not found: $HUMANEVAL_OVERRIDE_PATH" >&2
            exit 1
        }
        ;;
esac
case " $DATASETS_TO_TEST " in
    *" mbpp_plus "*|*" mbpp+ "*)
        [ -s "$MBPP_OVERRIDE_PATH" ] || {
            echo "ERROR: local MBPP+ dataset not found: $MBPP_OVERRIDE_PATH" >&2
            exit 1
        }
        ;;
esac
case " $DATASETS_TO_TEST " in
    *" livecodebench"*|*" lcb_"*)
        if [ ! -d "$LCB_REPO/lcb_runner" ]; then
            echo "ERROR: LiveCodeBench repo not found at $LCB_REPO" >&2
            exit 1
        fi
        for lcb_file in test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl test6.jsonl; do
            if [ ! -s "$LCB_CODEGEN_LITE_DIR/$lcb_file" ]; then
                echo "ERROR: LiveCodeBench v6 dataset file not found: $LCB_CODEGEN_LITE_DIR/$lcb_file" >&2
                exit 1
            fi
        done
        ;;
esac

require_module() {
    "$PYTHON_BIN" - "$1" <<'PYMOD'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PYMOD
}

update_result() {
    "$PYTHON_BIN" - "$@" <<'PYUPDATE'
import json
import os
import sys
results_file, model_name, model_path, dataset, avg, passed, num_problems = sys.argv[1:]
avg = None if avg == '' else float(avg)
passed = None if passed == '' else float(passed)
num_problems = None if num_problems == '' else int(num_problems)
data = {}
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
pass_k = os.environ.get("PASS_K", "16")
entry = data.setdefault(model_name, {})
entry["model_path"] = model_path
if avg is not None:
    entry[f"{dataset}_avg{pass_k}"] = avg
if passed is not None:
    entry[f"{dataset}_pass{pass_k}"] = passed
if num_problems is not None:
    entry[f"{dataset}_num_problems"] = num_problems
tmp = results_file + '.tmp'
os.makedirs(os.path.dirname(results_file) or '.', exist_ok=True)
with open(tmp, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
os.replace(tmp, results_file)
PYUPDATE
}

evalplus_dataset() {
    local model_path="$1" model_name="$2" results_file="$3" out_dir="$4" dataset_name="$5" evalplus_name="$6"
    require_module evalplus || { echo "ERROR: evalplus is not installed. Install: pip install --upgrade 'evalplus[vllm] @ git+https://github.com/evalplus/evalplus'" >&2; exit 1; }
    local root="$out_dir/evalplus/$EVAL_SIGNATURE"
    mkdir -p "$root"
    local metrics avg pass num_problems
    if metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_evalplus_metrics.py" --root "$root" --dataset "$evalplus_name" --pass_k "$PASS_K" 2>/dev/null); then
        echo "Existing EvalPlus $evalplus_name eval results found under $root; skipping generation/evaluation."
    else
        "$PYTHON_BIN" "$SCRIPT_DIR/run_evalplus_vllm.py" \
            --dataset "$evalplus_name" \
            --model "$model_path" \
            --root "$root" \
            --tp "$GEN_TP" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
            --max_tokens "$MAX_TOKENS" \
            --max_model_len "$MAX_MODEL_LEN" \
            --max_num_seqs "$CODE_EVAL_MAX_NUM_SEQS" \
            --n_samples "$PASS_K" \
            --seed "$EVAL_SEED" \
            --trust_remote_code \
            --force_base_prompt \
            ${EVALPLUS_EXTRA_ARGS:-}
        metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_evalplus_metrics.py" --root "$root" --dataset "$evalplus_name" --pass_k "$PASS_K")
    fi
    avg=$(echo "$metrics" | awk -F= '/^avg=/{print $2}')
    pass=$(echo "$metrics" | awk -F= '/^pass=/{print $2}')
    num_problems=$(echo "$metrics" | awk -F= '/^num_problems=/{print $2}')
    update_result "$results_file" "$model_name" "$model_path" "$dataset_name" "$avg" "$pass" "$num_problems"
}

lcb_dataset() {
    local model_path="$1" model_name="$2" results_file="$3" out_dir="$4"
    local lcb_repo="$LCB_REPO"
    if ! (cd "$lcb_repo" && require_module lcb_runner); then
        echo "ERROR: lcb_runner is not importable from $lcb_repo" >&2
        exit 1
    fi
    # --model selects only LiveCodeBench's prompt/parser style because
    # --local_model_path supplies the actual checkpoint. GenericBase emits a
    # raw completion prompt and does not add Qwen instruct/chat markers.
    local lcb_model_key="${LCB_MODEL_KEY:-bigcode/starcoder2-3b}"
    local enforce_eager_arg=""
    if [ "$CODE_EVAL_LCB_ENFORCE_EAGER" = "true" ]; then
        enforce_eager_arg="--enforce_eager"
    fi
    local root="$out_dir/livecodebench/$EVAL_SIGNATURE"
    mkdir -p "$root"
    local metrics avg pass num_problems
    local lcb_extract_args=(
        --root "$root"
        --pass_k "$PASS_K"
        --expected_tasks "${LCB_EXPECTED_TASKS:-1055}"
    )
    if [ "${LCB_AGGREGATE_ALL:-false}" = "true" ]; then
        lcb_extract_args+=(--aggregate_all)
    fi
    if metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_lcb_metrics.py" "${lcb_extract_args[@]}" 2>/dev/null); then
        echo "Existing LiveCodeBench v6 eval results found under $root; skipping generation/evaluation."
        avg=$(echo "$metrics" | awk -F= '/^avg=/{print $2}')
        pass=$(echo "$metrics" | awk -F= '/^pass=/{print $2}')
        num_problems=$(echo "$metrics" | awk -F= '/^num_problems=/{print $2}')
        update_result "$results_file" "$model_name" "$model_path" "livecodebench_v6" "$avg" "$pass" "$num_problems"
        return
    fi
    local runtime_dir="$root/runtime"
    mkdir -p "$runtime_dir"
    # LiveCodeBench loads its few-shot prompt fixtures through paths relative to
    # the current working directory.  Keep outputs isolated in runtime_dir while
    # making those repository-relative paths available there.
    if [ ! -e "$runtime_dir/lcb_runner" ]; then
        ln -s "$lcb_repo/lcb_runner" "$runtime_dir/lcb_runner"
    fi
    (
        cd "$runtime_dir"
        export PYTHONPATH="$lcb_repo:${PYTHONPATH:-}"
        export LCB_BASE_PROMPT_MAX_TOKENS="$MAX_PROMPT_TOKENS"
        export LCB_BASE_SEED="$EVAL_SEED"
        "$PYTHON_BIN" "$SCRIPT_DIR/run_lcb_base.py" \
            --model "$lcb_model_key" \
            --local_model_path "$model_path" \
            --trust_remote_code \
            --scenario codegeneration \
            --evaluate \
            --release_version release_v6 \
            --n "$PASS_K" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_tokens "$MAX_TOKENS" \
            --max_model_len "$MAX_MODEL_LEN" \
            --max_num_seqs "$CODE_EVAL_MAX_NUM_SEQS" \
            --tensor_parallel_size "$GEN_TP" \
            --continue_existing \
            --use_cache \
            --cache_batch_size "${LCB_CACHE_BATCH_SIZE:-32}" \
            $enforce_eager_arg \
            ${LCB_EXTRA_ARGS:-}
    )
    metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_lcb_metrics.py" "${lcb_extract_args[@]}")
    avg=$(echo "$metrics" | awk -F= '/^avg=/{print $2}')
    pass=$(echo "$metrics" | awk -F= '/^pass=/{print $2}')
    num_problems=$(echo "$metrics" | awk -F= '/^num_problems=/{print $2}')
    update_result "$results_file" "$model_name" "$model_path" "livecodebench_v6" "$avg" "$pass" "$num_problems"
}

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    if [ "$(basename "$MODEL_PATH")" = "hf_merged" ]; then
        FULL_MODEL_DIR=$(dirname "$MODEL_PATH")
        FULL_MODEL_NAME=$(basename "$FULL_MODEL_DIR")
    else
        FULL_MODEL_DIR="$MODEL_PATH"
        FULL_MODEL_NAME=$(basename "$MODEL_PATH")
    fi
    BASE_MODEL_NAME="${EVAL_BASE_MODEL_NAME:-$(echo "$FULL_MODEL_NAME" | sed -E 's/_kl_.*$//')}"
    MODEL_NAME="${EVAL_MODEL_NAME:-$FULL_MODEL_NAME}"
    GEN_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$GEN_OUTPUT_BASE_DIR/$MODEL_NAME}"
    RESULTS_FILE="${EVAL_RESULTS_FILE:-$RESULTS_BASE_DIR/${BASE_MODEL_NAME}_code.json}"
    mkdir -p "$GEN_OUTPUT_DIR" "$(dirname "$RESULTS_FILE")"

    echo "################################################################################"
    echo "# Code Model Benchmark"
    echo "# Model:      $MODEL_PATH"
    echo "# Model Name: $MODEL_NAME"
    echo "# Datasets:   $DATASETS_TO_TEST"
    echo "# Results:    $RESULTS_FILE"
    echo "# Sampling:   pass_k=$PASS_K temperature=$TEMPERATURE top_p=$TOP_P seed=$EVAL_SEED prompt_format=base_completion max_prompt_tokens=$MAX_PROMPT_TOKENS max_response_tokens=$MAX_TOKENS max_model_len=$MAX_MODEL_LEN max_num_seqs=$CODE_EVAL_MAX_NUM_SEQS lcb_enforce_eager=$CODE_EVAL_LCB_ENFORCE_EAGER"
    echo "# GPU layout: NGPUS_PER_NODE=$NGPUS_PER_NODE GEN_TP=$GEN_TP (one vLLM instance)"
    echo "# EvalPlus:   humaneval=$HUMANEVAL_OVERRIDE_PATH mbpp=$MBPP_OVERRIDE_PATH"
    echo "# LCB repo:   $LCB_REPO"
    echo "# LCB v6 data:$LCB_CODEGEN_LITE_DIR"
    echo "################################################################################"

    if [ "$DRY_RUN" = "true" ]; then
        echo "Dry run: configuration validated; no model was loaded."
        continue
    fi

    for ds in $DATASETS_TO_TEST; do
        case "$ds" in
            humaneval_plus|humaneval+) evalplus_dataset "$MODEL_PATH" "$MODEL_NAME" "$RESULTS_FILE" "$GEN_OUTPUT_DIR" humaneval_plus humaneval ;;
            mbpp_plus|mbpp+) evalplus_dataset "$MODEL_PATH" "$MODEL_NAME" "$RESULTS_FILE" "$GEN_OUTPUT_DIR" mbpp_plus mbpp ;;
            livecodebench_v6|lcb_v6|livecodebench) lcb_dataset "$MODEL_PATH" "$MODEL_NAME" "$RESULTS_FILE" "$GEN_OUTPUT_DIR" ;;
            *) echo "ERROR: unsupported code dataset: $ds" >&2; exit 1 ;;
        esac
    done

    METRICS_FILE="${EVAL_METRICS_FILE:-${GEN_OUTPUT_DIR}/metrics.json}"
    METRICS_CMD=(
        "$PYTHON_BIN" "$SCRIPT_DIR/export_metrics.py"
        --results_file "$RESULTS_FILE"
        --output_file "$METRICS_FILE"
        --model_name "$MODEL_NAME"
        --model_path "$MODEL_PATH"
        --datasets "$(echo "$METRICS_DATASETS" | tr ' ' ',')"
        --n_samples "$PASS_K"
        --prompt_length "$MAX_PROMPT_TOKENS"
        --response_length "$MAX_TOKENS"
        --temperature "$TEMPERATURE"
        --top_p "$TOP_P"
        --seed "$EVAL_SEED"
    )
    if [ -n "${EVAL_STEP:-${WANDB_GLOBAL_STEP:-}}" ]; then
        METRICS_CMD+=(--step "${EVAL_STEP:-${WANDB_GLOBAL_STEP}}")
    fi
    if [ -n "${EVAL_MILESTONE_FRACTION:-}" ]; then
        METRICS_CMD+=(--milestone_fraction "$EVAL_MILESTONE_FRACTION")
    fi
    "${METRICS_CMD[@]}"
    if [ -n "${WANDB_RUN_ID:-}" ]; then
        if [ -z "${WANDB_PROJECT:-}" ] || [ -z "${WANDB_GLOBAL_STEP:-}" ]; then
            echo "ERROR: WANDB_RUN_ID requires WANDB_PROJECT and WANDB_GLOBAL_STEP" >&2
            exit 1
        fi
        if [ "${EVAL_KIND:-milestone}" != "base" ] && [ -z "${EVAL_MILESTONE_FRACTION:-}" ]; then
            echo "ERROR: milestone W&B eval logging requires EVAL_MILESTONE_FRACTION" >&2
            exit 1
        fi
        "$PYTHON_BIN" "$VERL_ROOT/recipe/math_evaluation/log_metrics_wandb.py" --metrics_file "$METRICS_FILE"
    fi

    if [ "$WRITE_RESULTS_CSV" = "true" ]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/results_json_to_csv.py" \
            --results_file "$RESULTS_FILE" \
            --output_file "${EVAL_RESULTS_CSV_FILE:-${RESULTS_FILE%.json}.csv}" \
            --pass_k "$PASS_K"
    fi
    echo "Results saved to: $RESULTS_FILE"
done
