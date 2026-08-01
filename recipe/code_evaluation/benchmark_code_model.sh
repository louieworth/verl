#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
for mp in "${MODEL_PATHS[@]}"; do
    [ -d "$mp" ] || { echo "ERROR: model not found: $mp" >&2; exit 1; }
done

DATASETS_TO_TEST=${DATASETS:-"humaneval_plus mbpp_plus livecodebench_v6"}
PASS_K=${PASS_K:-16}
export PASS_K
TEMPERATURE=${CODE_EVAL_TEMPERATURE:-0.6}
TOP_P=${CODE_EVAL_TOP_P:-0.95}
MAX_PROMPT_TOKENS=${CODE_EVAL_MAX_PROMPT_TOKENS:-2048}
MAX_TOKENS=${CODE_EVAL_MAX_RESPONSE_TOKENS:-${CODE_EVAL_MAX_TOKENS:-16384}}
MAX_MODEL_LEN=${CODE_EVAL_MAX_MODEL_LEN:-$((MAX_PROMPT_TOKENS + MAX_TOKENS))}
CODE_EVAL_MAX_NUM_SEQS=${CODE_EVAL_MAX_NUM_SEQS:-128}
CODE_EVAL_LCB_ENFORCE_EAGER=${CODE_EVAL_LCB_ENFORCE_EAGER:-false}
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
CODE_EVAL_DATA_ROOT=${CODE_EVAL_DATA_ROOT:-$VERL_ROOT/data/eval_dataset/code}
LCB_REPO=${LCB_REPO:-$CODE_EVAL_DATA_ROOT/LiveCodeBench}
LCB_CODEGEN_LITE_DIR=${LCB_CODEGEN_LITE_DIR:-$CODE_EVAL_DATA_ROOT/livecodebench/code_generation_lite}
HUMANEVAL_OVERRIDE_PATH=${HUMANEVAL_OVERRIDE_PATH:-$CODE_EVAL_DATA_ROOT/evalplus/HumanEvalPlus-v0.1.10.jsonl}
MBPP_OVERRIDE_PATH=${MBPP_OVERRIDE_PATH:-$CODE_EVAL_DATA_ROOT/evalplus/MbppPlus-v0.2.0.jsonl}
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
results_file, model_name, model_path, dataset, avg, passed = sys.argv[1:]
avg = None if avg == '' else float(avg)
passed = None if passed == '' else float(passed)
data = {}
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
pass_k = os.environ.get("PASS_K", "4")
entry = data.setdefault(model_name, {})
entry["model_path"] = model_path
if avg is not None:
    entry[f"{dataset}_avg{pass_k}"] = avg
if passed is not None:
    entry[f"{dataset}_pass{pass_k}"] = passed
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
    mkdir -p "$out_dir/evalplus"
    local root="$out_dir/evalplus"
    local metrics avg pass
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
            --max_tokens "$MAX_TOKENS" \
            --max_model_len "$MAX_MODEL_LEN" \
            --max_num_seqs "$CODE_EVAL_MAX_NUM_SEQS" \
            --n_samples "$PASS_K" \
            --trust_remote_code \
            ${EVALPLUS_EXTRA_ARGS:-}
        metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_evalplus_metrics.py" --root "$root" --dataset "$evalplus_name" --pass_k "$PASS_K")
    fi
    avg=$(echo "$metrics" | awk -F= '/^avg=/{print $2}')
    pass=$(echo "$metrics" | awk -F= '/^pass=/{print $2}')
    update_result "$results_file" "$model_name" "$model_path" "$dataset_name" "$avg" "$pass"
}

lcb_dataset() {
    local model_path="$1" model_name="$2" results_file="$3" out_dir="$4"
    local lcb_repo="$LCB_REPO"
    if ! (cd "$lcb_repo" && require_module lcb_runner); then
        echo "ERROR: lcb_runner is not importable from $lcb_repo" >&2
        exit 1
    fi
    local lcb_model_key="${LCB_MODEL_KEY:-Qwen/Qwen3-235B-A22B}"
    local enforce_eager_arg=""
    if [ "$CODE_EVAL_LCB_ENFORCE_EAGER" = "true" ]; then
        enforce_eager_arg="--enforce_eager"
    fi
    mkdir -p "$out_dir/livecodebench"
    local metrics avg pass
    local lcb_extract_args=(--root "$out_dir/livecodebench" --pass_k "$PASS_K")
    if [ "${LCB_AGGREGATE_ALL:-false}" = "true" ]; then
        lcb_extract_args+=(--aggregate_all)
        if [ -n "${LCB_EXPECTED_TASKS:-}" ]; then
            lcb_extract_args+=(--expected_tasks "$LCB_EXPECTED_TASKS")
        fi
    fi
    if metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_lcb_metrics.py" "${lcb_extract_args[@]}" 2>/dev/null); then
        echo "Existing LiveCodeBench v6 eval results found under $out_dir/livecodebench; skipping generation/evaluation."
        avg=$(echo "$metrics" | awk -F= '/^avg=/{print $2}')
        pass=$(echo "$metrics" | awk -F= '/^pass=/{print $2}')
        update_result "$results_file" "$model_name" "$model_path" "livecodebench_v6" "$avg" "$pass"
        return
    fi
    local runtime_dir="$out_dir/livecodebench/runtime"
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
        "$PYTHON_BIN" -m lcb_runner.runner.main \
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
    update_result "$results_file" "$model_name" "$model_path" "livecodebench_v6" "$avg" "$pass"
}

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    FULL_MODEL_DIR=$(dirname "$MODEL_PATH")
    FULL_MODEL_NAME=$(basename "$FULL_MODEL_DIR")
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
    echo "# Sampling:   pass_k=$PASS_K temperature=$TEMPERATURE top_p=$TOP_P max_prompt_tokens=$MAX_PROMPT_TOKENS max_response_tokens=$MAX_TOKENS max_model_len=$MAX_MODEL_LEN max_num_seqs=$CODE_EVAL_MAX_NUM_SEQS lcb_enforce_eager=$CODE_EVAL_LCB_ENFORCE_EAGER"
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

    if [ "$WRITE_RESULTS_CSV" = "true" ]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/results_json_to_csv.py" \
            --results_file "$RESULTS_FILE" \
            --output_file "${EVAL_RESULTS_CSV_FILE:-${RESULTS_FILE%.json}.csv}" \
            --pass_k "$PASS_K"
    fi
    echo "Results saved to: $RESULTS_FILE"
done
