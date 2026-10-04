#!/usr/bin/env bash
set -euo pipefail

# Narval / Compute Canada environment fixes
unset ROCR_VISIBLE_DEVICES
# vLLM V1 uses its own CUDA memory pool and is incompatible with PyTorch's
# expandable segments allocator, which the DPO training wrapper enables.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
# export VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-skip}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${SCRIPT_DIR}/../run/common.sh"
WORKDIR="${WORKDIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PENS_PYTHON_BIN}}"
EVAL_MASTER_ADDR="${EVAL_MASTER_ADDR:-127.0.0.1}"
EVAL_MASTER_PORT="${EVAL_MASTER_PORT:-}"

find_free_port() {
    local port
    while true; do
        port="$(shuf -i 12000-65000 -n 1)"
        if ! (echo > "/dev/tcp/${EVAL_MASTER_ADDR}/${port}") 2>/dev/null; then
            echo "${port}"
            return 0
        fi
    done
}

if [[ -z "${EVAL_MASTER_PORT}" ]]; then
    EVAL_MASTER_PORT="$(find_free_port)"
fi

derive_model_slug() {
    local normalized="$1"
    local candidate=""

    normalized="${normalized%/}"
    if [[ "${normalized}" == *"/snapshots/"* ]]; then
        local before_snapshots="${normalized%%/snapshots/*}"
        candidate="$(basename "${before_snapshots}")"
    else
        candidate="$(basename "${normalized}")"
        case "${candidate}" in
            hf_merged|hf_merged_final|hf_merged_fixed|huggingface|actor)
                local actor_dir
                local step_parent
                local experiment_parent
                actor_dir="$(dirname "${normalized}")"
                step_parent="$(dirname "${actor_dir}")"
                experiment_parent="$(dirname "${step_parent}")"
                candidate="$(basename "${experiment_parent}")_$(basename "${step_parent}")_${candidate}"
                ;;
        esac
    fi

    if [[ "${candidate}" == models--* ]]; then
        candidate="${candidate##*--}"
    fi

    printf '%s' "${candidate}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9_-]+/_/g; s/__+/_/g; s/^[_-]+//; s/[_-]+$//'
}

MODEL_PATH="${MODEL_PATH:-${PENS_MODEL_DIR}}"
TEST_FILE="${TEST_FILE:-${PENS_EVAL_TEST_FILE}}"
PROMPT_FILE="${PROMPT_FILE:-${PENS_EVAL_PROMPT_FILE}}"
DEFAULT_MODEL_SLUG="$(derive_model_slug "${MODEL_PATH}")"
MODEL_KEY="${MODEL_KEY:-${DEFAULT_MODEL_SLUG}}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING,,}"


# Date + thinking label for filenames and W&B keys. Keep in sync with the
# tagged key written by run_pens_personalized_eval.py.
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
case "${VLLM_ENABLE_THINKING}" in
    true) _THINKING_TAG="thinkON" ;;
    false) _THINKING_TAG="thinkOFF" ;;
    skip) _THINKING_TAG="thinkNA" ;;
    *)
        echo "VLLM_ENABLE_THINKING must be true, false, or skip; got ${VLLM_ENABLE_THINKING}" >&2
        exit 1
        ;;
esac
OUTPUT_STEM="${OUTPUT_STEM:-${MODEL_KEY}_${_THINKING_TAG}_${DATE_TAG}}"

RESULTS_GEN_DIR="${RESULTS_GEN_DIR:-${PENS_OUTPUT_ROOT}/${PENS_EXPERIMENT_NAME:-${MODEL_KEY}}}"
RAW_FILE="${RAW_FILE:-${RESULTS_GEN_DIR}/${OUTPUT_STEM}.parquet}"

BACKEND="${BACKEND:-rouge}"
RESPONSE_INDEX="${RESPONSE_INDEX:-0}"
ALIGN_BY_ORDER="${ALIGN_BY_ORDER:-false}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
GEN_TP="${GEN_TP:-1}"
PASS_K="${PASS_K:-1}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}"
GEN_TOP_P="${GEN_TOP_P:-0.8}"
GEN_TOP_K="${GEN_TOP_K:-20}"
GEN_PROMPT_LENGTH="${GEN_PROMPT_LENGTH:-8000}"
GEN_RESPONSE_LENGTH="${GEN_RESPONSE_LENGTH:-128}"
# Covers 99.52% of prompts; the ~99 outliers >8000 tokens are rejected by vLLM
# with a 400 error and recorded as empty responses by main_generation_server.py.
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-8300}"
GEN_MAX_NUM_SEQS="${GEN_MAX_NUM_SEQS:-32}"
# Keep long 8k prompts via chunked prefill, but do not profile/compile all
# 8192 prefill tokens at once on 40 GiB A100s.  The 8192 default can consume
# the entire vLLM V1 memory budget before any KV-cache blocks are allocated.
GEN_MAX_NUM_BATCHED_TOKENS="${GEN_MAX_NUM_BATCHED_TOKENS:-4096}"
# Anti-repetition (see missing_structured_output analysis: ~70% of the 524
# failures were degenerate loops). 1.1 is a safe low value for short outputs.
GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.1}"
GEN_ENABLE_PREFIX_CACHING="${GEN_ENABLE_PREFIX_CACHING:-}"
GEN_ENFORCE_EAGER="${GEN_ENFORCE_EAGER:-}"
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING,,}"
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE="${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-}"
VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-}"
VLLM_USE_V1="${VLLM_USE_V1:-false}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER="${VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER:-}"
GEN_STREAM="${GEN_STREAM:-false}"
GEN_REQUEST_CONCURRENCY="${GEN_REQUEST_CONCURRENCY:-}"

if [[ -z "${RAY_NUM_CPUS}" ]]; then
    # Keep Ray startup bounded on high-core hosts. Leaving this unset on a 256-core
    # machine can trigger a worker startup storm before runtime_env_agent is ready.
    ray_cpu_budget=$((NGPUS_PER_NODE * 4))
    if (( ray_cpu_budget < 4 )); then
        ray_cpu_budget=4
    fi
    RAY_NUM_CPUS="${ray_cpu_budget}"
fi

append_hydra_override() {
    local key="$1"
    local value="${2:-}"
    if [[ -n "${value}" ]]; then
        GENERATION_CMD+=("${key}=${value}")
    fi
}

append_eval_arg() {
    local flag="$1"
    local value="${2:-}"
    if [[ -n "${value}" ]]; then
        EVAL_CMD+=("${flag}" "${value}")
    fi
}

mkdir -p "${RESULTS_GEN_DIR}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Missing prompt parquet: ${PROMPT_FILE}" >&2
    exit 1
fi

if [[ ! -f "${TEST_FILE}" ]]; then
    echo "Missing test file: ${TEST_FILE}" >&2
    exit 1
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${REPO_ROOT}"
fi

GENERATION_CMD=(
    "${PYTHON_BIN}"
    "-m"
    "verl.trainer.main_generation_server"
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE}"
    "actor_rollout_ref.rollout.temperature=${GEN_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${GEN_TOP_P}"
    "actor_rollout_ref.rollout.top_k=${GEN_TOP_K}"
    "actor_rollout_ref.rollout.prompt_length=${GEN_PROMPT_LENGTH}"
    "actor_rollout_ref.rollout.response_length=${GEN_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GEN_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=${PASS_K}"
    "data.train_files=['${PROMPT_FILE}']"
    "data.prompt_key=prompt"
    "+data.output_path=${RAW_FILE}"
)

append_hydra_override "actor_rollout_ref.rollout.max_model_len" "${GEN_MAX_MODEL_LEN}"
append_hydra_override "actor_rollout_ref.rollout.max_num_seqs" "${GEN_MAX_NUM_SEQS}"
append_hydra_override "actor_rollout_ref.rollout.max_num_batched_tokens" "${GEN_MAX_NUM_BATCHED_TOKENS}"
append_hydra_override "actor_rollout_ref.rollout.enable_prefix_caching" "${GEN_ENABLE_PREFIX_CACHING}"
append_hydra_override "actor_rollout_ref.rollout.enforce_eager" "${GEN_ENFORCE_EAGER}"
append_hydra_override "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only" "${VLLM_LANGUAGE_MODEL_ONLY}"

if [[ -n "${VLLM_ENABLE_THINKING}" ]]; then
    thinking="${VLLM_ENABLE_THINKING,,}"
    if [[ "${thinking}" == "true" || "${thinking}" == "false" ]]; then
        # Per-request chat_template_kwargs (vLLM 0.12+ accepts this in the
        # /v1/chat/completions request body; main_generation_server.py forwards it).
        # Stored under data.* because RolloutConfig is a strict dataclass.
        GENERATION_CMD+=(
            "+data.chat_template_kwargs={enable_thinking:${thinking}}"
        )
    fi
elif [[ -n "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS}" ]]; then
    GENERATION_CMD+=(
        "+data.chat_template_kwargs=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS}"
    )
fi

append_hydra_override "+data.repetition_penalty" "${GEN_REPETITION_PENALTY}"

if [[ "${GEN_STREAM,,}" == "true" ]]; then
    append_hydra_override "+data.stream" "true"
fi
append_hydra_override "+data.request_concurrency" "${GEN_REQUEST_CONCURRENCY}"

append_hydra_override "ray_kwargs.ray_init.num_cpus" "${RAY_NUM_CPUS}"
append_hydra_override "+ray_kwargs.ray_init.include_dashboard" "${RAY_INCLUDE_DASHBOARD}"
if [[ -n "${PENS_EVAL_RAY_ADDRESS:-}" ]]; then
    # Explicitly isolate this evaluation from other Ray heads on shared hosts.
    unset RAY_ADDRESS
    append_hydra_override "+ray_kwargs.ray_init.address" "${PENS_EVAL_RAY_ADDRESS}"
fi
append_hydra_override "+ray_kwargs.ray_init._temp_dir" "${PENS_EVAL_RAY_TMPDIR:-}"
append_hydra_override \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_hybrid_kv_cache_manager" \
    "${VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER}"
append_hydra_override \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.max_cudagraph_capture_size" \
    "${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE}"
append_hydra_override "+actor_rollout_ref.rollout.engine_kwargs.vllm.block_size" "${VLLM_BLOCK_SIZE}"
append_hydra_override "+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode" "${VLLM_MAMBA_CACHE_MODE}"
append_hydra_override "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1" "\"${VLLM_USE_V1}\""
if [[ -n "${VLLM_CACHE_ROOT:-}" ]]; then
    append_hydra_override \
        "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_CACHE_ROOT" \
        "\"${VLLM_CACHE_ROOT}\""
fi

echo "Ray config: num_cpus=${RAY_NUM_CPUS}, include_dashboard=${RAY_INCLUDE_DASHBOARD}"
echo "Running generation -> ${RAW_FILE}"
export MASTER_ADDR="${EVAL_MASTER_ADDR}"
export MASTER_PORT="${EVAL_MASTER_PORT}"
export DIST_INIT_METHOD="${DIST_INIT_METHOD:-env://}"
echo "Torch distributed master: ${EVAL_MASTER_ADDR}:${EVAL_MASTER_PORT}"

# Give every Ray/vLLM descendant of this one evaluation a unique marker.  If a
# driver or actor exits abruptly, the fallback below can identify only this
# evaluation's orphaned processes instead of touching another user's workload.
PENS_EVAL_RUN_ID="pens-eval-${BASHPID}-${RANDOM}-${RANDOM}"

process_has_eval_run_id() {
    local environ_file="$1"
    local entry
    while IFS= read -r -d '' entry; do
        if [[ "${entry}" == "PENS_EVAL_RUN_ID=${PENS_EVAL_RUN_ID}" ]]; then
            return 0
        fi
    done < "${environ_file}" 2>/dev/null || true
    return 1
}

cleanup_eval_descendants() {
    local environ_file
    local pid
    local -a residual_pids=()

    for environ_file in /proc/[0-9]*/environ; do
        [[ -r "${environ_file}" ]] || continue
        if process_has_eval_run_id "${environ_file}"; then
            pid="${environ_file#/proc/}"
            pid="${pid%/environ}"
            [[ "${pid}" == "${BASHPID}" ]] || residual_pids+=("${pid}")
        fi
    done

    if (( ${#residual_pids[@]} == 0 )); then
        echo "Evaluation process cleanup: no marked descendants remain"
        return 0
    fi

    echo "Evaluation process cleanup: terminating marked residual PIDs ${residual_pids[*]}"
    kill -TERM "${residual_pids[@]}" 2>/dev/null || true
    for _ in {1..20}; do
        local -a alive=()
        for pid in "${residual_pids[@]}"; do
            kill -0 "${pid}" 2>/dev/null && alive+=("${pid}")
        done
        (( ${#alive[@]} == 0 )) && return 0
        residual_pids=("${alive[@]}")
        sleep 0.5
    done

    echo "Evaluation process cleanup: force-killing marked residual PIDs ${residual_pids[*]}"
    kill -KILL "${residual_pids[@]}" 2>/dev/null || true
}

set +e
(
    cd "${REPO_ROOT}"
    PENS_EVAL_RUN_ID="${PENS_EVAL_RUN_ID}" "${GENERATION_CMD[@]}"
)
generation_status=$?
set -e
cleanup_eval_descendants
if (( generation_status != 0 )); then
    echo "Generation failed with exit code ${generation_status}" >&2
    exit "${generation_status}"
fi

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

export GEN_TEMPERATURE GEN_TOP_P GEN_TOP_K GEN_PROMPT_LENGTH GEN_RESPONSE_LENGTH
export GEN_MAX_MODEL_LEN GEN_MAX_NUM_SEQS GEN_MAX_NUM_BATCHED_TOKENS GEN_REPETITION_PENALTY
export GEN_GPU_MEMORY_UTILIZATION GEN_TP PASS_K VLLM_ENABLE_THINKING NNODES NGPUS_PER_NODE
export PROMPT_FILE TEST_FILE

EVAL_CMD=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_pens_personalized_eval.py"
    "--workdir" "${WORKDIR}"
    "--model-path" "${MODEL_PATH}"
    "--test-file" "${TEST_FILE}"
    "--results-gen-dir" "${RESULTS_GEN_DIR}"
    "--raw-file" "${RAW_FILE}"
    "--backend" "${BACKEND}"
    "--response-index" "${RESPONSE_INDEX}"
    "--thinking" "${VLLM_ENABLE_THINKING:-false}"
    "--date-tag" "${DATE_TAG}"
)

append_eval_arg "--model-key" "${MODEL_KEY}"
append_eval_arg "--checkpoint-state-file" "${PENS_EVAL_STATE_FILE:-}"

if [[ "${ALIGN_BY_ORDER,,}" == "true" ]]; then
    EVAL_CMD+=("--align-by-order")
fi

echo "Running evaluation -> W&B"
"${EVAL_CMD[@]}"
