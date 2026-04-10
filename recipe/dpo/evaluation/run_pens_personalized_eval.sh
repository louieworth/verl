#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WORKDIR="${WORKDIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

# MODEL_PATH="${MODEL_PATH:-/data/data/jiangli/ckpt/PENS/single_wise_dpo_click_hist_negative_only_lora_qwen3_5-4b/global_step_5340/actor/hf_merged}"
# MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-/data/data/jiangli/ckpt/PENS/single_wise_dpo_click_hist_all_lora_qwen3_5-4b/global_step_10682/actor/hf_merged}"
TEST_FILE="${TEST_FILE:-/data/data/jiangli/data/pens/extract/personalized_test.tsv}"
PROMPT_FILE="${PROMPT_FILE:-/data/data/jiangli/data/pens/eval/prompts.parquet}"
DEFAULT_MODEL_SLUG="$(derive_model_slug "${MODEL_PATH}")"
MODEL_KEY="${MODEL_KEY:-${DEFAULT_MODEL_SLUG}}"
OUTPUT_STEM="${OUTPUT_STEM:-${MODEL_KEY}}"

RESULTS_GEN_DIR="${RESULTS_GEN_DIR:-${WORKDIR}/gen_results}"
RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results}"
RAW_FILE="${RAW_FILE:-${RESULTS_GEN_DIR}/${OUTPUT_STEM}.parquet}"
RESULT_JSON_FILE="${RESULT_JSON_FILE:-${RESULTS_DIR}/result.json}"

BACKEND="${BACKEND:-auto}"
RESPONSE_INDEX="${RESPONSE_INDEX:-0}"
ALIGN_BY_ORDER="${ALIGN_BY_ORDER:-false}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
GEN_TP="${GEN_TP:-1}"
PASS_K="${PASS_K:-1}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}"
GEN_TOP_P="${GEN_TOP_P:-0.8}"
GEN_TOP_K="${GEN_TOP_K:-20}"
GEN_PROMPT_LENGTH="${GEN_PROMPT_LENGTH:-5120}"
GEN_RESPONSE_LENGTH="${GEN_RESPONSE_LENGTH:-64}"
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-}"
GEN_MAX_NUM_SEQS="${GEN_MAX_NUM_SEQS:-}"
GEN_ENABLE_PREFIX_CACHING="${GEN_ENABLE_PREFIX_CACHING:-}"
GEN_ENFORCE_EAGER="${GEN_ENFORCE_EAGER:-}"
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}"
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE="${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-}"
VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER="${VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER:-}"

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

mkdir -p "${RESULTS_GEN_DIR}" "${RESULTS_DIR}"

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
append_hydra_override "actor_rollout_ref.rollout.enable_prefix_caching" "${GEN_ENABLE_PREFIX_CACHING}"
append_hydra_override "actor_rollout_ref.rollout.enforce_eager" "${GEN_ENFORCE_EAGER}"
append_hydra_override "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only" "${VLLM_LANGUAGE_MODEL_ONLY}"

if [[ -n "${VLLM_ENABLE_THINKING}" ]]; then
    thinking="${VLLM_ENABLE_THINKING,,}"
    if [[ "${thinking}" == "true" || "${thinking}" == "false" ]]; then
        GENERATION_CMD+=(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.default_chat_template_kwargs={enable_thinking:${thinking}}"
        )
    fi
elif [[ -n "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS}" ]]; then
    GENERATION_CMD+=(
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.default_chat_template_kwargs=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS}"
    )
fi

append_hydra_override "ray_kwargs.ray_init.num_cpus" "${RAY_NUM_CPUS}"
append_hydra_override "+ray_kwargs.ray_init.include_dashboard" "${RAY_INCLUDE_DASHBOARD}"
append_hydra_override \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_hybrid_kv_cache_manager" \
    "${VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER}"
append_hydra_override \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.max_cudagraph_capture_size" \
    "${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE}"
append_hydra_override "+actor_rollout_ref.rollout.engine_kwargs.vllm.block_size" "${VLLM_BLOCK_SIZE}"
append_hydra_override "+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode" "${VLLM_MAMBA_CACHE_MODE}"

echo "Ray config: num_cpus=${RAY_NUM_CPUS}, include_dashboard=${RAY_INCLUDE_DASHBOARD}"
echo "Running generation -> ${RAW_FILE}"
(
    cd "${REPO_ROOT}"
    "${GENERATION_CMD[@]}"
)

EVAL_CMD=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_pens_personalized_eval.py"
    "--workdir" "${WORKDIR}"
    "--model-path" "${MODEL_PATH}"
    "--test-file" "${TEST_FILE}"
    "--results-gen-dir" "${RESULTS_GEN_DIR}"
    "--results-dir" "${RESULTS_DIR}"
    "--raw-file" "${RAW_FILE}"
    "--result-json-file" "${RESULT_JSON_FILE}"
    "--backend" "${BACKEND}"
    "--response-index" "${RESPONSE_INDEX}"
)

append_eval_arg "--model-key" "${MODEL_KEY}"

if [[ "${ALIGN_BY_ORDER,,}" == "true" ]]; then
    EVAL_CMD+=("--align-by-order")
fi

echo "Running evaluation -> ${RESULT_JSON_FILE}"
"${EVAL_CMD[@]}"
