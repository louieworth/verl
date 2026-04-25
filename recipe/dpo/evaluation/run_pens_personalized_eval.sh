#!/usr/bin/env bash
set -euo pipefail

# Narval / Compute Canada environment fixes
unset ROCR_VISIBLE_DEVICES
# export VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-skip}"
_VERL_ENV_BIN=/project/def-y7ding/lijiang3/envs/verl/bin
if [[ -x "${_VERL_ENV_BIN}/python" ]]; then
    export PATH="${_VERL_ENV_BIN}:${PATH}"
fi
# pyarrow cannot be installed into the venv on Compute Canada (only a dummy
# wheel exists); borrow it from the CVMFS arrow module instead.
_CVMFS_ARROW=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/19.0.1/lib/python3.12/site-packages
export PYTHONPATH="${_CVMFS_ARROW}:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WORKDIR="${WORKDIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${_VERL_ENV_BIN}/python}"

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

# Strip the DPO/SFT experiment wrapper to recover the underlying base-model slug.
# Convention in run_single_wise_dpo.sh: EXPERIMENT_NAME = "${LOSS}_${INPUT}_${SAMPLE}_${FT_VARIANT}_${MODEL_SLUG}"
# and the checkpoint path appends "/global_step_<N>/actor/<hf_merged>" — so the
# full slug becomes "<LOSS>_<INPUT>_<SAMPLE>_<lora|fullft>_<BASE>_global_step_<N>_<leaf>".
# Extract the BASE between "_(lora|fullft)_" and "_global_step_". For raw HF
# snapshots or base models the full slug is already the base slug.
#
# After stripping, also collapse SFT-warmup variants ("sft_warmup_pos_<model>")
# to the bare model name so DPO ckpts and raw HF ckpts of the same family land
# in the same results/<model>.json file.
derive_base_model_slug() {
    local full_slug
    full_slug="$(derive_model_slug "$1")"
    local base
    if [[ "${full_slug}" =~ _(lora|fullft)_(.+)_global_step_ ]]; then
        base="${BASH_REMATCH[2]}"
    else
        base="${full_slug}"
    fi
    # Strip SFT warmup prefix so results/<model>.json stays stable across
    # pre-SFT, post-SFT, and post-DPO ckpts of the same base family.
    base="${base#sft_warmup_pos_}"
    base="${base#sft_warmup_neg_}"
    base="${base#sft_warmup_}"
    printf '%s' "${base}"
}

# MODEL_PATH="${MODEL_PATH:-/data/data/jiangli/ckpt/PENS/single_wise_dpo_click_hist_negative_only_lora_qwen3_5-4b/global_step_5340/actor/hf_merged}"
# MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
# MODEL_PATH="${MODEL_PATH:-/scratch/lijiang3/ckpt/PENS/prospect_dpo_click_hist_all_lora_qwen3-8b/global_step_5341/actor/hf_merged}"
TEST_FILE="${TEST_FILE:-/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/Microsoft-PeNS/PENS/personalized_test.tsv}"
PROMPT_FILE="${PROMPT_FILE:-/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/eval/prompts.parquet}"
DEFAULT_MODEL_SLUG="$(derive_model_slug "${MODEL_PATH}")"
MODEL_KEY="${MODEL_KEY:-${DEFAULT_MODEL_SLUG}}"
# Base-model slug groups every ckpt derived from the same base into a single
# results/<BASE_MODEL_SLUG>.json (e.g. qwen3-8b.json, qwen3_5-4b.json). Override
# via BASE_MODEL_SLUG if the auto-derived value is wrong (e.g. cross-init runs).
DEFAULT_BASE_MODEL_SLUG="$(derive_base_model_slug "${MODEL_PATH}")"
BASE_MODEL_SLUG="${BASE_MODEL_SLUG:-${DEFAULT_BASE_MODEL_SLUG}}"

# Date + thinking label for filenames and JSON keys. Keep in sync with the
# tagged key written by run_pens_personalized_eval.py.
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
if [[ "${VLLM_ENABLE_THINKING,,}" == "true" ]]; then
    _THINKING_TAG="thinkON"
else
    _THINKING_TAG="thinkOFF"
fi
OUTPUT_STEM="${OUTPUT_STEM:-${MODEL_KEY}_${_THINKING_TAG}_${DATE_TAG}}"

RESULTS_GEN_DIR="${RESULTS_GEN_DIR:-${WORKDIR}/gen_results}"
RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results}"
RAW_FILE="${RAW_FILE:-${RESULTS_GEN_DIR}/${OUTPUT_STEM}.parquet}"
RESULT_JSON_FILE="${RESULT_JSON_FILE:-${RESULTS_DIR}/${BASE_MODEL_SLUG}.json}"

BACKEND="${BACKEND:-rouge}"
RESPONSE_INDEX="${RESPONSE_INDEX:-0}"
ALIGN_BY_ORDER="${ALIGN_BY_ORDER:-false}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
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
# Anti-repetition (see missing_structured_output analysis: ~70% of the 524
# failures were degenerate loops). 1.1 is a safe low value for short outputs.
GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.1}"
GEN_ENABLE_PREFIX_CACHING="${GEN_ENABLE_PREFIX_CACHING:-}"
GEN_ENFORCE_EAGER="${GEN_ENFORCE_EAGER:-}"
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}"
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.90}"
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

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

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
    "--thinking" "${VLLM_ENABLE_THINKING:-false}"
    "--date-tag" "${DATE_TAG}"
)

append_eval_arg "--model-key" "${MODEL_KEY}"

if [[ "${ALIGN_BY_ORDER,,}" == "true" ]]; then
    EVAL_CMD+=("--align-by-order")
fi

echo "Running evaluation -> ${RESULT_JSON_FILE}"
"${EVAL_CMD[@]}"
