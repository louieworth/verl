#!/usr/bin/env bash
# Canonical dispatcher for recipe/opd/scripts_math and
# recipe/opd/script_code experiment matrices.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [ -x /data2/conda/envs/verl/bin/python ]; then
    PYTHON_BIN="${PYTHON_BIN:-/data2/conda/envs/verl/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHON_BIN
source "$REPO_ROOT/recipe/opd/scripts_math/lib/model_path_validation.sh"

TASK="${OPD_TASK:?OPD_TASK must be math or code}"
FAMILY="${OPD_FAMILY:?OPD_FAMILY must be baseline, opd, or opsd}"
VARIANT="${OPD_VARIANT:?OPD_VARIANT is required}"
MODEL_SIZE="${OPD_MODEL_SIZE:?OPD_MODEL_SIZE must be 1B, 4B, or 8B}"

case "$TASK" in math|code) ;; *) echo "ERROR: invalid OPD_TASK=$TASK" >&2; exit 2 ;; esac
case "$FAMILY" in baseline|opd|opsd) ;; *) echo "ERROR: invalid OPD_FAMILY=$FAMILY" >&2; exit 2 ;; esac
case "$MODEL_SIZE" in 1B|4B|8B) ;; *) echo "ERROR: invalid OPD_MODEL_SIZE=$MODEL_SIZE" >&2; exit 2 ;; esac

contract_error() {
    echo "ERROR: explicit launcher contract: $*" >&2
    exit 2
}

require_setting() {
    local name="$1"
    if [ ! -v "$name" ] || [ -z "${!name}" ]; then
        contract_error "$name must be declared by the leaf launcher"
    fi
}

require_value() {
    local name="$1"
    local expected="$2"
    require_setting "$name"
    if [ "${!name}" != "$expected" ]; then
        contract_error "$name=${!name} is incompatible with $FAMILY/$VARIANT (expected $expected)"
    fi
}

require_zero() {
    local name="$1"
    require_setting "$name"
    case "${!name}" in
        0|0.0|0.00|0.000) ;;
        *) contract_error "$name=${!name} is incompatible with $FAMILY/$VARIANT (expected zero)" ;;
    esac
}

require_positive_number() {
    local name="$1"
    local value
    require_setting "$name"
    value="${!name}"
    if ! [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || \
       ! awk -v value="$value" 'BEGIN { exit !(value > 0) }'; then
        contract_error "$name must be positive (got $value)"
    fi
}

require_positive_integer() {
    local name="$1"
    require_setting "$name"
    if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        contract_error "$name must be a positive integer (got ${!name})"
    fi
}

require_nonnegative_integer() {
    local name="$1"
    require_setting "$name"
    if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
        contract_error "$name must be a non-negative integer (got ${!name})"
    fi
}

require_probability() {
    local name="$1"
    local value
    require_setting "$name"
    value="${!name}"
    if ! [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || \
       ! awk -v value="$value" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
        contract_error "$name must be in (0, 1] (got $value)"
    fi
}

validate_explicit_variant_contract() {
    # MULTI_STEP=0 selects the bottom-level default of 512 prompts/update.
    # A positive value is the requested total number of policy optimizer
    # steps; the bottom-level runner partitions the full dataset accordingly.
    require_nonnegative_integer MULTI_STEP
    if [ "$FAMILY" = baseline ]; then
        require_value MULTI_STEP 0
        if [ "$VARIANT" = grpo ]; then
            require_value ROLLOUT_N 8
        fi
        if [ "$TASK" = code ] && [ "$VARIANT" = grpo ]; then
            require_value CODE_GRPO_REWARD_CONTRACT deepcoder_binary_15_longest_v1
            require_value CODE_GRPO_MAX_TEST_CASES 15
            require_value CODE_GRPO_EXEC_TIMEOUT_SECONDS 10
        fi
        return 0
    fi
    require_value DISTILL_MODE "$FAMILY"
    require_value KL_METHOD full_vocab
    require_zero BETA
    require_setting KL_TYPE
    require_setting Y_MODE
    require_setting TEACHER_TRAINING_PROMPT
    require_setting Y_O_ROLLOUT_MODE
    require_setting TOP_K
    require_setting KL_TOKEN_CLIP

    case "$VARIANT" in
        vanilla)
            require_value KL_TYPE reverse
            require_value Y_MODE y_o
            require_value TEACHER_TRAINING_PROMPT vanilla
            require_value Y_O_ROLLOUT_MODE student
            require_zero TOP_K
            require_zero KL_TOKEN_CLIP
            ;;
        top_k)
            require_value KL_TYPE reverse
            require_value Y_MODE y_o
            require_value TEACHER_TRAINING_PROMPT vanilla
            require_value Y_O_ROLLOUT_MODE student
            require_positive_integer TOP_K
            require_zero KL_TOKEN_CLIP
            ;;
        clip)
            require_value KL_TYPE forward
            require_value Y_MODE y_o
            require_value TEACHER_TRAINING_PROMPT vanilla
            require_value Y_O_ROLLOUT_MODE student
            require_zero TOP_K
            require_positive_number KL_TOKEN_CLIP
            ;;
        skd)
            require_value KL_TYPE forward
            require_value Y_MODE y_o
            require_value TEACHER_TRAINING_PROMPT vanilla
            case "$Y_O_ROLLOUT_MODE" in
                skd|skd_vllm|skd_vllm_internal) ;;
                *) contract_error "Y_O_ROLLOUT_MODE=$Y_O_ROLLOUT_MODE is incompatible with $FAMILY/skd" ;;
            esac
            require_zero TOP_K
            require_zero KL_TOKEN_CLIP
            require_positive_integer SKD_GAMMA
            require_positive_integer SKD_ACCEPT_TOP_K
            require_probability SKD_ACCEPT_TOP_P
            require_positive_number SKD_STUDENT_TEMPERATURE
            require_probability SKD_STUDENT_TOP_P
            require_positive_number SKD_TEACHER_TEMPERATURE
            require_probability SKD_TEACHER_TOP_P
            require_positive_integer SKD_ROLLOUT_BATCH_SIZE
            require_positive_integer SKD_PIPELINE_LANES
            require_value SKD_PARALLEL_STUDENT_TEACHER true
            ;;
        trd)
            require_value KL_TYPE forward
            require_value Y_MODE y_r
            require_value TEACHER_TRAINING_PROMPT refine
            require_value Y_O_ROLLOUT_MODE student
            require_zero TOP_K
            require_zero KL_TOKEN_CLIP
            ;;
    esac
}

case "$MODEL_SIZE" in
    1B)
        DEFAULT_MODEL_PATH="Qwen/Qwen3-1.7B-Base"
        DEFAULT_MODEL_ALIAS="Qwen3-1.7B-Base"
        EXPECTED_HIDDEN_SIZE=2048
        EXPECTED_NUM_LAYERS=28
        ;;
    4B)
        DEFAULT_MODEL_PATH="Qwen/Qwen3-4B-Base"
        DEFAULT_MODEL_ALIAS="Qwen3-4B-Base"
        EXPECTED_HIDDEN_SIZE=2560
        EXPECTED_NUM_LAYERS=36
        ;;
    8B)
        DEFAULT_MODEL_PATH="Qwen/Qwen3-8B-Base"
        DEFAULT_MODEL_ALIAS="Qwen3-8B-Base"
        EXPECTED_HIDDEN_SIZE=4096
        EXPECTED_NUM_LAYERS=36
        ;;
esac
MODEL_PATH="$(
    opd_resolve_base_model_path \
        "${MODEL_PATH:-$DEFAULT_MODEL_PATH}" \
        "$DEFAULT_MODEL_PATH" \
        "$EXPECTED_HIDDEN_SIZE" \
        "$EXPECTED_NUM_LAYERS" \
        MODEL_PATH
)" || exit $?
MODEL_ALIAS="${MODEL_ALIAS:-$DEFAULT_MODEL_ALIAS}"

if [ "$FAMILY" = "baseline" ]; then
    case "$VARIANT" in base|sft|grpo) ;; *) echo "ERROR: baseline supports base/sft/grpo" >&2; exit 2 ;; esac
else
    case "$VARIANT" in vanilla|top_k|clip|skd|trd) ;; *) echo "ERROR: unsupported distillation variant=$VARIANT" >&2; exit 2 ;; esac
fi

# The leaf launcher is the single source of truth for the algorithm. Validate
# its literal declarations before dry-run output or any data/GPU side effects.
validate_explicit_variant_contract

export TASK MODEL_PATH MODEL_ALIAS
export MODEL_NAME="${MODEL_NAME:-$MODEL_ALIAS}"
export STUDENT_MODEL="${STUDENT_MODEL:-$MODEL_ALIAS}"
if [ "$FAMILY" = baseline ]; then
    export TEACHER_MODEL_PATH=""
    export TEACHER_MODEL=""
elif [ "$FAMILY" = opd ]; then
    TEACHER_MODEL_PATH="$(
        opd_resolve_base_model_path \
            "${TEACHER_MODEL_PATH:-Qwen/Qwen3-14B}" \
            Qwen/Qwen3-14B \
            5120 \
            40 \
            TEACHER_MODEL_PATH
    )" || exit $?
    export TEACHER_MODEL_PATH
    export TEACHER_MODEL="${TEACHER_MODEL:-Qwen3-14B}"
else
    TEACHER_MODEL_PATH="$(
        opd_resolve_base_model_path \
            "${TEACHER_MODEL_PATH:-$MODEL_PATH}" \
            "$DEFAULT_MODEL_PATH" \
            "$EXPECTED_HIDDEN_SIZE" \
            "$EXPECTED_NUM_LAYERS" \
            TEACHER_MODEL_PATH
    )" || exit $?
    if [ "$TEACHER_MODEL_PATH" != "$MODEL_PATH" ]; then
        echo "ERROR: OPSD is self-distillation: TEACHER_MODEL_PATH must equal MODEL_PATH" >&2
        exit 2
    fi
    export TEACHER_MODEL_PATH
    export TEACHER_MODEL="$MODEL_ALIAS"
fi

if [ "$FAMILY" = baseline ]; then
    TEACHER_ENABLE_THINKING=false
else
    TEACHER_ENABLE_THINKING="${TEACHER_ENABLE_THINKING:-false}"
fi
case "$TEACHER_ENABLE_THINKING" in
    true|false) ;;
    *) contract_error "TEACHER_ENABLE_THINKING must be true or false (got $TEACHER_ENABLE_THINKING)" ;;
esac
if [ "$FAMILY" = opsd ] && [ "$TEACHER_ENABLE_THINKING" = true ]; then
    contract_error "OPSD uses a frozen Base self-teacher and cannot enable teacher thinking mode"
fi
if [ "$FAMILY" = opd ]; then
    if [ "$TEACHER_ENABLE_THINKING" = true ]; then
        TEACHER_SUPERVISION_RENDER_MODE=qwen3_chat_thinking
        TEACHER_ROLLOUT_RENDER_MODE=qwen3_chat_thinking
        TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER="${TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER:-16}"
    else
        TEACHER_SUPERVISION_RENDER_MODE=plain_completion
        TEACHER_ROLLOUT_RENDER_MODE=plain_completion
        TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER=0
    fi
else
    TEACHER_ROLLOUT_RENDER_MODE=base_completion
    TEACHER_SUPERVISION_RENDER_MODE=base_completion
    TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER=0
fi
TEACHER_PROMPT_RENDER_CONTRACT="supervision-${TEACHER_SUPERVISION_RENDER_MODE}_rollout-${TEACHER_ROLLOUT_RENDER_MODE}_v1"
if ! [[ "$TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER" =~ ^[0-9]+$ ]]; then
    contract_error "TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER must be a non-negative integer"
fi
export TEACHER_ENABLE_THINKING TEACHER_SUPERVISION_RENDER_MODE TEACHER_ROLLOUT_RENDER_MODE
export TEACHER_PROMPT_RENDER_CONTRACT TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER

# Shared profile from arXiv:2607.04751 Table 5. These values are intentionally
# common to the distillation variants; do not substitute variant-specific
# optimizer settings from unrelated distillation recipes.
export BASE_PROMPT_LENGTH="${BASE_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export EVAL_PROMPT_LENGTH="${EVAL_PROMPT_LENGTH:-2048}"
export EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-16384}"
export EXPERT_SOLUTION_PROMPT_LENGTH="${EXPERT_SOLUTION_PROMPT_LENGTH:-4096}"
export MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-32768}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.7}"
export PASS_K="${PASS_K:-16}"
export GLOBAL_PROMPT_BATCH_SIZE="${GLOBAL_PROMPT_BATCH_SIZE:-512}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export USE_LORA="${USE_LORA:-true}"
export LORA_RANK="${LORA_RANK:-64}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"
export EVAL_FRACTIONS="${EVAL_FRACTIONS:-0.25,0.5,0.75,1.0}"
case "$TASK/$FAMILY" in
    math/opd) DEFAULT_WANDB_PROJECT="opd-math" ;;
    math/opsd) DEFAULT_WANDB_PROJECT="opsd-math" ;;
    code/opd) DEFAULT_WANDB_PROJECT="opd-code" ;;
    code/opsd) DEFAULT_WANDB_PROJECT="opsd-code" ;;
    # Baselines are shared by OPD and OPSD. Keep exactly four projects and put
    # the single shared baseline copy beside the OPD task runs; family=baseline
    # remains explicit in its name, group, tags, and config.
    math/baseline) DEFAULT_WANDB_PROJECT="opd-math" ;;
    code/baseline) DEFAULT_WANDB_PROJECT="opd-code" ;;
esac
export WANDB_PROJECT="${WANDB_PROJECT:-$DEFAULT_WANDB_PROJECT}"
export PROJECT_NAME="${PROJECT_NAME:-$WANDB_PROJECT}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export SEED="${SEED:-42}"
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [ "$TASK" = "math" ]; then
    export TRAIN_FILE="${TRAIN_FILE:-data/train_dataset/openthoughts_math_30k_opsd/train_grpo.parquet}"
    export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$TRAIN_FILE}"
    export SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-data/train_dataset/openthoughts_math_30k_opsd/train_sft.parquet}"
    export TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-siyanzhao/Openthoughts_math_30k_opsd}"
    export EVAL_DATASETS="${EVAL_DATASETS:-aime25 aime26 hmmt26 amobench}"
    export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-data/eval_dataset/math}"
else
    export TRAIN_FILE="${TRAIN_FILE:-data/train_dataset/taco/canonical/train_grpo.parquet}"
    if [ "$FAMILY" = baseline ]; then
        export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$TRAIN_FILE}"
    else
        # Code distillation does not execute TACO tests. Use the compact
        # prompt/expert parquet; only the GRPO baseline needs train_grpo's
        # large reward payload.
        export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-data/train_dataset/taco/canonical/train_distill.parquet}"
    fi
    export SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-data/train_dataset/taco/canonical/train_sft.parquet}"
    export TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-BAAI/TACO}"
    export EVAL_DATASETS="${EVAL_DATASETS:-humaneval_plus mbpp_plus livecodebench_v6}"
    # Code keeps the repository's established stochastic evaluation profile;
    # only the response cap and Base-completion formatting are standardized.
    export CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE:-0.6}"
    export CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-0.95}"
    export CODE_EVAL_SEED="${CODE_EVAL_SEED:-42}"
    export CODE_EVAL_MAX_PROMPT_TOKENS="${CODE_EVAL_MAX_PROMPT_TOKENS:-$EVAL_PROMPT_LENGTH}"
    export CODE_EVAL_MAX_RESPONSE_TOKENS="${CODE_EVAL_MAX_RESPONSE_TOKENS:-$EVAL_RESPONSE_LENGTH}"
    export CODE_EVAL_SIGNATURE="${CODE_EVAL_SIGNATURE:-n${PASS_K}_t${CODE_EVAL_TEMPERATURE}_p${CODE_EVAL_TOP_P}_prompt${CODE_EVAL_MAX_PROMPT_TOKENS}_response${CODE_EVAL_MAX_RESPONSE_TOKENS}_seed${CODE_EVAL_SEED}_base}"
fi
if [ "$TASK" = code ] && [ -f recipe/opd/script_code/runtime.env ]; then
    # Machine-local absolute paths and Python runtime written by prepare_code.sh.
    # shellcheck disable=SC1091
    source recipe/opd/script_code/runtime.env
fi
export PRECOMPUTED_STAGE1_PROMPTS_PATH="${PRECOMPUTED_STAGE1_PROMPTS_PATH:-$TRAIN_DATA_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
if [ "$FAMILY" != opd ]; then
    TEACHER_THINKING_NAME_TAG=""
    EXPERIMENT_TEACHER_SUFFIX=""
else
    TEACHER_THINKING_NAME_TAG="teacher-thinking-${TEACHER_ENABLE_THINKING}"
    EXPERIMENT_TEACHER_SUFFIX="-${TEACHER_THINKING_NAME_TAG}"
fi
export TEACHER_THINKING_NAME_TAG
EXPERIMENT_ID="${EXPERIMENT_ID:-${FAMILY}-${TASK}-${VARIANT}-${MODEL_ALIAS}${EXPERIMENT_TEACHER_SUFFIX}-ms${MULTI_STEP}-${RUN_STAMP}}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$EXPERIMENT_ID}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$EXPERIMENT_ID}"
export WANDB_GROUP="${WANDB_GROUP:-${VARIANT}-${MODEL_SIZE}}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-${FAMILY}-${VARIANT}}"

declare -a canonical_wandb_tags=(
    "task=$TASK"
    "family=$FAMILY"
    "variant=$VARIANT"
    "model=$MODEL_ALIAS"
    "model-size=$MODEL_SIZE"
    "model-kind=base"
    "train-response=$MAX_RESPONSE_LENGTH"
    "eval-response=$EVAL_RESPONSE_LENGTH"
    "eval-pass-k=$PASS_K"
    "eval-fractions=${EVAL_FRACTIONS//,/-}"
    "lr=$LEARNING_RATE"
    "lora=$USE_LORA"
    "lora-rank=$LORA_RANK"
    "micro-batch=$MICRO_BATCH_SIZE_PER_GPU"
    "dynamic-batch=$USE_DYNAMIC_BSZ"
    "seed=$SEED"
)
if [ "$TASK" = math ]; then
    canonical_wandb_tags+=("eval-suite=aime25+aime26+hmmt26+amobench")
else
    canonical_wandb_tags+=("eval-suite=humaneval-plus+mbpp-plus+lcb-v6")
fi
if [ "$FAMILY" = baseline ]; then
    canonical_wandb_tags+=("teacher=none")
    if [ "$VARIANT" = grpo ]; then
        canonical_wandb_tags+=("grpo-group-size=$ROLLOUT_N")
    fi
else
    canonical_wandb_tags+=(
        "teacher=$TEACHER_MODEL"
        "distill=$DISTILL_MODE"
        "kl=$KL_TYPE"
        "kl-support=$KL_METHOD"
        "y-mode=$Y_MODE"
        "rollout=$Y_O_ROLLOUT_MODE"
        "teacher-top-k=$TOP_K"
        "token-clip=$KL_TOKEN_CLIP"
        "requested-ms=$MULTI_STEP"
        "global-prompt-batch=$GLOBAL_PROMPT_BATCH_SIZE"
    )
    if [ "$FAMILY" = opd ]; then
        canonical_wandb_tags+=(
            "teacher-thinking=$TEACHER_ENABLE_THINKING"
            "teacher-supervision-render=$TEACHER_SUPERVISION_RENDER_MODE"
            "teacher-rollout-render=$TEACHER_ROLLOUT_RENDER_MODE"
        )
    fi
    if [ "$VARIANT" = skd ]; then
        canonical_wandb_tags+=(
            "skd-gamma=$SKD_GAMMA"
            "skd-accept-k=$SKD_ACCEPT_TOP_K"
            "skd-teacher-t=$SKD_TEACHER_TEMPERATURE"
        )
    fi
fi
if [ -n "${WANDB_TAGS:-}" ]; then
    canonical_wandb_tags+=("$WANDB_TAGS")
fi
WANDB_TAGS="$(IFS=,; printf '%s' "${canonical_wandb_tags[*]}")"
export WANDB_TAGS WANDB_GROUP WANDB_JOB_TYPE
unset canonical_wandb_tags

if [ -n "$TEACHER_THINKING_NAME_TAG" ]; then
    DEFAULT_RUN_ROOT="outputs/$WANDB_PROJECT/$TASK/$FAMILY/$VARIANT/$MODEL_ALIAS/$TEACHER_THINKING_NAME_TAG/$RUN_STAMP"
    DEFAULT_RESULTS_FILE="results/$WANDB_PROJECT/$TASK/$FAMILY/$VARIANT/$MODEL_ALIAS/$TEACHER_THINKING_NAME_TAG/$RUN_STAMP/results.json"
else
    DEFAULT_RUN_ROOT="outputs/$WANDB_PROJECT/$TASK/$FAMILY/$VARIANT/$MODEL_ALIAS/$RUN_STAMP"
    DEFAULT_RESULTS_FILE="results/$WANDB_PROJECT/$TASK/$FAMILY/$VARIANT/$MODEL_ALIAS/$RUN_STAMP/results.json"
fi
export RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
export WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-$RUN_ROOT/wandb_run.json}"
export WANDB_RUN_IDENTITY="${WANDB_RUN_IDENTITY:-$EXPERIMENT_ID}"
export RESULTS_FILE="${RESULTS_FILE:-$DEFAULT_RESULTS_FILE}"
export EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-$RESULTS_FILE}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-gen_results/eval/$TASK/$EXPERIMENT_ID}"

dry_run_value="${DRY_RUN:-0}"
declare -a canonical_args=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run_value=1 ;;
        *) canonical_args+=("$arg") ;;
    esac
done
set -- "${canonical_args[@]}"
case "$dry_run_value" in
    1|true|TRUE|yes|YES) dry_run=true ;;
    0|false|FALSE|no|NO) dry_run=false ;;
    *) echo "ERROR: DRY_RUN must be 0/1 or false/true (got: $dry_run_value)" >&2; exit 2 ;;
esac

if [ "$#" -gt 0 ]; then
    if [ "$FAMILY" != baseline ] || [ "$VARIANT" = base ]; then
        echo "ERROR: $FAMILY/$VARIANT does not accept positional CLI overrides: $*" >&2
        echo "Configure this launcher through environment variables; only baseline SFT/GRPO accept Hydra overrides." >&2
        exit 2
    fi
fi

print_profile() {
    local profile_train_data="$TRAIN_DATA_PATH"
    if [ "$FAMILY" = baseline ]; then
        case "$VARIANT" in
            base) profile_train_data="n/a (evaluation only)" ;;
            sft) profile_train_data="$SFT_TRAIN_FILE" ;;
        esac
    fi
    cat <<EOF
Canonical OPD experiment
  task/family/variant: $TASK/$FAMILY/$VARIANT
  model:               $MODEL_PATH
  teacher:             ${TEACHER_MODEL_PATH:-n/a}
  train data:          $profile_train_data
  student prompt:      $BASE_PROMPT_LENGTH
  train/eval response: $MAX_RESPONSE_LENGTH/$EVAL_RESPONSE_LENGTH
  train sampling:      T=$ROLLOUT_TEMPERATURE top_p=$ROLLOUT_TOP_P top_k=$ROLLOUT_TOP_K
  micro/dynamic:       $MICRO_BATCH_SIZE_PER_GPU/$USE_DYNAMIC_BSZ
  global prompts:      $GLOBAL_PROMPT_BATCH_SIZE
  requested ms:        $MULTI_STEP (0=automatic 512 prompts/step)
  eval fractions:      $EVAL_FRACTIONS
  eval datasets:       $EVAL_DATASETS
  W&B project/run:     $WANDB_PROJECT / $WANDB_RUN_NAME
  W&B group/job:       $WANDB_GROUP / $WANDB_JOB_TYPE
  W&B tags:            $WANDB_TAGS
  run root:            $RUN_ROOT
  results file:        $RESULTS_FILE
  eval output:         $EVAL_OUTPUT_DIR
EOF
    if [ "$FAMILY" != baseline ]; then
        cat <<EOF
  distill mode:        $DISTILL_MODE
  KL objective:        $KL_TYPE/$KL_METHOD (beta=$BETA)
  y/teacher prompt:    $Y_MODE/$TEACHER_TRAINING_PROMPT
  teacher supervision: $TEACHER_SUPERVISION_RENDER_MODE
  teacher rollout:     $TEACHER_ROLLOUT_RENDER_MODE (chat-template buffer=$TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER)
  rollout mode:        $Y_O_ROLLOUT_MODE
  loss support top-k:  $TOP_K
  token KL clip:       $KL_TOKEN_CLIP
EOF
        if [ "$FAMILY" = opd ]; then
            echo "  teacher thinking:    $TEACHER_ENABLE_THINKING"
        fi
        if [ "$VARIANT" = skd ]; then
            cat <<EOF
  SKD draft/accept:    gamma=$SKD_GAMMA top_k=$SKD_ACCEPT_TOP_K top_p=$SKD_ACCEPT_TOP_P
  SKD sampling:        student_T=$SKD_STUDENT_TEMPERATURE student_p=$SKD_STUDENT_TOP_P teacher_T=$SKD_TEACHER_TEMPERATURE teacher_p=$SKD_TEACHER_TOP_P
  SKD execution:       batch=$SKD_ROLLOUT_BATCH_SIZE lanes=$SKD_PIPELINE_LANES parallel=$SKD_PARALLEL_STUDENT_TEACHER
EOF
        fi
    elif [ "$VARIANT" = grpo ]; then
        cat <<EOF
  GRPO group size:     $ROLLOUT_N responses/prompt
EOF
        if [ "$TASK" = code ]; then
        cat <<EOF
  code reward:         $CODE_GRPO_REWARD_CONTRACT
  reward test suite:   top $CODE_GRPO_MAX_TEST_CASES by input length; binary all-pass; timeout=${CODE_GRPO_EXEC_TIMEOUT_SECONDS}s
EOF
        fi
    fi
}

if [ "$dry_run" = true ]; then
    print_profile
    exit 0
fi

# Code training artifacts are committed as <=40MB byte chunks. Materialize the
# exact, hash-verified canonical parquets once per fresh checkout; every code
# baseline and distillation variant continues to consume the original paths.
if [ "$TASK" = code ] && [ "${OPD_CODE_BOOTSTRAP_READY:-0}" != 1 ] && \
   { [ "$FAMILY" != baseline ] || [ "$VARIANT" != base ]; }; then
    "$PYTHON_BIN" recipe/opd/script_code/materialize_train_bundle.py restore \
        --bundle-dir data/train_dataset/taco/bundle \
        --output-dir data/train_dataset/taco/canonical
fi

resolve_wandb_run_id() {
    export WANDB_RUN_ID="$($PYTHON_BIN - \
        "$WANDB_PROJECT" \
        "$WANDB_RUN_NAME" \
        "$WANDB_RUN_IDENTITY" \
        "$WANDB_RUN_ID_FILE" \
        "${WANDB_RUN_ID:-}" <<'PYWANDBID'
import sys

from recipe.opd.experiment_tracking import resolve_wandb_run_state

project, name, identity, state_path, explicit_id = sys.argv[1:]
state = resolve_wandb_run_state(
    project=project,
    run_name=name,
    run_identity=identity,
    state_path=state_path,
    explicit_run_id=explicit_id,
)
print(state.run_id)
PYWANDBID
)"
}

training_milestones() {
    local parquet_path="$1"
    local global_batch="$2"
    "$PYTHON_BIN" - "$parquet_path" "$global_batch" "$EVAL_FRACTIONS" <<'PYMILESTONES'
import math
import sys

import pyarrow.parquet as pq

from recipe.opd.experiment_tracking import parse_eval_fractions

path, batch_raw, fractions_raw = sys.argv[1:]
batch = int(batch_raw)
rows = pq.ParquetFile(path).metadata.num_rows
if rows % batch:
    raise SystemExit(
        f"canonical parquet has {rows} rows, not divisible by global batch {batch}; "
        "regenerate it with --pad-to-multiple matching GLOBAL_PROMPT_BATCH_SIZE"
    )
total = rows // batch
by_step = {}
for fraction in parse_eval_fractions(fractions_raw):
    step = min(total, math.ceil(total * fraction))
    by_step.setdefault(step, []).append(fraction)
for step in sorted(by_step):
    # Multiple fractions can coincide for a tiny smoke dataset. Log the largest
    # requested fraction while evaluating the shared checkpoint only once.
    print(step, max(by_step[step]), total)
PYMILESTONES
}

require_training_data() {
    local path="$1"
    if [ ! -s "$path" ]; then
        echo "ERROR: canonical training data is missing: $path" >&2
        echo "Prepare it with recipe/opd/dataset/prepare_experiment_data.py" >&2
        exit 1
    fi
}

run_base_eval() {
    export EVAL_KIND=base
    export WANDB_GLOBAL_STEP=0
    export EVAL_STEP=0
    unset EVAL_MILESTONE_FRACTION
    export EVAL_OUTPUT_DIR="$RUN_ROOT/eval/base/generations"
    export EVAL_RESULTS_FILE="$RUN_ROOT/eval/base/results.json"
    export EVAL_METRICS_FILE="$RUN_ROOT/eval/base/metrics.json"
    if [ "$TASK" = "math" ]; then
        DATASETS="$EVAL_DATASETS" \
        EVAL_BASE_MODEL_NAME="$MODEL_ALIAS" \
        EVAL_MODEL_NAME="$EXPERIMENT_ID" \
            bash recipe/math_evaluation/benchmark_kl_model.sh "$MODEL_PATH"
    else
        DATASETS="$EVAL_DATASETS" \
        EVAL_BASE_MODEL_NAME="$MODEL_ALIAS" \
        EVAL_MODEL_NAME="$EXPERIMENT_ID" \
            bash recipe/code_evaluation/benchmark_code_model.sh "$MODEL_PATH"
    fi
}

run_segmented_baseline() {
    local train_file="$1"
    local runner="$2"
    shift 2
    local schedule step fraction total_steps step_dir

    schedule="$(training_milestones "$train_file" "$GLOBAL_PROMPT_BATCH_SIZE")"
    while read -r step fraction total_steps; do
        [ -n "$step" ] || continue
        step_dir="$RUN_ROOT/eval/step_${step}"
        if [ -f "$step_dir/.complete" ]; then
            echo "Milestone step $step/$total_steps is already complete: $step_dir"
            continue
        fi
        export TOTAL_TRAINING_STEPS="$total_steps"
        export STOP_AT_STEP="$step"
        export WANDB_GLOBAL_STEP="$step"
        export EVAL_STEP="$step"
        export EVAL_MILESTONE_FRACTION="$fraction"
        export EVAL_KIND=milestone
        export EVAL_MODEL_NAME="${EXPERIMENT_ID}_step${step}"
        export EVAL_RESULTS_FILE="$step_dir/results.json"
        export EVAL_METRICS_FILE="$step_dir/metrics.json"
        export EVAL_OUTPUT_DIR="$step_dir/generations"
        export EVAL_RESULTS_CSV_FILE="$step_dir/results.csv"
        export RUN_EVAL_AFTER_TRAINING=true
        echo "Running training/eval milestone fraction=$fraction step=$step/$total_steps"
        bash "$runner" "$@"
        mkdir -p "$step_dir"
        touch "$step_dir/.complete"
    done <<< "$schedule"
}

if [ "$FAMILY" = baseline ]; then
    # Baselines do not pass through run_kl_training.sh, so resolve their
    # persisted W&B identity here.  Distillation resolves it only after its
    # resume-matching logic has selected the correct historical state file.
    resolve_wandb_run_id
    case "$VARIANT" in
        base)
            if [ "$#" -gt 0 ]; then
                echo "ERROR: base evaluation accepts configuration through environment variables, not positional arguments: $*" >&2
                exit 2
            fi
            run_base_eval
            ;;
        sft)
            require_training_data "$SFT_TRAIN_FILE"
            export TRAIN_FILE="$SFT_TRAIN_FILE"
            export MAX_LENGTH="$((BASE_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
            export MAX_TOKEN_LEN_PER_GPU="$MAX_LENGTH"
            export TRAIN_BATCH_SIZE="$GLOBAL_PROMPT_BATCH_SIZE"
            export SFT_DATASET_CLASS="recipe.opd.base_completion.BaseCompletionSFTDataset"
            export SFT_DRY_RUN=false
            if [ "$TASK" = math ]; then
                run_segmented_baseline \
                    "$SFT_TRAIN_FILE" \
                    recipe/opd/run/sft_deepscaler/_run_qwen3_deepscaler_sft.sh \
                    "$@"
            else
                run_segmented_baseline \
                    "$SFT_TRAIN_FILE" \
                    recipe/opd/run/sft_deepscaler/_run_qwen3_taco_sft.sh \
                    "$@"
            fi
            ;;
        grpo)
            require_training_data "$TRAIN_FILE"
            export TRAIN_BATCH_SIZE="$GLOBAL_PROMPT_BATCH_SIZE"
            export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
            export MAX_TRAIN_DURATION_SECONDS="${MAX_TRAIN_DURATION_SECONDS:-31536000}"
            export TOTAL_EPOCHS=1
            export GRPO_DRY_RUN=false
            export TRAIN_DIR="${TRAIN_DIR:-$RUN_ROOT/training}"
            export CKPTS_DIR="${CKPTS_DIR:-$RUN_ROOT/checkpoints}"
            run_segmented_baseline \
                "$TRAIN_FILE" \
                recipe/opd/run/grpo/_run_qwen3_grpo_8h100.sh \
                "$@"
            ;;
    esac
    exit 0
fi

require_training_data "$TRAIN_DATA_PATH"
if [ "$#" -gt 0 ]; then
    echo "ERROR: OPD/OPSD launchers accept configuration through environment variables, not positional arguments: $*" >&2
    exit 2
fi
if [ "$Y_MODE" = y_r ]; then
    # Qwen3 Base checkpoints expose a 32K native context. Reserve the full
    # 16K target response and truncate the derived rewrite prompt to the
    # remaining context instead of requesting an invalid 38K/43K sequence.
    export MAX_PROMPT_LENGTH="$((MODEL_CONTEXT_LENGTH - MAX_RESPONSE_LENGTH - TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER))"
    if [ "$MAX_PROMPT_LENGTH" -le 0 ]; then
        echo "ERROR: teacher chat-template buffer leaves no y_r prompt budget" >&2
        exit 2
    fi
    export MAX_LENGTH="$MODEL_CONTEXT_LENGTH"
    export STAGE2_PROMPT_LENGTH="$MAX_PROMPT_LENGTH"
else
    if [ "$FAMILY" = opd ]; then
        export MAX_PROMPT_LENGTH="$BASE_PROMPT_LENGTH"
        export MAX_LENGTH="$((BASE_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER))"
    else
        export MAX_PROMPT_LENGTH="$((BASE_PROMPT_LENGTH + EXPERT_SOLUTION_PROMPT_LENGTH))"
        export MAX_LENGTH="$((BASE_PROMPT_LENGTH + EXPERT_SOLUTION_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER))"
    fi
    export STAGE2_PROMPT_LENGTH="$((BASE_PROMPT_LENGTH + EXPERT_SOLUTION_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
fi
export MAX_TOKEN_LEN_PER_GPU="$MAX_LENGTH"
# The bottom-level KL runner derives both rollout partition size and gradient
# accumulation from MULTI_STEP (or from the default 512 prompts when ms=0).
export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"
export PIPELINE_LOCAL_KEEP_POLICY="${PIPELINE_LOCAL_KEEP_POLICY:-rolling_last_hf}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/training}"
export MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$RUN_ROOT/models}"
export EVAL_DATASETS="${EVAL_DATASETS// /,}"

exec bash recipe/opd/run/run_kl_training.sh "$@"
