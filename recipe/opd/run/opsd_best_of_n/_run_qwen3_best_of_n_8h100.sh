#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

TASK="${TASK:?TASK must be math or code}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_ALIAS="${MODEL_ALIAS:?MODEL_ALIAS is required}"
SAMPLING_BUDGET_MODE="${SAMPLING_BUDGET_MODE:-limited}"
SAMPLING_BUDGET_SECONDS="${SAMPLING_BUDGET_SECONDS:-}"
BEST_OF_N="${BEST_OF_N:-4}"
RUN_NAMESPACE="${RUN_NAMESPACE:-opsd_best_of_n}"
RUN_VARIANT_TAG="${RUN_VARIANT_TAG:-}"

case "$TASK" in
    math|code) ;;
    *) echo "ERROR: TASK must be math or code, got: $TASK" >&2; exit 1 ;;
esac
case "$SAMPLING_BUDGET_MODE" in
    limited)
        case "$SAMPLING_BUDGET_SECONDS" in
            *[!0-9]*|"") echo "ERROR: limited sampling requires a positive SAMPLING_BUDGET_SECONDS" >&2; exit 1 ;;
        esac
        if [ "$SAMPLING_BUDGET_SECONDS" -le 0 ]; then
            echo "ERROR: SAMPLING_BUDGET_SECONDS must be positive in limited mode" >&2
            exit 1
        fi
        ;;
    unlimited) ;;
    *)
        echo "ERROR: SAMPLING_BUDGET_MODE must be limited or unlimited, got: $SAMPLING_BUDGET_MODE" >&2
        exit 1
        ;;
esac
case "$BEST_OF_N" in
    *[!0-9]*|"") echo "ERROR: BEST_OF_N must be an integer greater than 1" >&2; exit 1 ;;
esac
if [ "$BEST_OF_N" -le 1 ]; then
    echo "ERROR: BEST_OF_N must be greater than 1" >&2
    exit 1
fi
case "$RUN_NAMESPACE" in
    *[!a-zA-Z0-9_.-]*|"") echo "ERROR: invalid RUN_NAMESPACE: $RUN_NAMESPACE" >&2; exit 1 ;;
esac
case "$RUN_VARIANT_TAG" in
    *[!a-zA-Z0-9_.-]*) echo "ERROR: invalid RUN_VARIANT_TAG: $RUN_VARIANT_TAG" >&2; exit 1 ;;
esac

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

require_file() {
    if [ ! -s "$1" ]; then
        echo "ERROR: missing $2: $1" >&2
        exit 1
    fi
}

require_model() {
    if [ ! -d "$1" ] || [ ! -s "$1/config.json" ]; then
        echo "ERROR: missing local base model or config.json: $1" >&2
        echo "Run recipe/opd/run/grpo/prepare/download_models.sh or set MODEL_PATH." >&2
        exit 1
    fi
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

MODEL_PATH="$(absolute_path "$MODEL_PATH")"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d.%H%M%S)}"
RUN_VARIANT_SUFFIX=""
if [ -n "$RUN_VARIANT_TAG" ]; then
    RUN_VARIANT_SUFFIX="_${RUN_VARIANT_TAG}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_ALIAS}_${TASK}_opsd_best_of_${BEST_OF_N}${RUN_VARIANT_SUFFIX}_${TIMESTAMP}}"
RUN_DATE="${RUN_DATE:-bon${BEST_OF_N}${RUN_VARIANT_SUFFIX}_${TIMESTAMP}}"

if [ "$TASK" = "math" ]; then
    TRAIN_FILE="$(absolute_path "${TRAIN_FILE:-data/train_dataset/deepscaler/train_grpo.parquet}")"
    BASE_PROMPT_LENGTH="${BASE_PROMPT_LENGTH:-2048}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
    EXPERT_SOLUTION_PROMPT_LENGTH="${EXPERT_SOLUTION_PROMPT_LENGTH:-3072}"
    TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-deepscaleR}"
    EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25,beyondaime,amobench}"
    EVAL_DATASETS_DIR="$(absolute_path "${EVAL_DATASETS_DIR:-data/eval_dataset/math}")"
else
    TRAIN_FILE="$(absolute_path "${TRAIN_FILE:-data/train_dataset/taco/train_grpo.parquet}")"
    BASE_PROMPT_LENGTH="${BASE_PROMPT_LENGTH:-4096}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
    EXPERT_SOLUTION_PROMPT_LENGTH="${EXPERT_SOLUTION_PROMPT_LENGTH:-4096}"
    TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-BAAI/TACO}"
    EVAL_DATASETS="${EVAL_DATASETS:-humaneval_plus,mbpp_plus,livecodebench_v6}"
    EVAL_DATASETS_DIR="$(absolute_path "${EVAL_DATASETS_DIR:-data/eval_dataset/code}")"
fi
export HF_HOME="${OPSD_HF_HOME:-$EVAL_DATASETS_DIR/huggingface_cache}"
export HF_DATASETS_CACHE="${OPSD_HF_DATASETS_CACHE:-$HF_HOME/datasets}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER="${ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER:-4096}"
ROLLOUT_MAX_MODEL_LEN=$((BASE_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER))
if [ "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" -lt "$ROLLOUT_MAX_MODEL_LEN" ]; then
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_MODEL_LEN"
fi

PASS_K="${PASS_K:-16}"
GEN_TP="${GEN_TP:-1}"
EVAL_GEN_TP="${EVAL_GEN_TP:-1}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-64}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GENERATION_MAX_CONCURRENCY="${GENERATION_MAX_CONCURRENCY:-128}"

RUN_ROOT="$(absolute_path "${RUN_ROOT:-outputs/$RUN_NAMESPACE/$EXPERIMENT_NAME}")"
SAMPLING_DIR="$(absolute_path "${SAMPLING_DIR:-gen_results/$RUN_NAMESPACE/$EXPERIMENT_NAME}")"
PROMPTS_FILE="${PROMPTS_FILE:-$SAMPLING_DIR/stage1_prompts.parquet}"
CANDIDATES_FILE="${CANDIDATES_FILE:-$SAMPLING_DIR/stage1_best_of_${BEST_OF_N}_candidates.parquet}"
SELECTED_TRAIN_FILE="${SELECTED_TRAIN_FILE:-$SAMPLING_DIR/train_best_of_${BEST_OF_N}_correct_only.parquet}"
MODEL_SAVE_DIR="$(absolute_path "${MODEL_SAVE_DIR:-model/trained/$RUN_NAMESPACE}")"
GEN_RESULTS_ROOT="$(absolute_path "${GEN_RESULTS_ROOT:-gen_results}")"

PREPARE_COMMAND=(
    python3 -m recipe.opd.run.opsd_best_of_n.prepare_best_of_n_prompts
    --input "$TRAIN_FILE"
    --output "$PROMPTS_FILE"
)
if [ -n "$MAX_SAMPLES" ]; then
    PREPARE_COMMAND+=(--max-samples "$MAX_SAMPLES")
fi

SAMPLING_COMMAND_PREFIX=(
    env -u PYTORCH_CUDA_ALLOC_CONF
    python3 -m verl.trainer.main_generation_server
    trainer.nnodes="$NNODES"
    trainer.n_gpus_per_node="$NGPUS_PER_NODE"
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=true
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.prompt_length="$BASE_PROMPT_LENGTH"
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH"
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN"
    actor_rollout_ref.rollout.tensor_model_parallel_size="$GEN_TP"
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n="$BEST_OF_N"
    data.train_files="['$PROMPTS_FILE']"
    data.prompt_key=prompt
    +data.output_path="$CANDIDATES_FILE"
)

SELECT_COMMAND=(
    python3 -m recipe.opd.run.opsd_best_of_n.select_best_of_n
    --input "$CANDIDATES_FILE"
    --output "$SELECTED_TRAIN_FILE"
    --n "$BEST_OF_N"
)

MODEL_RUN_NAME="y_o_kl_forward_full_vocab_clip0_vanilla_ms1_${RUN_DATE}"
EXPECTED_MODEL_PATH="$MODEL_SAVE_DIR/OPSD_${TASK^^}/$MODEL_ALIAS/$MODEL_RUN_NAME/epoch1/ms1/batch00001/hf_merged"
RESULTS_FILE="$(absolute_path "${RESULTS_FILE:-results/OPSD/$TASK/${MODEL_ALIAS}_${TASK}.json}")"
RESULTS_MODEL_KEY_BASE="${RESULTS_MODEL_KEY:-${MODEL_ALIAS}_OPSD_${TASK^^}_${MODEL_RUN_NAME}_LORA}"
EXPECTED_RESULTS_MODEL_KEY="${RESULTS_MODEL_KEY_BASE}_step00001of00001"

if [ "${OPSD_BEST_OF_N_DRY_RUN:-false}" = "true" ]; then
    echo "Task:                    $TASK"
    echo "Base model:              $MODEL_PATH"
    echo "Source train parquet:    $TRAIN_FILE"
    echo "Best-of-N:               $BEST_OF_N"
    if [ "$SAMPLING_BUDGET_MODE" = "limited" ]; then
        echo "Sampling-only budget:    ${SAMPLING_BUDGET_SECONDS}s"
    else
        echo "Sampling-only budget:    none (generate the complete dataset)"
    fi
    echo "Run namespace:           $RUN_NAMESPACE"
    echo "Run variant tag:         ${RUN_VARIANT_TAG:-<none>}"
    echo "Rollout memory:          $ROLLOUT_GPU_MEMORY_UTILIZATION"
    echo "Rollout tensor parallel: $GEN_TP"
    echo "Rollout max model len:   $ROLLOUT_MAX_MODEL_LEN"
    echo "Candidate parquet:       $CANDIDATES_FILE"
    echo "Selected train parquet:  $SELECTED_TRAIN_FILE"
    echo "Training output:         $RUN_ROOT"
    echo "Training gen-results:    $GEN_RESULTS_ROOT"
    echo "Final merged model:      $EXPECTED_MODEL_PATH"
    echo "Eval results JSON:       $RESULTS_FILE"
    echo "Results model key:       $EXPECTED_RESULTS_MODEL_KEY"
    echo "Eval datasets:           $EVAL_DATASETS"
    echo "Eval Avg/Pass K:         $PASS_K"
    echo "Periodic saves:          disabled; final save only"
    print_command "${PREPARE_COMMAND[@]}"
    if [ "$SAMPLING_BUDGET_MODE" = "limited" ]; then
        echo "Sampling command receives +data.generation_deadline_epoch_seconds=<start+$SAMPLING_BUDGET_SECONDS>:"
    else
        echo "Sampling command has no generation deadline:"
    fi
    print_command "${SAMPLING_COMMAND_PREFIX[@]}"
    print_command "${SELECT_COMMAND[@]}"
    exit 0
fi

require_model "$MODEL_PATH"
require_file "$TRAIN_FILE" "training parquet"
mkdir -p "$RUN_ROOT" "$SAMPLING_DIR" "$MODEL_SAVE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

if [ ! -s "$PROMPTS_FILE" ]; then
    "${PREPARE_COMMAND[@]}"
else
    echo "Reusing prepared prompts: $PROMPTS_FILE"
fi

if [ ! -s "$CANDIDATES_FILE" ]; then
    export GENERATION_MAX_CONCURRENCY
    if [ "$SAMPLING_BUDGET_MODE" = "limited" ]; then
        sampling_start_epoch="$(date +%s)"
        sampling_deadline_epoch=$((sampling_start_epoch + SAMPLING_BUDGET_SECONDS))
        echo "Starting Best-of-$BEST_OF_N sampling; deadline=$sampling_deadline_epoch budget=${SAMPLING_BUDGET_SECONDS}s"
        "${SAMPLING_COMMAND_PREFIX[@]}" \
            +data.generation_deadline_epoch_seconds="$sampling_deadline_epoch" \
            2>&1 | tee "$RUN_ROOT/sampling.log"
    else
        echo "Starting complete Best-of-$BEST_OF_N sampling with no wall-clock deadline"
        "${SAMPLING_COMMAND_PREFIX[@]}" \
            2>&1 | tee "$RUN_ROOT/sampling.log"
    fi
else
    echo "Reusing completed candidate parquet: $CANDIDATES_FILE"
fi

if [ ! -s "$SELECTED_TRAIN_FILE" ]; then
    "${SELECT_COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/select_best_of_n.log"
else
    echo "Reusing selected training parquet: $SELECTED_TRAIN_FILE"
fi

export TASK MODEL_PATH
export MODEL_NAME="$MODEL_ALIAS"
export DATA_PATH="$SELECTED_TRAIN_FILE"
export TRAIN_DATA_PATH="$TRAIN_FILE"
export TRAIN_DATA_SOURCE
export DISTILL_MODE=opsd
export Y_MODE=y_o
export MULTI_STEP=1
export TOTAL_EPOCHS=1
export TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND:-1}"
export BASE_PROMPT_LENGTH MAX_RESPONSE_LENGTH EXPERT_SOLUTION_PROMPT_LENGTH
export NGPUS_PER_NODE NNODES EVAL_GEN_TP EVAL_MAX_NUM_SEQS EVAL_GPU_MEMORY_UTILIZATION
export PASS_K EVAL_DATASETS EVAL_DATASETS_DIR
export RUN_DATE
export RESULTS_FILE GEN_RESULTS_ROOT
export RESULTS_MODEL_KEY="$RESULTS_MODEL_KEY_BASE"
export OUTPUT_DIR="$RUN_ROOT/training"
export MODEL_SAVE_DIR
export PIPELINE_ARCHIVE_MODEL_ROOT="$MODEL_SAVE_DIR/archive"
export SAVE_MERGED_MODEL=true
export SAVE_STEPS=-1
export KEEP_LAST_N_CHECKPOINTS=1
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
export PIPELINE_RESUME_MODE=fresh
export NTFY_ENABLED="${NTFY_ENABLED:-false}"

if [ "$TASK" = "code" ]; then
    CODE_EVAL_ROOT="$EVAL_DATASETS_DIR"
    export HUMANEVAL_OVERRIDE_PATH="${HUMANEVAL_OVERRIDE_PATH:-$CODE_EVAL_ROOT/evalplus/HumanEvalPlus-v0.1.10.jsonl}"
    export MBPP_OVERRIDE_PATH="${MBPP_OVERRIDE_PATH:-$CODE_EVAL_ROOT/evalplus/MbppPlus-v0.2.0.jsonl}"
    export LCB_REPO="${LCB_REPO:-$CODE_EVAL_ROOT/LiveCodeBench}"
    export LCB_CODEGEN_LITE_DIR="${LCB_CODEGEN_LITE_DIR:-$CODE_EVAL_ROOT/livecodebench/code_generation_lite}"
    export PYTHONPATH="$LCB_REPO:$REPO_ROOT:${PYTHONPATH:-}"
fi

exec bash "$REPO_ROOT/recipe/opd/run/opsd/forward_y_o.sh" "$@"
