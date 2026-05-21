#!/bin/bash
# =============================================================================
# Megatron OPD compatibility runner.
#
# This maps the non-ablation recipe/opd/run knobs onto the official
# recipe/gkd/megatron implementation. The default mode is one-step
# optimization: one full-dataset rollout/teacher/update cycle.
# =============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(dirname "$SCRIPT_DIR")"
VERL_ROOT="$(dirname "$(dirname "$(dirname "$MEGATRON_DIR")")")"

PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
        export "$arg"
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done
set -- "${PASSTHROUGH_ARGS[@]}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"
MODEL_NAME="${MODEL_PATH##*/}"

DISTILL_MODE="${DISTILL_MODE:-opd}"
KL_TYPE="${KL_TYPE:-forward}"              # forward | reverse | jsd
KL_METHOD="${KL_METHOD:-full_vocab}"       # full_vocab uses vLLM teacher prompt logprobs
Y_MODE="${Y_MODE:-y_r}"                    # y_o | y_r
TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-refine}" # refine | vanilla
APPEND_INSTRUCTION_TO_PROMPT="${APPEND_INSTRUCTION_TO_PROMPT:-True}"
KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0}"
BETA="${BETA:-0}"
TOP_K="${TOP_K:-0}"                        # 0 = use all teacher-server top-k rows
TEMPERATURE="${TEMPERATURE:-1.0}"

LEARNING_RATE="${LEARNING_RATE:-2e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MIN_LR="${MIN_LR:-0.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" # compatibility only; Megatron uses micro-batches
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${TRAIN_EPOCHS_PER_ROUND:-1}}"
OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-one_step}"
SCHEDULER="${SCHEDULER:-auto}"
TEACHER_INFLIGHT_SAMPLES="${TEACHER_INFLIGHT_SAMPLES:-0}"
MAX_POLICY_LAG="${MAX_POLICY_LAG:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}" # actor update microbatch; one_step may later expand this to the full-dataset batch
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-$TRAIN_BATCH_SIZE}" # online y_o/y_r generation batch
ACTOR_UPDATE_BATCH_SIZE="${ACTOR_UPDATE_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"    # internal fallback; USE_DYNAMIC_BSZ=True uses STUDENT_ACTOR_MAX_TOKENS_PER_GPU as the real limiter
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DATA_PATH="${DATA_PATH:-}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/scratch/l/luli/data/DeepScaleR-Cleaned}"
PROMPT_KEY="${PROMPT_KEY:-auto}"           # auto chooses prompt, problem, then question
DATA_SOURCE="${DATA_SOURCE:-math}"         # used when the dataset has no data_source column

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${NGPUS_PER_NODE:-4}}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-$N_GPUS_PER_NODE}" # compatibility alias; async rollout is colocated
PP="${PP:-1}"
TP="${TP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"
INFER_TP="${INFER_TP:-${GEN_TP:-1}}"
SP="${SP:-False}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
STUDENT_ACTOR_MAX_TOKENS_PER_GPU="${STUDENT_ACTOR_MAX_TOKENS_PER_GPU:-49152}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-${NUM_WORKERS:-8}}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
# prefix_caching default off: vLLM 0.11.0 V1 + Qwen3 + flashinfer autotuner has
# reproduced an IMA at the very first scheduled prompt (rms_norm in q_norm) with
# prefix_caching=True; the 14-min variant of the same crash was unaffected, but
# this newer first-batch variant looks distinct. Keep disabled by default until
# vLLM is upgraded out of 0.11.0.
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-True}"
# Megatron actor stays on GPU during vLLM generation (sleep_mode offloads
# vLLM only, not the actor's params/optimizer), so leave headroom.
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"

TEACHER_SERVER_HOST="${TEACHER_SERVER_HOST:-127.0.0.1}"
TEACHER_SERVER_PORT="${TEACHER_SERVER_PORT:-15555}"
TEACHER_BACKEND="${TEACHER_BACKEND:-auto}" # auto | vllm_server | local_hf
TEACHER_N_SERVER_WORKERS="${TEACHER_N_SERVER_WORKERS:-1}"
TEACHER_CLIENT_TIMEOUT_MS="${TEACHER_CLIENT_TIMEOUT_MS:-3600000}"
TEACHER_REQUEST_BATCH_SIZE="${TEACHER_REQUEST_BATCH_SIZE:-}"
PROJECT_NAME="${PROJECT_NAME:-${WANDB_PROJECT:-on-policy-distill-opd}}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M%S)}"
SAVE_FREQ_REQUESTED="${SAVE_FREQ:-${SAVE_STEPS:-auto}}"
LOGGER="${LOGGER:-['console']}"
RAY_NO_WAIT="${RAY_NO_WAIT:-false}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-${MODEL_SAVE_DIR:-/scratch/l/luli/jiangli/ckpt}}"

is_true() {
    case "${1:-}" in
        true|True|1|yes|Yes|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

sanitize_path_component() {
    local value="${1:-unknown}"
    value="${value%/}"
    value="${value##*/}"
    value="${value// /_}"
    value="${value//[^A-Za-z0-9._-]/_}"
    if [ -z "$value" ]; then
        value="unknown"
    fi
    printf '%s' "$value"
}

case "$OPTIMIZATION_MODE" in
    one_step|multi_step) ;;
    *) echo "ERROR: OPTIMIZATION_MODE must be one_step or multi_step, got $OPTIMIZATION_MODE" >&2; exit 1 ;;
esac
case "$SCHEDULER" in
    auto|one_step|one_step_off|two_step_off|three_step_off|bounded_lag_y_r) ;;
    *) echo "ERROR: SCHEDULER must be auto, one_step, one_step_off, two_step_off, three_step_off, or bounded_lag_y_r, got $SCHEDULER" >&2; exit 1 ;;
esac
if [ "$OPTIMIZATION_MODE" = "one_step" ] && [ "$SCHEDULER" != "auto" ] && [ "$SCHEDULER" != "one_step" ]; then
    echo "ERROR: OPTIMIZATION_MODE=one_step requires SCHEDULER=auto or SCHEDULER=one_step." >&2
    echo "       SCHEDULER=$SCHEDULER is an async pipeline scheduler and would not run one full-dataset update." >&2
    exit 1
fi

case "$DISTILL_MODE" in
    opd|opsd) ;;
    *) echo "ERROR: DISTILL_MODE must be opd or opsd, got $DISTILL_MODE" >&2; exit 1 ;;
esac
case "$KL_TYPE" in
    forward|reverse|jsd) ;;
    *) echo "ERROR: KL_TYPE must be forward, reverse, or jsd, got $KL_TYPE" >&2; exit 1 ;;
esac
case "$TEACHER_BACKEND" in
    auto|vllm_server|local_hf) ;;
    *) echo "ERROR: TEACHER_BACKEND must be auto, vllm_server, or local_hf, got $TEACHER_BACKEND" >&2; exit 1 ;;
esac
if [ "$TEACHER_BACKEND" = "auto" ]; then
    TEACHER_BACKEND="vllm_server"
fi
case "$Y_MODE" in
    y_o|y_r) ;;
    y_raw) Y_MODE="y_o" ;;
    y_cor) Y_MODE="y_r" ;;
    *) echo "ERROR: Y_MODE must be y_o or y_r, got $Y_MODE" >&2; exit 1 ;;
esac
case "$TEACHER_TRAINING_PROMPT" in
    refine|vanilla) ;;
    *) echo "ERROR: TEACHER_TRAINING_PROMPT must be refine or vanilla, got $TEACHER_TRAINING_PROMPT" >&2; exit 1 ;;
esac
if [ -z "${MAX_PROMPT_LENGTH+x}" ]; then
    MAX_PROMPT_LENGTH=1024
fi
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
if [ -z "${TEACHER_MAX_PROMPT_LENGTH+x}" ]; then
    if [ "$Y_MODE" = "y_r" ] && [ "$TEACHER_TRAINING_PROMPT" = "refine" ]; then
        TEACHER_MAX_PROMPT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    else
        TEACHER_MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH"
    fi
fi
if [ -z "${TEACHER_SEQ_LEN+x}" ]; then
    TEACHER_SEQ_LEN=$((TEACHER_MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
fi
EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME:-${WANDB_RUN_NAME:-}}"

RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-false}"
SAVE_MERGED_MODEL="${SAVE_MERGED_MODEL:-$RUN_EVAL_AFTER_TRAINING}"
MERGE_TIE_WORD_EMBEDDING="${MERGE_TIE_WORD_EMBEDDING:-False}"
MERGE_HF_MODEL_CONFIG_PATH="${MERGE_HF_MODEL_CONFIG_PATH:-}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-}"
EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt24,hmmt25,beyondaime,amobench}"
EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-/scratch/l/luli/jiangli/datasets/eval}"
PASS_K="${PASS_K:-16}"
EVAL_NNODES="${EVAL_NNODES:-$NNODES}"
EVAL_N_GPUS_PER_NODE="${EVAL_N_GPUS_PER_NODE:-$N_GPUS_PER_NODE}"
EVAL_GEN_TP="${EVAL_GEN_TP:-$INFER_TP}"

if [ "$DISTILL_MODE" = "opd" ] && [ -z "$TEACHER_MODEL_PATH" ]; then
    echo "ERROR: DISTILL_MODE=opd requires TEACHER_MODEL_PATH for the external teacher server." >&2
    exit 1
fi

for unsupported_var in \
    USE_LORA LORA_RANK LORA_ALPHA FSDP_STRATEGY FSDP_SIZE SP_SIZE \
    PARAM_OFFLOAD OPTIMIZER_OFFLOAD OFFLOAD_POLICY \
    MAX_LENGTH STAGE2_PROMPT_LENGTH CORRECTED_RESPONSES_PATH \
    PROMPT_TRUNCATION FORWARD_STAGE2_MODE FORWARD_FILTER_STAGE2 \
    FORWARD_FILTER_THRESHOLD FORWARD_FILTER_REQUIRE_STAGE1_FAILED \
    GRAD_COSINE_INTERVAL CORRECTION_TOKEN_PHRASES CORRECTION_TOKEN_IDS \
    LOG_DIFFICULTY_BUCKETS KEEP_LAST_N_CHECKPOINTS \
    NODE_RANK MASTER_ADDR MASTER_PORT
do
    if [ -n "${!unsupported_var:-}" ]; then
        echo "WARNING: $unsupported_var is a recipe/opd FSDP/offline-pipeline knob and is not implemented by the official Megatron GKD path." >&2
    fi
done

check_server_ready() {
    python3 - "$TEACHER_SERVER_HOST" "$TEACHER_SERVER_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=5):
        pass
except OSError:
    sys.exit(1)
PY
}

echo "Checking teacher server at ${TEACHER_SERVER_HOST}:${TEACHER_SERVER_PORT} ..."
if ! check_server_ready; then
    echo "ERROR: teacher server is not reachable. Start recipe/gkd/megatron/teacher first." >&2
    exit 1
fi

if [ -z "$DATA_PATH" ]; then
    DATA_PATH="$TRAIN_DATA_PATH"
fi
if [ -d "$DATA_PATH" ]; then
    if [ ! -f "$DATA_PATH/dataset_info.json" ] && [ ! -f "$DATA_PATH/state.json" ]; then
        echo "ERROR: DATA_PATH directory must be a HuggingFace dataset saved by save_to_disk(): $DATA_PATH" >&2
        exit 1
    fi
else
    case "$DATA_PATH" in
        *.parquet|*.json|*.jsonl) ;;
        *)
            echo "ERROR: DATA_PATH/TRAIN_DATA_PATH must point to raw prompt parquet/json/jsonl or a HuggingFace dataset directory." >&2
            echo "       This runner no longer consumes or creates offline y_o/y_r generation files." >&2
            exit 1
            ;;
    esac
fi
if [ ! -e "$DATA_PATH" ]; then
    echo "ERROR: raw prompt data path does not exist: $DATA_PATH" >&2
    exit 1
fi

export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

resolve_dataset_size() {
    "$PYTHON_BIN" - "$DATA_PATH" <<'PY'
import json
import os
import sys

path = sys.argv[1]
if os.path.isdir(path):
    state_path = os.path.join(path, "state.json")
    if os.path.exists(state_path):
        import pyarrow.ipc as ipc

        with open(state_path) as f:
            state = json.load(f)
        total_rows = 0
        for data_file in state.get("_data_files", []):
            filename = data_file.get("filename")
            if not filename:
                continue
            arrow_path = os.path.join(path, filename)
            with ipc.open_stream(arrow_path) as reader:
                total_rows += sum(batch.num_rows for batch in reader)
        if total_rows:
            print(total_rows)
        else:
            from datasets import load_from_disk

            print(len(load_from_disk(path)))
    else:
        from datasets import load_from_disk

        print(len(load_from_disk(path)))
elif path.endswith(".parquet"):
    try:
        import pyarrow.parquet as pq

        print(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        import pandas as pd

        print(len(pd.read_parquet(path)))
elif path.endswith(".jsonl"):
    with open(path) as f:
        print(sum(1 for line in f if line.strip()))
elif path.endswith(".json"):
    with open(path) as f:
        data = json.load(f)
    print(len(data))
else:
    raise ValueError(f"Unsupported data path: {path}")
PY
}

DATASET_NUM_ROWS="$(resolve_dataset_size)"
case "$DATASET_NUM_ROWS" in
    ""|*[!0-9]*) echo "ERROR: failed to resolve dataset size from $DATA_PATH" >&2; exit 1 ;;
esac
if [ "$DATASET_NUM_ROWS" -lt 1 ]; then
    echo "ERROR: training dataset is empty: $DATA_PATH" >&2
    exit 1
fi

if [ "$MAX_SAMPLES" -gt 0 ] && [ "$MAX_SAMPLES" -lt "$DATASET_NUM_ROWS" ]; then
    EFFECTIVE_TRAIN_SAMPLES="$MAX_SAMPLES"
else
    EFFECTIVE_TRAIN_SAMPLES="$DATASET_NUM_ROWS"
fi

case "$TRAIN_BATCH_SIZE" in
    ""|*[!0-9]*) echo "ERROR: TRAIN_BATCH_SIZE must be a positive integer, got $TRAIN_BATCH_SIZE" >&2; exit 1 ;;
esac
if [ "$TRAIN_BATCH_SIZE" -lt 1 ]; then
    echo "ERROR: TRAIN_BATCH_SIZE must be positive, got $TRAIN_BATCH_SIZE" >&2
    exit 1
fi
case "$ROLLOUT_BATCH_SIZE" in
    ""|*[!0-9]*) echo "ERROR: ROLLOUT_BATCH_SIZE must be a positive integer, got $ROLLOUT_BATCH_SIZE" >&2; exit 1 ;;
esac
if [ "$ROLLOUT_BATCH_SIZE" -lt 1 ]; then
    echo "ERROR: ROLLOUT_BATCH_SIZE must be positive, got $ROLLOUT_BATCH_SIZE" >&2
    exit 1
fi
case "$ACTOR_UPDATE_BATCH_SIZE" in
    ""|*[!0-9]*) echo "ERROR: ACTOR_UPDATE_BATCH_SIZE must be a positive integer, got $ACTOR_UPDATE_BATCH_SIZE" >&2; exit 1 ;;
esac
if [ "$ACTOR_UPDATE_BATCH_SIZE" -lt 1 ]; then
    echo "ERROR: ACTOR_UPDATE_BATCH_SIZE must be positive, got $ACTOR_UPDATE_BATCH_SIZE" >&2
    exit 1
fi
case "$TEACHER_INFLIGHT_SAMPLES" in
    ""|*[!0-9]*) echo "ERROR: TEACHER_INFLIGHT_SAMPLES must be a non-negative integer, got $TEACHER_INFLIGHT_SAMPLES" >&2; exit 1 ;;
esac
case "$MAX_POLICY_LAG" in
    ""|*[!0-9]*) echo "ERROR: MAX_POLICY_LAG must be a non-negative integer, got $MAX_POLICY_LAG" >&2; exit 1 ;;
esac
if [ "$OPTIMIZATION_MODE" != "one_step" ] && [ $((ROLLOUT_BATCH_SIZE % ACTOR_UPDATE_BATCH_SIZE)) -ne 0 ]; then
    echo "ERROR: ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE must be divisible by ACTOR_UPDATE_BATCH_SIZE=$ACTOR_UPDATE_BATCH_SIZE" >&2
    exit 1
fi
EFFECTIVE_ACTOR_BATCH_SIZE=$((ACTOR_UPDATE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [ "$OPTIMIZATION_MODE" != "one_step" ] && [ "$SCHEDULER" = "three_step_off" ] && [ "$ROLLOUT_BATCH_SIZE" -gt "$ACTOR_UPDATE_BATCH_SIZE" ]; then
    if [ $((EFFECTIVE_ACTOR_BATCH_SIZE % ROLLOUT_BATCH_SIZE)) -ne 0 ]; then
        echo "ERROR: streaming three_step_off requires EFFECTIVE_ACTOR_BATCH_SIZE=$EFFECTIVE_ACTOR_BATCH_SIZE to be a multiple of ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE." >&2
        echo "       With ROLLOUT_BATCH_SIZE>$ACTOR_UPDATE_BATCH_SIZE, otherwise optimizer.step can happen before the rollout window is fully consumed." >&2
        exit 1
    fi
fi
case "$TOTAL_EPOCHS" in
    ""|*[!0-9]*) echo "ERROR: TOTAL_EPOCHS must be a positive integer, got $TOTAL_EPOCHS" >&2; exit 1 ;;
esac
if [ "$TOTAL_EPOCHS" -lt 1 ]; then
    echo "ERROR: TOTAL_EPOCHS must be positive, got $TOTAL_EPOCHS" >&2
    exit 1
fi

if [ "$OPTIMIZATION_MODE" = "one_step" ]; then
    STEPS_PER_EPOCH=1
    DROPPED_TRAIN_SAMPLES=0
else
    ROLLOUT_STEPS_PER_EPOCH=$((EFFECTIVE_TRAIN_SAMPLES / ROLLOUT_BATCH_SIZE))
    DROPPED_TRAIN_SAMPLES=$((EFFECTIVE_TRAIN_SAMPLES % ROLLOUT_BATCH_SIZE))
    if [ "$ROLLOUT_STEPS_PER_EPOCH" -lt 1 ]; then
        echo "ERROR: multi_step train dataloader would be empty: effective_samples=$EFFECTIVE_TRAIN_SAMPLES rollout_batch_size=$ROLLOUT_BATCH_SIZE drop_last=true" >&2
        exit 1
    fi
    if [ "$SCHEDULER" = "three_step_off" ]; then
        TRAINED_SAMPLES_PER_EPOCH=$((ROLLOUT_STEPS_PER_EPOCH * ROLLOUT_BATCH_SIZE))
        STEPS_PER_EPOCH=$((TRAINED_SAMPLES_PER_EPOCH / ACTOR_UPDATE_BATCH_SIZE))
    else
        STEPS_PER_EPOCH=$ROLLOUT_STEPS_PER_EPOCH
    fi
fi
TOTAL_TRAINING_STEPS=$((STEPS_PER_EPOCH * TOTAL_EPOCHS))

if [ "$SAVE_FREQ_REQUESTED" = "auto" ]; then
    SAVE_FREQ=$(((TOTAL_TRAINING_STEPS + 4) / 5))
    if [ "$SAVE_FREQ" -lt 1 ]; then
        SAVE_FREQ=1
    fi
else
    SAVE_FREQ="$SAVE_FREQ_REQUESTED"
fi
case "$SAVE_FREQ" in
    ""|*[!0-9]*) echo "ERROR: SAVE_FREQ/SAVE_STEPS must be a positive integer or auto, got $SAVE_FREQ_REQUESTED" >&2; exit 1 ;;
esac
if [ "$SAVE_FREQ" -lt 1 ]; then
    echo "ERROR: SAVE_FREQ must be positive when saving OPD Megatron checkpoints, got $SAVE_FREQ" >&2
    exit 1
fi

if is_true "$RUN_EVAL_AFTER_TRAINING" && is_true "$RAY_NO_WAIT"; then
    echo "ERROR: RUN_EVAL_AFTER_TRAINING=true cannot be used with RAY_NO_WAIT=true because evaluation needs the completed checkpoint." >&2
    exit 1
fi

STUDENT_MODEL_ID="$(sanitize_path_component "${STUDENT_MODEL:-$MODEL_PATH}")"
TEACHER_MODEL_ID="$(sanitize_path_component "${TEACHER_MODEL:-$TEACHER_MODEL_PATH}")"
DISTILL_MODE_ID="$(printf '%s' "$DISTILL_MODE" | tr '[:lower:]' '[:upper:]')"
RESOLVED_SCHEDULER="$SCHEDULER"
if [ "$RESOLVED_SCHEDULER" = "auto" ] || [ -z "$RESOLVED_SCHEDULER" ]; then
    if [ "$OPTIMIZATION_MODE" = "one_step" ]; then
        RESOLVED_SCHEDULER="one_step"
    elif [ "$Y_MODE" = "y_r" ]; then
        RESOLVED_SCHEDULER="three_step_off"
    else
        RESOLVED_SCHEDULER="one_step_off"
    fi
fi
if [ "$OPTIMIZATION_MODE" = "multi_step" ]; then
    STEP_TAG="${OPTIMIZATION_MODE}_steps${TOTAL_TRAINING_STEPS}_spe${STEPS_PER_EPOCH}_rbs${ROLLOUT_BATCH_SIZE}_abs${ACTOR_UPDATE_BATCH_SIZE}"
    if [ "$SCHEDULER" = "bounded_lag_y_r" ]; then
        STEP_TAG="${STEP_TAG}_blag${TEACHER_INFLIGHT_SAMPLES}_lag${MAX_POLICY_LAG}"
    fi
else
    STEP_TAG="${OPTIMIZATION_MODE}_steps${TOTAL_TRAINING_STEPS}"
fi
GENERATED_EXPERIMENT_NAME="${STUDENT_MODEL_ID}_${DISTILL_MODE}_${Y_MODE}_${KL_TYPE}_${STEP_TAG}_teacher-${TEACHER_MODEL_ID}_${TEACHER_TRAINING_PROMPT}_${RUN_DATE}"
EXPERIMENT_NAME="${EXPERIMENT_NAME_OVERRIDE:-$GENERATED_EXPERIMENT_NAME}"

DEFAULT_LOCAL_DIR="${OUTPUT_DIR:-${CKPT_BASE_DIR}/${DISTILL_MODE_ID}/${STUDENT_MODEL_ID}/${EXPERIMENT_NAME}}"
if [[ "$DEFAULT_LOCAL_DIR" = /* ]]; then
    DEFAULT_LOCAL_DIR_ABS="$DEFAULT_LOCAL_DIR"
else
    DEFAULT_LOCAL_DIR_ABS="$MEGATRON_DIR/$DEFAULT_LOCAL_DIR"
fi
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-$DEFAULT_LOCAL_DIR_ABS/hf_merged}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$DEFAULT_LOCAL_DIR_ABS/eval/gen_results}"
EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-$VERL_ROOT/results/${DISTILL_MODE_ID}_result.json}"
RUN_METADATA_FILE="${RUN_METADATA_FILE:-$DEFAULT_LOCAL_DIR_ABS/run_metadata.json}"
mkdir -p "$DEFAULT_LOCAL_DIR_ABS"

write_run_metadata() {
    local final_actor_checkpoint="${1:-}"
    RUN_METADATA_FILE="$RUN_METADATA_FILE" \
    MODEL_PATH="$MODEL_PATH" \
    MODEL_NAME="$MODEL_NAME" \
    STUDENT_MODEL_ID="$STUDENT_MODEL_ID" \
    TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
    TEACHER_MODEL_ID="$TEACHER_MODEL_ID" \
    TEACHER_BACKEND="$TEACHER_BACKEND" \
    TEACHER_SERVER_HOST="$TEACHER_SERVER_HOST" \
    TEACHER_SERVER_PORT="$TEACHER_SERVER_PORT" \
    TEACHER_N_SERVER_WORKERS="$TEACHER_N_SERVER_WORKERS" \
    TEACHER_CLIENT_TIMEOUT_MS="$TEACHER_CLIENT_TIMEOUT_MS" \
    TEACHER_REQUEST_BATCH_SIZE="$TEACHER_REQUEST_BATCH_SIZE" \
    DISTILL_MODE="$DISTILL_MODE" \
    DISTILL_MODE_ID="$DISTILL_MODE_ID" \
    Y_MODE="$Y_MODE" \
    KL_TYPE="$KL_TYPE" \
    KL_METHOD="$KL_METHOD" \
    KL_TOKEN_CLIP="$KL_TOKEN_CLIP" \
    TOP_K="$TOP_K" \
    BETA="$BETA" \
    TEMPERATURE="$TEMPERATURE" \
    TEACHER_TRAINING_PROMPT="$TEACHER_TRAINING_PROMPT" \
    OPTIMIZATION_MODE="$OPTIMIZATION_MODE" \
    SCHEDULER="$SCHEDULER" \
    RESOLVED_SCHEDULER="$RESOLVED_SCHEDULER" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
    ACTOR_UPDATE_BATCH_SIZE="$ACTOR_UPDATE_BATCH_SIZE" \
    TEACHER_INFLIGHT_SAMPLES="$TEACHER_INFLIGHT_SAMPLES" \
    MAX_POLICY_LAG="$MAX_POLICY_LAG" \
    EFFECTIVE_ACTOR_BATCH_SIZE="$EFFECTIVE_ACTOR_BATCH_SIZE" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    USE_DYNAMIC_BSZ="$USE_DYNAMIC_BSZ" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    TEACHER_MAX_PROMPT_LENGTH="$TEACHER_MAX_PROMPT_LENGTH" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    TEACHER_SEQ_LEN="$TEACHER_SEQ_LEN" \
    STUDENT_ACTOR_MAX_TOKENS_PER_GPU="$STUDENT_ACTOR_MAX_TOKENS_PER_GPU" \
    TOTAL_EPOCHS="$TOTAL_EPOCHS" \
    DATA_PATH="$DATA_PATH" \
    DATASET_NUM_ROWS="$DATASET_NUM_ROWS" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    EFFECTIVE_TRAIN_SAMPLES="$EFFECTIVE_TRAIN_SAMPLES" \
    STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
    TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
    DROPPED_TRAIN_SAMPLES="$DROPPED_TRAIN_SAMPLES" \
    SAVE_FREQ="$SAVE_FREQ" \
    RUN_DATE="$RUN_DATE" \
    PROJECT_NAME="$PROJECT_NAME" \
    EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    DEFAULT_LOCAL_DIR_ABS="$DEFAULT_LOCAL_DIR_ABS" \
    MERGED_MODEL_DIR="$MERGED_MODEL_DIR" \
    EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
    EVAL_RESULTS_FILE="$EVAL_RESULTS_FILE" \
    PASS_K="$PASS_K" \
    FINAL_ACTOR_CHECKPOINT="$final_actor_checkpoint" \
    "$PYTHON_BIN" - <<'PY'
import json
import os

def env_int(name):
    value = os.environ.get(name)
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return value

def env_float_or_str(name):
    value = os.environ.get(name)
    if value in (None, ""):
        return value
    try:
        return float(value)
    except ValueError:
        return value

def env_bool(name):
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "y"}

metadata = {
    "date": os.environ["RUN_DATE"],
    "project_name": os.environ["PROJECT_NAME"],
    "experiment_name": os.environ["EXPERIMENT_NAME"],
    "student_model": {
        "path": os.environ["MODEL_PATH"],
        "name": os.environ["MODEL_NAME"],
        "id": os.environ["STUDENT_MODEL_ID"],
    },
    "teacher_model": {
        "path": os.environ["TEACHER_MODEL_PATH"],
        "id": os.environ["TEACHER_MODEL_ID"],
        "backend": os.environ["TEACHER_BACKEND"],
        "local_chunk_size": env_int("TEACHER_LOCAL_CHUNK_SIZE"),
        "server_host": os.environ["TEACHER_SERVER_HOST"],
        "server_port": env_int("TEACHER_SERVER_PORT"),
        "n_server_workers": env_int("TEACHER_N_SERVER_WORKERS"),
        "client_timeout_ms": env_int("TEACHER_CLIENT_TIMEOUT_MS"),
        "request_batch_size": env_int("TEACHER_REQUEST_BATCH_SIZE"),
    },
    "opd": {
        "distill_mode": os.environ["DISTILL_MODE"],
        "distill_mode_id": os.environ["DISTILL_MODE_ID"],
        "y_mode": os.environ["Y_MODE"],
        "teacher_training_prompt": os.environ["TEACHER_TRAINING_PROMPT"],
        "kl_type": os.environ["KL_TYPE"],
        "kl_method": os.environ["KL_METHOD"],
        "kl_token_clip": env_float_or_str("KL_TOKEN_CLIP"),
        "top_k": env_int("TOP_K"),
        "beta": env_float_or_str("BETA"),
        "temperature": env_float_or_str("TEMPERATURE"),
    },
    "optimization": {
        "mode": os.environ["OPTIMIZATION_MODE"],
        "scheduler": os.environ["SCHEDULER"],
        "resolved_scheduler": os.environ["RESOLVED_SCHEDULER"],
        "train_batch_size": env_int("TRAIN_BATCH_SIZE"),
        "rollout_batch_size": env_int("ROLLOUT_BATCH_SIZE"),
        "actor_update_batch_size": env_int("ACTOR_UPDATE_BATCH_SIZE"),
        "teacher_inflight_samples": env_int("TEACHER_INFLIGHT_SAMPLES"),
        "max_policy_lag": env_int("MAX_POLICY_LAG"),
        "effective_actor_batch_size": env_int("EFFECTIVE_ACTOR_BATCH_SIZE"),
        "micro_batch_size": env_int("MICRO_BATCH_SIZE"),
        "use_dynamic_bsz": env_bool("USE_DYNAMIC_BSZ"),
        "max_prompt_length": env_int("MAX_PROMPT_LENGTH"),
        "teacher_max_prompt_length": env_int("TEACHER_MAX_PROMPT_LENGTH"),
        "max_response_length": env_int("MAX_RESPONSE_LENGTH"),
        "teacher_seq_len": env_int("TEACHER_SEQ_LEN"),
        "student_actor_max_tokens_per_gpu": env_int("STUDENT_ACTOR_MAX_TOKENS_PER_GPU"),
        "total_epochs": env_int("TOTAL_EPOCHS"),
        "dataset_rows": env_int("DATASET_NUM_ROWS"),
        "max_samples": env_int("MAX_SAMPLES"),
        "effective_train_samples": env_int("EFFECTIVE_TRAIN_SAMPLES"),
        "steps_per_epoch": env_int("STEPS_PER_EPOCH"),
        "total_training_steps": env_int("TOTAL_TRAINING_STEPS"),
        "dropped_train_samples": env_int("DROPPED_TRAIN_SAMPLES"),
        "save_freq": env_int("SAVE_FREQ"),
    },
    "paths": {
        "data_path": os.environ["DATA_PATH"],
        "checkpoint_dir": os.environ["DEFAULT_LOCAL_DIR_ABS"],
        "merged_model_dir": os.environ["MERGED_MODEL_DIR"],
        "eval_output_dir": os.environ["EVAL_OUTPUT_DIR"],
        "eval_results_file": os.environ["EVAL_RESULTS_FILE"],
        "final_actor_checkpoint": os.environ.get("FINAL_ACTOR_CHECKPOINT") or None,
    },
    "evaluation": {
        "pass_k": env_int("PASS_K"),
    },
}

path = os.environ["RUN_METADATA_FILE"]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
PY
}

write_run_metadata

resolve_latest_actor_checkpoint() {
    local ckpt_root="$1"
    local latest_file="$ckpt_root/latest_checkpointed_iteration.txt"
    if [ -f "$latest_file" ]; then
        local latest_step
        latest_step="$(tr -d '[:space:]' < "$latest_file")"
        if [ -n "$latest_step" ] && [ -d "$ckpt_root/global_step_${latest_step}/actor" ]; then
            echo "$ckpt_root/global_step_${latest_step}/actor"
            return 0
        fi
    fi

    local best_step=-1
    local best_dir=""
    local step_dir step_name step
    for step_dir in "$ckpt_root"/global_step_*; do
        [ -d "$step_dir/actor" ] || continue
        step_name="${step_dir##*/}"
        step="${step_name#global_step_}"
        case "$step" in
            ""|*[!0-9]*) continue ;;
        esac
        if [ "$step" -gt "$best_step" ]; then
            best_step="$step"
            best_dir="$step_dir/actor"
        fi
    done

    if [ -n "$best_dir" ]; then
        echo "$best_dir"
        return 0
    fi
    return 1
}

hf_model_dir_ready() {
    local model_dir="$1"
    [ -d "$model_dir" ] || return 1
    [ -f "$model_dir/config.json" ] || return 1
    find "$model_dir" -maxdepth 1 \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .
}

merge_megatron_checkpoint() {
    local actor_ckpt="$1"
    local target_dir="$2"

    if hf_model_dir_ready "$target_dir"; then
        echo "HuggingFace model already exists at $target_dir; skipping merge."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Merging Megatron checkpoint to HuggingFace"
    echo "  Checkpoint: $actor_ckpt"
    echo "  Target:     $target_dir"
    echo "=========================================="

    local merge_cmd=(
        "$PYTHON_BIN" -m verl.model_merger merge
        --backend megatron
        --local_dir "$actor_ckpt"
        --target_dir "$target_dir"
        --trust-remote-code
    )
    if [ -n "$MERGE_HF_MODEL_CONFIG_PATH" ]; then
        merge_cmd+=(--hf_model_config_path "$MERGE_HF_MODEL_CONFIG_PATH")
    fi
    if is_true "$MERGE_TIE_WORD_EMBEDDING"; then
        merge_cmd+=(--tie-word-embedding)
    fi

    (cd "$VERL_ROOT" && "${merge_cmd[@]}")
    if ! hf_model_dir_ready "$target_dir"; then
        echo "ERROR: merged HuggingFace model is missing or incomplete at $target_dir" >&2
        exit 1
    fi
}

eval_dataset_rel_path() {
    case "$1" in
        aime24) echo "aime24/aime24_test.parquet" ;;
        aime25) echo "aime25/aime25_test.parquet" ;;
        math500) echo "math500/math500_test.parquet" ;;
        hmmt24) echo "hmmt24/hmmt24_test.parquet" ;;
        hmmt25) echo "hmmt25/hmmt25_test.parquet" ;;
        amc23) echo "amc23/amc23_test.parquet" ;;
        beyondaime|bytedance-seed/beyondaime) echo "beyondaime/beyondaime_test.parquet" ;;
        amobench|amo-bench|meituan-longcat/amo-bench) echo "amobench/amobench_test.parquet" ;;
        gsm8k|openai/gsm8k) echo "gsm8k/gsm8k_test.parquet" ;;
        *) return 1 ;;
    esac
}

check_eval_datasets() {
    local datasets_space
    datasets_space="$(echo "$EVAL_DATASETS" | tr ',' ' ')"
    local missing=""
    local ds rel_path
    for ds in $datasets_space; do
        if ! rel_path="$(eval_dataset_rel_path "$ds")"; then
            echo "ERROR: unsupported eval dataset: $ds" >&2
            exit 1
        fi
        if [ ! -s "$EVAL_DATASETS_DIR/$rel_path" ]; then
            missing="$missing $ds"
        fi
    done
    if [ -n "$missing" ]; then
        echo "ERROR: missing eval dataset parquet under $EVAL_DATASETS_DIR:$missing" >&2
        echo "       Expected layout: \$EVAL_DATASETS_DIR/<name>/<name>_test.parquet" >&2
        exit 1
    fi
}

wait_for_gpu_release() {
    local wait_max="${EVAL_GPU_RELEASE_WAIT_SECONDS:-60}"
    local waited=0
    local threshold_mb=$((EVAL_N_GPUS_PER_NODE * 1024))
    local visible_gpus="${CUDA_VISIBLE_DEVICES:-}"
    local nvidia_smi_args=()
    if [ -n "$visible_gpus" ] && [ "$visible_gpus" != "NoDevFiles" ]; then
        nvidia_smi_args+=(--id="$visible_gpus")
    fi
    while [ "$waited" -lt "$wait_max" ]; do
        local used_mb
        used_mb=$(nvidia-smi "${nvidia_smi_args[@]}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
        if [ "$used_mb" -lt "$threshold_mb" ]; then
            echo "GPUs released for eval (total ${used_mb} MB used, threshold ${threshold_mb} MB)."
            return 0
        fi
        echo "GPUs still busy before eval (total ${used_mb} MB used); waiting..."
        sleep 5
        waited=$((waited + 5))
    done
    echo "WARNING: GPU memory still high after ${wait_max}s; evaluation may OOM." >&2
}

run_math_evaluation() {
    local eval_model_path="$1"
    local datasets_space
    datasets_space="$(echo "$EVAL_DATASETS" | tr ',' ' ')"

    check_eval_datasets
    wait_for_gpu_release

    echo ""
    echo "=========================================="
    echo "Running async math evaluation"
    echo "  Model:    $eval_model_path"
    echo "  Datasets: $datasets_space"
    echo "  pass_k:   $PASS_K"
    echo "  Results:  $EVAL_RESULTS_FILE"
    echo "  Outputs:  $EVAL_OUTPUT_DIR"
    echo "=========================================="

    (
        cd "$VERL_ROOT"
        unset PYTORCH_CUDA_ALLOC_CONF
        PYTHON_BIN="$PYTHON_BIN" \
        NGPUS_PER_NODE="$EVAL_N_GPUS_PER_NODE" \
        NNODES="$EVAL_NNODES" \
        GEN_TP="$EVAL_GEN_TP" \
        EVAL_MODEL_NAME="$EXPERIMENT_NAME" \
        EVAL_BASE_MODEL_NAME="$STUDENT_MODEL_ID" \
        EVAL_METADATA_FILE="$RUN_METADATA_FILE" \
        EVAL_DATASETS_DIR="$EVAL_DATASETS_DIR" \
        DATASETS="$datasets_space" \
        PASS_K="$PASS_K" \
        EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
        EVAL_RESULTS_FILE="$EVAL_RESULTS_FILE" \
        bash "$VERL_ROOT/recipe/math_evaluation/benchmark_kl_model.sh" "$eval_model_path"
    )
}

export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
if ! is_true "${KEEP_NVTE_ATTENTION_FLAGS:-false}"; then
    export NVTE_FLASH_ATTN=1
    export NVTE_FUSED_ATTN=1
    export NVTE_UNFUSED_ATTN=1
fi
export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"

RUNTIME_ENV="${RUNTIME_ENV:-$MEGATRON_DIR/config/runtime_env.yaml}"
NO_WAIT_ARGS=()
if is_true "$RAY_NO_WAIT"; then
    NO_WAIT_ARGS+=(--no-wait)
fi
ROLLOUT_OPTIONAL_ARGS=()
if [ -n "$ROLLOUT_MAX_MODEL_LEN" ]; then
    ROLLOUT_OPTIONAL_ARGS+=(actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN")
fi
if [ -n "$ROLLOUT_MAX_NUM_SEQS" ]; then
    ROLLOUT_OPTIONAL_ARGS+=(actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS")
fi

echo "=========================================="
echo "Megatron OPD configuration"
echo "  model:       $MODEL_PATH"
echo "  teacher:     $TEACHER_MODEL_PATH backend=$TEACHER_BACKEND"
echo "  teacher rpc: ${TEACHER_SERVER_HOST}:${TEACHER_SERVER_PORT} shards=$TEACHER_N_SERVER_WORKERS request_batch_size=${TEACHER_REQUEST_BATCH_SIZE:-auto} client_timeout_ms=$TEACHER_CLIENT_TIMEOUT_MS"
echo "  data:        $DATA_PATH (raw prompts; y_o/y_r generated online)"
echo "  data keys:   prompt_key=$PROMPT_KEY default_data_source=$DATA_SOURCE"
echo "  mode:        distill=$DISTILL_MODE y=$Y_MODE kl=$KL_TYPE prompt=$TEACHER_TRAINING_PROMPT"
echo "  math prompt: append_instruction=$APPEND_INSTRUCTION_TO_PROMPT"
echo "  schedule:    optimization_mode=$OPTIMIZATION_MODE scheduler=$SCHEDULER"
if [ "$SCHEDULER" = "bounded_lag_y_r" ]; then
    echo "  bounded lag: teacher_inflight_samples=$TEACHER_INFLIGHT_SAMPLES max_policy_lag=$MAX_POLICY_LAG"
fi
if is_true "$USE_DYNAMIC_BSZ"; then
    echo "  batch:       rollout=$ROLLOUT_BATCH_SIZE actor_update=$ACTOR_UPDATE_BATCH_SIZE dynamic_bsz=True max_samples=$MAX_SAMPLES effective_samples=$EFFECTIVE_TRAIN_SAMPLES"
else
    echo "  batch:       rollout=$ROLLOUT_BATCH_SIZE actor_update=$ACTOR_UPDATE_BATCH_SIZE micro=$MICRO_BATCH_SIZE max_samples=$MAX_SAMPLES effective_samples=$EFFECTIVE_TRAIN_SAMPLES"
fi
echo "  steps:       steps_per_epoch=$STEPS_PER_EPOCH total_steps=$TOTAL_TRAINING_STEPS save_freq=$SAVE_FREQ"
echo "  optim:       lr=$LEARNING_RATE warmup=$WARMUP_RATIO weight_decay=$WEIGHT_DECAY min_lr=$MIN_LR grad_accum_compat=$GRADIENT_ACCUMULATION_STEPS effective_actor_batch=$EFFECTIVE_ACTOR_BATCH_SIZE"
echo "  lengths:     student_max_prompt=$MAX_PROMPT_LENGTH teacher_max_prompt=$TEACHER_MAX_PROMPT_LENGTH teacher_seq_len=$TEACHER_SEQ_LEN max_response=$MAX_RESPONSE_LENGTH student_actor_max_tokens_per_gpu=$STUDENT_ACTOR_MAX_TOKENS_PER_GPU"
echo "  resources:   actor_rollout_gpus=$N_GPUS_PER_NODE nnodes=$NNODES infer_tp=$INFER_TP agent_workers=$AGENT_NUM_WORKERS rollout_max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-null} rollout_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL rollout_prefix_cache=$ROLLOUT_ENABLE_PREFIX_CACHING rollout_enforce_eager=$ROLLOUT_ENFORCE_EAGER rollout_free_cache=$ROLLOUT_FREE_CACHE_ENGINE rollout_sleep_mode=$ROLLOUT_ENABLE_SLEEP_MODE"
echo "  weight sync: VERL_VLLM_USE_SHM_WEIGHT_SYNC=${VERL_VLLM_USE_SHM_WEIGHT_SYNC:-0}"
echo "  checkpoint:  save_freq=$SAVE_FREQ dir=$DEFAULT_LOCAL_DIR_ABS"
echo "  metadata:    $RUN_METADATA_FILE"
if is_true "$SAVE_MERGED_MODEL" || is_true "$RUN_EVAL_AFTER_TRAINING"; then
    echo "  merged hf:   $MERGED_MODEL_DIR"
fi
if is_true "$RUN_EVAL_AFTER_TRAINING"; then
    echo "  eval:        datasets=$EVAL_DATASETS pass_k=$PASS_K results=$EVAL_RESULTS_FILE"
fi
echo "=========================================="

ray job submit "${NO_WAIT_ARGS[@]}" --runtime-env="$RUNTIME_ENV" \
    --working-dir "$MEGATRON_DIR" \
    -- /usr/bin/env \
    ROLLOUT_GPUS_PER_NODE="$ROLLOUT_GPUS_PER_NODE" \
    TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-$N_GPUS_PER_NODE}" \
    N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
    NGPUS_PER_NODE="${NGPUS_PER_NODE:-$N_GPUS_PER_NODE}" \
    VERL_VLLM_USE_SHM_WEIGHT_SYNC="${VERL_VLLM_USE_SHM_WEIGHT_SYNC:-0}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
    "$PYTHON_BIN" -m main_gkd --config-name on_policy_distill_trainer \
    data.train_files="$DATA_PATH" \
    data.prompt_key="$PROMPT_KEY" \
    data.default_data_source="$DATA_SOURCE" \
    data.train_batch_size="$ROLLOUT_BATCH_SIZE" \
    data.max_samples="$MAX_SAMPLES" \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.hybrid_engine="${HYBRID_ENGINE:-True}" \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$WARMUP_RATIO" \
    actor_rollout_ref.actor.optim.weight_decay="$WEIGHT_DECAY" \
    actor_rollout_ref.actor.optim.min_lr="$MIN_LR" \
    actor_rollout_ref.actor.micro_batch_size="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
    actor_rollout_ref.actor.max_token_len="$STUDENT_ACTOR_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE" \
    actor_rollout_ref.actor.megatron.sequence_parallel="$SP" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="$PP" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$TP" \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="$EP" \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="$ETP" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.99}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K:--1}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$INFER_TP" \
    actor_rollout_ref.rollout.load_format="$LOAD_FORMAT" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.enable_chunked_prefill="$ROLLOUT_ENABLE_CHUNKED_PREFILL" \
    actor_rollout_ref.rollout.enable_prefix_caching="$ROLLOUT_ENABLE_PREFIX_CACHING" \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER:-False}" \
    actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.enable_sleep_mode="$ROLLOUT_ENABLE_SLEEP_MODE" \
    "${ROLLOUT_OPTIONAL_ARGS[@]}" \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-2048}" \
    actor_rollout_ref.teacher.server_ip="$TEACHER_SERVER_HOST" \
    actor_rollout_ref.teacher.server_port="$TEACHER_SERVER_PORT" \
    actor_rollout_ref.teacher.backend="$TEACHER_BACKEND" \
    actor_rollout_ref.teacher.model_path="$TEACHER_MODEL_PATH" \
    actor_rollout_ref.teacher.n_server_workers="$TEACHER_N_SERVER_WORKERS" \
    actor_rollout_ref.teacher.client_timeout_ms="$TEACHER_CLIENT_TIMEOUT_MS" \
    actor_rollout_ref.teacher.request_batch_size="$TEACHER_REQUEST_BATCH_SIZE" \
    actor_rollout_ref.teacher.local_chunk_size="${TEACHER_LOCAL_CHUNK_SIZE:-128}" \
    actor_rollout_ref.teacher.local_prefill_chunk_size="${TEACHER_LOCAL_PREFILL_CHUNK_SIZE:-512}" \
    actor_rollout_ref.teacher.local_attn_implementation="${TEACHER_LOCAL_ATTN_IMPLEMENTATION:-flash_attention_2}" \
    actor_rollout_ref.teacher.temperature="$TEMPERATURE" \
    opd.y_mode="$Y_MODE" \
    opd.distill_mode="$DISTILL_MODE" \
    opd.teacher_training_prompt="$TEACHER_TRAINING_PROMPT" \
    opd.teacher_max_prompt_length="$TEACHER_MAX_PROMPT_LENGTH" \
    opd.append_instruction_to_prompt="$APPEND_INSTRUCTION_TO_PROMPT" \
    opd.kl_type="$KL_TYPE" \
    opd.kl_method="$KL_METHOD" \
    opd.kl_token_clip="$KL_TOKEN_CLIP" \
    opd.beta="$BETA" \
    opd.top_k="$TOP_K" \
    opd.temperature="$TEMPERATURE" \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.nnodes="$NNODES" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node="$ROLLOUT_GPUS_PER_NODE" \
    trainer.scheduler="$SCHEDULER" \
    trainer.optimization_mode="$OPTIMIZATION_MODE" \
    trainer.actor_update_batch_size="$ACTOR_UPDATE_BATCH_SIZE" \
    trainer.teacher_inflight_samples="$TEACHER_INFLIGHT_SAMPLES" \
    trainer.max_policy_lag="$MAX_POLICY_LAG" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    "$@"

TRAINED_ACTOR_CKPT=""
if TRAINED_ACTOR_CKPT="$(resolve_latest_actor_checkpoint "$DEFAULT_LOCAL_DIR_ABS")"; then
    write_run_metadata "$TRAINED_ACTOR_CKPT"
fi
if is_true "$SAVE_MERGED_MODEL" || { is_true "$RUN_EVAL_AFTER_TRAINING" && [ -z "$EVAL_MODEL_PATH" ]; }; then
    if [ -z "$TRAINED_ACTOR_CKPT" ]; then
        echo "ERROR: no saved Megatron actor checkpoint found under $DEFAULT_LOCAL_DIR_ABS" >&2
        echo "       Check trainer.save_freq/SAVE_FREQ and the training log." >&2
        exit 1
    fi
    merge_megatron_checkpoint "$TRAINED_ACTOR_CKPT" "$MERGED_MODEL_DIR"
fi

if is_true "$RUN_EVAL_AFTER_TRAINING"; then
    FINAL_EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-$MERGED_MODEL_DIR}"
    if ! hf_model_dir_ready "$FINAL_EVAL_MODEL_PATH"; then
        echo "ERROR: eval requested but HuggingFace model is missing or incomplete at $FINAL_EVAL_MODEL_PATH" >&2
        exit 1
    fi
    run_math_evaluation "$FINAL_EVAL_MODEL_PATH"
fi
