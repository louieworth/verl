#!/bin/bash
# =============================================================================
# KL Divergence Training for Math Reasoning
# =============================================================================
#
# This script runs token-level KL divergence training with 4 variants:
#   1. Reverse KL + Monte Carlo
#   2. Reverse KL + Full Vocabulary
#   3. Forward KL + Monte Carlo
#   4. Forward KL + Full Vocabulary
#
# Multi-epoch behavior:
#   - TOTAL_EPOCHS controls the outer pipeline epochs
#   - Each epoch generates its own stage1/stage2 data under a run-qualified
#     gen_results directory to avoid mixing y_o/y_r, KL variants, teachers, etc.
#   - Each epoch trains from the current model and saves to an epoch-specific checkpoint dir
#   - Epoch N+1 loads the model saved by epoch N
# =============================================================================

set -e
set -o pipefail

RUN_KL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_KL_SCRIPT_DIR/hf_export_validation.sh"

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

# `expandable_segments:True` avoids a ~6GiB reserved-but-unallocated block
# during FSDP backward, but it is incompatible with vLLM's CuMemAllocator
# memory pool, so we only set it around the torchrun training command below
# (not exported here, otherwise stage1/stage2 generation and eval — which
# all spawn vLLM — crash with "Expandable segments are not compatible with
# memory pool").

# Disable OpenTelemetry exporters in Ray/vLLM worker processes. The OTLP gRPC
# metric exporter can segfault inside grpc/opentelemetry-cpp during long rollout
# jobs; metrics/traces are not needed for this pipeline.
export OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}"
export OTEL_METRICS_EXPORTER="${OTEL_METRICS_EXPORTER:-none}"
export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-none}"
export OTEL_LOGS_EXPORTER="${OTEL_LOGS_EXPORTER:-none}"
export VERL_FORCE_BASE_COMPLETION="${VERL_FORCE_BASE_COMPLETION:-true}"
PYTHON_BIN=${PYTHON_BIN:-python3}

# =============================================================================
# Configuration
# =============================================================================
# Task / Distillation Mode
TASK=${TASK:-"math"}  # math | code
case "$TASK" in
    math|code) ;;
    *) echo "ERROR: TASK must be math or code (got: $TASK)" >&2; exit 1 ;;
esac

DISTILL_MODE=${DISTILL_MODE:-"opsd"}   # OPD: external Qwen3-14B teacher, no y*.
                                        # OPSD: frozen step-0 Base self-teacher with y*.
case "$DISTILL_MODE" in
    opsd|opd) ;;
    *) echo "ERROR: DISTILL_MODE must be opsd or opd (got: $DISTILL_MODE)" >&2; exit 1 ;;
esac
TEACHER_ENABLE_THINKING=${TEACHER_ENABLE_THINKING:-false}
case "$TEACHER_ENABLE_THINKING" in
    true|false) ;;
    *) echo "ERROR: TEACHER_ENABLE_THINKING must be true or false (got: $TEACHER_ENABLE_THINKING)" >&2; exit 1 ;;
esac
if [ "$DISTILL_MODE" = "opsd" ] && [ "$TEACHER_ENABLE_THINKING" = "true" ]; then
    echo "ERROR: OPSD uses a frozen Base self-teacher and cannot enable teacher thinking mode" >&2
    exit 1
fi
if [ "$DISTILL_MODE" = "opd" ]; then
    if [ "$TEACHER_ENABLE_THINKING" = "true" ]; then
        TEACHER_SUPERVISION_RENDER_MODE=qwen3_chat_thinking
        TEACHER_ROLLOUT_RENDER_MODE=qwen3_chat_thinking
        TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER=${TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER:-16}
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
if [ "$DISTILL_MODE" = "opd" ]; then
    TEACHER_THINKING_NAME_TAG="teacher-thinking-${TEACHER_ENABLE_THINKING}"
    TEACHER_THINKING_FILE_SUFFIX="_${TEACHER_THINKING_NAME_TAG}"
    TEACHER_THINKING_RUN_SEGMENT="${TEACHER_THINKING_NAME_TAG}_"
else
    TEACHER_THINKING_NAME_TAG=""
    TEACHER_THINKING_FILE_SUFFIX=""
    TEACHER_THINKING_RUN_SEGMENT=""
fi
TEACHER_PROMPT_RENDER_CONTRACT="supervision-${TEACHER_SUPERVISION_RENDER_MODE}_rollout-${TEACHER_ROLLOUT_RENDER_MODE}_v1"
if ! [[ "$TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER must be a non-negative integer" >&2
    exit 1
fi
export TEACHER_ENABLE_THINKING TEACHER_THINKING_NAME_TAG
export TEACHER_SUPERVISION_RENDER_MODE TEACHER_ROLLOUT_RENDER_MODE
export TEACHER_PROMPT_RENDER_CONTRACT TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER

run_teacher_generation() {
    if [ "$DISTILL_MODE" = "opd" ] && [ "$TEACHER_ENABLE_THINKING" = "true" ]; then
        env -u PYTORCH_CUDA_ALLOC_CONF \
            VERL_FORCE_BASE_COMPLETION=false \
            VERL_ENABLE_THINKING=true \
            "$@"
    else
        env -u PYTORCH_CUDA_ALLOC_CONF -u VERL_ENABLE_THINKING \
            VERL_FORCE_BASE_COMPLETION=true \
            "$@"
    fi
}

# KL Training Settings
KL_TYPE=${KL_TYPE:-"reverse"}          # reverse | forward | jsd (OPSD generalized JSD)
KL_METHOD=${KL_METHOD:-"full_vocab"}   # monte_carlo or full_vocab (JSD at beta∈(0,1) requires full_vocab)
KL_TOKEN_CLIP=${KL_TOKEN_CLIP:-0.0}     # Per-token KL/JSD clip; 0 disables.
TEMPERATURE=${TEMPERATURE:-1.0}         # KL softmax temperature
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
BETA=${BETA:-0}                         # JSD mixture coefficient (0=forward KL, 1=reverse KL, 0<β<1=mixture)
Y_MODE=${Y_MODE:-"y_o"}    # y_o (formerly y_raw): train on stage1 student rollouts, teacher prompt = π(·|x, y*).
                            # y_r (formerly y_cor): train on stage2 teacher rewrites, teacher prompt = π(·|x, y_o, y*).
                            # Drives both data routing AND training-time teacher prompt construction.
                            # Legacy aliases y_raw / y_cor are accepted but emit a deprecation warning.
case "$Y_MODE" in
    y_raw)
        echo "WARNING: Y_MODE=y_raw is deprecated; use Y_MODE=y_o (alias accepted for backward compat)." >&2
        Y_MODE="y_o" ;;
    y_cor)
        echo "WARNING: Y_MODE=y_cor is deprecated; use Y_MODE=y_r (alias accepted for backward compat)." >&2
        Y_MODE="y_r" ;;
esac
PROMPT_TRUNCATION=${PROMPT_TRUNCATION:-"false"}  # Preserve the exact rollout prefix; prepared prompts already fit the native context budget.
FORWARD_STAGE2_MODE=${FORWARD_STAGE2_MODE:-"rewrite_all"}  # y_o only: rewrite_all | stage1_reward_0_only | stage1_reward_1_only (filter stage1 parquet by reward)
FORWARD_FILTER_STAGE2=${FORWARD_FILTER_STAGE2:-"false"}    # y_r only: true=score y_r after stage2 gen and keep reward>=threshold
FORWARD_FILTER_THRESHOLD=${FORWARD_FILTER_THRESHOLD:-1.0}  # Minimum stage2_reward to keep (1.0 = correct)
FORWARD_FILTER_REQUIRE_STAGE1_FAILED=${FORWARD_FILTER_REQUIRE_STAGE1_FAILED:-"false"}  # true: also require extra_info.reward==0 (lets you reuse rewrite_all parquet as reward0_only+filter)

# Diagnostic metrics (T1–T4 from the y vs y' study).
# T2 is FSDP-shard aware and cheap; T3 only fires for KL_METHOD=full_vocab;
# T4 requires score_stage1_reward.py to have populated extra_info.reward.
GRAD_COSINE_INTERVAL=${GRAD_COSINE_INTERVAL:-0}                # T2: optimizer steps between cosine evals (0=off)
CORRECTION_TOKEN_PHRASES=${CORRECTION_TOKEN_PHRASES:-""}        # T3: comma-separated correction phrases (e.g. "Wait,But,Actually,However,Hmm")
CORRECTION_TOKEN_IDS=${CORRECTION_TOKEN_IDS:-""}                # T3: comma-separated explicit ids; overrides phrases
LOG_DIFFICULTY_BUCKETS=${LOG_DIFFICULTY_BUCKETS:-"false"}        # T4: split kl_loss by stage1 reward bucket
SCORE_STAGE1=${SCORE_STAGE1:-"auto"}                      # auto/true: score only when a downstream feature needs stage1 reward; force=always

should_score_stage1() {
    case "$SCORE_STAGE1" in
        force|FORCE|always|ALWAYS) return 0 ;;
        false|0|no|NO|off|OFF) return 1 ;;
        true|1|yes|YES|on|ON|auto|AUTO) ;;
    esac
    if [ "$LOG_DIFFICULTY_BUCKETS" = "true" ]; then
        return 0
    fi
    if [ "$FORWARD_FILTER_REQUIRE_STAGE1_FAILED" = "true" ]; then
        return 0
    fi
    if [ "$Y_MODE" = "y_o" ]; then
        case "$FORWARD_STAGE2_MODE" in
            stage1_reward_0_only|stage1_reward_1_only) return 0 ;;
        esac
    fi
    return 1
}

# Top-K teacher local support matching (Fu et al. 2026, arXiv:2603.25562).
# When > 0 (and KL_METHOD=full_vocab), the per-position KL is replaced by a
# truncated KL on the top-K teacher-supported tokens, with both teacher and
# student renormalized inside the support. Paper default K=32. JSD is not
# supported with TOP_K because the mixture only makes sense over full vocab.
TOP_K=${TOP_K:-0}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B-Base"}
if [ -z "${TEACHER_MODEL_PATH:-}" ]; then
    if [ "${DISTILL_MODE:-opsd}" = "opd" ]; then
        TEACHER_MODEL_PATH="Qwen/Qwen3-14B"
    else
        TEACHER_MODEL_PATH="$MODEL_PATH"
    fi
fi

# Enforce the model-family contract at the bottom-level runner as well as in
# the canonical launchers. For OPSD, MODEL_PATH remains the immutable step-0
# identity even when later policy chunks load an updated checkpoint internally.
model_identity() {
    local path="$1"
    if [ -d "$path" ]; then
        (cd "$path" && pwd -P)
    else
        printf '%s\n' "$path"
    fi
}
if [ "$DISTILL_MODE" = "opsd" ]; then
    if [ "$(model_identity "$TEACHER_MODEL_PATH")" != "$(model_identity "$MODEL_PATH")" ]; then
        echo "ERROR: OPSD is self-distillation: TEACHER_MODEL_PATH must be the frozen step-0 MODEL_PATH." >&2
        exit 1
    fi
elif [ "$TEACHER_MODEL_PATH" != "Qwen/Qwen3-14B" ] && [ ! -d "$TEACHER_MODEL_PATH" ]; then
    echo "ERROR: OPD teacher must be Qwen/Qwen3-14B or a local snapshot of that checkpoint." >&2
    exit 1
fi
USE_LORA=${USE_LORA:-true}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}

# Training Settings
LEARNING_RATE=${LEARNING_RATE:-1e-6}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-false}
USER_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-64}
KL_FULL_VOCAB_CHUNK_SIZE=${KL_FULL_VOCAB_CHUNK_SIZE:-512}
GRADIENT_ACCUMULATION_SOURCE="default"
[ -n "$USER_GRADIENT_ACCUMULATION_STEPS" ] && GRADIENT_ACCUMULATION_SOURCE="user"
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}  # Outer pipeline epochs
TRAIN_EPOCHS_PER_ROUND=${TRAIN_EPOCHS_PER_ROUND:-1}  # Trainer epochs for each pipeline epoch
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.005}
PROMPT_CONTRACT_VERSION=${PROMPT_CONTRACT_VERSION:-plain_base_completion_v3}

# Data Settings
DATA_PATH=${DATA_PATH:-""}  # Optional override; reused across epochs if set
CORRECTED_RESPONSES_PATH=${CORRECTED_RESPONSES_PATH:-""}  # Optional legacy two-file mode
PRECOMPUTED_Y_O_TRAJECTORY_PATH=${PRECOMPUTED_Y_O_TRAJECTORY_PATH:-""}
PRECOMPUTED_STAGE1_PROMPTS_PATH=${PRECOMPUTED_STAGE1_PROMPTS_PATH:-""}
MAX_SAMPLES=${MAX_SAMPLES:-""}  # For testing, leave empty for full data
# Offline on-policy optimization. MULTI_STEP is the requested total number of
# policy optimizer steps. 0 selects the default 512 prompts/step; >0 partitions
# the full dataset into exactly that many balanced rollout/training updates.
MULTI_STEP=${MULTI_STEP:-0}
USER_MULTI_STEP="$MULTI_STEP"
USER_PIPELINE_BATCH_SIZE="${PIPELINE_BATCH_SIZE:-}"
DEFAULT_PIPELINE_BATCH_SIZE=${DEFAULT_PIPELINE_BATCH_SIZE:-512}
PIPELINE_STEP_MODE="default_batch"
PIPELINE_BALANCED_REMAINDER=0
PIPELINE_BALANCED_INCREMENT=1
PIPELINE_MAX_CHUNK_SIZE=0
PIPELINE_AUTO_CHUNK_SIZE=0  # Internal auto-computed chunk size when MULTI_STEP>0.
PIPELINE_KEEP_STEPS=${PIPELINE_KEEP_STEPS:-""}      # Optional comma list, e.g. 0,8,16,24,32,39.
USER_PIPELINE_KEEP_STEPS="$PIPELINE_KEEP_STEPS"
PIPELINE_KEEP_INTERVAL=${PIPELINE_KEEP_INTERVAL:-"0"}  # 0 = final only; >0 keeps every N updates.
PIPELINE_LOCAL_KEEP_POLICY=${PIPELINE_LOCAL_KEEP_POLICY:-"last_hf_only"}  # last_hf_only | all_kept
PIPELINE_ARCHIVE_PRUNED_MODE=${PIPELINE_ARCHIVE_PRUNED_MODE:-"delete"}  # delete | move
PIPELINE_ARCHIVE_MODEL_ROOT=${PIPELINE_ARCHIVE_MODEL_ROOT:-"model/archive"}
PIPELINE_ARCHIVE_MODEL_DIR=${PIPELINE_ARCHIVE_MODEL_DIR:-""}  # resolved after DISTILL_TASK_FAMILY/MODEL_NAME are known
PIPELINE_ARCHIVE_KEEP_MODE=${PIPELINE_ARCHIVE_KEEP_MODE:-"off"}  # copy | off; fixed checkpoints for eval, not resume
PIPELINE_CLEANUP_BATCH_DATA=${PIPELINE_CLEANUP_BATCH_DATA:-"true"}
PIPELINE_STORE_RESUME_MODEL_IN_GEN_RESULTS=${PIPELINE_STORE_RESUME_MODEL_IN_GEN_RESULTS:-"true"}
PIPELINE_CLEANUP_GEN_RESULTS_ON_COMPLETE=${PIPELINE_CLEANUP_GEN_RESULTS_ON_COMPLETE:-"true"}
PIPELINE_TEMP_MODEL_DIR=${PIPELINE_TEMP_MODEL_DIR:-""}
PIPELINE_EPHEMERAL_MODELS=${PIPELINE_EPHEMERAL_MODELS:-"false"}  # Eval consumes and deletes all model artifacts.
PIPELINE_DONE_MARKER_SCHEMA="opd_pipeline_update/v1"
# Resume behavior:
#   resume_matching : default; find an older gen_results run with the same semantic signature,
#                     reuse its model_save_base_dir/gen_results_base_dir, and continue.
#   fresh           : do not search older runs. A normal new RUN_DATE creates a fresh run.
PIPELINE_RESUME_MODE=${PIPELINE_RESUME_MODE:-"resume_matching"}
PIPELINE_AUTO_RESUME=${PIPELINE_AUTO_RESUME:-"false"}  # Skip completed chunks inside the selected run.
PIPELINE_STOP_AFTER_UPDATE=${PIPELINE_STOP_AFTER_UPDATE:-""}
if [ "$PIPELINE_RESUME_MODE" = "resume_matching" ]; then
    PIPELINE_AUTO_RESUME="true"
fi

echo ""
echo "############################################################"
echo "# OPD PIPELINE REQUESTED START MODE: $PIPELINE_RESUME_MODE"
if [ "$PIPELINE_RESUME_MODE" = "resume_matching" ]; then
    echo "# AUTO-RESUME: enabled; will search for a matching previous run."
else
    echo "# FRESH/REFRESH: enabled; will not search previous runs."
fi
echo "############################################################"
echo ""

# Distributed Training
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
# auto: use srun to launch one torchrun agent per node when NNODES>1 under Slurm.
USE_SLURM_TORCHRUN=${USE_SLURM_TORCHRUN:-"auto"}
SLURM_GPU_TYPE=${SLURM_GPU_TYPE:-"h100"}
# Use every GPU in one tensor-parallel vLLM replica by default for both
# training-time generation and post-training evaluation.
GEN_TP=${GEN_TP:-$NGPUS_PER_NODE}
EVAL_GEN_TP=${EVAL_GEN_TP:-$NGPUS_PER_NODE}

# verl FSDP Settings
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}
FSDP_SIZE=${FSDP_SIZE:--1}
SP_SIZE=${SP_SIZE:-1}
# Resolved after Y_MODE / TEACHER_TRAINING_PROMPT are known.
MAX_LENGTH=${MAX_LENGTH:-}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-}
STAGE2_PROMPT_LENGTH=${STAGE2_PROMPT_LENGTH:-}  # y_r stage2 rewrite prompt length, resolved below.
NUM_WORKERS=${NUM_WORKERS:-4}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-"true"}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-"false"}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-"false"}
OFFLOAD_POLICY=${OFFLOAD_POLICY:-"false"}

# Output Settings
OUTPUT_DIR=${OUTPUT_DIR:-""}  # Base output dir; epoch subdirs are appended
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"model/trained"}  # Base model save dir; epoch subdirs are appended
WANDB_PROJECT=${WANDB_PROJECT:-"trd"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}  # Stable display name shared by all train/eval segments.
WANDB_RUN_ID=${WANDB_RUN_ID:-""}
WANDB_RUN_ID_FILE=${WANDB_RUN_ID_FILE:-""}
WANDB_RUN_IDENTITY=${WANDB_RUN_IDENTITY:-""}
WANDB_RESUME=${WANDB_RESUME:-"allow"}
EVAL_FRACTIONS=${EVAL_FRACTIONS:-"0.25,0.5,0.75,1"}
export WANDB_MODE="${WANDB_MODE:-online}"
SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-"true"}  # Merge LoRA after training
SAVE_STEPS=${SAVE_STEPS:-100}                   # FSDP ckpt every N optimizer steps (crash recovery)
KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS:-1}  # Rolling window; per-epoch hf_merged is always preserved

# Resident student rollout keeps ordinary pipeline updates on rolling FSDP
# state, so they do not need a merged Hugging Face export.
RESIDENT_STUDENT_ROLLOUT=${RESIDENT_STUDENT_ROLLOUT:-""}
RESIDENT_YO_MANIFEST=${RESIDENT_YO_MANIFEST:-""}
RESIDENT_YO_AUTOSTART=${RESIDENT_YO_AUTOSTART:-"true"}
RESIDENT_YO_LOAD_FORMAT=${RESIDENT_YO_LOAD_FORMAT:-"auto"}
ASYNC_HF_EXPORT=${ASYNC_HF_EXPORT:-"false"}
RESIDENT_YO_RELEASE_AFTER_STAGE1="false"

# Evaluation Settings
RUN_EVAL_AFTER_TRAINING=${RUN_EVAL_AFTER_TRAINING:-"true"}
# EVAL_DATASETS=${EVAL_DATASETS:-"aime24,aime25,math500,hmmt25"} DEFAULT_DATASETS="math500 hmmt25 beyondaime amobench gsm8k"
if [ "$TASK" = "code" ]; then
    EVAL_DATASETS=${EVAL_DATASETS:-"humaneval_plus,mbpp_plus,livecodebench_v6"}
    PASS_K=${PASS_K:-16}
    CODE_EVAL_TEMPERATURE=${CODE_EVAL_TEMPERATURE:-0.6}
    CODE_EVAL_TOP_P=${CODE_EVAL_TOP_P:-0.95}
    CODE_EVAL_SEED=${CODE_EVAL_SEED:-42}
    CODE_EVAL_MAX_PROMPT_TOKENS=${CODE_EVAL_MAX_PROMPT_TOKENS:-2048}
    CODE_EVAL_MAX_RESPONSE_TOKENS=${CODE_EVAL_MAX_RESPONSE_TOKENS:-${CODE_EVAL_MAX_TOKENS:-16384}}
    CODE_EVAL_SIGNATURE=${CODE_EVAL_SIGNATURE:-"n${PASS_K}_t${CODE_EVAL_TEMPERATURE}_p${CODE_EVAL_TOP_P}_prompt${CODE_EVAL_MAX_PROMPT_TOKENS}_response${CODE_EVAL_MAX_RESPONSE_TOKENS}_seed${CODE_EVAL_SEED}_base"}
    export CODE_EVAL_TEMPERATURE CODE_EVAL_TOP_P CODE_EVAL_SEED
    export CODE_EVAL_MAX_PROMPT_TOKENS CODE_EVAL_MAX_RESPONSE_TOKENS CODE_EVAL_SIGNATURE
else
    EVAL_DATASETS=${EVAL_DATASETS:-"aime25,aime26,hmmt26,amobench"}
    PASS_K=${PASS_K:-16}
fi

# Notification Settings
NTFY_ENABLED=${NTFY_ENABLED:-"true"}
NTFY_TOPIC=${NTFY_TOPIC:-"ec2_lijiang_2026"}
NTFY_URL=${NTFY_URL:-"https://ntfy.sh/$NTFY_TOPIC"}
NTFY_TIMEOUT_SECONDS=${NTFY_TIMEOUT_SECONDS:-10}
NTFY_JOB_STARTED="false"
NTFY_JOB_START_TS=$(date +%s)

# Paths
# Layout: recipe/opd/run/<this_script>.sh
#         recipe/opd/<run_training.py, run_eval_suite.py>
#         recipe/opd/generation/<stage1/2 prep, scoring>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$SCRIPT_DIR")"                           # recipe/opd
VERL_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"               # repo root
cd "$VERL_ROOT"
SCRIPT_DIR="recipe/opd/run"
RECIPE_DIR="recipe/opd"
PIPELINE_DIR="recipe/opd/generation"
export HF_HOME="${OPD_HF_HOME:-data/download_cache/huggingface}"
export HF_DATASETS_CACHE="${OPD_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${OPD_HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE

if [ "$TASK" = "code" ]; then
    EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-"data/eval_dataset/code"}
else
    EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-"data/eval_dataset/math"}
fi

# Training data path (for prompts)
if [ "$TASK" = "code" ]; then
    # OPD/OPSD do not consume TACO execution tests; those remain isolated in
    # train_grpo.parquet for the GRPO baseline.
    TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"data/train_dataset/taco/canonical/train_distill.parquet"}
    TRAIN_DATA_SOURCE=${TRAIN_DATA_SOURCE:-"BAAI/TACO"}
    SCORE_STAGE1=${SCORE_STAGE1:-"auto"}
else
    TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"data/train_dataset/openthoughts_math_30k_opsd/train_grpo.parquet"}
    TRAIN_DATA_SOURCE=${TRAIN_DATA_SOURCE:-"siyanzhao/Openthoughts_math_30k_opsd"}
    SCORE_STAGE1=${SCORE_STAGE1:-"auto"}
fi

# Model name from the original base model, not from epoch checkpoints. Preserve
# an explicit outer value so local snapshot paths can still get readable names.
MODEL_NAME="${MODEL_NAME:-${MODEL_PATH##*/}}"
if [ "$Y_MODE" != "y_o" ] && [ "$Y_MODE" != "y_r" ]; then
    echo "ERROR: Y_MODE must be one of: y_o, y_r (got: $Y_MODE)"
    exit 1
fi

# TEACHER_TRAINING_PROMPT picks the teacher-side prompt variant *at training time*.
# The generation-time prompt is FIXED (always the refine variant — for OPD it
# always contains initial_response; for OPSD it always contains expert_solution
# + initial_response). This knob only changes what the teacher conditions on
# when computing logprobs over y_r tokens during the KL loss.
#
#   vanilla — teacher trained on π_T(·|x)         (OPD) or π_T(·|x, y*)         (OPSD)
#   refine  — teacher trained on π_T(·|x, y_o)    (OPD) or π_T(·|x, y*, y_o)    (OPSD)
#
# Default: y_r data → refine (matches generation-time conditioning);
#          y_o data → vanilla (stage1 parquet has no initial_response field).
if [ -n "${TEACHER_TRAINING_PROMPT:-}" ]; then
    case "$TEACHER_TRAINING_PROMPT" in
        vanilla) USE_INITIAL_RESPONSE="false" ;;
        refine)  USE_INITIAL_RESPONSE="true" ;;
        *) echo "ERROR: TEACHER_TRAINING_PROMPT must be 'vanilla' or 'refine' (got: $TEACHER_TRAINING_PROMPT)" >&2; exit 1 ;;
    esac
else
    if [ "$Y_MODE" = "y_r" ]; then
        TEACHER_TRAINING_PROMPT="refine"; USE_INITIAL_RESPONSE="true"
    else
        TEACHER_TRAINING_PROMPT="vanilla"; USE_INITIAL_RESPONSE="false"
    fi
fi

# refine + y_o is invalid: stage1 parquet has no initial_response field to embed.
if [ "$Y_MODE" = "y_o" ] && [ "$USE_INITIAL_RESPONSE" = "true" ]; then
    echo "ERROR: TEACHER_TRAINING_PROMPT=refine is incompatible with Y_MODE=y_o (no initial response in stage1 parquet)." >&2
    exit 1
fi

# Length budget policy:
#   - Student prompt is fixed to the base problem budget.
#   - Student/teacher generated responses share one fixed response budget.
#   - OPSD teacher prompts additionally reserve an expert-solution budget.
# Derived budgets are intentionally conservative so the default path does not
# clip target responses. If a model cannot support the derived context length,
# lower EXPERT_SOLUTION_PROMPT_LENGTH or MAX_RESPONSE_LENGTH explicitly.
if [ "$TASK" = "code" ]; then
    BASE_PROMPT_LENGTH=${BASE_PROMPT_LENGTH:-2048}
    MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
    EXPERT_SOLUTION_PROMPT_LENGTH=${EXPERT_SOLUTION_PROMPT_LENGTH:-4096}
else
    BASE_PROMPT_LENGTH=${BASE_PROMPT_LENGTH:-2048}
    MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
    EXPERT_SOLUTION_PROMPT_LENGTH=${EXPERT_SOLUTION_PROMPT_LENGTH:-4096}
fi
STAGE1_PROMPT_LENGTH=$BASE_PROMPT_LENGTH
MODEL_CONTEXT_LENGTH=${MODEL_CONTEXT_LENGTH:-32768}
if [ "$MAX_RESPONSE_LENGTH" -ge "$MODEL_CONTEXT_LENGTH" ]; then
    echo "ERROR: MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH must be below MODEL_CONTEXT_LENGTH=$MODEL_CONTEXT_LENGTH" >&2
    exit 1
fi
MAX_DERIVED_PROMPT_LENGTH=$((MODEL_CONTEXT_LENGTH - MAX_RESPONSE_LENGTH - TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER))
if [ "$MAX_DERIVED_PROMPT_LENGTH" -le 0 ]; then
    echo "ERROR: teacher chat-template buffer leaves no prompt budget" >&2
    exit 1
fi

if [ "$DISTILL_MODE" = "opsd" ]; then
    if [ "$USE_INITIAL_RESPONSE" = "true" ]; then
        DERIVED_TEACHER_PROMPT_LENGTH=$((BASE_PROMPT_LENGTH + EXPERT_SOLUTION_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    else
        DERIVED_TEACHER_PROMPT_LENGTH=$((BASE_PROMPT_LENGTH + EXPERT_SOLUTION_PROMPT_LENGTH))
    fi
    DERIVED_STAGE2_PROMPT_LENGTH=$((BASE_PROMPT_LENGTH + EXPERT_SOLUTION_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
else
    if [ "$USE_INITIAL_RESPONSE" = "true" ]; then
        DERIVED_TEACHER_PROMPT_LENGTH=$((BASE_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    else
        DERIVED_TEACHER_PROMPT_LENGTH=$BASE_PROMPT_LENGTH
    fi
    DERIVED_STAGE2_PROMPT_LENGTH=$((BASE_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
fi

# MAX_PROMPT_LENGTH is the training-time teacher prompt budget. Stage2 y_r
# generation has its own prompt budget because y_r generation always uses refine
# prompts even when TEACHER_TRAINING_PROMPT=vanilla.
if [ -z "${MAX_PROMPT_LENGTH+x}" ] || [ -z "$MAX_PROMPT_LENGTH" ]; then
    MAX_PROMPT_LENGTH=$DERIVED_TEACHER_PROMPT_LENGTH
fi
if [ -z "${STAGE2_PROMPT_LENGTH+x}" ] || [ -z "$STAGE2_PROMPT_LENGTH" ]; then
    STAGE2_PROMPT_LENGTH=$DERIVED_STAGE2_PROMPT_LENGTH
fi
if [ "$MAX_PROMPT_LENGTH" -gt "$MAX_DERIVED_PROMPT_LENGTH" ]; then
    echo "WARNING: capping teacher prompt $MAX_PROMPT_LENGTH -> $MAX_DERIVED_PROMPT_LENGTH to preserve the 16K response within native context." >&2
    MAX_PROMPT_LENGTH=$MAX_DERIVED_PROMPT_LENGTH
fi
if [ "$STAGE2_PROMPT_LENGTH" -gt "$MAX_DERIVED_PROMPT_LENGTH" ]; then
    echo "WARNING: capping stage2 prompt $STAGE2_PROMPT_LENGTH -> $MAX_DERIVED_PROMPT_LENGTH to preserve the 16K response within native context." >&2
    STAGE2_PROMPT_LENGTH=$MAX_DERIVED_PROMPT_LENGTH
fi

TRAIN_MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
if [ "$BASE_PROMPT_LENGTH" -gt "$TRAIN_MAX_PROMPT_LENGTH" ]; then
    TRAIN_MAX_PROMPT_LENGTH=$BASE_PROMPT_LENGTH
fi
DERIVED_MAX_LENGTH=$((TRAIN_MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER))
if [ -z "$MAX_LENGTH" ]; then
    MAX_LENGTH=$DERIVED_MAX_LENGTH
elif [ "$MAX_LENGTH" -lt "$DERIVED_MAX_LENGTH" ]; then
    echo "WARNING: MAX_LENGTH=$MAX_LENGTH is below derived no-clip budget $DERIVED_MAX_LENGTH; response or prompt truncation may occur." >&2
fi
if [ "$MAX_LENGTH" -gt "$MODEL_CONTEXT_LENGTH" ]; then
    echo "ERROR: MAX_LENGTH=$MAX_LENGTH exceeds native MODEL_CONTEXT_LENGTH=$MODEL_CONTEXT_LENGTH" >&2
    exit 1
fi

ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER=${ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER:-0}
STAGE1_ROLLOUT_MAX_MODEL_LEN=$((STAGE1_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER))
STAGE2_ROLLOUT_MAX_MODEL_LEN=$((STAGE2_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER))
TEACHER_TRAJECTORY_PROMPT_LENGTH=${TEACHER_TRAJECTORY_PROMPT_LENGTH:-$MAX_PROMPT_LENGTH}
TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN=$((TEACHER_TRAJECTORY_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER))
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
# y_o rollout source. `student` is the historical student-only rollout.
# `teacher` samples y_t from a fixed trajectory policy while leaving the KL
# teacher unchanged. Its conditioning is selected independently:
# `pi_T_x_only_v1` samples pi_T(.|x), while
# `opsd_x_y_star_expert_rewrite_v1` samples pi_T(.|x,y*).
# `expert` uses the dataset's non-empty expert_cot as y* without generation.
# `skd`/`skd_vllm` runs student draft + teacher accept/replacement during y_o rollout.
# `skd_vllm_internal` installs a repo-local vLLM sampler patch and uses vLLM
# speculative decoding when the installed vLLM supports a plain student draft model.
Y_O_ROLLOUT_MODE=${Y_O_ROLLOUT_MODE:-"student"}
TRAJECTORY_MODEL_PATH=${TRAJECTORY_MODEL_PATH:-""}
TEACHER_TRAJECTORY_PROMPT_PATH=${TEACHER_TRAJECTORY_PROMPT_PATH:-""}
TEACHER_TRAJECTORY_CONDITIONING=${TEACHER_TRAJECTORY_CONDITIONING:-""}
TEACHER_TRAJECTORY_CACHE_MODE=${TEACHER_TRAJECTORY_CACHE_MODE:-"read_write"}
TEACHER_TRAJECTORY_CACHE_ROOT=${TEACHER_TRAJECTORY_CACHE_ROOT:-"gen_results/fixed_teacher_trajectory_cache"}
if [ -n "${TRAJECTORY_MODEL:-}" ]; then
    TRAJECTORY_MODEL_NAME="$TRAJECTORY_MODEL"
elif [ -n "$TRAJECTORY_MODEL_PATH" ]; then
    TRAJECTORY_MODEL_NAME="${TRAJECTORY_MODEL_PATH##*/}"
else
    TRAJECTORY_MODEL_NAME=""
fi
case "$Y_O_ROLLOUT_MODE" in
    student|expert) ;;
    teacher)
        if [ -z "$TRAJECTORY_MODEL_PATH" ]; then
            echo "ERROR: Y_O_ROLLOUT_MODE=teacher requires TRAJECTORY_MODEL_PATH." >&2
            exit 1
        fi
        if [ -z "$TEACHER_TRAJECTORY_CONDITIONING" ]; then
            TEACHER_TRAJECTORY_CONDITIONING="pi_T_x_only_v1"
        fi
        case "$TEACHER_TRAJECTORY_CONDITIONING" in
            pi_T_x_only_v1) ;;
            opsd_x_y_star_expert_rewrite_v1)
                if [ -z "$TEACHER_TRAJECTORY_PROMPT_PATH" ] || [ ! -s "$TEACHER_TRAJECTORY_PROMPT_PATH" ]; then
                    echo "ERROR: expert-rewrite teacher rollout requires a prepared pi(.|x,y*) prompt parquet." >&2
                    echo "       Missing TEACHER_TRAJECTORY_PROMPT_PATH: ${TEACHER_TRAJECTORY_PROMPT_PATH:-<empty>}" >&2
                    exit 1
                fi
                ;;
            *)
                echo "ERROR: unsupported TEACHER_TRAJECTORY_CONDITIONING=$TEACHER_TRAJECTORY_CONDITIONING" >&2
                echo "       Expected pi_T_x_only_v1 or opsd_x_y_star_expert_rewrite_v1." >&2
                exit 1
                ;;
        esac
        ;;
    skd|skd_vllm) ;;
    skd_vllm_internal)
        if [ "$DISTILL_MODE" = "opsd" ]; then
            echo "ERROR: skd_vllm_internal cannot give student x and OPSD teacher x+y* distinct prefixes; use Y_O_ROLLOUT_MODE=skd_vllm." >&2
            exit 1
        fi
        ;;
    *) echo "ERROR: Y_O_ROLLOUT_MODE must be one of: student, teacher, expert, skd, skd_vllm, skd_vllm_internal (got: $Y_O_ROLLOUT_MODE)" >&2; exit 1 ;;
esac

case "$PIPELINE_EPHEMERAL_MODELS" in
    true|false) ;;
    *) echo "ERROR: PIPELINE_EPHEMERAL_MODELS must be true or false (got: $PIPELINE_EPHEMERAL_MODELS)." >&2; exit 1 ;;
esac

if [ -z "$RESIDENT_STUDENT_ROLLOUT" ]; then
    if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ] && [ "$Y_O_ROLLOUT_MODE" = "student" ]; then
        RESIDENT_STUDENT_ROLLOUT="true"
    else
        RESIDENT_STUDENT_ROLLOUT="false"
    fi
fi
case "$RESIDENT_STUDENT_ROLLOUT" in
    true|false) ;;
    *) echo "ERROR: RESIDENT_STUDENT_ROLLOUT must be true or false (got: $RESIDENT_STUDENT_ROLLOUT)." >&2; exit 1 ;;
esac

if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
    SAVE_MERGED_MODEL="false"
    if [ "$PIPELINE_ARCHIVE_KEEP_MODE" != "off" ]; then
        echo "ERROR: ephemeral model mode is incompatible with PIPELINE_ARCHIVE_KEEP_MODE=$PIPELINE_ARCHIVE_KEEP_MODE." >&2
        exit 1
    fi
    case "$Y_O_ROLLOUT_MODE" in
        student)
            if [ "$RESIDENT_STUDENT_ROLLOUT" != "true" ]; then
                echo "ERROR: student rollout needs RESIDENT_STUDENT_ROLLOUT=true in ephemeral model mode." >&2
                exit 1
            fi
            ;;
        skd|skd_vllm)
            if [ "$USE_LORA" != "true" ]; then
                echo "ERROR: SKD ephemeral model mode requires USE_LORA=true for adapter rollout." >&2
                exit 1
            fi
            RESIDENT_STUDENT_ROLLOUT="false"
            ;;
        teacher|expert)
            RESIDENT_STUDENT_ROLLOUT="false"
            ;;
        skd_vllm_internal)
            echo "ERROR: skd_vllm_internal cannot consume a rolling LoRA adapter; use skd_vllm." >&2
            exit 1
            ;;
    esac
fi

# A y_r teacher rollout needs all Ray GPU leases. In ephemeral mode the
# managed student server is stopped after stage 1 and restarted from the next
# rolling FSDP checkpoint. An externally managed server cannot be stopped here.
if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$Y_MODE" = "y_r" ]; then
    if [ "$PIPELINE_EPHEMERAL_MODELS" != "true" ]; then
        echo "ERROR: resident y_r rollout requires PIPELINE_EPHEMERAL_MODELS=true." >&2
        exit 1
    fi
    if [ -n "$RESIDENT_YO_MANIFEST" ]; then
        echo "ERROR: ephemeral y_r rollout requires a runner-managed resident server." >&2
        exit 1
    fi
    RESIDENT_YO_RELEASE_AFTER_STAGE1="true"
fi
export PIPELINE_EPHEMERAL_MODELS SAVE_MERGED_MODEL RESIDENT_STUDENT_ROLLOUT

if [ "$Y_O_ROLLOUT_MODE" != "teacher" ]; then
    TEACHER_TRAJECTORY_CONDITIONING=""
elif [ "$TEACHER_TRAJECTORY_CONDITIONING" = "pi_T_x_only_v1" ]; then
    # x-only rollout uses the ordinary problem prompt, not the larger OPSD
    # training-time prompt budget that reserves space for y*.
    TEACHER_TRAJECTORY_PROMPT_LENGTH=$BASE_PROMPT_LENGTH
    TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN=$STAGE1_ROLLOUT_MAX_MODEL_LEN
fi
if [ -n "$PRECOMPUTED_Y_O_TRAJECTORY_PATH" ]; then
    if [ "$Y_MODE" != "y_o" ] || [ "$Y_O_ROLLOUT_MODE" != "expert" ]; then
        echo "ERROR: PRECOMPUTED_Y_O_TRAJECTORY_PATH is currently supported only for Y_MODE=y_o and Y_O_ROLLOUT_MODE=expert." >&2
        exit 1
    fi
    if [ ! -s "$PRECOMPUTED_Y_O_TRAJECTORY_PATH" ]; then
        echo "ERROR: precomputed y_o trajectory parquet is missing: $PRECOMPUTED_Y_O_TRAJECTORY_PATH" >&2
        exit 1
    fi
fi
case "$TEACHER_TRAJECTORY_CACHE_MODE" in
    off|read_only|read_write) ;;
    *) echo "ERROR: TEACHER_TRAJECTORY_CACHE_MODE must be one of: off, read_only, read_write (got: $TEACHER_TRAJECTORY_CACHE_MODE)" >&2; exit 1 ;;
esac
SKD_ACCEPT_TOP_K=${SKD_ACCEPT_TOP_K:-25}
SKD_ACCEPT_TOP_P=${SKD_ACCEPT_TOP_P:-1.0}
SKD_GAMMA=${SKD_GAMMA:-5}
SKD_ROLLOUT_BATCH_SIZE=${SKD_ROLLOUT_BATCH_SIZE:-128}
SKD_STUDENT_TEMPERATURE=${SKD_STUDENT_TEMPERATURE:-$ROLLOUT_TEMPERATURE}
SKD_STUDENT_TOP_P=${SKD_STUDENT_TOP_P:-$ROLLOUT_TOP_P}
SKD_TEACHER_TEMPERATURE=${SKD_TEACHER_TEMPERATURE:-$ROLLOUT_TEMPERATURE}
SKD_TEACHER_TOP_P=${SKD_TEACHER_TOP_P:-$ROLLOUT_TOP_P}
SKD_STUDENT_GPUS=${SKD_STUDENT_GPUS:-0,1,2,3}
SKD_TEACHER_GPUS=${SKD_TEACHER_GPUS:-4,5,6,7}
SKD_SHARED_GPUS=${SKD_SHARED_GPUS:-0,1,2,3,4,5,6,7}
SKD_STUDENT_TP=${SKD_STUDENT_TP:-0}
SKD_TEACHER_TP=${SKD_TEACHER_TP:-0}
SKD_VLLM_GPU_MEMORY_UTILIZATION=${SKD_VLLM_GPU_MEMORY_UTILIZATION:-$ROLLOUT_GPU_MEMORY_UTILIZATION}
SKD_VLLM_MAX_NUM_SEQS=${SKD_VLLM_MAX_NUM_SEQS:-$SKD_ROLLOUT_BATCH_SIZE}
SKD_VLLM_MAX_NUM_BATCHED_TOKENS=${SKD_VLLM_MAX_NUM_BATCHED_TOKENS:-$ROLLOUT_MAX_NUM_BATCHED_TOKENS}
SKD_PARALLEL_STUDENT_TEACHER=${SKD_PARALLEL_STUDENT_TEACHER:-true}
SKD_PIPELINE_LANES=${SKD_PIPELINE_LANES:-2}
if [ "$DISTILL_MODE" = "opsd" ]; then
    SKD_TEACHER_PROMPT_CONTRACT=${SKD_TEACHER_PROMPT_CONTRACT:-"opsd_x_y_star_v1"}
    SKD_TEACHER_PROMPT_LENGTH=${SKD_TEACHER_PROMPT_LENGTH:-$((BASE_PROMPT_LENGTH + EXPERT_SOLUTION_PROMPT_LENGTH))}
else
    if [ "$TEACHER_ENABLE_THINKING" = "true" ]; then
        SKD_TEACHER_PROMPT_CONTRACT=${SKD_TEACHER_PROMPT_CONTRACT:-"opd_x_only_qwen3_thinking_v1"}
    else
        SKD_TEACHER_PROMPT_CONTRACT=${SKD_TEACHER_PROMPT_CONTRACT:-"opd_x_only_plain_completion_v1"}
    fi
    SKD_TEACHER_PROMPT_LENGTH=${SKD_TEACHER_PROMPT_LENGTH:-$((BASE_PROMPT_LENGTH + TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER))}
fi
if [ "$SKD_TEACHER_PROMPT_LENGTH" -gt "$MAX_DERIVED_PROMPT_LENGTH" ]; then
    echo "WARNING: capping SKD teacher prompt $SKD_TEACHER_PROMPT_LENGTH -> $MAX_DERIVED_PROMPT_LENGTH to preserve the response budget." >&2
    SKD_TEACHER_PROMPT_LENGTH=$MAX_DERIVED_PROMPT_LENGTH
fi
SKD_ROLLOUT_MAX_MODEL_LEN=$((SKD_TEACHER_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER))
if [ "$STAGE1_ROLLOUT_MAX_MODEL_LEN" -gt "$SKD_ROLLOUT_MAX_MODEL_LEN" ]; then
    SKD_ROLLOUT_MAX_MODEL_LEN=$STAGE1_ROLLOUT_MAX_MODEL_LEN
fi
if [ "$SKD_ROLLOUT_MAX_MODEL_LEN" -gt "$MODEL_CONTEXT_LENGTH" ]; then
    echo "ERROR: SKD_ROLLOUT_MAX_MODEL_LEN=$SKD_ROLLOUT_MAX_MODEL_LEN exceeds MODEL_CONTEXT_LENGTH=$MODEL_CONTEXT_LENGTH" >&2
    exit 1
fi
# Persist the semantic SKD rollout configuration only for SKD runs. Keeping this
# block empty for student/teacher/expert prevents operationally irrelevant SKD
# environment variables from invalidating their gen_results identity.
SKD_ROLLOUT_METADATA_YAML=""
case "$Y_O_ROLLOUT_MODE" in
    skd|skd_vllm|skd_vllm_internal)
        SKD_ROLLOUT_METADATA_YAML="$(cat <<EOF
skd_gamma: $SKD_GAMMA
skd_accept_top_k: $SKD_ACCEPT_TOP_K
skd_accept_top_p: $SKD_ACCEPT_TOP_P
skd_student_temperature: $SKD_STUDENT_TEMPERATURE
skd_student_top_p: $SKD_STUDENT_TOP_P
skd_teacher_temperature: $SKD_TEACHER_TEMPERATURE
skd_teacher_top_p: $SKD_TEACHER_TOP_P
skd_rollout_batch_size: $SKD_ROLLOUT_BATCH_SIZE
skd_pipeline_lanes: $SKD_PIPELINE_LANES
skd_parallel_student_teacher: $SKD_PARALLEL_STUDENT_TEACHER
skd_teacher_prompt_contract: $SKD_TEACHER_PROMPT_CONTRACT
skd_teacher_enable_thinking: $TEACHER_ENABLE_THINKING
skd_teacher_prompt_length: $SKD_TEACHER_PROMPT_LENGTH
skd_rollout_max_model_len: $SKD_ROLLOUT_MAX_MODEL_LEN
EOF
)"
        ;;
esac
STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
if [ "$STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS" -lt "$STAGE1_ROLLOUT_MAX_MODEL_LEN" ]; then
    STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS=$STAGE1_ROLLOUT_MAX_MODEL_LEN
fi
TEACHER_TRAJECTORY_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
if [ "$TEACHER_TRAJECTORY_MAX_NUM_BATCHED_TOKENS" -lt "$TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN" ]; then
    TEACHER_TRAJECTORY_MAX_NUM_BATCHED_TOKENS=$TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN
fi
if [ "$SKD_VLLM_MAX_NUM_BATCHED_TOKENS" -lt "$SKD_ROLLOUT_MAX_MODEL_LEN" ]; then
    SKD_VLLM_MAX_NUM_BATCHED_TOKENS=$SKD_ROLLOUT_MAX_MODEL_LEN
fi
STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
if [ "$STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS" -lt "$STAGE2_ROLLOUT_MAX_MODEL_LEN" ]; then
    STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS=$STAGE2_ROLLOUT_MAX_MODEL_LEN
fi
# Keep the dynamic batching token cap aligned with the training truncation
# budget. A smaller external value can fail before KLTrainingDataset truncation
# takes effect when long code samples exceed the per-GPU cap.
MAX_TOKEN_LEN_PER_GPU=$MAX_LENGTH
PROMPT_MODE_TAG="$Y_MODE"   # y_o or y_r — kept named PROMPT_MODE_TAG for backward compat with downstream string handling
CLIP_TAG="clip$(echo $KL_TOKEN_CLIP | sed 's/\.//')"  # e.g. 0.1 -> clip01, 0.06 -> clip006
if [ "$KL_TYPE" = "jsd" ]; then
    BETA_TAG="_beta$(echo $BETA | sed 's/\.//')"
else
    BETA_TAG=""
fi
if [ "${TOP_K:-0}" -gt 0 ]; then
    TOPK_TAG="_topk${TOP_K}"
else
    TOPK_TAG=""
fi
# y_o + FORWARD_STAGE2_MODE = stage1_reward_{0,1}_only filters the stage1
# parquet by extra_info.reward to produce a per-bucket training set.
Y_O_FILTER_TAG=""
if [ "$Y_MODE" = "y_o" ]; then
    case "${FORWARD_STAGE2_MODE:-}" in
        "stage1_reward_0_only") Y_O_FILTER_TAG="_reward0" ;;
        "stage1_reward_1_only") Y_O_FILTER_TAG="_reward1" ;;
        ""|"none"|"all"|"rewrite_all") ;;  # default = no filter
        *) echo "ERROR: y_o FORWARD_STAGE2_MODE must be one of: stage1_reward_0_only, stage1_reward_1_only, all (got: $FORWARD_STAGE2_MODE)"; exit 1 ;;
    esac
fi
# Teacher model name — used to qualify stage2 file paths and the experiment tag.
if [ -n "${TEACHER_MODEL:-}" ]; then
    TEACHER_MODEL_NAME="$TEACHER_MODEL"
elif [ -n "${TEACHER_MODEL_PATH:-}" ]; then
    TEACHER_MODEL_NAME="${TEACHER_MODEL_PATH##*/}"
else
    TEACHER_MODEL_NAME="${MODEL_NAME}"
fi

EXPERIMENT_TAG="kl_${KL_TYPE}_${KL_METHOD}_${PROMPT_MODE_TAG}_${CLIP_TAG}${BETA_TAG}${TOPK_TAG}${Y_O_FILTER_TAG}_${DISTILL_MODE}_${TEACHER_TRAINING_PROMPT}"
EXPERIMENT_TAG="${EXPERIMENT_TAG}_teacher${TEACHER_MODEL_NAME}${TEACHER_THINKING_FILE_SUFFIX}"

# y_r path no longer reads FORWARD_STAGE2_MODE — y_r_prepare.py always processes
# every stage1 row. Reward-based filtering of y_r happens post-generation via
# FORWARD_FILTER_STAGE2=true (recipe/opd/dataset/filter_stage2_by_reward.py).

# OPD requires its external Qwen3-14B teacher; OPSD explicitly passes the
# student's frozen step-0 Base checkpoint as its self-teacher.
if [ -z "$TEACHER_MODEL_PATH" ]; then
    echo "ERROR: TEACHER_MODEL_PATH is required for both OPD and OPSD."
    exit 1
fi

if [ "$TOTAL_EPOCHS" -lt 1 ]; then
    echo "ERROR: TOTAL_EPOCHS must be >= 1"
    exit 1
fi

if [ "$TOTAL_EPOCHS" -gt 1 ] && [ "$SAVE_MERGED_MODEL" != "true" ]; then
    echo "ERROR: Multi-epoch training requires SAVE_MERGED_MODEL=true so epoch N+1 can load epoch N as HuggingFace weights."
    exit 1
fi

if ! [[ "$MULTI_STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MULTI_STEP must be a non-negative integer (0 means automatic 512 prompts/step; got: $MULTI_STEP)."
    exit 1
fi
if ! [[ "$DEFAULT_PIPELINE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: DEFAULT_PIPELINE_BATCH_SIZE must be a positive integer (got: $DEFAULT_PIPELINE_BATCH_SIZE)."
    exit 1
fi
if [ -n "$USER_PIPELINE_BATCH_SIZE" ] && ! [[ "$USER_PIPELINE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PIPELINE_BATCH_SIZE must be a positive integer (got: $USER_PIPELINE_BATCH_SIZE)."
    exit 1
fi

if [ "$USER_MULTI_STEP" -gt 0 ]; then
    PIPELINE_STEP_MODE="explicit_steps"
    if [ -n "$USER_PIPELINE_BATCH_SIZE" ]; then
        echo "WARNING: MULTI_STEP=$USER_MULTI_STEP overrides PIPELINE_BATCH_SIZE=$USER_PIPELINE_BATCH_SIZE; deriving balanced partitions from ms." >&2
    fi
    USER_PIPELINE_BATCH_SIZE=""
    if [ "$TRAIN_EPOCHS_PER_ROUND" -ne 1 ]; then
        echo "ERROR: MULTI_STEP>0 means the exact total policy optimizer steps and requires TRAIN_EPOCHS_PER_ROUND=1." >&2
        exit 1
    fi
else
    PIPELINE_STEP_MODE="default_batch"
    USER_PIPELINE_BATCH_SIZE="${USER_PIPELINE_BATCH_SIZE:-$DEFAULT_PIPELINE_BATCH_SIZE}"
    # Enter the shared pipeline setup; the actual update count is derived from
    # dataset_rows / prompts_per_step after the dataset is inspected.
    MULTI_STEP=1
fi
PIPELINE_AUTO_CHUNK_SIZE=0

case "$PIPELINE_RESUME_MODE" in
    fresh|resume_matching) ;;
    *) echo "ERROR: PIPELINE_RESUME_MODE must be fresh or resume_matching (got: $PIPELINE_RESUME_MODE)."; exit 1 ;;
esac
case "$PIPELINE_AUTO_RESUME" in
    true|false) ;;
    *) echo "ERROR: PIPELINE_AUTO_RESUME must be true or false (got: $PIPELINE_AUTO_RESUME)."; exit 1 ;;
esac
if [ -n "$PIPELINE_STOP_AFTER_UPDATE" ] && ! [[ "$PIPELINE_STOP_AFTER_UPDATE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PIPELINE_STOP_AFTER_UPDATE must be empty or a positive integer (got: $PIPELINE_STOP_AFTER_UPDATE)."
    exit 1
fi

for _positive_integer_setting in \
    "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE" \
    "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS" \
    "TRAIN_EPOCHS_PER_ROUND=$TRAIN_EPOCHS_PER_ROUND" \
    "NNODES=$NNODES" \
    "NGPUS_PER_NODE=$NGPUS_PER_NODE" \
    "MICRO_BATCH_SIZE_PER_GPU=$MICRO_BATCH_SIZE_PER_GPU"; do
    _positive_integer_name="${_positive_integer_setting%%=*}"
    _positive_integer_value="${_positive_integer_setting#*=}"
    if ! [[ "$_positive_integer_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $_positive_integer_name must be a positive integer (got: $_positive_integer_value)."
        exit 1
    fi
done

pipeline_optimizer_steps_for_samples() {
    # Mirror KLTrainer's DistributedSampler/DataLoader/gradient-accumulation
    # ceil behavior. This lets every independently launched chunk append to one
    # truthful optimizer-step axis in W&B, including a short final tail chunk.
    local sample_count="$1"
    local world_size=$((NNODES * NGPUS_PER_NODE))
    local samples_per_rank=$(((sample_count + world_size - 1) / world_size))
    local batches_per_rank=$(((samples_per_rank + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
    local steps_per_epoch=$(((batches_per_rank + GRADIENT_ACCUMULATION_STEPS - 1) / GRADIENT_ACCUMULATION_STEPS))
    echo $((steps_per_epoch * TRAIN_EPOCHS_PER_ROUND))
}

pipeline_partition_start_size() {
    local batch_index="$1"
    local start size remaining extra_before

    if [ "$PIPELINE_STEP_MODE" = "explicit_steps" ]; then
        extra_before=$((batch_index - 1))
        if [ "$extra_before" -gt "$PIPELINE_BALANCED_REMAINDER" ]; then
            extra_before="$PIPELINE_BALANCED_REMAINDER"
        fi
        start=$(((batch_index - 1) * PIPELINE_AUTO_CHUNK_SIZE + extra_before * PIPELINE_BALANCED_INCREMENT))
        size="$PIPELINE_AUTO_CHUNK_SIZE"
        if [ "$batch_index" -le "$PIPELINE_BALANCED_REMAINDER" ]; then
            size=$((size + PIPELINE_BALANCED_INCREMENT))
        fi
    else
        start=$(((batch_index - 1) * PIPELINE_AUTO_CHUNK_SIZE))
        size="$PIPELINE_AUTO_CHUNK_SIZE"
        remaining=$((TOTAL_TRAIN_SAMPLES - start))
        if [ "$remaining" -lt "$size" ]; then
            size="$remaining"
        fi
    fi
    printf '%s %s\n' "$start" "$size"
}

if [ "$MULTI_STEP" -gt 0 ]; then
    if [ "$SAVE_MERGED_MODEL" != "true" ] && [ "$PIPELINE_EPHEMERAL_MODELS" != "true" ]; then
        echo "ERROR: MULTI_STEP>0 requires SAVE_MERGED_MODEL=true so the next chunk can load the updated policy."
        exit 1
    fi
    if [ "$ASYNC_HF_EXPORT" = "true" ]; then
        echo "ERROR: ASYNC_HF_EXPORT=true is not crash-safe for MULTI_STEP>0." >&2
        echo "       Use synchronous exports so completion markers only reference fully committed HF/FSDP artifacts." >&2
        exit 1
    fi
    if [ -n "$CORRECTED_RESPONSES_PATH" ]; then
        echo "ERROR: MULTI_STEP>0 is incompatible with CORRECTED_RESPONSES_PATH legacy override."
        exit 1
    fi
    if [ -n "$PIPELINE_KEEP_INTERVAL" ] && ! [[ "$PIPELINE_KEEP_INTERVAL" =~ ^[0-9]+$ ]]; then
        echo "ERROR: PIPELINE_KEEP_INTERVAL must be a non-negative integer when set (got: $PIPELINE_KEEP_INTERVAL)."
        exit 1
    fi
fi

PIPELINE_TOTAL_STEPS=0
PIPELINE_BATCHES_PER_EPOCH=0
PIPELINE_DROPPED_SAMPLES=0
PIPELINE_FULL_BATCH_OPTIMIZER_STEPS=0
PIPELINE_TAIL_BATCH_SIZE=0
PIPELINE_TAIL_OPTIMIZER_STEPS=0
PIPELINE_TOTAL_OPTIMIZER_STEPS=0
OPTIMIZER_MILESTONE_STEPS=""
MULTISTEP_TAG=""
if [ "$MULTI_STEP" -gt 0 ]; then
    PIPELINE_LENGTH_DATA_PATH="${DATA_PATH:-$TRAIN_DATA_PATH}"
    TOTAL_TRAIN_SAMPLES="$("$PYTHON_BIN" - "$PIPELINE_LENGTH_DATA_PATH" "${MAX_SAMPLES:-}" <<'PYDATASETLEN_EARLY'
import os
import sys
import datasets

path, max_samples = sys.argv[1], sys.argv[2]
if os.path.isfile(path) and path.lower().endswith(".parquet"):
    import pyarrow.parquet as pq

    n = pq.ParquetFile(path).metadata.num_rows
else:
    try:
        ds_loaded = datasets.load_from_disk(path)
        ds_raw = ds_loaded["train"] if isinstance(ds_loaded, datasets.DatasetDict) else ds_loaded
    except (ValueError, FileNotFoundError):
        ds_raw = datasets.load_dataset(path, split="train")
    n = len(ds_raw)
if max_samples:
    n = min(n, int(max_samples))
print(n)
PYDATASETLEN_EARLY
)"
    if [ "$TOTAL_TRAIN_SAMPLES" -lt 1 ]; then
        echo "ERROR: training dataset has no rows: $PIPELINE_LENGTH_DATA_PATH"
        exit 1
    fi
    if [ "$PIPELINE_STEP_MODE" = "explicit_steps" ] && [ "$USER_MULTI_STEP" -gt "$TOTAL_TRAIN_SAMPLES" ]; then
        echo "ERROR: MULTI_STEP=$USER_MULTI_STEP is larger than available samples ($TOTAL_TRAIN_SAMPLES)."
        exit 1
    fi
    if [ "$TOTAL_EPOCHS" -ne 1 ]; then
        echo "ERROR: MULTI_STEP>0 expects TOTAL_EPOCHS=1; use MULTI_STEP as the total number of policy updates."
        exit 1
    fi
    if [ "$PIPELINE_STEP_MODE" = "default_batch" ]; then
        PIPELINE_AUTO_CHUNK_SIZE="$USER_PIPELINE_BATCH_SIZE"
        PIPELINE_MAX_CHUNK_SIZE="$PIPELINE_AUTO_CHUNK_SIZE"
        PIPELINE_TOTAL_STEPS=$(((TOTAL_TRAIN_SAMPLES + PIPELINE_AUTO_CHUNK_SIZE - 1) / PIPELINE_AUTO_CHUNK_SIZE))
        MULTI_STEP="$PIPELINE_TOTAL_STEPS"
        PIPELINE_BATCHES_PER_EPOCH="$PIPELINE_TOTAL_STEPS"
        PIPELINE_DROPPED_SAMPLES=0
        PIPELINE_BALANCED_REMAINDER=0
        if [ -z "$USER_GRADIENT_ACCUMULATION_STEPS" ]; then
            _world_size=$((NNODES * NGPUS_PER_NODE))
            _samples_per_rank=$(((PIPELINE_AUTO_CHUNK_SIZE + _world_size - 1) / _world_size))
            GRADIENT_ACCUMULATION_STEPS=$(((_samples_per_rank + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
            GRADIENT_ACCUMULATION_SOURCE="derived_from_default_batch"
        fi
    else
        PIPELINE_TOTAL_STEPS="$USER_MULTI_STEP"
        MULTI_STEP="$USER_MULTI_STEP"
        PIPELINE_BATCHES_PER_EPOCH="$USER_MULTI_STEP"
        _world_size=$((NNODES * NGPUS_PER_NODE))
        _partition_quantum=$((_world_size * TRAIN_BATCH_SIZE))
        _partition_units=$((TOTAL_TRAIN_SAMPLES / _partition_quantum))
        if [ $((TOTAL_TRAIN_SAMPLES % _partition_quantum)) -eq 0 ] && \
           [ "$_partition_units" -ge "$USER_MULTI_STEP" ]; then
            # Keep canonical partitions divisible by the data-parallel world
            # size so DistributedSampler does not pad them with duplicate rows.
            PIPELINE_AUTO_CHUNK_SIZE=$(((_partition_units / USER_MULTI_STEP) * _partition_quantum))
            PIPELINE_BALANCED_REMAINDER=$((_partition_units % USER_MULTI_STEP))
            PIPELINE_BALANCED_INCREMENT="$_partition_quantum"
        else
            PIPELINE_AUTO_CHUNK_SIZE=$((TOTAL_TRAIN_SAMPLES / USER_MULTI_STEP))
            PIPELINE_BALANCED_REMAINDER=$((TOTAL_TRAIN_SAMPLES % USER_MULTI_STEP))
            PIPELINE_BALANCED_INCREMENT=1
        fi
        PIPELINE_MAX_CHUNK_SIZE="$PIPELINE_AUTO_CHUNK_SIZE"
        if [ "$PIPELINE_BALANCED_REMAINDER" -gt 0 ]; then
            PIPELINE_MAX_CHUNK_SIZE=$((PIPELINE_AUTO_CHUNK_SIZE + PIPELINE_BALANCED_INCREMENT))
        fi
        PIPELINE_DROPPED_SAMPLES=0

        # One balanced data partition is exactly one optimizer step.  Override
        # the 512-prompt default (and any inherited GA) so MULTI_STEP retains
        # its user-facing meaning as total policy optimizer steps.
        _samples_per_rank=$(((PIPELINE_MAX_CHUNK_SIZE + _world_size - 1) / _world_size))
        GRADIENT_ACCUMULATION_STEPS=$(((_samples_per_rank + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
        GRADIENT_ACCUMULATION_SOURCE="derived_from_ms${USER_MULTI_STEP}"
    fi
    if [ "$PIPELINE_AUTO_CHUNK_SIZE" -lt 1 ]; then
        echo "ERROR: computed pipeline chunk size is <1 (samples=$TOTAL_TRAIN_SAMPLES, MULTI_STEP=$MULTI_STEP)."
        exit 1
    fi
    PIPELINE_TAIL_BATCH_SIZE="$PIPELINE_AUTO_CHUNK_SIZE"
    if [ "$PIPELINE_STEP_MODE" = "default_batch" ]; then
        PIPELINE_TAIL_BATCH_SIZE=$((TOTAL_TRAIN_SAMPLES - (PIPELINE_TOTAL_STEPS - 1) * PIPELINE_AUTO_CHUNK_SIZE))
    fi
    PIPELINE_FULL_BATCH_OPTIMIZER_STEPS="$(pipeline_optimizer_steps_for_samples "$PIPELINE_MAX_CHUNK_SIZE")"
    PIPELINE_TAIL_OPTIMIZER_STEPS="$(pipeline_optimizer_steps_for_samples "$PIPELINE_TAIL_BATCH_SIZE")"
    if [ "$PIPELINE_TOTAL_STEPS" -eq 1 ]; then
        PIPELINE_TOTAL_OPTIMIZER_STEPS="$PIPELINE_TAIL_OPTIMIZER_STEPS"
    else
        PIPELINE_TOTAL_OPTIMIZER_STEPS=$((
            (PIPELINE_TOTAL_STEPS - 1) * PIPELINE_FULL_BATCH_OPTIMIZER_STEPS
            + PIPELINE_TAIL_OPTIMIZER_STEPS
        ))
    fi
    if [ -n "$DATA_PATH" ] && [ "$PIPELINE_TOTAL_STEPS" -ne 1 ]; then
        echo "ERROR: DATA_PATH with multi-step training is supported only for one pre-generated update."
        exit 1
    fi
    if [ -z "$PIPELINE_KEEP_INTERVAL" ]; then
        PIPELINE_KEEP_INTERVAL=0
    fi
    PIPELINE_MILESTONE_STEPS="$(
        "$PYTHON_BIN" - "$PIPELINE_TOTAL_STEPS" "$EVAL_FRACTIONS" <<'PYMILESTONES'
import sys

from recipe.opd.experiment_tracking import compute_eval_milestones

total_steps = int(sys.argv[1])
fractions = sys.argv[2]
print(",".join(str(step) for step in compute_eval_milestones(total_steps, fractions)))
PYMILESTONES
    )"
    OPTIMIZER_MILESTONE_STEPS="$(
        "$PYTHON_BIN" - "$PIPELINE_TOTAL_OPTIMIZER_STEPS" "$EVAL_FRACTIONS" <<'PYOPTIMIZERMILESTONES'
import sys

from recipe.opd.experiment_tracking import compute_eval_milestones

total_steps = int(sys.argv[1])
fractions = sys.argv[2]
print(",".join(str(step) for step in compute_eval_milestones(total_steps, fractions)))
PYOPTIMIZERMILESTONES
    )"
    if [ -n "$USER_PIPELINE_KEEP_STEPS" ] && [ "$USER_PIPELINE_KEEP_STEPS" != "$PIPELINE_MILESTONE_STEPS" ]; then
        echo "WARNING: overriding PIPELINE_KEEP_STEPS=$USER_PIPELINE_KEEP_STEPS with eval milestones $PIPELINE_MILESTONE_STEPS" >&2
    fi
    PIPELINE_KEEP_STEPS="$PIPELINE_MILESTONE_STEPS"
    PIPELINE_KEEP_INTERVAL=0
    # A canonical ms=1 run has one rollout/training round but many optimizer
    # steps.  Preserve its four optimizer milestone checkpoints until the
    # post-training evaluator has consumed them.
    if [ "$PIPELINE_TOTAL_STEPS" -eq 1 ] && [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
        OPTIMIZER_MILESTONE_COUNT="$(awk -F, '{print NF}' <<< "$OPTIMIZER_MILESTONE_STEPS")"
        if [ "$KEEP_LAST_N_CHECKPOINTS" -lt "$OPTIMIZER_MILESTONE_COUNT" ]; then
            KEEP_LAST_N_CHECKPOINTS="$OPTIMIZER_MILESTONE_COUNT"
        fi
    fi
    MULTISTEP_TAG="ms${PIPELINE_TOTAL_STEPS}"
    EXPERIMENT_TAG="${EXPERIMENT_TAG}_${MULTISTEP_TAG}"
fi

if [ -n "$PRECOMPUTED_Y_O_TRAJECTORY_PATH" ]; then
    PRECOMPUTED_Y_O_TRAJECTORY_ROWS="$(
        "$PYTHON_BIN" - "$PRECOMPUTED_Y_O_TRAJECTORY_PATH" <<'PYTRAJECTORYLEN'
import sys
import pyarrow.parquet as pq

print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PYTRAJECTORYLEN
    )"
    REQUIRED_PRECOMPUTED_ROWS="${TOTAL_TRAIN_SAMPLES:-${MAX_SAMPLES:-0}}"
    if [ "$REQUIRED_PRECOMPUTED_ROWS" -gt 0 ] && \
       [ "$PRECOMPUTED_Y_O_TRAJECTORY_ROWS" -lt "$REQUIRED_PRECOMPUTED_ROWS" ]; then
        echo "ERROR: precomputed trajectory has $PRECOMPUTED_Y_O_TRAJECTORY_ROWS rows, but training requires at least $REQUIRED_PRECOMPUTED_ROWS." >&2
        exit 1
    fi
fi

if ! [[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GRADIENT_ACCUMULATION_STEPS must be a positive integer (got: $GRADIENT_ACCUMULATION_STEPS)."
    exit 1
fi

# Base directories and run naming
DISTILL_FAMILY="$(echo "$DISTILL_MODE" | tr '[:lower:]' '[:upper:]')"
TASK_UPPER="$(echo "$TASK" | tr '[:lower:]' '[:upper:]')"
DISTILL_TASK_FAMILY="${DISTILL_FAMILY}_${TASK_UPPER}"
TASK_PATH_SUFFIX="/$TASK"
TASK_FILE_SUFFIX="_${TASK}"
TASK_RESULT_SUFFIX="_${TASK_UPPER}"
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d-%H%M%S)}
OPTIMIZATION_STEP_TAG="${MULTISTEP_TAG:-ms1}"
case "$Y_O_ROLLOUT_MODE" in
    student) Y_O_ROLLOUT_TAG="" ;;
    teacher)
        case "$TEACHER_TRAJECTORY_CONDITIONING" in
            pi_T_x_only_v1)
                Y_O_ROLLOUT_TAG="_y_t_$(sanitize_path_component "$TRAJECTORY_MODEL_NAME")"
                ;;
            opsd_x_y_star_expert_rewrite_v1)
                Y_O_ROLLOUT_TAG="_y_t_expert_rewrite_$(sanitize_path_component "$TRAJECTORY_MODEL_NAME")"
                ;;
        esac
        ;;
    expert) Y_O_ROLLOUT_TAG="_y_star" ;;
    skd|skd_vllm) Y_O_ROLLOUT_TAG="_skd" ;;
    skd_vllm_internal) Y_O_ROLLOUT_TAG="_skd_internal" ;;
esac
RUN_DESCRIPTOR="${PROMPT_MODE_TAG}${Y_O_ROLLOUT_TAG}_kl_${KL_TYPE}_${KL_METHOD}_${CLIP_TAG}${BETA_TAG}${TOPK_TAG}${Y_O_FILTER_TAG}_${TEACHER_TRAINING_PROMPT}${TEACHER_THINKING_FILE_SUFFIX}_${OPTIMIZATION_STEP_TAG}_${RUN_DATE}"
if [ "$DISTILL_MODE" = "opd" ]; then
    MODEL_RUN_NAME="teacher${TEACHER_MODEL_NAME}_${RUN_DESCRIPTOR}"
else
    MODEL_RUN_NAME="$RUN_DESCRIPTOR"
fi

case "$USE_LORA" in
    true|True|1|yes|Yes|y|Y)
        TUNING_MODE_TAG="lora"
        RESULTS_TUNING_SUFFIX="_LORA"
        SIGNATURE_LORA_RANK="$LORA_RANK"
        SIGNATURE_LORA_ALPHA="$LORA_ALPHA"
        ;;
    *)
        TUNING_MODE_TAG="nolora"
        RESULTS_TUNING_SUFFIX="_NO_LORA"
        SIGNATURE_LORA_RANK="0"
        SIGNATURE_LORA_ALPHA="0"
        ;;
esac
RESULTS_MODEL_KEY="${RESULTS_MODEL_KEY:-${MODEL_NAME}_${DISTILL_FAMILY}${TASK_RESULT_SUFFIX}_${MODEL_RUN_NAME}${RESULTS_TUNING_SUFFIX}}"
RESULTS_BASE_MODEL_NAME="${DISTILL_FAMILY}${TASK_PATH_SUFFIX}/${MODEL_NAME}"
RESULTS_FILE="${RESULTS_FILE:-results/${DISTILL_FAMILY}${TASK_PATH_SUFFIX}/${MODEL_NAME}${TASK_FILE_SUFFIX}${TEACHER_THINKING_FILE_SUFFIX}.json}"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_BASE_DIR="outputs/${DISTILL_FAMILY}${TASK_PATH_SUFFIX}/${MODEL_NAME}/${MODEL_RUN_NAME}"
else
    OUTPUT_BASE_DIR="$OUTPUT_DIR"
fi

MODEL_SAVE_ROOT="${MODEL_SAVE_DIR:-model/trained}"
MODEL_SAVE_BASE_DIR="$MODEL_SAVE_ROOT/${DISTILL_TASK_FAMILY}/${MODEL_NAME}/${MODEL_RUN_NAME}"
if [ -z "$PIPELINE_ARCHIVE_MODEL_DIR" ]; then
    PIPELINE_ARCHIVE_MODEL_DIR="$PIPELINE_ARCHIVE_MODEL_ROOT/${DISTILL_TASK_FAMILY}/${MODEL_NAME}"
fi

RESIDENT_YO_MANIFEST_USER_VALUE="$RESIDENT_YO_MANIFEST"

# gen_results uses a short random run id by default. In fresh mode the id is
# intentionally date-free (gen_uid-*) because semantic identity is carried by
# the signature below, not by RUN_DATE. In resume_matching mode we scan existing
# gen_results metadata for the same signature, switch back to that run's
# model_save_base_dir/gen_results_base_dir, and continue from its checkpoints.
GEN_RESULTS_ROOT="${GEN_RESULTS_ROOT:-gen_results}"
GEN_RESULTS_STUDENT_TAG="$(sanitize_path_component "$MODEL_NAME")"
GEN_RESULTS_TEACHER_TAG="$(sanitize_path_component "$TEACHER_MODEL_NAME")"
GEN_RESULTS_MAX_SAMPLES_TAG="${MAX_SAMPLES:-all}"
[ -z "$GEN_RESULTS_MAX_SAMPLES_TAG" ] && GEN_RESULTS_MAX_SAMPLES_TAG="all"
GEN_RESULTS_MAX_SAMPLES_TAG="$(sanitize_path_component "$GEN_RESULTS_MAX_SAMPLES_TAG")"
GEN_RESULTS_YO_FILTER_TAG="${Y_O_FILTER_TAG#_}"
[ -z "$GEN_RESULTS_YO_FILTER_TAG" ] && GEN_RESULTS_YO_FILTER_TAG="all"
GEN_RESULTS_STAGE2_FILTER_TAG="s2filter-${FORWARD_FILTER_STAGE2}"
if [ "$FORWARD_FILTER_STAGE2" = "true" ]; then
    GEN_RESULTS_STAGE2_FILTER_TAG="s2filter-thr${FORWARD_FILTER_THRESHOLD}-s1fail${FORWARD_FILTER_REQUIRE_STAGE1_FAILED}"
fi

gen_results_metadata_value() {
    local file="$1"
    local key="$2"
    awk -v key="$key" '
        index($0, key ":") == 1 {
            sub(/^[^:]*:[[:space:]]*/, "")
            print
            exit
        }
    ' "$file"
}

find_matching_gen_results_metadata() {
    local signature="$1"
    local best=""
    local best_mtime=0
    local best_completed=0
    local meta sig mtime details_file details_sig model_base completed_marker candidate_completed

    [ -d "$GEN_RESULTS_ROOT" ] || return 0
    while IFS= read -r meta; do
        sig="$(gen_results_metadata_value "$meta" gen_results_signature)"
        if [ "$sig" != "$signature" ]; then
            # Compatibility: MAX_LENGTH controls KL training truncation, not the
            # generated data. Older runs persisted it in the semantic signature;
            # compare normalized signature details with max_length removed.
            details_file="$(gen_results_metadata_value "$meta" gen_results_signature_details_file)"
            [ -s "$details_file" ] || details_file="$(dirname "$meta")/run_signature.txt"
            if [ -s "$details_file" ]; then
                details_sig="$(grep -v '^max_length=' "$details_file" | sha256sum | awk '{print $1}')"
            else
                details_sig=""
            fi
            [ "$details_sig" = "$signature" ] || continue
        fi
        mtime="$(stat -c %Y "$meta" 2>/dev/null || printf '0')"
        model_base="$(gen_results_metadata_value "$meta" model_save_base_dir)"
        completed_marker="$model_base/epoch1/ms${MULTI_STEP}/batch00001/hf_merged"
        candidate_completed=0
        if [ -n "$model_base" ] && hf_export_complete "$completed_marker"; then
            candidate_completed=1
        fi
        if [ -z "$best" ] || \
           [ "$candidate_completed" -gt "$best_completed" ] || \
           { [ "$candidate_completed" -eq "$best_completed" ] && [ "$mtime" -gt "$best_mtime" ]; }; then
            best="$meta"
            best_mtime="$mtime"
            best_completed="$candidate_completed"
        fi
    done < <(find "$GEN_RESULTS_ROOT" -mindepth 2 -maxdepth 3 -type f -name run_metadata.yaml 2>/dev/null)

    [ -n "$best" ] && printf '%s\n' "$best"
}

compute_gen_results_signature() {
    # Guard resume against accidentally reusing a gen_results tree for a different
    # algorithm/data target. Operational knobs used for recovery, such as
    # TRAIN_BATCH_SIZE and KL_FULL_VOCAB_CHUNK_SIZE, are intentionally excluded.
    local teacher_trajectory_signature_content=""
    local skd_rollout_signature_content=""
    if [ "$Y_O_ROLLOUT_MODE" = "teacher" ]; then
        teacher_trajectory_signature_content="
teacher_trajectory_conditioning=$TEACHER_TRAJECTORY_CONDITIONING
teacher_trajectory_prompt_path=$TEACHER_TRAJECTORY_PROMPT_PATH
teacher_trajectory_prompt_length=$TEACHER_TRAJECTORY_PROMPT_LENGTH
teacher_trajectory_rollout_max_model_len=$TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN"
    fi
    case "$Y_O_ROLLOUT_MODE" in
        skd|skd_vllm|skd_vllm_internal)
            skd_rollout_signature_content="
skd_gamma=$SKD_GAMMA
skd_accept_top_k=$SKD_ACCEPT_TOP_K
skd_accept_top_p=$SKD_ACCEPT_TOP_P
skd_student_temperature=$SKD_STUDENT_TEMPERATURE
skd_student_top_p=$SKD_STUDENT_TOP_P
skd_teacher_temperature=$SKD_TEACHER_TEMPERATURE
skd_teacher_top_p=$SKD_TEACHER_TOP_P
skd_rollout_batch_size=$SKD_ROLLOUT_BATCH_SIZE
skd_pipeline_lanes=$SKD_PIPELINE_LANES
skd_parallel_student_teacher=$SKD_PARALLEL_STUDENT_TEACHER
skd_teacher_prompt_contract=$SKD_TEACHER_PROMPT_CONTRACT
skd_teacher_prompt_length=$SKD_TEACHER_PROMPT_LENGTH
skd_rollout_max_model_len=$SKD_ROLLOUT_MAX_MODEL_LEN"
            ;;
    esac
    GEN_RESULTS_RUN_SIGNATURE_CONTENT="$(cat <<EOF
task=$TASK
distill_family=$DISTILL_FAMILY
distill_mode=$DISTILL_MODE
model_name=$MODEL_NAME
model_path=$MODEL_PATH
teacher_model_name=$TEACHER_MODEL_NAME
teacher_model_path=${TEACHER_MODEL_PATH:-$MODEL_PATH}
teacher_enable_thinking=$TEACHER_ENABLE_THINKING
teacher_thinking_name_tag=$TEACHER_THINKING_NAME_TAG
teacher_supervision_render_mode=$TEACHER_SUPERVISION_RENDER_MODE
teacher_rollout_render_mode=$TEACHER_ROLLOUT_RENDER_MODE
teacher_prompt_render_contract=$TEACHER_PROMPT_RENDER_CONTRACT
teacher_chat_template_token_buffer=$TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER
tuning_mode=$TUNING_MODE_TAG
use_lora=$USE_LORA
lora_rank=$SIGNATURE_LORA_RANK
lora_alpha=$SIGNATURE_LORA_ALPHA
prompt_mode=$PROMPT_MODE_TAG
prompt_contract_version=$PROMPT_CONTRACT_VERSION
y_mode=$Y_MODE
y_o_rollout_mode=$Y_O_ROLLOUT_MODE
trajectory_model_name=${TRAJECTORY_MODEL_NAME:-}
trajectory_model_path=${TRAJECTORY_MODEL_PATH:-}${teacher_trajectory_signature_content}${skd_rollout_signature_content}
precomputed_y_o_trajectory_path=${PRECOMPUTED_Y_O_TRAJECTORY_PATH:-}
kl_type=$KL_TYPE
kl_method=$KL_METHOD
teacher_training_prompt=$TEACHER_TRAINING_PROMPT
use_initial_response=$USE_INITIAL_RESPONSE
kl_token_clip=$KL_TOKEN_CLIP
beta=$BETA
top_k=$TOP_K
forward_stage2_mode=$FORWARD_STAGE2_MODE
forward_filter_stage2=$FORWARD_FILTER_STAGE2
forward_filter_threshold=$FORWARD_FILTER_THRESHOLD
forward_filter_require_stage1_failed=$FORWARD_FILTER_REQUIRE_STAGE1_FAILED
multi_step=$MULTI_STEP
pipeline_step_mode=$PIPELINE_STEP_MODE
pipeline_auto_chunk_size=$PIPELINE_AUTO_CHUNK_SIZE
pipeline_max_chunk_size=$PIPELINE_MAX_CHUNK_SIZE
pipeline_balanced_remainder=$PIPELINE_BALANCED_REMAINDER
pipeline_balanced_increment=$PIPELINE_BALANCED_INCREMENT
pipeline_total_steps=${PIPELINE_TOTAL_STEPS:-0}
pipeline_batches_per_epoch=${PIPELINE_BATCHES_PER_EPOCH:-$TOTAL_EPOCHS}
resident_student_rollout=$RESIDENT_STUDENT_ROLLOUT
pipeline_ephemeral_models=$PIPELINE_EPHEMERAL_MODELS
base_prompt_length=$BASE_PROMPT_LENGTH
max_prompt_length=$MAX_PROMPT_LENGTH
expert_solution_prompt_length=$EXPERT_SOLUTION_PROMPT_LENGTH
stage2_prompt_length=$STAGE2_PROMPT_LENGTH
max_response_length=$MAX_RESPONSE_LENGTH
stage1_rollout_max_model_len=$STAGE1_ROLLOUT_MAX_MODEL_LEN
stage2_rollout_max_model_len=$STAGE2_ROLLOUT_MAX_MODEL_LEN
rollout_chat_template_token_buffer=$ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER
rollout_max_num_seqs=$ROLLOUT_MAX_NUM_SEQS
stage1_rollout_max_num_batched_tokens=$STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS
stage2_rollout_max_num_batched_tokens=$STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS
temperature=$TEMPERATURE
rollout_temperature=$ROLLOUT_TEMPERATURE
rollout_top_p=$ROLLOUT_TOP_P
rollout_top_k=$ROLLOUT_TOP_K
learning_rate=$LEARNING_RATE
warmup_ratio=$WARMUP_RATIO
weight_decay=$WEIGHT_DECAY
train_batch_size=$TRAIN_BATCH_SIZE
micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU
use_dynamic_bsz=$USE_DYNAMIC_BSZ
gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS
train_epochs_per_round=$TRAIN_EPOCHS_PER_ROUND
model_context_length=$MODEL_CONTEXT_LENGTH
max_samples=${MAX_SAMPLES:-all}
data_path=${DATA_PATH:-}
train_data_path=$TRAIN_DATA_PATH
corrected_responses_path=${CORRECTED_RESPONSES_PATH:-}
EOF
)"
    GEN_RESULTS_RUN_SIGNATURE="$(printf '%s\n' "$GEN_RESULTS_RUN_SIGNATURE_CONTENT" | sha256sum | awk '{print $1}')"
}

compute_gen_results_signature

case "${KL_CONFIG_DRY_RUN:-false}" in
    true)
        echo "KL configuration dry run"
        echo "  task:                $TASK"
        echo "  distill mode:        $DISTILL_MODE"
        echo "  y mode:              $Y_MODE"
        echo "  rollout mode:        $Y_O_ROLLOUT_MODE"
        echo "  train data:          $TRAIN_DATA_PATH"
        echo "  train samples:       ${TOTAL_TRAIN_SAMPLES:-unknown}"
        echo "  step mode:           $PIPELINE_STEP_MODE"
        echo "  pipeline batch size: ${PIPELINE_AUTO_CHUNK_SIZE:-0}"
        echo "  pipeline batch max:  ${PIPELINE_MAX_CHUNK_SIZE:-0}"
        echo "  larger partitions:  ${PIPELINE_BALANCED_REMAINDER:-0} (increment=${PIPELINE_BALANCED_INCREMENT:-1})"
        echo "  pipeline updates:    ${PIPELINE_TOTAL_STEPS:-0}"
        if [ "${PIPELINE_TOTAL_STEPS:-0}" -gt 0 ]; then
            echo "  pipeline tail size:  $PIPELINE_TAIL_BATCH_SIZE"
            echo "  optimizer steps:     $PIPELINE_TOTAL_OPTIMIZER_STEPS (full chunk=$PIPELINE_FULL_BATCH_OPTIMIZER_STEPS, tail=$PIPELINE_TAIL_OPTIMIZER_STEPS)"
        fi
        echo "  gradient accum:      $GRADIENT_ACCUMULATION_STEPS ($GRADIENT_ACCUMULATION_SOURCE)"
        echo "  pipeline milestones: ${PIPELINE_KEEP_STEPS:-none}"
        echo "  optimizer eval ckpt: ${OPTIMIZER_MILESTONE_STEPS:-none}"
        echo "  checkpoint policy:   $PIPELINE_LOCAL_KEEP_POLICY"
        echo "  ephemeral models:    $PIPELINE_EPHEMERAL_MODELS"
        echo "  resident rollout:    $RESIDENT_STUDENT_ROLLOUT"
        echo "  eval fractions:      $EVAL_FRACTIONS"
        echo "  eval root:           $EVAL_DATASETS_DIR"
        echo "  hf cache:            $HF_HOME"
        echo "  eval datasets:       $EVAL_DATASETS"
        echo "  eval pass_k:         $PASS_K"
        echo "  student model:       $MODEL_PATH"
        echo "  teacher model:       ${TEACHER_MODEL_PATH:-$MODEL_PATH}"
        if [ "$DISTILL_MODE" = "opd" ]; then
            echo "  teacher thinking:    $TEACHER_ENABLE_THINKING"
        fi
        echo "  teacher supervision: $TEACHER_SUPERVISION_RENDER_MODE"
        echo "  teacher rollout:     $TEACHER_ROLLOUT_RENDER_MODE"
        echo "  teacher prompt/max:  $MAX_PROMPT_LENGTH/$MAX_LENGTH (rollout chat buffer=$TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER)"
        if [[ "$Y_O_ROLLOUT_MODE" == skd* ]]; then
            echo "  skd teacher prompt:  $SKD_TEACHER_PROMPT_CONTRACT"
            echo "  skd teacher length:  $SKD_TEACHER_PROMPT_LENGTH"
            echo "  skd max model len:   $SKD_ROLLOUT_MAX_MODEL_LEN"
        fi
        echo "  run name:            $MODEL_RUN_NAME"
        echo "  output dir:          $OUTPUT_BASE_DIR"
        echo "  model dir:           $MODEL_SAVE_BASE_DIR"
        echo "  results key:         $RESULTS_MODEL_KEY"
        echo "  results file:        $RESULTS_FILE"
        echo "  gen results prefix:  gen_${TASK}_${TEACHER_THINKING_RUN_SEGMENT}${OPTIMIZATION_STEP_TAG:-ms1}"
        exit 0
        ;;
    false) ;;
    *) echo "ERROR: KL_CONFIG_DRY_RUN must be true or false" >&2; exit 1 ;;
esac

GEN_RESULTS_RUN_ID_FILE_USER_VALUE="${GEN_RESULTS_RUN_ID_FILE:-}"
GEN_RESULTS_RUN_SIGNATURE_FILE_USER_VALUE="${GEN_RESULTS_RUN_SIGNATURE_FILE:-}"
GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE_USER_VALUE="${GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE:-}"
GEN_RESULTS_ENV_RUN_ID="${GEN_RESULTS_RUN_ID:-}"
[ -n "$GEN_RESULTS_ENV_RUN_ID" ] && GEN_RESULTS_ENV_RUN_ID="$(sanitize_path_component "$GEN_RESULTS_ENV_RUN_ID")"
GEN_RESULTS_ENV_BASE_DIR="${GEN_RESULTS_BASE_DIR:-}"
GEN_RESULTS_ENV_BASE_ID=""
if [ -n "$GEN_RESULTS_ENV_BASE_DIR" ]; then
    GEN_RESULTS_ENV_BASE_ID="$(sanitize_path_component "${GEN_RESULTS_ENV_BASE_DIR##*/}")"
fi

if [ "$PIPELINE_RESUME_MODE" = "resume_matching" ] && [ -z "$GEN_RESULTS_ENV_RUN_ID" ] && [ -z "$GEN_RESULTS_ENV_BASE_DIR" ]; then
    MATCHED_GEN_RESULTS_METADATA="$(find_matching_gen_results_metadata "$GEN_RESULTS_RUN_SIGNATURE" || true)"
    if [ -n "$MATCHED_GEN_RESULTS_METADATA" ]; then
        MATCHED_GEN_RESULTS_BASE_DIR="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" gen_results_base_dir)"
        [ -z "$MATCHED_GEN_RESULTS_BASE_DIR" ] && MATCHED_GEN_RESULTS_BASE_DIR="$(dirname "$MATCHED_GEN_RESULTS_METADATA")"
        MATCHED_GEN_RESULTS_RUN_ID="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" gen_results_run_id)"
        [ -z "$MATCHED_GEN_RESULTS_RUN_ID" ] && MATCHED_GEN_RESULTS_RUN_ID="$(sanitize_path_component "${MATCHED_GEN_RESULTS_BASE_DIR##*/}")"
        MATCHED_MODEL_SAVE_BASE_DIR="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" model_save_base_dir)"
        MATCHED_OUTPUT_BASE_DIR="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" output_base_dir)"
        MATCHED_MODEL_RUN_NAME="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" model_run_name)"
        MATCHED_RESULTS_MODEL_KEY="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" results_model_key)"
        MATCHED_RESULTS_FILE="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" results_file)"
        MATCHED_RUN_DESCRIPTOR="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" run_descriptor)"
        MATCHED_WANDB_RUN_ID="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" wandb_run_id)"
        MATCHED_WANDB_RUN_ID_FILE="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" wandb_run_id_file)"
        MATCHED_WANDB_RUN_IDENTITY="$(gen_results_metadata_value "$MATCHED_GEN_RESULTS_METADATA" wandb_run_identity)"

        if [ -z "$MATCHED_MODEL_SAVE_BASE_DIR" ]; then
            echo "WARNING: matched gen_results metadata has no model_save_base_dir: $MATCHED_GEN_RESULTS_METADATA" >&2
            echo "         Starting a fresh run instead." >&2
        else
            echo ""
            echo "############################################################"
            echo "# OPD PIPELINE EFFECTIVE START MODE: RESUME_MATCHING"
            echo "# Matched an existing run; continuing from completed markers."
            echo "############################################################"
            echo "  metadata:    $MATCHED_GEN_RESULTS_METADATA"
            echo "  signature:   $GEN_RESULTS_RUN_SIGNATURE"
            echo "  model save:  $MATCHED_MODEL_SAVE_BASE_DIR"
            echo "  gen results: $MATCHED_GEN_RESULTS_BASE_DIR"
            echo "############################################################"
            MODEL_SAVE_BASE_DIR="$MATCHED_MODEL_SAVE_BASE_DIR"
            [ -n "$MATCHED_OUTPUT_BASE_DIR" ] && OUTPUT_BASE_DIR="$MATCHED_OUTPUT_BASE_DIR"
            [ -n "$MATCHED_MODEL_RUN_NAME" ] && MODEL_RUN_NAME="$MATCHED_MODEL_RUN_NAME"
            [ -n "$MATCHED_RESULTS_MODEL_KEY" ] && RESULTS_MODEL_KEY="$MATCHED_RESULTS_MODEL_KEY"
            [ -n "$MATCHED_RESULTS_FILE" ] && RESULTS_FILE="$MATCHED_RESULTS_FILE"
            [ -n "$MATCHED_RUN_DESCRIPTOR" ] && RUN_DESCRIPTOR="$MATCHED_RUN_DESCRIPTOR"
            [ -n "$MATCHED_WANDB_RUN_ID" ] && WANDB_RUN_ID="$MATCHED_WANDB_RUN_ID"
            [ -n "$MATCHED_WANDB_RUN_ID_FILE" ] && WANDB_RUN_ID_FILE="$MATCHED_WANDB_RUN_ID_FILE"
            [ -n "$MATCHED_WANDB_RUN_IDENTITY" ] && WANDB_RUN_IDENTITY="$MATCHED_WANDB_RUN_IDENTITY"
            GEN_RESULTS_ENV_RUN_ID="$MATCHED_GEN_RESULTS_RUN_ID"
            GEN_RESULTS_ENV_BASE_DIR="$MATCHED_GEN_RESULTS_BASE_DIR"
            GEN_RESULTS_ENV_BASE_ID="$(sanitize_path_component "${GEN_RESULTS_ENV_BASE_DIR##*/}")"
        fi
    else
        echo ""
        echo "############################################################"
        echo "# OPD PIPELINE EFFECTIVE START MODE: FRESH/REFRESH"
        echo "# PIPELINE_RESUME_MODE=resume_matching, but no matching gen_results run was found."
        echo "# Starting a new run."
        echo "############################################################"
        echo ""
    fi
elif [ "$PIPELINE_RESUME_MODE" = "fresh" ]; then
    echo ""
    echo "############################################################"
    echo "# OPD PIPELINE EFFECTIVE START MODE: FRESH/REFRESH"
    echo "# PIPELINE_RESUME_MODE=fresh; matching previous runs was skipped."
    echo "############################################################"
    echo ""
fi

GEN_RESULTS_RUN_PREFIX_DEFAULT="gen_${TASK}_${TEACHER_THINKING_RUN_SEGMENT}${OPTIMIZATION_STEP_TAG:-ms1}"
GEN_RESULTS_RUN_PREFIX="$(sanitize_path_component "${GEN_RESULTS_RUN_PREFIX:-$GEN_RESULTS_RUN_PREFIX_DEFAULT}")"
GEN_RESULTS_RUN_ID_FILE="${GEN_RESULTS_RUN_ID_FILE_USER_VALUE:-$MODEL_SAVE_BASE_DIR/gen_results_run_id.txt}"

if [ -s "$GEN_RESULTS_RUN_ID_FILE" ]; then
    GEN_RESULTS_PERSISTED_RUN_ID="$(tr -d '[:space:]' < "$GEN_RESULTS_RUN_ID_FILE")"
    if [ -n "$GEN_RESULTS_ENV_RUN_ID" ] && [ "$GEN_RESULTS_ENV_RUN_ID" != "$GEN_RESULTS_PERSISTED_RUN_ID" ]; then
        echo "ERROR: GEN_RESULTS_RUN_ID=$GEN_RESULTS_ENV_RUN_ID conflicts with persisted run id $GEN_RESULTS_PERSISTED_RUN_ID" >&2
        echo "       Persisted file: $GEN_RESULTS_RUN_ID_FILE" >&2
        echo "       Use PIPELINE_RESUME_MODE=resume_matching to find and continue a matching run, or use fresh output dirs." >&2
        exit 1
    fi
    if [ -n "$GEN_RESULTS_ENV_BASE_ID" ] && [ "$GEN_RESULTS_ENV_BASE_ID" != "$GEN_RESULTS_PERSISTED_RUN_ID" ]; then
        echo "ERROR: GEN_RESULTS_BASE_DIR=$GEN_RESULTS_ENV_BASE_DIR conflicts with persisted run id $GEN_RESULTS_PERSISTED_RUN_ID" >&2
        echo "       Persisted file: $GEN_RESULTS_RUN_ID_FILE" >&2
        echo "       Use the persisted gen_results directory to resume, or use fresh output dirs." >&2
        exit 1
    fi
    GEN_RESULTS_RUN_ID="$GEN_RESULTS_PERSISTED_RUN_ID"
elif [ -n "$GEN_RESULTS_ENV_RUN_ID" ]; then
    GEN_RESULTS_RUN_ID="$GEN_RESULTS_ENV_RUN_ID"
    if [ -n "$GEN_RESULTS_ENV_BASE_ID" ] && [ "$GEN_RESULTS_ENV_BASE_ID" != "$GEN_RESULTS_RUN_ID" ]; then
        echo "ERROR: GEN_RESULTS_BASE_DIR=$GEN_RESULTS_ENV_BASE_DIR conflicts with GEN_RESULTS_RUN_ID=$GEN_RESULTS_RUN_ID" >&2
        echo "       GEN_RESULTS_BASE_DIR must end with the same run id." >&2
        exit 1
    fi
elif [ -n "$GEN_RESULTS_ENV_BASE_DIR" ]; then
    GEN_RESULTS_RUN_ID="$GEN_RESULTS_ENV_BASE_ID"
else
    GEN_RESULTS_RANDOM_UID="$("$PYTHON_BIN" - <<'PYGENUID'
import secrets
print(secrets.token_hex(12))
PYGENUID
)"
    GEN_RESULTS_RUN_ID="${GEN_RESULTS_RUN_PREFIX}_uid-${GEN_RESULTS_RANDOM_UID}"
fi
GEN_RESULTS_RUN_ID="$(sanitize_path_component "$GEN_RESULTS_RUN_ID")"
mkdir -p "$(dirname "$GEN_RESULTS_RUN_ID_FILE")"
printf '%s\n' "$GEN_RESULTS_RUN_ID" > "$GEN_RESULTS_RUN_ID_FILE"
GEN_RESULTS_BASE_DIR="${GEN_RESULTS_ENV_BASE_DIR:-$GEN_RESULTS_ROOT/$GEN_RESULTS_RUN_ID}"
GEN_RESULTS_METADATA_FILE="$GEN_RESULTS_BASE_DIR/run_metadata.yaml"
GEN_RESULTS_RUN_SIGNATURE_FILE="${GEN_RESULTS_RUN_SIGNATURE_FILE_USER_VALUE:-$MODEL_SAVE_BASE_DIR/gen_results_run_signature.sha256}"
GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE="${GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE_USER_VALUE:-$MODEL_SAVE_BASE_DIR/gen_results_run_signature.txt}"
GEN_RESULTS_LOCAL_SIGNATURE_FILE="$GEN_RESULTS_BASE_DIR/run_signature.sha256"
GEN_RESULTS_LOCAL_SIGNATURE_DETAILS_FILE="$GEN_RESULTS_BASE_DIR/run_signature.txt"
TASK_INTERMEDIATE_PREFIX="${TASK}_stage"
FULL_STAGE1_PROMPTS="$GEN_RESULTS_BASE_DIR/${TASK_INTERMEDIATE_PREFIX}1_prompts.parquet"

if [ -s "$GEN_RESULTS_RUN_SIGNATURE_FILE" ]; then
    GEN_RESULTS_PERSISTED_SIGNATURE="$(tr -d '[:space:]' < "$GEN_RESULTS_RUN_SIGNATURE_FILE")"
    GEN_RESULTS_PERSISTED_SIGNATURE_COMPAT=""
    if [ "$GEN_RESULTS_PERSISTED_SIGNATURE" != "$GEN_RESULTS_RUN_SIGNATURE" ]; then
        if [ -s "$GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE" ]; then
            GEN_RESULTS_PERSISTED_SIGNATURE_COMPAT="$(grep -v '^max_length=' "$GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE" | sha256sum | awk '{print $1}')"
        elif [ -s "$GEN_RESULTS_LOCAL_SIGNATURE_DETAILS_FILE" ]; then
            GEN_RESULTS_PERSISTED_SIGNATURE_COMPAT="$(grep -v '^max_length=' "$GEN_RESULTS_LOCAL_SIGNATURE_DETAILS_FILE" | sha256sum | awk '{print $1}')"
        fi
    fi
    if [ "$GEN_RESULTS_PERSISTED_SIGNATURE" != "$GEN_RESULTS_RUN_SIGNATURE" ] && \
       [ "$GEN_RESULTS_PERSISTED_SIGNATURE_COMPAT" != "$GEN_RESULTS_RUN_SIGNATURE" ] && \
       [ "${GEN_RESULTS_ALLOW_CONFIG_MISMATCH:-false}" != "true" ]; then
        echo "ERROR: gen_results semantic signature mismatch for resume." >&2
        echo "       Persisted file: $GEN_RESULTS_RUN_SIGNATURE_FILE" >&2
        echo "       Persisted details: $GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE" >&2
        echo "       Current signature: $GEN_RESULTS_RUN_SIGNATURE" >&2
        echo "       Persisted signature: $GEN_RESULTS_PERSISTED_SIGNATURE" >&2
        echo "       Use PIPELINE_RESUME_MODE=fresh with fresh output dirs, or set GEN_RESULTS_ALLOW_CONFIG_MISMATCH=true only if intentional." >&2
        exit 1
    fi
fi
mkdir -p "$(dirname "$GEN_RESULTS_RUN_SIGNATURE_FILE")"
mkdir -p "$GEN_RESULTS_BASE_DIR"
printf '%s\n' "$GEN_RESULTS_RUN_SIGNATURE" > "$GEN_RESULTS_RUN_SIGNATURE_FILE"
printf '%s\n' "$GEN_RESULTS_RUN_SIGNATURE_CONTENT" > "$GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE"
printf '%s\n' "$GEN_RESULTS_RUN_SIGNATURE" > "$GEN_RESULTS_LOCAL_SIGNATURE_FILE"
printf '%s\n' "$GEN_RESULTS_RUN_SIGNATURE_CONTENT" > "$GEN_RESULTS_LOCAL_SIGNATURE_DETAILS_FILE"

if [ -z "$RESIDENT_YO_MANIFEST_USER_VALUE" ]; then
    RESIDENT_YO_MANIFEST="$MODEL_SAVE_BASE_DIR/resident_y_o/manifest.json"
fi
RESIDENT_YO_LOG_DIR="$MODEL_SAVE_BASE_DIR/resident_y_o"
RESIDENT_YO_PID_FILE="$RESIDENT_YO_LOG_DIR/server.pid"
RESIDENT_YO_SYNC_MARKER="$RESIDENT_YO_LOG_DIR/current_weights.env"
RESIDENT_YO_RAY_NAMESPACE="${RESIDENT_YO_RAY_NAMESPACE:-opd_resident_y_o}"
if [ -z "$PIPELINE_TEMP_MODEL_DIR" ]; then
    if [ -n "${OPD_JOB_WORK_DIR:-}" ]; then
        PIPELINE_TEMP_MODEL_DIR="$OPD_JOB_WORK_DIR/pipeline_tmp_checkpoints"
    else
        PIPELINE_TEMP_MODEL_DIR="$OUTPUT_BASE_DIR/pipeline_tmp_checkpoints"
    fi
fi

if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME_BASE="$RESULTS_MODEL_KEY"
else
    WANDB_RUN_NAME_BASE="$WANDB_RUN_NAME"
fi
WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-$MODEL_SAVE_BASE_DIR/wandb_run.json}"
WANDB_RUN_IDENTITY="${WANDB_RUN_IDENTITY:-$RESULTS_MODEL_KEY}"
WANDB_RUN_ID="$(
    "$PYTHON_BIN" - "$WANDB_PROJECT" "$WANDB_RUN_NAME_BASE" "$WANDB_RUN_IDENTITY" "$WANDB_RUN_ID_FILE" "$WANDB_RUN_ID" <<'PYWANDBRUN'
import sys

from recipe.opd.experiment_tracking import resolve_wandb_run_state

project, run_name, run_identity, state_path, explicit_run_id = sys.argv[1:]
state = resolve_wandb_run_state(
    project=project,
    run_name=run_name,
    run_identity=run_identity,
    state_path=state_path,
    explicit_run_id=explicit_run_id,
)
print(state.run_id)
PYWANDBRUN
)"
export WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_ID_FILE WANDB_RUN_IDENTITY WANDB_RESUME WANDB_MODE

# =============================================================================
# Helper Functions
# =============================================================================

file_exists_and_nonempty() {
    local file="$1"
    [ -f "$file" ] && [ -s "$file" ]
}

epoch_gen_results_dir() {
    local epoch="$1"
    echo "$GEN_RESULTS_BASE_DIR/epoch${epoch}"
}

epoch_output_dir() {
    local epoch="$1"
    echo "$OUTPUT_BASE_DIR/epoch${epoch}"
}

epoch_model_save_dir() {
    local epoch="$1"
    echo "$MODEL_SAVE_BASE_DIR/epoch${epoch}"
}

resolve_data_path_in_dir() {
    local data_dir="$1"

    if [ -n "$DATA_PATH" ]; then
        echo "$DATA_PATH"
        return
    fi

    # y_r: train on stage2 teacher rewrites. Qualified by DISTILL_MODE and the
    #      TEACHER_MODEL_NAME because:
    #        - OPSD vs OPD use different teacher prompts (with/without y*)
    #        - Different teachers (e.g. Qwen3-8B vs Qwen3-32B) produce different y_r
    # y_o: train on stage1 rollouts. SKD y_o is path-qualified to avoid
    #      accidentally reusing ordinary student rollout parquet.
    if [ "$Y_MODE" = "y_r" ]; then
        echo "$data_dir/${TASK_INTERMEDIATE_PREFIX}2_${PROMPT_MODE_TAG}_${DISTILL_MODE}_${TEACHER_MODEL_NAME}${TEACHER_THINKING_FILE_SUFFIX}_responses.parquet"
    else
        local stage1_response_stem="${TASK_INTERMEDIATE_PREFIX}1${Y_O_ROLLOUT_TAG}_responses"
        case "${FORWARD_STAGE2_MODE:-}" in
            "stage1_reward_0_only") echo "$data_dir/${stage1_response_stem}_reward0.parquet" ;;
            "stage1_reward_1_only") echo "$data_dir/${stage1_response_stem}_reward1.parquet" ;;
            *) echo "$data_dir/${stage1_response_stem}.parquet" ;;
        esac
    fi
}

resolve_epoch_data_path() {
    local epoch="$1"
    resolve_data_path_in_dir "$(epoch_gen_results_dir "$epoch")"
}

pipeline_batching_enabled() {
    [ "${MULTI_STEP:-0}" -gt 0 ]
}

format_pipeline_batch_id() {
    printf "%05d" "$1"
}

pipeline_batch_rel_dir() {
    local batch_index="$1"
    echo "${MULTISTEP_TAG}/batch$(format_pipeline_batch_id "$batch_index")"
}

update_gen_results_dir() {
    local epoch="$1"
    local batch_index="${2:-}"
    if [ -n "$batch_index" ]; then
        echo "$(epoch_gen_results_dir "$epoch")/$(pipeline_batch_rel_dir "$batch_index")"
    else
        epoch_gen_results_dir "$epoch"
    fi
}

update_output_dir() {
    local epoch="$1"
    local batch_index="${2:-}"
    if [ -n "$batch_index" ]; then
        echo "$(epoch_output_dir "$epoch")/$(pipeline_batch_rel_dir "$batch_index")"
    else
        epoch_output_dir "$epoch"
    fi
}

update_model_save_dir() {
    local epoch="$1"
    local batch_index="${2:-}"
    if [ -n "$batch_index" ]; then
        echo "$(epoch_model_save_dir "$epoch")/$(pipeline_batch_rel_dir "$batch_index")"
    else
        epoch_model_save_dir "$epoch"
    fi
}

pipeline_temp_model_save_dir() {
    local update="$1"
    echo "$PIPELINE_TEMP_MODEL_DIR/step$(format_pipeline_batch_id "$update")"
}

pipeline_final_alias_dir() {
    echo "$MODEL_SAVE_BASE_DIR/final"
}

find_latest_fsdp_checkpoint() {
    local model_dir="$1"
    [ -d "$model_dir" ] || return 0
    find "$model_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1
}

pipeline_marker_model_path() {
    local update="$1"
    local marker
    marker="$(pipeline_done_marker "$update")"
    [ -f "$marker" ] || return 0
    awk -F= '$1 == "model_path" {print substr($0, index($0, "=") + 1); exit}' "$marker"
}

pipeline_checkpoint_for_update() {
    local update="$1"
    local batches_per_epoch="$2"
    local marker_path ckpt

    marker_path="$(pipeline_marker_model_path "$update")"
    if [ -n "$marker_path" ] && [ -d "$marker_path" ]; then
        case "${marker_path##*/}" in
            global_step_*) echo "$marker_path"; return 0 ;;
            *)
                ckpt="$(find_latest_fsdp_checkpoint "$marker_path")"
                [ -n "$ckpt" ] && { echo "$ckpt"; return 0; }
                ;;
        esac
    fi

    ckpt="$(find_latest_fsdp_checkpoint "$(pipeline_temp_model_save_dir "$update")")"
    [ -n "$ckpt" ] && { echo "$ckpt"; return 0; }

    # Backward compatibility for older runs that wrote temporary FSDP checkpoints
    # directly under batch000XX/.
    ckpt="$(find_latest_fsdp_checkpoint "$(model_dir_for_global_update "$update" "$batches_per_epoch")")"
    [ -n "$ckpt" ] && echo "$ckpt"
}

resident_y_o_synced_to_checkpoint() {
    local checkpoint="$1"
    local pid marker_pid marker_checkpoint

    [ -s "$RESIDENT_YO_SYNC_MARKER" ] || return 1
    [ -s "$RESIDENT_YO_PID_FILE" ] || return 1
    pid="$(cat "$RESIDENT_YO_PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1

    marker_pid="$(awk -F= '$1 == "pid" {print substr($0, index($0, "=") + 1); exit}' "$RESIDENT_YO_SYNC_MARKER" 2>/dev/null || true)"
    marker_checkpoint="$(awk -F= '$1 == "checkpoint_path" {print substr($0, index($0, "=") + 1); exit}' "$RESIDENT_YO_SYNC_MARKER" 2>/dev/null || true)"

    [ "$marker_pid" = "$pid" ] && [ "$marker_checkpoint" = "$checkpoint" ]
}

mark_resident_y_o_synced() {
    local checkpoint="$1"
    local pid

    [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] || return
    [ -n "$checkpoint" ] || return
    [ -s "$RESIDENT_YO_PID_FILE" ] || return
    pid="$(cat "$RESIDENT_YO_PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] || return

    mkdir -p "$RESIDENT_YO_LOG_DIR"
    {
        printf "pid=%s\n" "$pid"
        printf "checkpoint_path=%s\n" "$checkpoint"
        printf "synced_at=%s\n" "$(date -Is)"
    } > "$RESIDENT_YO_SYNC_MARKER"
}

run_resident_sync_torchrun() {
    local sync_cmd="$1"
    local sync_log_dir="$2"
    local sync_key="$3"
    local use_slurm_torchrun="$USE_SLURM_TORCHRUN"

    if [ "$use_slurm_torchrun" = "auto" ]; then
        if [ "$NNODES" -gt 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
            use_slurm_torchrun="true"
        else
            use_slurm_torchrun="false"
        fi
    fi

    mkdir -p "$sync_log_dir"
    if [ "$use_slurm_torchrun" = "true" ]; then
        local sync_cmd_file="$sync_log_dir/resident_sync_${sync_key}.sh"
        cat > "$sync_cmd_file" <<EOF_RESIDENT_SYNC
#!/bin/bash
set -e
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:\${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_MODE="\${WANDB_MODE:-online}"
export NODE_RANK="\${SLURM_PROCID:-$NODE_RANK}"
$sync_cmd
EOF_RESIDENT_SYNC
        chmod +x "$sync_cmd_file"
        echo "Resident sync command file: $sync_cmd_file"
        local srun_job_arg=()
        if [ -n "${SLURM_JOB_ID:-}" ]; then
            srun_job_arg=(--jobid="$SLURM_JOB_ID")
        fi
        (
            srun "${srun_job_arg[@]}" --overlap \
                --cpu-bind=none \
                --nodes="$NNODES" \
                --ntasks="$NNODES" \
                --ntasks-per-node=1 \
                --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
                --gres="gpu:${SLURM_GPU_TYPE}:${NGPUS_PER_NODE}" \
                --kill-on-bad-exit=1 \
                bash "$sync_cmd_file"
        ) 2>&1 | tee "$sync_log_dir/resident_sync_${sync_key}_$(date +%Y%m%d_%H%M%S).log"
    else
        (
            cd "$VERL_ROOT"
            export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
            export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
            export WANDB_MODE="${WANDB_MODE:-online}"
            eval "$sync_cmd"
        ) 2>&1 | tee "$sync_log_dir/resident_sync_${sync_key}_$(date +%Y%m%d_%H%M%S).log"
    fi
}

sync_resident_y_o_from_checkpoint() {
    local checkpoint="$1"
    local sync_log_dir="$2"
    local sync_key="$3"
    local checkpoint_parent

    RESIDENT_YO_PRE_SYNC_PERFORMED="false"
    [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] || return
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "ERROR: resident rollout pre-generation sync checkpoint is missing: $checkpoint" >&2
        exit 1
    fi

    ensure_resident_y_o_server
    if resident_y_o_synced_to_checkpoint "$checkpoint"; then
        echo "Resident y_o rollout already synced to checkpoint: $checkpoint"
        return
    fi

    checkpoint_parent="$(dirname "$checkpoint")"
    echo ""
    echo "=========================================="
    echo "Pre-generation resident y_o weight sync"
    echo "  Checkpoint: $checkpoint"
    echo "  Manifest:   $RESIDENT_YO_MANIFEST"
    echo "=========================================="

    local sync_cmd="$PYTHON_BIN -m torch.distributed.run \
        --nproc-per-node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node-rank=\${NODE_RANK} \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        $RECIPE_DIR/run_training.py \
        --sync_resident_rollout_only true \
        --nnodes $NNODES \
        --n_gpus_per_node $NGPUS_PER_NODE \
        --task $TASK \
        --distill_mode $DISTILL_MODE \
        --kl_type $KL_TYPE \
        --kl_method $KL_METHOD \
        --kl_token_clip $KL_TOKEN_CLIP \
        --beta $BETA \
        --temperature $TEMPERATURE \
        --student_model_path $MODEL_PATH \
        ${TEACHER_MODEL_PATH:+--teacher_model_path $TEACHER_MODEL_PATH} \
        --teacher_enable_thinking $TEACHER_ENABLE_THINKING \
        --base_model_name $MODEL_NAME \
        --use_lora $USE_LORA \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --learning_rate $LEARNING_RATE \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --total_epochs 1 \
        --max_length $MAX_LENGTH \
        --warmup_steps_ratio $WARMUP_RATIO \
        --weight_decay $WEIGHT_DECAY \
        --min_lr_ratio 0.1 \
        --num_workers $NUM_WORKERS \
        --fsdp_strategy $FSDP_STRATEGY \
        --fsdp_size $FSDP_SIZE \
        --ulysses_sequence_parallel_size $SP_SIZE \
        --max_token_len_per_gpu $MAX_TOKEN_LEN_PER_GPU \
        --use_dynamic_bsz $USE_DYNAMIC_BSZ \
        --micro_batch_size_per_gpu $MICRO_BATCH_SIZE_PER_GPU \
        --use_torch_compile $USE_TORCH_COMPILE \
        --param_offload $PARAM_OFFLOAD \
        --optimizer_offload $OPTIMIZER_OFFLOAD \
        --offload_policy $OFFLOAD_POLICY \
        --epoch_index 1 \
        --output_dir $sync_log_dir \
        --model_save_dir $checkpoint_parent \
        --gen_results_dir $sync_log_dir \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name ${WANDB_RUN_NAME_BASE}_resident_sync_${sync_key} \
        --wandb_run_id $WANDB_RUN_ID \
        --wandb_run_id_file $WANDB_RUN_ID_FILE \
        --wandb_run_identity $WANDB_RUN_IDENTITY \
        --wandb_global_step_offset 0 \
        --save_merged_model false \
        --save_steps $SAVE_STEPS \
        --max_ckpt_to_keep $KEEP_LAST_N_CHECKPOINTS \
        --resume_checkpoint_path $checkpoint \
        --resume_checkpoint_mode initialize \
        --resident_rollout_manifest $RESIDENT_YO_MANIFEST \
        --sync_resident_rollout true \
        --async_hf_export false \
        --run_eval_after_training false \
        --eval_datasets $EVAL_DATASETS \
        --eval_datasets_dir $EVAL_DATASETS_DIR \
        --eval_fractions $EVAL_FRACTIONS \
        --use_initial_response $USE_INITIAL_RESPONSE \
        --prompt_truncation $PROMPT_TRUNCATION \
        --grad_cosine_interval 0 \
        --log_difficulty_buckets false \
        --top_k $TOP_K"

    run_resident_sync_torchrun "$sync_cmd" "$sync_log_dir" "$sync_key"
    mark_resident_y_o_synced "$checkpoint"
    RESIDENT_YO_PRE_SYNC_PERFORMED="true"
}

invalidate_resident_generated_data_after_pre_sync() {
    local data_dir="$1"

    case "$data_dir" in
        "$GEN_RESULTS_BASE_DIR"/epoch*/ms*/batch*) ;;
        *)
            echo "WARNING: refusing to invalidate unexpected resident gen dir: $data_dir" >&2
            return
            ;;
    esac

    echo "Invalidating existing resident-generated responses after pre-generation sync: $data_dir"
    rm -f "$data_dir/${TASK_INTERMEDIATE_PREFIX}1${Y_O_ROLLOUT_TAG}_responses.parquet"
    rm -f "$data_dir"/${TASK_INTERMEDIATE_PREFIX}2_*_responses*.parquet
    rm -f "$data_dir/deepscaleR_stage1_responses.parquet"
    rm -f "$data_dir"/deepscaleR_stage2_*_responses*.parquet
}

ensure_resident_y_o_server() {
    if [ "$RESIDENT_STUDENT_ROLLOUT" != "true" ]; then
        return
    fi
    if [ -s "$RESIDENT_YO_MANIFEST" ]; then
        if [ -s "$RESIDENT_YO_PID_FILE" ] && kill -0 "$(cat "$RESIDENT_YO_PID_FILE")" 2>/dev/null; then
            echo "Resident y_o server manifest: $RESIDENT_YO_MANIFEST"
            return
        fi
        echo "Resident y_o manifest exists but server pid is not alive; restarting resident y_o server..."
        rm -f "$RESIDENT_YO_MANIFEST" "$RESIDENT_YO_PID_FILE" "$RESIDENT_YO_SYNC_MARKER"
    fi
    if [ "$RESIDENT_YO_AUTOSTART" != "true" ]; then
        echo "ERROR: RESIDENT_STUDENT_ROLLOUT=true but manifest is missing: $RESIDENT_YO_MANIFEST" >&2
        echo "       Start recipe.opd.resident_y_o_server first or set RESIDENT_YO_AUTOSTART=true." >&2
        exit 1
    fi

    mkdir -p "$RESIDENT_YO_LOG_DIR"
    rm -f "$RESIDENT_YO_SYNC_MARKER"
    local resident_lora_rank=0
    if [ "$USE_LORA" = "true" ]; then
        resident_lora_rank="$LORA_RANK"
    fi

    echo "Starting resident y_o vLLM server..."
    (
        cd "$VERL_ROOT"
        export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
        export VERL_ENABLE_STANDALONE_VLLM_SLEEP=1
        "$PYTHON_BIN" -m recipe.opd.resident_y_o_server \
            --manifest "$RESIDENT_YO_MANIFEST" \
            --model_path "$MODEL_PATH" \
            --nnodes "$NNODES" \
            --n_gpus_per_node "$NGPUS_PER_NODE" \
            --tensor_model_parallel_size "$GEN_TP" \
            --gpu_memory_utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
            --max_num_seqs "$ROLLOUT_MAX_NUM_SEQS" \
            --max_num_batched_tokens "$STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
            --prompt_length "$BASE_PROMPT_LENGTH" \
            --response_length "$MAX_RESPONSE_LENGTH" \
            --max_model_len "$STAGE1_ROLLOUT_MAX_MODEL_LEN" \
            --lora_rank "$resident_lora_rank" \
            --lora_alpha "$LORA_ALPHA" \
            --load_format "$RESIDENT_YO_LOAD_FORMAT" \
            --free_cache_engine true \
            --enable_sleep_mode true \
            --ray_address "$RAY_ADDRESS" \
            --ray_namespace "$RESIDENT_YO_RAY_NAMESPACE"
    ) > "$RESIDENT_YO_LOG_DIR/server.log" 2>&1 &
    echo $! > "$RESIDENT_YO_PID_FILE"

    local _i
    for _i in $(seq 1 180); do
        if [ -s "$RESIDENT_YO_MANIFEST" ]; then
            echo "Resident y_o server ready: $RESIDENT_YO_MANIFEST"
            return
        fi
        if ! kill -0 "$(cat "$RESIDENT_YO_PID_FILE")" 2>/dev/null; then
            echo "ERROR: resident y_o server exited before writing manifest. Log: $RESIDENT_YO_LOG_DIR/server.log" >&2
            exit 1
        fi
        sleep 2
    done
    echo "ERROR: timed out waiting for resident y_o server manifest: $RESIDENT_YO_MANIFEST" >&2
    echo "       Log: $RESIDENT_YO_LOG_DIR/server.log" >&2
    exit 1
}

sleep_resident_y_o_server() {
    if [ "$RESIDENT_STUDENT_ROLLOUT" != "true" ]; then
        return
    fi
    "$PYTHON_BIN" -m recipe.opd.resident_y_o_control \
        --manifest "$RESIDENT_YO_MANIFEST" \
        --action sleep
}

stop_managed_resident_y_o_server() {
    local pid attempt

    [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] || return 0
    # A caller-provided manifest denotes an externally managed server.
    [ -z "$RESIDENT_YO_MANIFEST_USER_VALUE" ] || return 0
    if [ -s "$RESIDENT_YO_PID_FILE" ]; then
        pid="$(cat "$RESIDENT_YO_PID_FILE" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Stopping managed resident y_o server: pid=$pid"
            kill "$pid" 2>/dev/null || true
            for attempt in $(seq 1 30); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "WARNING: resident y_o server did not stop gracefully; terminating pid=$pid" >&2
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    fi
    rm -f "$RESIDENT_YO_MANIFEST" "$RESIDENT_YO_PID_FILE" "$RESIDENT_YO_SYNC_MARKER"
}

release_resident_y_o_after_generation() {
    [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] || return 0
    if [ "$RESIDENT_YO_RELEASE_AFTER_STAGE1" = "true" ]; then
        stop_managed_resident_y_o_server
    else
        sleep_resident_y_o_server
    fi
}

ensure_full_stage1_prompts() {
    if file_exists_and_nonempty "$FULL_STAGE1_PROMPTS"; then
        echo "  Full Stage 1 prompts already prepared: $FULL_STAGE1_PROMPTS"
        return
    fi

    if [ -n "$PRECOMPUTED_STAGE1_PROMPTS_PATH" ]; then
        if [ ! -s "$PRECOMPUTED_STAGE1_PROMPTS_PATH" ]; then
            echo "ERROR: repo-local Stage 1 prompt parquet is missing: $PRECOMPUTED_STAGE1_PROMPTS_PATH" >&2
            exit 1
        fi
        echo "  [Stage 1 prompt cache] Loading prepared prompts: $PRECOMPUTED_STAGE1_PROMPTS_PATH"
        "$PYTHON_BIN" - \
            "$PRECOMPUTED_STAGE1_PROMPTS_PATH" \
            "$FULL_STAGE1_PROMPTS" \
            "${MAX_SAMPLES:-}" \
            "$TEACHER_TRAJECTORY_CONDITIONING" <<'PYPREPAREDSTAGE1'
import os
import sys

import pandas as pd

input_path, output_path, max_samples, conditioning = sys.argv[1:5]
frame = pd.read_parquet(input_path)
required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
missing = required - set(frame.columns)
if missing:
    raise SystemExit(
        f"prepared Stage 1 parquet is missing columns: {sorted(missing)}"
    )
if max_samples:
    frame = frame.iloc[: int(max_samples)].copy()
else:
    frame = frame.copy()

# A y* trajectory parquet may also contain its target responses. They are never
# rollout inputs and must not be mistaken for teacher-generated y_t.
if "responses" in frame.columns:
    frame = frame.drop(columns=["responses"])

if conditioning == "pi_T_x_only_v1":
    for index, row in frame.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        messages = prompt if isinstance(prompt, list) else [prompt]
        prompt_text = "\n".join(
            str(message.get("content", "")) if isinstance(message, dict) else str(message)
            for message in messages
        )
        extra = row["extra_info"]
        expert = extra.get("expert_cot", "") if isinstance(extra, dict) else ""
        if expert and str(expert) in prompt_text:
            raise SystemExit(
                f"x-only trajectory prompt at row {index} contains expert_cot"
            )
        if "Given the expert solution below" in prompt_text:
            raise SystemExit(
                f"x-only trajectory prompt at row {index} contains the expert-rewrite instruction"
            )

os.makedirs(os.path.dirname(output_path), exist_ok=True)
temporary = output_path + ".tmp"
frame.to_parquet(temporary, index=False)
os.replace(temporary, output_path)
print(f"Wrote {len(frame)} prepared Stage 1 prompts -> {output_path}")
PYPREPAREDSTAGE1
        return
    fi

    echo "  [Stage 1 prompt cache] Preparing all prompts once: $FULL_STAGE1_PROMPTS"
    local args=(
        --input_path "$TRAIN_DATA_PATH"
        --output_file "$FULL_STAGE1_PROMPTS"
        --task "$TASK"
        --data_source "$TRAIN_DATA_SOURCE"
    )
    if [ -n "$MAX_SAMPLES" ]; then
        args+=(--max_samples "$MAX_SAMPLES")
    fi
    "$PYTHON_BIN" "$PIPELINE_DIR/y_o_prepare.py" "${args[@]}"
}

prepare_stage1_prompts() {
    local output_file="$1"
    local batch_start="${2:-}"
    local batch_size="${3:-}"

    if [ -n "$batch_start" ]; then
        ensure_full_stage1_prompts
        if file_exists_and_nonempty "$output_file"; then
            echo "  Chunk Stage 1 prompts already prepared: $output_file"
            return
        fi
        echo "  [Stage 1 prompt chunk] rows ${batch_start}..$((batch_start + batch_size - 1)) -> $output_file"
        "$PYTHON_BIN" - "$FULL_STAGE1_PROMPTS" "$output_file" "$batch_start" "$batch_size" <<'PYCHUNKPROMPTS'
import os
import sys
import pandas as pd

input_path, output_path, start, size = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
df = pd.read_parquet(input_path)
chunk = df.iloc[start:start + size].reset_index(drop=True)
if len(chunk) != size:
    raise SystemExit(f"requested {size} rows from {start}, got {len(chunk)}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
chunk.to_parquet(output_path)
print(f"wrote {len(chunk)} prompt rows -> {output_path}")
PYCHUNKPROMPTS
        return
    fi

    local args=(
        --input_path "$TRAIN_DATA_PATH"
        --output_file "$output_file"
        --task "$TASK"
        --data_source "$TRAIN_DATA_SOURCE"
    )
    if [ -n "$MAX_SAMPLES" ]; then
        args+=(--max_samples "$MAX_SAMPLES")
    fi
    "$PYTHON_BIN" "$PIPELINE_DIR/y_o_prepare.py" "${args[@]}"
}

prepare_teacher_trajectory_prompt_slice() {
    local stage1_prompts="$1"
    local output_file="$2"
    local batch_start="${3:-}"
    local batch_size="${4:-}"
    local -a args=(
        --input "$TEACHER_TRAJECTORY_PROMPT_PATH"
        --output "$output_file"
        --task "$TASK"
        --alignment-input "$stage1_prompts"
    )

    if [ -n "$batch_start" ]; then
        args+=(--start-index "$batch_start")
        args+=(--num-samples "$batch_size")
    elif [ -n "$MAX_SAMPLES" ]; then
        args+=(--start-index 0)
        args+=(--num-samples "$MAX_SAMPLES")
    fi

    "$PYTHON_BIN" -m recipe.opd.generation.teacher_y_t_prepare "${args[@]}"
}

teacher_trajectory_cache_path() {
    local teacher_prompts="$1"
    local prompt_digest config_digest model_tag

    prompt_digest="$(sha256sum "$teacher_prompts" | awk '{print $1}')"
    config_digest="$(
        printf '%s\n' \
            "trajectory_model_path=$TRAJECTORY_MODEL_PATH" \
            "trajectory_conditioning=$TEACHER_TRAJECTORY_CONDITIONING" \
            "task=$TASK" \
            "temperature=$ROLLOUT_TEMPERATURE" \
            "top_p=$ROLLOUT_TOP_P" \
            "top_k=$ROLLOUT_TOP_K" \
            "n=1" \
            "teacher_trajectory_prompt_length=$TEACHER_TRAJECTORY_PROMPT_LENGTH" \
            "max_response_length=$MAX_RESPONSE_LENGTH" \
            "teacher_trajectory_rollout_max_model_len=$TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN" |
            sha256sum |
            awk '{print $1}'
    )"
    model_tag="$(sanitize_path_component "$TRAJECTORY_MODEL_NAME")"
    printf '%s/%s/%s/%s.parquet\n' \
        "$TEACHER_TRAJECTORY_CACHE_ROOT" \
        "$model_tag" \
        "$config_digest" \
        "$prompt_digest"
}

teacher_trajectory_cache_is_valid() {
    local cache_path="$1"
    local stage1_prompts="$2"
    [ -s "$cache_path" ] || return 1

    "$PYTHON_BIN" - "$cache_path" "$stage1_prompts" <<'PYCACHEVALIDATE'
import sys
import pyarrow.parquet as pq

cache_path, prompts_path = sys.argv[1:3]
cache = pq.ParquetFile(cache_path)
prompts = pq.ParquetFile(prompts_path)
if cache.metadata.num_rows != prompts.metadata.num_rows:
    raise SystemExit(1)
if "responses" not in cache.schema_arrow.names:
    raise SystemExit(1)
PYCACHEVALIDATE
}

write_teacher_trajectory_cache() {
    local source_path="$1"
    local cache_path="$2"
    local source_prompt_path="$3"
    local cache_dir temporary_cache

    cache_dir="$(dirname "$cache_path")"
    mkdir -p "$cache_dir"
    temporary_cache="$(mktemp "$cache_dir/.trajectory.XXXXXX.parquet")"
    cp "$source_path" "$temporary_cache"
    chmod 664 "$temporary_cache"
    mv -f "$temporary_cache" "$cache_path"
    cat > "${cache_path%.parquet}.metadata.txt" <<EOF
trajectory_model_path=$TRAJECTORY_MODEL_PATH
trajectory_model_name=$TRAJECTORY_MODEL_NAME
trajectory_conditioning=$TEACHER_TRAJECTORY_CONDITIONING
task=$TASK
temperature=$ROLLOUT_TEMPERATURE
top_p=$ROLLOUT_TOP_P
top_k=$ROLLOUT_TOP_K
n=1
teacher_trajectory_prompt_path=$TEACHER_TRAJECTORY_PROMPT_PATH
teacher_trajectory_prompt_length=$TEACHER_TRAJECTORY_PROMPT_LENGTH
max_response_length=$MAX_RESPONSE_LENGTH
teacher_trajectory_rollout_max_model_len=$TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN
source_prompt_parquet=$source_prompt_path
EOF
}


generate_stage1_y_o_responses() {
    local stage1_prompts="$1"
    local stage1_output="$2"
    local current_model_path="$3"
    local current_teacher_model_path="${4:-}"
    local batch_start="${5:-}"
    local batch_size="${6:-}"
    local student_lora_adapter_path="${7:-}"
    local teacher_cache_path=""
    local expert_trajectory_input=""
    local -a expert_slice_args=()

    case "$Y_O_ROLLOUT_MODE" in
        expert)
            expert_trajectory_input="$stage1_prompts"
            if [ -n "$PRECOMPUTED_Y_O_TRAJECTORY_PATH" ]; then
                expert_trajectory_input="$PRECOMPUTED_Y_O_TRAJECTORY_PATH"
                if [ -n "$batch_start" ]; then
                    expert_slice_args+=(--start-index "$batch_start")
                    expert_slice_args+=(--num-samples "$batch_size")
                elif [ -n "$MAX_SAMPLES" ]; then
                    expert_slice_args+=(--num-samples "$MAX_SAMPLES")
                fi
                echo "  [Stage 1] Slicing precomputed y* trajectory parquet: $expert_trajectory_input"
            else
                echo "  [Stage 1] Building y* responses from prompt extra_info..."
            fi
            "$PYTHON_BIN" -m recipe.opd.generation.expert_y_star_generate \
                --input "$expert_trajectory_input" \
                --output "$stage1_output" \
                "${expert_slice_args[@]}"
            ;;
        teacher)
            local teacher_rollout_prompts="$stage1_prompts"
            local trajectory_distribution="pi_T(.|x)"
            if [ "$TEACHER_TRAJECTORY_CONDITIONING" = "opsd_x_y_star_expert_rewrite_v1" ]; then
                teacher_rollout_prompts="${stage1_output%.parquet}_expert_rewrite_prompts.parquet"
                trajectory_distribution="pi_T(.|x,y*)"
                echo "  [Stage 1] Preparing aligned pi_T(.|x,y*) expert-rewrite prompts..."
                prepare_teacher_trajectory_prompt_slice \
                    "$stage1_prompts" \
                    "$teacher_rollout_prompts" \
                    "$batch_start" \
                    "$batch_size"
            else
                echo "  [Stage 1] Using original problem prompts for pi_T(.|x) rollout."
            fi

            if [ "$TEACHER_TRAJECTORY_CACHE_MODE" != "off" ]; then
                teacher_cache_path="$(teacher_trajectory_cache_path "$teacher_rollout_prompts")"
                if teacher_trajectory_cache_is_valid "$teacher_cache_path" "$teacher_rollout_prompts"; then
                    echo "  [Stage 1] Reusing fixed-teacher y_t cache: $teacher_cache_path"
                    cp "$teacher_cache_path" "$stage1_output"
                    return 0
                fi
                echo "  [Stage 1] No matching fixed-teacher y_t cache: $teacher_cache_path"
            fi

            echo "  [Stage 1] Generating y_t ~ $trajectory_distribution with fixed trajectory model: $TRAJECTORY_MODEL_PATH"
            run_teacher_generation "$PYTHON_BIN" -m verl.trainer.main_generation_server \
                trainer.nnodes="${NNODES}" \
                trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                actor_rollout_ref.model.path="${TRAJECTORY_MODEL_PATH}" \
                actor_rollout_ref.model.trust_remote_code=true \
                actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
                actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
                actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K" \
                actor_rollout_ref.rollout.prompt_length="${TEACHER_TRAJECTORY_PROMPT_LENGTH}" \
                actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
                actor_rollout_ref.rollout.max_model_len="${TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN}" \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
                actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
                actor_rollout_ref.rollout.max_num_batched_tokens="${TEACHER_TRAJECTORY_MAX_NUM_BATCHED_TOKENS}" \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.n=1 \
                data.train_files="['${teacher_rollout_prompts}']" \
                data.prompt_key=prompt \
                +data.output_path="${stage1_output}"
            if [ "$TEACHER_TRAJECTORY_CACHE_MODE" = "read_write" ]; then
                write_teacher_trajectory_cache "$stage1_output" "$teacher_cache_path" "$teacher_rollout_prompts"
                echo "  [Stage 1] Saved fixed-teacher y_t cache: $teacher_cache_path"
            fi
            ;;
        skd|skd_vllm)
            echo "  [Stage 1] Generating SKD y_o responses with vLLM..."
            echo "    student_gpus=$SKD_STUDENT_GPUS teacher_gpus=$SKD_TEACHER_GPUS shared_gpus=$SKD_SHARED_GPUS"
            echo "    skd_top_k=$SKD_ACCEPT_TOP_K skd_top_p=$SKD_ACCEPT_TOP_P gamma=$SKD_GAMMA parallel_student_teacher=$SKD_PARALLEL_STUDENT_TEACHER"
            PYTHONPATH="local_vllm_patch:${PYTHONPATH:-}" env -u PYTORCH_CUDA_ALLOC_CONF "$PYTHON_BIN" -m recipe.opd.generation.skd_vllm_y_o_generate \
                --input "$stage1_prompts" \
                --output "$stage1_output" \
                --prompt_key prompt \
                --student_model_path "$current_model_path" \
                --student_lora_adapter_path "$student_lora_adapter_path" \
                --teacher_model_path "${current_teacher_model_path:-}" \
                --tokenizer_path "$current_model_path" \
                --teacher_tokenizer_path "${current_teacher_model_path:-$current_model_path}" \
                --teacher_enable_thinking "$TEACHER_ENABLE_THINKING" \
                --distill_mode "$DISTILL_MODE" \
                --task "$TASK" \
                --max_tokens "$MAX_RESPONSE_LENGTH" \
                --prompt_length "$BASE_PROMPT_LENGTH" \
                --teacher_prompt_length "$SKD_TEACHER_PROMPT_LENGTH" \
                --max_model_len "$SKD_ROLLOUT_MAX_MODEL_LEN" \
                --batch_size "$SKD_ROLLOUT_BATCH_SIZE" \
                --gamma "$SKD_GAMMA" \
                --top_k "$SKD_ACCEPT_TOP_K" \
                --top_p "$SKD_ACCEPT_TOP_P" \
                --student_temperature "$SKD_STUDENT_TEMPERATURE" \
                --student_top_p "$SKD_STUDENT_TOP_P" \
                --teacher_temperature "$SKD_TEACHER_TEMPERATURE" \
                --teacher_top_p "$SKD_TEACHER_TOP_P" \
                --student_gpus "$SKD_STUDENT_GPUS" \
                --teacher_gpus "$SKD_TEACHER_GPUS" \
                --shared_gpus "$SKD_SHARED_GPUS" \
                --gpu_memory_utilization "$SKD_VLLM_GPU_MEMORY_UTILIZATION" \
                --max_num_seqs "$SKD_VLLM_MAX_NUM_SEQS" \
                --max_num_batched_tokens "$SKD_VLLM_MAX_NUM_BATCHED_TOKENS" \
                --parallel_student_teacher "$SKD_PARALLEL_STUDENT_TEACHER" \
                --pipeline_lanes "$SKD_PIPELINE_LANES"
            ;;
        skd_vllm_internal)
            echo "  [Stage 1] Generating SKD y_o responses with vLLM internal sampler..."
            echo "    shared_gpus=$SKD_SHARED_GPUS teacher_tp=${SKD_TEACHER_TP:-auto} student_tp=${SKD_STUDENT_TP:-auto}"
            echo "    skd_top_k=$SKD_ACCEPT_TOP_K skd_top_p=$SKD_ACCEPT_TOP_P gamma=$SKD_GAMMA"
            PYTHONPATH="local_vllm_patch:${PYTHONPATH:-}" env -u PYTORCH_CUDA_ALLOC_CONF "$PYTHON_BIN" -m recipe.opd.generation.skd_vllm_internal_y_o_generate \
                --input "$stage1_prompts" \
                --output "$stage1_output" \
                --prompt_key prompt \
                --student_model_path "$current_model_path" \
                --teacher_model_path "${current_teacher_model_path:-}" \
                --tokenizer_path "$current_model_path" \
                --distill_mode "$DISTILL_MODE" \
                --max_tokens "$MAX_RESPONSE_LENGTH" \
                --prompt_length "$BASE_PROMPT_LENGTH" \
                --max_model_len "$STAGE1_ROLLOUT_MAX_MODEL_LEN" \
                --batch_size "$SKD_ROLLOUT_BATCH_SIZE" \
                --gamma "$SKD_GAMMA" \
                --top_k "$SKD_ACCEPT_TOP_K" \
                --top_p "$SKD_ACCEPT_TOP_P" \
                --student_temperature "$SKD_STUDENT_TEMPERATURE" \
                --student_top_p "$SKD_STUDENT_TOP_P" \
                --teacher_temperature "$SKD_TEACHER_TEMPERATURE" \
                --teacher_top_p "$SKD_TEACHER_TOP_P" \
                --shared_gpus "$SKD_SHARED_GPUS" \
                --student_tp "$SKD_STUDENT_TP" \
                --teacher_tp "$SKD_TEACHER_TP" \
                --gpu_memory_utilization "$SKD_VLLM_GPU_MEMORY_UTILIZATION" \
                --max_num_seqs "$SKD_VLLM_MAX_NUM_SEQS" \
                --max_num_batched_tokens "$SKD_VLLM_MAX_NUM_BATCHED_TOKENS"
            ;;
        student)
            if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
                ensure_resident_y_o_server
                "$PYTHON_BIN" -m recipe.opd.resident_y_o_generate \
                    --manifest "$RESIDENT_YO_MANIFEST" \
                    --input "$stage1_prompts" \
                    --output "$stage1_output" \
                    --prompt_key prompt \
                    --model_path "$MODEL_PATH" \
                    --temperature "$ROLLOUT_TEMPERATURE" \
                    --top_p "$ROLLOUT_TOP_P" \
                    --max_tokens "$MAX_RESPONSE_LENGTH"
            else
                env -u PYTORCH_CUDA_ALLOC_CONF "$PYTHON_BIN" -m verl.trainer.main_generation_server \
                    trainer.nnodes="${NNODES}" \
                    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                    actor_rollout_ref.model.path="${current_model_path}" \
                    actor_rollout_ref.model.trust_remote_code=true \
                    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
                    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
                    actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K" \
                    actor_rollout_ref.rollout.prompt_length="${BASE_PROMPT_LENGTH}" \
                    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
                    actor_rollout_ref.rollout.max_model_len="${STAGE1_ROLLOUT_MAX_MODEL_LEN}" \
                    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
                    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
                    actor_rollout_ref.rollout.max_num_batched_tokens="${STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
                    actor_rollout_ref.rollout.name=vllm \
                    actor_rollout_ref.rollout.n=1 \
                    data.train_files="['${stage1_prompts}']" \
                    data.prompt_key=prompt \
                    +data.output_path="${stage1_output}"
            fi
            ;;
    esac
}
step1_reuse_ms_tag_for_path() {
    local target_path="$1"

    case "$target_path" in
        */epoch1/ms*/batch00001/*)
            local rest="${target_path#*/epoch1/}"
            printf '%s\n' "${rest%%/*}"
            return 0
            ;;
    esac

    return 1
}

step1_reuse_enabled_for_path() {
    local target_path="$1"

    case "${STEP1_RESPONSE_REUSE:-${MS1_RESPONSE_REUSE:-auto}}" in
        false|False|0|no|No|n|N|"") return 1 ;;
    esac

    step1_reuse_ms_tag_for_path "$target_path" >/dev/null
}

metadata_matches_step1_reuse_context() {
    local meta="$1"
    local expected_y_mode="$2"
    local value

    [ -f "$meta" ] || return 1

    value="$(gen_results_metadata_value "$meta" task)"
    if [ -n "$value" ]; then
        [ "$value" = "$TASK" ] || return 1
    fi
    value="$(gen_results_metadata_value "$meta" distill_mode)"
    [ "$value" = "$DISTILL_MODE" ] || return 1
    value="$(gen_results_metadata_value "$meta" prompt_contract_version)"
    [ "$value" = "$PROMPT_CONTRACT_VERSION" ] || return 1
    value="$(gen_results_metadata_value "$meta" student_model_name)"
    [ -n "$value" ] || value="$(gen_results_metadata_value "$meta" model_name)"
    [ "$value" = "$MODEL_NAME" ] || return 1
    value="$(gen_results_metadata_value "$meta" student_model_path)"
    [ -n "$value" ] || value="$(gen_results_metadata_value "$meta" model_path)"
    [ "$value" = "$MODEL_PATH" ] || return 1
    value="$(gen_results_metadata_value "$meta" teacher_model_name)"
    [ "$value" = "$TEACHER_MODEL_NAME" ] || return 1
    value="$(gen_results_metadata_value "$meta" teacher_model_path)"
    [ "$value" = "${TEACHER_MODEL_PATH:-$MODEL_PATH}" ] || return 1
    value="$(gen_results_metadata_value "$meta" teacher_enable_thinking)"
    [ "$value" = "$TEACHER_ENABLE_THINKING" ] || return 1
    value="$(gen_results_metadata_value "$meta" teacher_supervision_render_mode)"
    [ "$value" = "$TEACHER_SUPERVISION_RENDER_MODE" ] || return 1
    value="$(gen_results_metadata_value "$meta" teacher_rollout_render_mode)"
    [ "$value" = "$TEACHER_ROLLOUT_RENDER_MODE" ] || return 1
    value="$(gen_results_metadata_value "$meta" teacher_prompt_render_contract)"
    [ "$value" = "$TEACHER_PROMPT_RENDER_CONTRACT" ] || return 1
    value="$(gen_results_metadata_value "$meta" y_mode)"
    [ "$value" = "$expected_y_mode" ] || return 1
    value="$(gen_results_metadata_value "$meta" y_o_rollout_mode)"
    [ -n "$value" ] || value="student"
    [ "$value" = "$Y_O_ROLLOUT_MODE" ] || return 1
    value="$(gen_results_metadata_value "$meta" trajectory_model_path)"
    [ "$value" = "${TRAJECTORY_MODEL_PATH:-}" ] || return 1
    if [ "$Y_O_ROLLOUT_MODE" = "teacher" ]; then
        value="$(gen_results_metadata_value "$meta" teacher_trajectory_conditioning)"
        [ "$value" = "$TEACHER_TRAJECTORY_CONDITIONING" ] || return 1
        value="$(gen_results_metadata_value "$meta" teacher_trajectory_prompt_path)"
        [ "$value" = "${TEACHER_TRAJECTORY_PROMPT_PATH:-}" ] || return 1
    fi
    case "$Y_O_ROLLOUT_MODE" in
        skd|skd_vllm|skd_vllm_internal)
            value="$(gen_results_metadata_value "$meta" skd_gamma)"
            [ "$value" = "$SKD_GAMMA" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_accept_top_k)"
            [ "$value" = "$SKD_ACCEPT_TOP_K" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_accept_top_p)"
            [ "$value" = "$SKD_ACCEPT_TOP_P" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_student_temperature)"
            [ "$value" = "$SKD_STUDENT_TEMPERATURE" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_student_top_p)"
            [ "$value" = "$SKD_STUDENT_TOP_P" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_teacher_temperature)"
            [ "$value" = "$SKD_TEACHER_TEMPERATURE" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_teacher_top_p)"
            [ "$value" = "$SKD_TEACHER_TOP_P" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_rollout_batch_size)"
            [ "$value" = "$SKD_ROLLOUT_BATCH_SIZE" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_pipeline_lanes)"
            [ "$value" = "$SKD_PIPELINE_LANES" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_parallel_student_teacher)"
            [ "$value" = "$SKD_PARALLEL_STUDENT_TEACHER" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_teacher_prompt_contract)"
            [ "$value" = "$SKD_TEACHER_PROMPT_CONTRACT" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_teacher_prompt_length)"
            [ "$value" = "$SKD_TEACHER_PROMPT_LENGTH" ] || return 1
            value="$(gen_results_metadata_value "$meta" skd_rollout_max_model_len)"
            [ "$value" = "$SKD_ROLLOUT_MAX_MODEL_LEN" ] || return 1
            ;;
    esac
    value="$(gen_results_metadata_value "$meta" train_data_path)"
    [ "$value" = "$TRAIN_DATA_PATH" ] || return 1
    value="$(gen_results_metadata_value "$meta" max_samples)"
    [ "$value" = "${MAX_SAMPLES:-all}" ] || return 1
    value="$(gen_results_metadata_value "$meta" multi_step)"
    [ "$value" = "${MULTI_STEP:-0}" ] || return 1
    value="$(gen_results_metadata_value "$meta" base_prompt_length)"
    [ "$value" = "$BASE_PROMPT_LENGTH" ] || return 1
    value="$(gen_results_metadata_value "$meta" max_response_length)"
    [ "$value" = "$MAX_RESPONSE_LENGTH" ] || return 1
    value="$(gen_results_metadata_value "$meta" rollout_temperature)"
    [ "$value" = "$ROLLOUT_TEMPERATURE" ] || return 1
    value="$(gen_results_metadata_value "$meta" rollout_top_p)"
    [ "$value" = "$ROLLOUT_TOP_P" ] || return 1
    value="$(gen_results_metadata_value "$meta" rollout_top_k)"
    [ "$value" = "$ROLLOUT_TOP_K" ] || return 1
    if [ "$expected_y_mode" = "y_r" ]; then
        value="$(gen_results_metadata_value "$meta" stage2_prompt_length)"
        [ "$value" = "$STAGE2_PROMPT_LENGTH" ] || return 1
    fi

    return 0
}

find_canonical_step1_responses() {
    local stage="$1"
    local target_ms_tag="$2"
    local expected_y_mode pattern candidate base meta

    [ -d "$GEN_RESULTS_ROOT" ] || return 0
    [ -n "$target_ms_tag" ] || return 0

    case "$stage" in
        stage1)
            expected_y_mode="y_o"
            pattern="*/epoch1/${target_ms_tag}/batch00001/${TASK_INTERMEDIATE_PREFIX}1${Y_O_ROLLOUT_TAG}_responses.parquet"
            ;;
        stage2)
            expected_y_mode="y_r"
            pattern="*/epoch1/${target_ms_tag}/batch00001/${TASK_INTERMEDIATE_PREFIX}2_*_responses.parquet"
            ;;
        *)
            return 0
            ;;
    esac

    while IFS= read -r candidate; do
        base="${candidate%%/epoch1/*}"
        meta="$base/run_metadata.yaml"
        metadata_matches_step1_reuse_context "$meta" "$expected_y_mode" || continue
        printf '%s\n' "$candidate"
        return 0
    done < <(
        find "$GEN_RESULTS_ROOT" -path "$pattern" -type f -size +0c -printf '%T@ %p\n' 2>/dev/null \
            | sort -n \
            | awk '{ $1=""; sub(/^ /, ""); print }'
    )
}

reuse_step1_responses_if_available() {
    local stage="$1"
    local target_path="$2"

    step1_reuse_enabled_for_path "$target_path" || return 1
    if [ "$stage" = "stage1" ] && [ "${target_path##*/}" != "${TASK_INTERMEDIATE_PREFIX}1${Y_O_ROLLOUT_TAG}_responses.parquet" ]; then
        return 1
    fi

    local target_ms_tag setting source_path=""
    target_ms_tag="$(step1_reuse_ms_tag_for_path "$target_path" || true)"
    [ -n "$target_ms_tag" ] || return 1

    case "$stage" in
        stage1) setting="${STEP1_STAGE1_RESPONSE_REUSE_PATH:-${MS1_STAGE1_RESPONSE_REUSE_PATH:-${STEP1_RESPONSE_REUSE:-${MS1_RESPONSE_REUSE:-auto}}}}" ;;
        stage2) setting="${STEP1_STAGE2_RESPONSE_REUSE_PATH:-${MS1_STAGE2_RESPONSE_REUSE_PATH:-${STEP1_RESPONSE_REUSE:-${MS1_RESPONSE_REUSE:-auto}}}}" ;;
        *) return 1 ;;
    esac

    case "$setting" in
        false|False|0|no|No|n|N|"") return 1 ;;
    esac

    if [ "$setting" != "auto" ]; then
        source_path="$setting"
    else
        source_path="$(find_canonical_step1_responses "$stage" "$target_ms_tag" || true)"
    fi

    if [ -z "$source_path" ] || [ ! -s "$source_path" ]; then
        return 1
    fi
    source_path="$(realpath "$source_path")"

    if [ -e "$target_path" ] && [ "$target_path" -ef "$source_path" ]; then
        echo "  [step1 reuse] Already using canonical $stage responses: $target_path"
        return 0
    fi

    mkdir -p "$(dirname "$target_path")"
    if [ -e "$target_path" ] || [ -L "$target_path" ]; then
        echo "  [step1 reuse] Replacing non-canonical $stage responses: $target_path"
        rm -f "$target_path"
    fi
    ln -s "$source_path" "$target_path"
    if [ ! -s "$target_path" ]; then
        echo "ERROR: reused $stage response symlink is not readable: $target_path -> $source_path" >&2
        rm -f "$target_path"
        return 1
    fi
    echo "  [step1 reuse] Reusing canonical $stage responses:"
    echo "    source: $source_path"
    echo "    target: $target_path"
    return 0
}

reuse_ms1_responses_if_available() {
    reuse_step1_responses_if_available "$@"
}

train_dataset_num_rows() {
    "$PYTHON_BIN" - "$TRAIN_DATA_PATH" "${MAX_SAMPLES:-}" <<'PYDATASETLEN'
import os
import sys
import datasets

path, max_samples = sys.argv[1], sys.argv[2]
if os.path.isfile(path) and path.lower().endswith(".parquet"):
    import pyarrow.parquet as pq

    n = pq.ParquetFile(path).metadata.num_rows
else:
    try:
        ds_loaded = datasets.load_from_disk(path)
        ds_raw = ds_loaded["train"] if isinstance(ds_loaded, datasets.DatasetDict) else ds_loaded
    except (ValueError, FileNotFoundError):
        ds_raw = datasets.load_dataset(path, split="train")
    n = len(ds_raw)
if max_samples:
    n = min(n, int(max_samples))
print(n)
PYDATASETLEN
}

update_position_from_global_step() {
    local update="$1"
    local batches_per_epoch="$2"
    local out_epoch=$(((update - 1) / batches_per_epoch + 1))
    local out_batch=$(((update - 1) % batches_per_epoch + 1))
    echo "$out_epoch $out_batch"
}

model_dir_for_global_update() {
    local update="$1"
    local batches_per_epoch="$2"
    local pos epoch batch
    pos="$(update_position_from_global_step "$update" "$batches_per_epoch")"
    epoch="${pos%% *}"
    batch="${pos##* }"
    update_model_save_dir "$epoch" "$batch"
}

pipeline_should_keep_update() {
    local update="$1"
    local total_updates="$2"

    if [ "$update" -eq 0 ] || [ "$update" -eq "$total_updates" ]; then
        return 0
    fi
    if [ -n "$PIPELINE_KEEP_STEPS" ]; then
        case ",$PIPELINE_KEEP_STEPS," in
            *,"$update",*) return 0 ;;
            *) return 1 ;;
        esac
    fi
    if [ "$PIPELINE_KEEP_INTERVAL" -gt 0 ] && [ $((update % PIPELINE_KEEP_INTERVAL)) -eq 0 ]; then
        return 0
    fi
    return 1
}

pipeline_milestone_fraction() {
    local update="$1"
    local total_updates="$2"
    "$PYTHON_BIN" - "$update" "$total_updates" "$EVAL_FRACTIONS" <<'PYMILESTONEFRACTION'
import sys

from recipe.opd.experiment_tracking import compute_eval_milestone_fractions

update = int(sys.argv[1])
total = int(sys.argv[2])
fractions = compute_eval_milestone_fractions(total, sys.argv[3])
print(fractions.get(update, min(1.0, update / total)))
PYMILESTONEFRACTION
}

pipeline_optimizer_step_offset() {
    local update="$1"
    if [ "$update" -le 1 ]; then
        echo 0
        return
    fi

    # Prefer the actual persisted end step of the previous segment. Generated
    # rows can be filtered, so a row-count estimate is only a fallback for the
    # first run or legacy progress markers.
    local previous_marker previous_end
    previous_marker="$(pipeline_done_marker $((update - 1)))"
    if pipeline_done_marker_structurally_valid $((update - 1)); then
        previous_end="$(awk -F= '$1 == "global_optimizer_step" {print substr($0, index($0, "=") + 1); exit}' "$previous_marker")"
        if [[ "$previous_end" =~ ^[0-9]+$ ]]; then
            echo "$previous_end"
            return
        fi
    fi
    echo $(((update - 1) * PIPELINE_FULL_BATCH_OPTIMIZER_STEPS))
}

pipeline_optimizer_global_step_after_update() {
    local update="$1"
    local total_updates="$2"
    local offset marker persisted_end checkpoint checkpoint_step
    local current_steps="$PIPELINE_FULL_BATCH_OPTIMIZER_STEPS"
    marker="$(pipeline_done_marker "$update")"
    if pipeline_done_marker_structurally_valid "$update"; then
        persisted_end="$(awk -F= '$1 == "global_optimizer_step" {print substr($0, index($0, "=") + 1); exit}' "$marker")"
        if [[ "$persisted_end" =~ ^[0-9]+$ ]]; then
            echo "$persisted_end"
            return
        fi
    fi

    offset="$(pipeline_optimizer_step_offset "$update")"
    checkpoint="$(pipeline_checkpoint_for_update "$update" "$PIPELINE_BATCHES_PER_EPOCH")"
    checkpoint_step="${checkpoint##*/global_step_}"
    if [ -n "$checkpoint" ] && [[ "$checkpoint_step" =~ ^[0-9]+$ ]]; then
        echo $((offset + checkpoint_step))
        return
    fi
    if [ "$update" -eq "$total_updates" ]; then
        current_steps="$PIPELINE_TAIL_OPTIMIZER_STEPS"
    fi
    echo $((offset + current_steps))
}

pipeline_progress_dir() {
    echo "$GEN_RESULTS_BASE_DIR/pipeline_progress"
}

pipeline_latest_model_state_file() {
    echo "$GEN_RESULTS_BASE_DIR/latest_model.env"
}

pipeline_completed_steps_file() {
    echo "$GEN_RESULTS_BASE_DIR/completed_steps.tsv"
}

pipeline_resume_models_dir() {
    echo "$GEN_RESULTS_BASE_DIR/pipeline_models"
}

pipeline_resume_model_dir() {
    local update="$1"
    echo "$(pipeline_resume_models_dir)/step$(format_pipeline_batch_id "$update")/hf_merged"
}

cleanup_old_gen_results_resume_models() {
    local keep_update="$1"
    local keep_dir dir step_dir

    [ -d "$(pipeline_resume_models_dir)" ] || return 0
    keep_dir="$(dirname "$(pipeline_resume_model_dir "$keep_update")")"
    for step_dir in "$(pipeline_resume_models_dir)"/step*; do
        [ -d "$step_dir" ] || continue
        [ "$step_dir" = "$keep_dir" ] && continue
        echo "Deleting old gen_results resume model: $step_dir" >&2
        rm -rf "$step_dir"
    done
}

archive_pipeline_keep_model() {
    local update="$1"
    local total_updates="$2"
    local batches_per_epoch="$3"
    local source_model_dir="$4"
    local rel target_model_dir

    [ "$PIPELINE_ARCHIVE_KEEP_MODE" = "copy" ] || return 0
    pipeline_should_keep_update "$update" "$total_updates" || return 0
    [ -d "$source_model_dir" ] || {
        echo "WARNING: fixed keep checkpoint source missing, not archiving: $source_model_dir" >&2
        return 0
    }

    rel="${source_model_dir#$MODEL_SAVE_BASE_DIR/}"
    target_model_dir="$PIPELINE_ARCHIVE_MODEL_DIR/$(basename "$MODEL_SAVE_BASE_DIR")/$rel"
    rm -rf "$target_model_dir"
    mkdir -p "$(dirname "$target_model_dir")"
    echo "Archiving fixed keep checkpoint step $update: $source_model_dir -> $target_model_dir" >&2
    cp -a "$source_model_dir" "$target_model_dir"
}

persist_pipeline_model_for_resume() {
    local update="$1"
    local total_updates="$2"
    local source_model_dir="$3"
    local target_model_dir target_step_dir

    if [ "$PIPELINE_STORE_RESUME_MODEL_IN_GEN_RESULTS" != "true" ] || [ "$update" -ge "$total_updates" ]; then
        printf "%s\n" "$source_model_dir"
        return
    fi
    # Milestone checkpoints are durable experiment artifacts and must remain
    # at their planned model path for evaluation and later analysis.
    if pipeline_should_keep_update "$update" "$total_updates"; then
        printf "%s\n" "$source_model_dir"
        return
    fi

    target_model_dir="$(pipeline_resume_model_dir "$update")"
    target_step_dir="$(dirname "$target_model_dir")"
    if ! [ -d "$source_model_dir" ]; then
        if hf_export_complete "$target_model_dir"; then
            echo "Reusing strictly completed resume model after interrupted marker commit: $target_model_dir" >&2
            printf "%s\n" "$target_model_dir"
            return
        fi
        echo "ERROR: cannot persist missing resume model: $source_model_dir" >&2
        exit 1
    fi
    if ! hf_export_complete "$source_model_dir"; then
        echo "ERROR: refusing to persist incomplete resume model: $source_model_dir" >&2
        exit 1
    fi

    rm -rf "$target_step_dir"
    mkdir -p "$target_step_dir"
    echo "Moving resume model into gen_results: $source_model_dir -> $target_model_dir" >&2
    mv "$source_model_dir" "$target_model_dir"
    cleanup_old_gen_results_resume_models "$update"
    printf "%s\n" "$target_model_dir"
}

cleanup_pipeline_gen_results_after_complete() {
    [ "$PIPELINE_CLEANUP_GEN_RESULTS_ON_COMPLETE" = "true" ] || return 0
    [ "$MULTI_STEP" -gt 1 ] || return 0

    if [ -d "$(pipeline_resume_models_dir)" ]; then
        echo "Deleting gen_results resume models after completed pipeline: $(pipeline_resume_models_dir)"
        rm -rf "$(pipeline_resume_models_dir)"
    fi

    local ms_dir batch_dir
    for ms_dir in "$GEN_RESULTS_BASE_DIR"/epoch*/ms*; do
        [ -d "$ms_dir" ] || continue
        for batch_dir in "$ms_dir"/batch*; do
            [ -d "$batch_dir" ] || continue
            case "$batch_dir" in
                */batch00001)
                    echo "Preserving completed step1 parquet cache for response reuse: $batch_dir"
                    ;;
                *)
                    echo "Deleting completed multi-step gen_results batch data: $batch_dir"
                    rm -rf "$batch_dir"
                    ;;
            esac
        done
    done
}

write_pipeline_latest_model_state() {
    local update="$1"
    local total_updates="$2"
    local batches_per_epoch="$3"
    local model_path="$4"
    local status="$5"
    local epoch batch marker latest_file completed_file

    epoch="$(((update - 1) / batches_per_epoch + 1))"
    batch="$(((update - 1) % batches_per_epoch + 1))"
    latest_file="$(pipeline_latest_model_state_file)"
    completed_file="$(pipeline_completed_steps_file)"
    marker="$(pipeline_done_marker "$update")"

    mkdir -p "$(dirname "$latest_file")"
    {
        printf "latest_step=%s\n" "$update"
        printf "total_steps=%s\n" "$total_updates"
        printf "epoch=%s\n" "$epoch"
        printf "batch=%s\n" "$batch"
        printf "batch_id=%s\n" "$(format_pipeline_batch_id "$batch")"
        printf "status=%s\n" "$status"
        printf "model_path=%s\n" "$model_path"
        printf "model_save_base_dir=%s\n" "$MODEL_SAVE_BASE_DIR"
        printf "done_marker=%s\n" "$marker"
        printf "updated_at=%s\n" "$(date +%Y-%m-%dT%H:%M:%S%z)"
    } > "$latest_file"

    if [ ! -s "$completed_file" ]; then
        printf "step\tepoch\tbatch\tstatus\tmodel_path\n" > "$completed_file"
    fi
    awk -F "\t" -v step="$update" 'NR == 1 || $1 != step' "$completed_file" > "${completed_file}.tmp" 2>/dev/null || cp "$completed_file" "${completed_file}.tmp"
    printf "%s\t%s\t%s\t%s\t%s\n" "$update" "$epoch" "$batch" "$status" "$model_path" >> "${completed_file}.tmp"
    mv "${completed_file}.tmp" "$completed_file"
}

read_pipeline_latest_model_path_for_step() {
    local expected_step="$1"
    local latest_file latest_step latest_model
    latest_file="$(pipeline_latest_model_state_file)"
    [ -s "$latest_file" ] || return 1
    latest_step="$(awk -F= '$1 == "latest_step" {print substr($0, index($0, "=") + 1); exit}' "$latest_file")"
    latest_model="$(awk -F= '$1 == "model_path" {print substr($0, index($0, "=") + 1); exit}' "$latest_file")"
    [ "$latest_step" = "$expected_step" ] || return 1
    [ -n "$latest_model" ] || return 1
    [ -d "$latest_model" ] || return 1
    printf "%s\n" "$latest_model"
}

read_pipeline_done_model_path_for_step() {
    local expected_step="$1"
    local marker model_path
    marker="$(pipeline_done_marker "$expected_step")"
    pipeline_done_marker_semantically_valid "$expected_step" || return 1
    model_path="$(read_pipeline_done_marker_field "$marker" model_path)" || return 1
    printf "%s\n" "$model_path"
}

read_pipeline_done_marker_field() {
    local marker="$1"
    local field="$2"

    awk -v wanted="$field" '
        index($0, "=") > 0 && substr($0, 1, index($0, "=") - 1) == wanted {
            count += 1
            value = substr($0, index($0, "=") + 1)
        }
        END {
            if (count != 1 || value == "") {
                exit 1
            }
            print value
        }
    ' "$marker"
}

pipeline_done_marker_structurally_valid() {
    local expected_step="$1"
    local marker schema marker_step status model_path optimizer_offset segment_optimizer_steps global_optimizer_step

    marker="$(pipeline_done_marker "$expected_step")"
    [ -s "$marker" ] || return 1
    schema="$(read_pipeline_done_marker_field "$marker" schema)" || return 1
    marker_step="$(read_pipeline_done_marker_field "$marker" step)" || return 1
    status="$(read_pipeline_done_marker_field "$marker" status)" || return 1
    model_path="$(read_pipeline_done_marker_field "$marker" model_path)" || return 1
    optimizer_offset="$(read_pipeline_done_marker_field "$marker" optimizer_step_offset)" || return 1
    segment_optimizer_steps="$(read_pipeline_done_marker_field "$marker" segment_optimizer_steps)" || return 1
    global_optimizer_step="$(read_pipeline_done_marker_field "$marker" global_optimizer_step)" || return 1
    [ "$schema" = "$PIPELINE_DONE_MARKER_SCHEMA" ] || return 1
    [ "$marker_step" = "$expected_step" ] || return 1
    [[ "$marker_step" =~ ^[1-9][0-9]*$ ]] || return 1
    case "$status" in
        rolling_temp|consumed_ephemeral|prunable|kept|gen_results_resume) ;;
        *) return 1 ;;
    esac
    [ -n "$model_path" ] || return 1
    [[ "$optimizer_offset" =~ ^[0-9]+$ ]] || return 1
    [[ "$segment_optimizer_steps" =~ ^[1-9][0-9]*$ ]] || return 1
    [[ "$global_optimizer_step" =~ ^[1-9][0-9]*$ ]] || return 1
    [ "$global_optimizer_step" -eq $((optimizer_offset + segment_optimizer_steps)) ] || return 1
}

pipeline_fsdp_checkpoint_complete() {
    local model_path="$1"
    local checkpoint="$model_path"
    local world_size rank shard_kind

    [ -d "$checkpoint" ] || return 1
    case "${checkpoint##*/}" in
        global_step_*) ;;
        *) checkpoint="$(find_latest_fsdp_checkpoint "$checkpoint")" ;;
    esac
    [ -n "$checkpoint" ] && [ -d "$checkpoint" ] || return 1
    [ -s "$checkpoint/fsdp_config.json" ] || return 1
    world_size="$("$PYTHON_BIN" - "$checkpoint/fsdp_config.json" <<'PYFSDPWORLDSIZE'
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))["world_size"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
print(value)
PYFSDPWORLDSIZE
    )" || return 1
    [[ "$world_size" =~ ^[1-9][0-9]*$ ]] || return 1
    for ((rank = 0; rank < world_size; rank++)); do
        for shard_kind in model optim extra_state; do
            [ -s "$checkpoint/${shard_kind}_world_size_${world_size}_rank_${rank}.pt" ] || return 1
        done
    done
}

pipeline_done_model_artifact_complete() {
    local status="$1"
    local model_path="$2"

    case "$status" in
        rolling_temp) pipeline_fsdp_checkpoint_complete "$model_path" ;;
        consumed_ephemeral)
            [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ] && [ "$model_path" = "none" ]
            ;;
        prunable|kept|gen_results_resume) hf_export_complete "$model_path" ;;
        *) return 1 ;;
    esac
}

pipeline_update_resume_checkpoint_complete() {
    local update="$1"
    local total_updates="$2"
    local batches_per_epoch="$3"
    local checkpoint

    [ "$update" -lt "$total_updates" ] || return 0
    checkpoint="$(pipeline_checkpoint_for_update "$update" "$batches_per_epoch")"
    [ -n "$checkpoint" ] || return 1
    pipeline_fsdp_checkpoint_complete "$checkpoint"
}

pipeline_done_marker_semantically_valid() {
    local expected_step="$1"
    local marker status model_path

    pipeline_done_marker_structurally_valid "$expected_step" || return 1
    marker="$(pipeline_done_marker "$expected_step")"
    status="$(read_pipeline_done_marker_field "$marker" status)" || return 1
    model_path="$(read_pipeline_done_marker_field "$marker" model_path)" || return 1
    pipeline_done_model_artifact_complete "$status" "$model_path"
}

pipeline_done_marker_can_use_frontier() {
    case "$1" in
        rolling_temp|consumed_ephemeral|prunable|gen_results_resume) return 0 ;;
        *) return 1 ;;
    esac
}

pipeline_validated_completion_frontier() {
    local total_updates="$1"
    local batches_per_epoch="$2"
    local update frontier=0

    # Retention intentionally removes superseded, non-milestone model paths.
    # A later intact checkpoint/export covers those earlier sequential updates,
    # but every marker in the covered prefix must still be structurally valid.
    for update in $(seq 1 "$total_updates"); do
        pipeline_done_marker_structurally_valid "$update" || break
        if pipeline_done_marker_semantically_valid "$update" && \
           pipeline_update_resume_checkpoint_complete "$update" "$total_updates" "$batches_per_epoch"; then
            frontier="$update"
        fi
    done
    printf "%s\n" "$frontier"
}

pipeline_done_marker() {
    local update="$1"
    echo "$(pipeline_progress_dir)/step$(format_pipeline_batch_id "$update").done"
}

write_pipeline_checkpoint_plan() {
    local total_updates="$1"
    local batches_per_epoch="$2"
    local manifest="$MODEL_SAVE_BASE_DIR/pipeline_checkpoints.tsv"

    mkdir -p "$MODEL_SAVE_BASE_DIR"
    {
        printf "step\tstatus\tpath\n"
        printf "0\tbase_policy\t%s\n" "$MODEL_PATH"
        local update dir
        for update in $(seq 1 "$total_updates"); do
            if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
                if pipeline_should_keep_update "$update" "$total_updates" && \
                   [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
                    dir="$(pipeline_temp_model_save_dir "$update")/hf_merged"
                    printf "%s\ttransient_eval\t%s\n" "$update" "$dir"
                elif [ "$update" -eq 1 ]; then
                    dir="$(pipeline_temp_model_save_dir "$update")/global_step_*"
                    printf "%s\trolling_temp\t%s\n" "$update" "$dir"
                fi
                continue
            fi
            if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$update" -lt "$total_updates" ] && \
               ! pipeline_should_keep_update "$update" "$total_updates"; then
                if [ "$update" -eq 1 ]; then
                    dir="$(pipeline_temp_model_save_dir "$update")/global_step_*"
                    printf "%s\trolling_temp\t%s\n" "$update" "$dir"
                fi
                continue
            fi
            if pipeline_should_keep_update "$update" "$total_updates"; then
                dir="$(model_dir_for_global_update "$update" "$batches_per_epoch")/hf_merged"
                printf "%s\tplanned_keep\t%s\n" "$update" "$dir"
            fi
        done
    } > "$manifest"
    echo "Pipeline checkpoint plan: $manifest"
}

mark_pipeline_update_done() {
    local update="$1"
    local total_updates="$2"
    local batches_per_epoch="$3"
    local model_dir marker marker_tmp keep_status checkpoint checkpoint_step optimizer_offset global_optimizer_step

    optimizer_offset="$(pipeline_optimizer_step_offset "$update")"
    checkpoint="$(pipeline_checkpoint_for_update "$update" "$batches_per_epoch")"
    checkpoint_step="${checkpoint##*/global_step_}"
    if ! [[ "$checkpoint_step" =~ ^[0-9]+$ ]]; then
        checkpoint_step="$PIPELINE_FULL_BATCH_OPTIMIZER_STEPS"
        if [ "$update" -eq "$total_updates" ]; then
            checkpoint_step="$PIPELINE_TAIL_OPTIMIZER_STEPS"
        fi
    fi
    global_optimizer_step=$((optimizer_offset + checkpoint_step))

    if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
        if [ "$update" -eq "$total_updates" ] || \
           { [ "$RUN_EVAL_AFTER_TRAINING" = "true" ] && pipeline_should_keep_update "$update" "$total_updates"; }; then
            model_dir="none"
            keep_status="consumed_ephemeral"
        else
            model_dir="$checkpoint"
            keep_status="rolling_temp"
        fi
    elif [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$update" -lt "$total_updates" ] && \
       ! pipeline_should_keep_update "$update" "$total_updates"; then
        model_dir="$(find_latest_fsdp_checkpoint "$(pipeline_temp_model_save_dir "$update")")"
        [ -z "$model_dir" ] && model_dir="$(pipeline_temp_model_save_dir "$update")"
        keep_status="rolling_temp"
    else
        model_dir="$(model_dir_for_global_update "$update" "$batches_per_epoch")/hf_merged"
        keep_status="prunable"
        if pipeline_should_keep_update "$update" "$total_updates"; then
            keep_status="kept"
            archive_pipeline_keep_model "$update" "$total_updates" "$batches_per_epoch" "$model_dir"
        fi
        model_dir="$(persist_pipeline_model_for_resume "$update" "$total_updates" "$model_dir")"
        if [ "$model_dir" = "$(pipeline_resume_model_dir "$update")" ]; then
            keep_status="gen_results_resume"
        fi
    fi
    if ! pipeline_done_model_artifact_complete "$keep_status" "$model_dir"; then
        echo "ERROR: refusing to commit pipeline step $update with incomplete $keep_status artifact: $model_dir" >&2
        return 1
    fi
    if ! pipeline_update_resume_checkpoint_complete "$update" "$total_updates" "$batches_per_epoch"; then
        echo "ERROR: refusing to commit pipeline step $update without a complete next-step FSDP resume checkpoint" >&2
        return 1
    fi
    marker="$(pipeline_done_marker "$update")"

    mkdir -p "$(dirname "$marker")"
    # The marker is the completion commit record.  Publish it only after the
    # supporting state is durable, and use a same-directory rename so readers
    # can observe either the old complete marker or the new complete marker,
    # never a partially written file.
    write_pipeline_latest_model_state "$update" "$total_updates" "$batches_per_epoch" "$model_dir" "$keep_status"
    marker_tmp="$(mktemp "${marker}.tmp.XXXXXX")"
    if ! {
        printf "schema=%s\n" "$PIPELINE_DONE_MARKER_SCHEMA"
        printf "step=%s\n" "$update"
        printf "status=%s\n" "$keep_status"
        printf "model_path=%s\n" "$model_dir"
        printf "optimizer_step_offset=%s\n" "$optimizer_offset"
        printf "segment_optimizer_steps=%s\n" "$checkpoint_step"
        printf "global_optimizer_step=%s\n" "$global_optimizer_step"
    } > "$marker_tmp"; then
        rm -f -- "$marker_tmp"
        return 1
    fi
    if ! mv -f -- "$marker_tmp" "$marker"; then
        rm -f -- "$marker_tmp"
        return 1
    fi
}

finalize_resumed_pipeline_update() {
    local update="$1"
    local total_updates="$2"
    local batches_per_epoch="$3"
    local refresh_marker="$4"

    if [ "$total_updates" -eq 1 ]; then
        local position epoch batch
        position="$(update_position_from_global_step "$update" "$batches_per_epoch")"
        epoch="${position%% *}"
        batch="${position##* }"
        run_single_rollout_optimizer_milestone_evals \
            "$(update_model_save_dir "$epoch" "$batch")" \
            "$(update_output_dir "$epoch" "$batch")" \
            "$PIPELINE_TOTAL_OPTIMIZER_STEPS"
    else
        backfill_completed_pipeline_milestone_eval \
            "$update" \
            "$total_updates" \
            "$batches_per_epoch"
    fi
    if [ "$refresh_marker" = "true" ]; then
        mark_pipeline_update_done "$update" "$total_updates" "$batches_per_epoch"
    fi
    # Always run retention after a validated resume hit.  These functions are
    # idempotent, so a crash after marker commit but before pruning is repaired
    # on every subsequent invocation, including when the marker already exists.
    prune_pipeline_models "$update" "$total_updates" "$batches_per_epoch"
    prune_pipeline_temp_checkpoints "$update" "$total_updates"
}

archive_or_delete_pipeline_path() {
    local src="$1"
    local label="$2"

    [ -e "$src" ] || return 0
    case "$src" in
        "$MODEL_SAVE_BASE_DIR"/epoch*/ms*/batch*|"$MODEL_SAVE_BASE_DIR"/epoch*/ms*/batch*/*) ;;
        *) echo "WARNING: refusing to prune unexpected model path: $src" >&2; return 0 ;;
    esac

    if [ "$PIPELINE_ARCHIVE_PRUNED_MODE" = "delete" ]; then
        echo "Deleting $label: $src"
        rm -rf "$src"
        return
    fi

    if [ "$PIPELINE_ARCHIVE_PRUNED_MODE" != "move" ]; then
        echo "WARNING: unknown PIPELINE_ARCHIVE_PRUNED_MODE=$PIPELINE_ARCHIVE_PRUNED_MODE; leaving $src" >&2
        return
    fi

    local rel dst
    rel="${src#$MODEL_SAVE_BASE_DIR/}"
    dst="$PIPELINE_ARCHIVE_MODEL_DIR/$(basename "$MODEL_SAVE_BASE_DIR")/$rel"
    mkdir -p "$(dirname "$dst")"
    if [ -e "$dst" ]; then
        local suffix
        suffix="$(date +%Y%m%d-%H%M%S)"
        dst="${dst}.moved_${suffix}"
    fi
    echo "Archiving $label: $src -> $dst"
    mv "$src" "$dst"
}

prune_pipeline_models() {
    local current_update="$1"
    local total_updates="$2"
    local batches_per_epoch="$3"
    local update dir current_dir child

    case "$PIPELINE_LOCAL_KEEP_POLICY" in
        last_hf_only|rolling_last_hf) ;;
        all_kept)
        if [ "$current_update" -le 1 ]; then
            return
        fi
        for update in $(seq 1 $((current_update - 1))); do
            if pipeline_should_keep_update "$update" "$total_updates"; then
                continue
            fi
            dir="$(model_dir_for_global_update "$update" "$batches_per_epoch")"
            archive_or_delete_pipeline_path "$dir" "non-kept pipeline model step $update"
        done
        return ;;
        *) echo "WARNING: unknown PIPELINE_LOCAL_KEEP_POLICY=$PIPELINE_LOCAL_KEEP_POLICY; using last_hf_only" >&2 ;;
    esac

    # Local policy: durable milestone hf_merged exports remain reproducible,
    # while sharded model/optimizer state rolls forward one segment at a time.
    # Step N keeps its FSDP checkpoint only until step N+1 has completed.
    current_dir="$(model_dir_for_global_update "$current_update" "$batches_per_epoch")"

    if [ "$current_update" -gt 1 ]; then
        for update in $(seq 1 $((current_update - 1))); do
            dir="$(model_dir_for_global_update "$update" "$batches_per_epoch")"
            if pipeline_should_keep_update "$update" "$total_updates"; then
                if [ -d "$dir" ]; then
                    for child in "$dir"/*; do
                        [ -e "$child" ] || continue
                        [ "$(basename "$child")" = "hf_merged" ] && continue
                        archive_or_delete_pipeline_path \
                            "$child" "superseded milestone shard step $update"
                    done
                fi
                continue
            fi
            archive_or_delete_pipeline_path "$dir" "older pipeline model step $update"
        done
    fi

    if [ -d "$current_dir" ]; then
        for child in "$current_dir"/*; do
            [ -e "$child" ] || continue
            [ "$(basename "$child")" = "hf_merged" ] && continue
            # The next pipeline segment restores model, optimizer, and LR
            # scheduler from this checkpoint. Keep the latest FSDP state until
            # that next segment has completed; its subsequent prune pass will
            # then remove this older batch directory.
            if [ "$current_update" -lt "$total_updates" ] && \
               [[ "$(basename "$child")" == global_step_* ]]; then
                continue
            fi
            archive_or_delete_pipeline_path "$child" "current pipeline auxiliary checkpoint step $current_update"
        done
    fi
}

link_pipeline_final_model_alias() {
    local final_model_save_dir="$1"
    local alias_dir
    alias_dir="$(pipeline_final_alias_dir)"

    hf_export_complete "$final_model_save_dir/hf_merged" || return 0
    mkdir -p "$(dirname "$alias_dir")"
    ln -sfn "$(realpath "$final_model_save_dir")" "$alias_dir"
    echo "Final model alias: $alias_dir/hf_merged -> $final_model_save_dir/hf_merged"
}

prune_pipeline_temp_checkpoints() {
    local current_update="$1"
    local total_updates="$2"
    local dir dir_name dir_step

    if [ "$RESIDENT_STUDENT_ROLLOUT" != "true" ] && \
       [ "$PIPELINE_EPHEMERAL_MODELS" != "true" ]; then
        return 0
    fi
    [ -n "$PIPELINE_TEMP_MODEL_DIR" ] || return 0
    [ -d "$PIPELINE_TEMP_MODEL_DIR" ] || return 0

    if [ "$current_update" -ge "$total_updates" ]; then
        echo "Removing rolling temporary FSDP checkpoints after final save: $PIPELINE_TEMP_MODEL_DIR"
        rm -rf "$PIPELINE_TEMP_MODEL_DIR"
        return
    fi

    for dir in "$PIPELINE_TEMP_MODEL_DIR"/step*; do
        [ -d "$dir" ] || continue
        case "$dir" in
            "$PIPELINE_TEMP_MODEL_DIR"/step*) ;;
            *) echo "WARNING: refusing to prune unexpected temp checkpoint dir: $dir" >&2; continue ;;
        esac
        dir_name="${dir##*/step}"
        if ! [[ "$dir_name" =~ ^[0-9]+$ ]]; then
            echo "WARNING: refusing to prune unparseable temp checkpoint dir: $dir" >&2
            continue
        fi
        dir_step=$((10#$dir_name))
        # During resume we walk completed updates from step 1 toward the
        # validated frontier.  Never delete that future frontier checkpoint;
        # it is the proof and resume payload that covers already-pruned steps.
        [ "$dir_step" -lt "$current_update" ] || continue
        echo "Pruning rolling temporary FSDP checkpoint: $dir"
        rm -rf "$dir"
    done
}

cleanup_pipeline_batch_data() {
    local data_dir="$1"
    if [ -z "${pipeline_batch_index:-}" ] || [ "$PIPELINE_CLEANUP_BATCH_DATA" != "true" ]; then
        return
    fi
    case "$data_dir" in
        */epoch1/ms*/batch00001)
            echo "Preserving step1 pipeline batch parquet cache for response reuse: $data_dir"
            return
            ;;
    esac
    case "$data_dir" in
        "$GEN_RESULTS_BASE_DIR"/epoch*/ms*/batch*)
            if [ -d "$data_dir" ]; then
                echo "Cleaning pipeline batch parquet cache: $data_dir"
                rm -rf "$data_dir"
            fi
            ;;
        *)
            echo "WARNING: refusing to clean unexpected gen dir: $data_dir" >&2
            ;;
    esac
}

cleanup_ephemeral_hf_export() {
    local export_dir="$1"

    [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ] || return 0
    [ -e "$export_dir" ] || return 0
    case "$export_dir" in
        "$PIPELINE_TEMP_MODEL_DIR"/step*/hf_merged|\
        "$PIPELINE_TEMP_MODEL_DIR"/step*/optimizer_eval/step*/hf_merged) ;;
        *)
            echo "ERROR: refusing to delete unexpected ephemeral HF export: $export_dir" >&2
            return 1
            ;;
    esac
    echo "Deleting consumed temporary HF export: $export_dir"
    rm -rf -- "$export_dir"
}

export_latest_fsdp_checkpoint_after_training() {
    local model_save_dir="$1"
    local target_dir="$model_save_dir/hf_merged"
    local checkpoint_path export_lora_rank=0 export_lora_alpha=0

    hf_export_complete "$target_dir" && return 0
    checkpoint_path="$(find_latest_fsdp_checkpoint "$model_save_dir")"
    if [ -z "$checkpoint_path" ] || ! pipeline_fsdp_checkpoint_complete "$checkpoint_path"; then
        echo "ERROR: cannot export an incomplete FSDP checkpoint under $model_save_dir" >&2
        return 1
    fi
    if [ "$USE_LORA" = "true" ]; then
        export_lora_rank="$LORA_RANK"
        export_lora_alpha="$LORA_ALPHA"
    fi

    echo "Exporting HF model after all training ranks have exited"
    echo "  Checkpoint: $checkpoint_path"
    echo "  Target:     $target_dir"
    if command -v free >/dev/null 2>&1; then
        free -h
    fi
    "$PYTHON_BIN" -m recipe.opd.export_checkpoint \
        --local-dir "$checkpoint_path" \
        --base-model "$MODEL_PATH" \
        --target-dir "$target_dir" \
        --lora-rank "$export_lora_rank" \
        --lora-alpha "$export_lora_alpha" \
        --trust-remote-code
    if ! hf_export_complete "$target_dir"; then
        echo "ERROR: post-training HF export is incomplete: $target_dir" >&2
        return 1
    fi
}

resolve_update_model_path() {
    local epoch="$1"
    local batch_index="$2"
    local batches_per_epoch="$3"

    if [ "$epoch" -eq 1 ] && [ "$batch_index" -eq 1 ]; then
        echo "$MODEL_PATH"
        return
    fi

    local prev_epoch prev_batch
    if [ "$batch_index" -gt 1 ]; then
        prev_epoch="$epoch"
        prev_batch=$((batch_index - 1))
    else
        prev_epoch=$((epoch - 1))
        prev_batch="$batches_per_epoch"
    fi

    local prev_update
    prev_update=$(((epoch - 1) * batches_per_epoch + batch_index - 1))

    local done_model_path marker marker_model_path
    marker="$(pipeline_done_marker "$prev_update")"
    done_model_path="$(read_pipeline_done_model_path_for_step "$prev_update" || true)"
    if [ -n "$done_model_path" ]; then
        echo "$done_model_path"
        return
    fi

    marker_model_path="$(awk -F= '$1 == "model_path" {print substr($0, index($0, "=") + 1); exit}' "$marker" 2>/dev/null || true)"
    echo "ERROR: cannot resolve previous policy model from gen_results step marker." >&2
    echo "       Required previous step: $prev_update" >&2
    echo "       Step marker: $marker" >&2
    echo "       Recorded model_path: ${marker_model_path:-missing}" >&2
    echo "       Resume state must be fixed under gen_results/gen_uid-...; no local batch-path fallback is used." >&2
    exit 1
}

resolve_epoch_model_path() {
    local epoch="$1"

    if [ "$epoch" -eq 1 ]; then
        echo "$MODEL_PATH"
        return
    fi

    local prev_epoch=$((epoch - 1))
    local prev_model_dir
    prev_model_dir="$(epoch_model_save_dir "$prev_epoch")"

    local prev_model_path
    if [ "$SAVE_MERGED_MODEL" = "true" ]; then
        prev_model_path="$prev_model_dir/hf_merged"
    else
        prev_model_path="$prev_model_dir/final"
    fi

    if [ ! -d "$prev_model_path" ]; then
        echo "ERROR: Previous epoch model not found: $prev_model_path"
        exit 1
    fi

    echo "$prev_model_path"
}

print_base_configuration() {
    echo "=========================================="
    echo "KL Divergence Training Configuration"
    echo "=========================================="
    echo ""
    echo "KL Settings:"
    echo "  Type:     $KL_TYPE"
    echo "  Method:   $KL_METHOD"
    echo "  Temp:     $TEMPERATURE"
    echo "  Clip:     $KL_TOKEN_CLIP"
    echo "  Y Mode:   $Y_MODE  (prompt_tag=$PROMPT_MODE_TAG)"
    echo "  Distill:  $DISTILL_MODE  (teacher_training_prompt=$TEACHER_TRAINING_PROMPT, use_initial_response=$USE_INITIAL_RESPONSE, teacher_thinking=$TEACHER_ENABLE_THINKING)"
    if [ "$KL_TYPE" = "jsd" ]; then
        echo "  Beta:     $BETA"
    fi
    if [ "$Y_MODE" = "y_r" ]; then
        echo "  Stage2 Mode: $FORWARD_STAGE2_MODE"
        echo "  Stage2 Filter: $FORWARD_FILTER_STAGE2 (threshold=$FORWARD_FILTER_THRESHOLD)"
    fi
    echo ""
    echo "Model Settings:"
    echo "  Base Model:      $MODEL_PATH"
    echo "  Teacher:         ${TEACHER_MODEL_PATH:-<same as student>}"
    echo "  LoRA:            $USE_LORA (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
    echo ""
    echo "Training Settings:"
    echo "  Pipeline Epochs: $TOTAL_EPOCHS"
    if [ "$MULTI_STEP" -gt 0 ]; then
        echo "  Multi-step:      $MULTI_STEP policy updates"
        if [ "$PIPELINE_STEP_MODE" = "default_batch" ]; then
            echo "  Pipeline Batch:  $PIPELINE_AUTO_CHUNK_SIZE samples/update (final tail retained: $PIPELINE_TAIL_BATCH_SIZE)"
        else
            echo "  Balanced Batch:  $PIPELINE_AUTO_CHUNK_SIZE..$PIPELINE_MAX_CHUNK_SIZE samples/update ($PIPELINE_BALANCED_REMAINDER larger partitions, increment=$PIPELINE_BALANCED_INCREMENT, no rows dropped)"
        fi
        echo "  Optimizer Steps: planned total=$PIPELINE_TOTAL_OPTIMIZER_STEPS, full chunk=$PIPELINE_FULL_BATCH_OPTIMIZER_STEPS, tail=$PIPELINE_TAIL_OPTIMIZER_STEPS"
        echo "  Eval Checkpoints: optimizer steps ${OPTIMIZER_MILESTONE_STEPS:-<none>} (fractions=$EVAL_FRACTIONS)"
        echo "  Multi-step Tag:  $MULTISTEP_TAG"
        echo "  Auto Resume:     $PIPELINE_AUTO_RESUME"
        echo "  Resume Mode:     $PIPELINE_RESUME_MODE"
        if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
            echo "  Model Retention: temporary FSDP only; HF exports deleted after milestone eval"
        else
            echo "  Keep Models:     local=${PIPELINE_LOCAL_KEEP_POLICY}, planned=${PIPELINE_KEEP_STEPS:-every ${PIPELINE_KEEP_INTERVAL} plus final} (base step 0 is recorded, not copied)"
        fi
    else
        echo "  Multi-step:      disabled; all samples (one-step)"
    fi
    echo "  Train Epochs/Round: $TRAIN_EPOCHS_PER_ROUND"
    echo "  LR:             $LEARNING_RATE"
    echo "  Batch:          $TRAIN_BATCH_SIZE"
    echo "  Student Prompt: $BASE_PROMPT_LENGTH"
    echo "  Response Len:   $MAX_RESPONSE_LENGTH"
    echo "  Expert Budget:  $EXPERT_SOLUTION_PROMPT_LENGTH"
    echo "  Teacher Prompt: $MAX_PROMPT_LENGTH (training-time)"
    echo "  Stage2 Prompt:  $STAGE2_PROMPT_LENGTH (y_r generation)"
    echo "  Max Len:        $MAX_LENGTH (KL train total)"
    echo "  Rollout Lens:   stage1=$STAGE1_ROLLOUT_MAX_MODEL_LEN stage2=$STAGE2_ROLLOUT_MAX_MODEL_LEN (chat_template_buffer=$ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER)"
    echo "  Rollout Batch:  max_num_seqs=$ROLLOUT_MAX_NUM_SEQS max_tokens(stage1/stage2)=$STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS/$STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS"
    echo "  Rollout Sample: temperature=$ROLLOUT_TEMPERATURE top_p=$ROLLOUT_TOP_P top_k=$ROLLOUT_TOP_K"
    echo "  y_o Rollout:    $Y_O_ROLLOUT_MODE"
if [ "$Y_O_ROLLOUT_MODE" = "teacher" ]; then
    echo "  Trajectory Model: $TRAJECTORY_MODEL_PATH"
    if [ "$TEACHER_TRAJECTORY_CONDITIONING" = "pi_T_x_only_v1" ]; then
        echo "  Trajectory Prompt: pi_T(.|x), original problem prompt"
    else
        echo "  Trajectory Prompt: pi_T(.|x,y*) expert rewrite ($TEACHER_TRAJECTORY_PROMPT_PATH)"
    fi
    echo "  Trajectory Cache: $TEACHER_TRAJECTORY_CACHE_MODE ($TEACHER_TRAJECTORY_CACHE_ROOT)"
elif [ "$Y_O_ROLLOUT_MODE" = "expert" ]; then
    echo "  Trajectory Model: <dataset expert_cot / y*>"
fi
if [[ "$Y_O_ROLLOUT_MODE" == skd* ]]; then
    echo "  SKD Rollout:    batch=$SKD_ROLLOUT_BATCH_SIZE max_num_seqs=$SKD_VLLM_MAX_NUM_SEQS pipeline_lanes=$SKD_PIPELINE_LANES gamma=$SKD_GAMMA top_k=$SKD_ACCEPT_TOP_K top_p=$SKD_ACCEPT_TOP_P"
    echo "  SKD Teacher:    contract=$SKD_TEACHER_PROMPT_CONTRACT prompt=$SKD_TEACHER_PROMPT_LENGTH max_model_len=$SKD_ROLLOUT_MAX_MODEL_LEN"
fi
    echo "  Grad Accum:     $GRADIENT_ACCUMULATION_STEPS ($GRADIENT_ACCUMULATION_SOURCE)"
    echo "  FSDP:           $FSDP_STRATEGY (size=$FSDP_SIZE, sp=$SP_SIZE)"
    echo "  Token/GPU:      $MAX_TOKEN_LEN_PER_GPU"
    echo ""
    echo "Data Settings:"
    echo "  Train Data:     $TRAIN_DATA_PATH"
    echo "  Precomputed y_o: ${PRECOMPUTED_Y_O_TRAJECTORY_PATH:-<none>}"
    echo "  Manual Data Override: ${DATA_PATH:-<auto>}"
    if [ "$Y_MODE" = "y_r" ] && [ -n "$CORRECTED_RESPONSES_PATH" ]; then
        echo "  Legacy Rewrite Targets: $CORRECTED_RESPONSES_PATH"
    fi
    echo "  Max Samples:    ${MAX_SAMPLES:-all}"
    if [ "$MULTI_STEP" -gt 0 ]; then
        echo "  Full Prompts:   $FULL_STAGE1_PROMPTS"
        echo "  Chunk Cache:    temporary under $GEN_RESULTS_BASE_DIR/epochN/${MULTISTEP_TAG}/batchXXXXX (cleaned after train=$PIPELINE_CLEANUP_BATCH_DATA)"
    fi
    echo ""
    echo "Output Base:"
    echo "  Run Name:       $MODEL_RUN_NAME"
    echo "  Result Key:     $RESULTS_MODEL_KEY"
    echo "  Output Dir:     $OUTPUT_BASE_DIR"
    echo "  Model Save:     $MODEL_SAVE_BASE_DIR"
    if [ "$MULTI_STEP" -gt 0 ] && [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
        echo "  Temp FSDP:      $PIPELINE_TEMP_MODEL_DIR (rolling, max one intermediate checkpoint)"
        echo "  Final Alias:    $(pipeline_final_alias_dir)/hf_merged"
    fi
    echo "  Gen Results ID: $GEN_RESULTS_RUN_ID"
    echo "  Gen ID File:    $GEN_RESULTS_RUN_ID_FILE"
    echo "  Gen Signature:  $GEN_RESULTS_RUN_SIGNATURE"
    echo "  Gen Results:    $GEN_RESULTS_BASE_DIR"
    echo "  Gen Metadata:   $GEN_RESULTS_METADATA_FILE"
    echo "  Wandb:          $WANDB_PROJECT / $WANDB_RUN_NAME_BASE (id=$WANDB_RUN_ID, mode=$WANDB_MODE)"
    echo "  Wandb State:    $WANDB_RUN_ID_FILE"
    echo ""
    echo "Evaluation:"
    echo "  Run After Training: $RUN_EVAL_AFTER_TRAINING"
    echo "  Eval Fractions:     $EVAL_FRACTIONS"
    echo "  Results File:       $RESULTS_FILE"
    if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
        echo "  Datasets:           $EVAL_DATASETS"
        echo "  Dataset Root:       $EVAL_DATASETS_DIR"
    fi
    echo ""
    echo "Diagnostic Metrics (T1–T4):"
    echo "  T2 grad_cosine_interval: $GRAD_COSINE_INTERVAL (0=off)"
    if [ -n "$CORRECTION_TOKEN_IDS" ]; then
        echo "  T3 correction_token_ids: $CORRECTION_TOKEN_IDS"
    elif [ -n "$CORRECTION_TOKEN_PHRASES" ]; then
        echo "  T3 correction_token_phrases: $CORRECTION_TOKEN_PHRASES"
    else
        echo "  T3 correction tokens: <off>"
    fi
    echo "  T4 log_difficulty_buckets: $LOG_DIFFICULTY_BUCKETS"
    if [ "${TOP_K:-0}" -gt 0 ]; then
        echo "  Top-K teacher local support matching: K=$TOP_K (paper Eq. 7-8)"
    fi
    echo ""
    echo "=========================================="
}

run_epoch() {
    local epoch="$1"
    local current_model_path="$2"
    local current_teacher_model_path="$3"
    local pipeline_batch_index="${4:-}"
    local pipeline_batch_start="${5:-}"
    local pipeline_current_batch_size="${6:-}"
    local update_index="${7:-$epoch}"
    local total_updates="${8:-$TOTAL_EPOCHS}"

    local current_gen_results_dir
    current_gen_results_dir="$(update_gen_results_dir "$epoch" "$pipeline_batch_index")"
    local current_output_dir
    current_output_dir="$(update_output_dir "$epoch" "$pipeline_batch_index")"
    local current_final_model_save_dir
    current_final_model_save_dir="$(update_model_save_dir "$epoch" "$pipeline_batch_index")"
    local current_model_save_dir="$current_final_model_save_dir"
    local current_wandb_run_name="$WANDB_RUN_NAME_BASE"
    local wandb_global_step_offset=0
    local wandb_total_training_steps_arg=""
    local wandb_total_training_steps=""
    local is_milestone_update="true"
    if [ -n "$pipeline_batch_index" ]; then
        wandb_global_step_offset="$(pipeline_optimizer_step_offset "$update_index")"
        wandb_total_training_steps="$PIPELINE_TOTAL_OPTIMIZER_STEPS"
        wandb_total_training_steps_arg="--wandb_total_training_steps $PIPELINE_TOTAL_OPTIMIZER_STEPS"
        if ! pipeline_should_keep_update "$update_index" "$total_updates"; then
            is_milestone_update="false"
        fi
    fi
    local current_data_path
    current_data_path="$(resolve_data_path_in_dir "$current_gen_results_dir")"

    local run_eval_this_update="$RUN_EVAL_AFTER_TRAINING"
    if [ -n "$pipeline_batch_index" ] && [ "$is_milestone_update" != "true" ]; then
        run_eval_this_update="false"
    fi

    local save_merged_this_update="$SAVE_MERGED_MODEL"
    local sync_resident_rollout_this_update="false"
    local resume_checkpoint_arg=""
    local prev_ckpt=""
    local rollout_lora_adapter_path=""
    local student_model_path_for_train="$current_model_path"
    if [ -n "$pipeline_batch_index" ]; then
        # Every rollout chunk is one segment of the same optimizer schedule.
        # Build the worker from the stable Base config and restore model,
        # optimizer, and LR scheduler from the previous segment checkpoint.
        student_model_path_for_train="$MODEL_PATH"
        if [ "$update_index" -gt 1 ]; then
            prev_ckpt="$(pipeline_checkpoint_for_update $((update_index - 1)) "$PIPELINE_BATCHES_PER_EPOCH")"
            if [ -z "$prev_ckpt" ]; then
                echo "ERROR: pipeline update $update_index needs the previous FSDP checkpoint" >&2
                echo "       Previous marker: $(pipeline_done_marker $((update_index - 1)))" >&2
                exit 1
            fi
            resume_checkpoint_arg="--resume_checkpoint_path $prev_ckpt --resume_checkpoint_mode initialize"
        fi
    fi
    if [ -n "$pipeline_batch_index" ] && [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
        current_model_save_dir="$(pipeline_temp_model_save_dir "$update_index")"
        save_merged_this_update="false"
        if [ "$run_eval_this_update" = "true" ]; then
            save_merged_this_update="true"
        fi
        case "$Y_O_ROLLOUT_MODE" in
            skd|skd_vllm)
                if [ "$update_index" -gt 1 ]; then
                    rollout_lora_adapter_path="$prev_ckpt/lora_adapter"
                    if [ ! -s "$rollout_lora_adapter_path/adapter_model.safetensors" ] || \
                       [ ! -s "$rollout_lora_adapter_path/adapter_config.json" ]; then
                        echo "ERROR: rolling SKD LoRA adapter is incomplete: $rollout_lora_adapter_path" >&2
                        exit 1
                    fi
                fi
                ;;
        esac
    fi
    if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
        student_model_path_for_train="$MODEL_PATH"
        sync_resident_rollout_this_update="true"
        if [ "$Y_MODE" = "y_r" ] && [ "$RESIDENT_YO_RELEASE_AFTER_STAGE1" = "true" ]; then
            sync_resident_rollout_this_update="false"
        elif [ -n "$pipeline_batch_index" ] && [ "$PIPELINE_EPHEMERAL_MODELS" != "true" ] && \
             [ "$is_milestone_update" != "true" ]; then
            save_merged_this_update="false"
            current_model_save_dir="$(pipeline_temp_model_save_dir "$update_index")"
        fi
    fi

    mkdir -p "$current_gen_results_dir"
    mkdir -p "$current_output_dir"
    mkdir -p "$current_output_dir/logs"
    mkdir -p "$current_model_save_dir"
    ensure_resident_y_o_server
    if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$update_index" -gt 1 ]; then
        sync_resident_y_o_from_checkpoint \
            "$prev_ckpt" \
            "$current_output_dir/logs" \
            "step$(format_pipeline_batch_id "$update_index")"
        if [ "${RESIDENT_YO_PRE_SYNC_PERFORMED:-false}" = "true" ]; then
            invalidate_resident_generated_data_after_pre_sync "$current_gen_results_dir"
        fi
    fi

    echo ""
    echo "=========================================="
    echo "Epoch ${epoch}/${TOTAL_EPOCHS}"
    if [ -n "$pipeline_batch_index" ]; then
        echo "Pipeline Batch: $(format_pipeline_batch_id "$pipeline_batch_index") (samples ${pipeline_batch_start}..$((pipeline_batch_start + pipeline_current_batch_size - 1)), update ${update_index}/${total_updates})"
    fi
    echo "=========================================="
    echo "Student Model: $current_model_path"
    echo "Teacher Model: ${current_teacher_model_path:-<same as student>}"
    echo "Gen Results Dir: $current_gen_results_dir"
    echo "Output Dir: $current_output_dir"
    echo "Model Save Dir: $current_model_save_dir"
    if [ "$current_model_save_dir" != "$current_final_model_save_dir" ]; then
        echo "Final Model Dir: $current_final_model_save_dir"
    fi
    echo "Training Data: $current_data_path"
    echo ""

    echo "=========================================="
    echo "Checking Training Data"
    echo "=========================================="

    if [ "$Y_MODE" = "y_r" ]; then
        if file_exists_and_nonempty "$current_data_path"; then
            echo "Found existing Stage 2 $PROMPT_MODE_TAG data: $current_data_path"
        else
            echo "Stage 2 $PROMPT_MODE_TAG data not found. Generating..."

            local stage1_output="$current_gen_results_dir/${TASK_INTERMEDIATE_PREFIX}1${Y_O_ROLLOUT_TAG}_responses.parquet"
            local stage1_prompts="$current_gen_results_dir/${TASK_INTERMEDIATE_PREFIX}1_prompts.parquet"

            if reuse_ms1_responses_if_available stage2 "$current_data_path"; then
                echo "  [step1 reuse] Stage 2 $PROMPT_MODE_TAG data ready: $current_data_path"
            else
                if file_exists_and_nonempty "$stage1_output"; then
                    echo "  Stage 1 already done: $stage1_output"
                elif reuse_ms1_responses_if_available stage1 "$stage1_output"; then
                    echo "  [step1 reuse] Stage 1 data ready: $stage1_output"
                else
                    echo "  [Stage 1] Generating initial responses..."
                    prepare_stage1_prompts "$stage1_prompts" "$pipeline_batch_start" "$pipeline_current_batch_size"

                    generate_stage1_y_o_responses \
                        "$stage1_prompts" \
                        "$stage1_output" \
                        "$current_model_path" \
                        "${current_teacher_model_path:-}" \
                        "$pipeline_batch_start" \
                        "$pipeline_current_batch_size" \
                        "$rollout_lora_adapter_path"
                fi

                release_resident_y_o_after_generation


            # Score stage1 responses only when a downstream feature needs
            # extra_info.reward. Code scoring executes tests, so it is intentionally
            # skipped for the common rewrite_all path.
            if should_score_stage1; then
                echo "  [Stage 1 score] Ensuring extra_info.reward is populated..."
                "$PYTHON_BIN" -m recipe.opd.dataset.score_stage1_reward \
                    --parquet "$stage1_output"
            else
                echo "  [Stage 1 score] Skipping; no enabled downstream feature needs extra_info.reward."
            fi

            local stage2_prompts="$current_gen_results_dir/${TASK_INTERMEDIATE_PREFIX}2_${PROMPT_MODE_TAG}_${DISTILL_MODE}_${TEACHER_MODEL_NAME}${TEACHER_THINKING_FILE_SUFFIX}_prompts.parquet"
            if file_exists_and_nonempty "$stage2_prompts"; then
                echo "  Stage 2 prompts already prepared: $stage2_prompts"
            else
                echo "  [Stage 2] Generating prompts over all stage1 rows..."
                # Generation prompt is fixed (always refine variant) — y_r_prepare.py
                # hardcodes use_initial_response=True for both OPSD and OPD. The
                # training-side teacher conditioning (TEACHER_TRAINING_PROMPT) is
                # independent and goes to run_training.py, not here.
                "$PYTHON_BIN" "$PIPELINE_DIR/y_r_prepare.py" \
                    --stage1_output "$stage1_output" \
                    --task "$TASK" \
                    --distill_mode "$DISTILL_MODE" \
                    --tokenizer_path "${current_teacher_model_path:-$current_model_path}" \
                    --max_prompt_tokens "$STAGE2_PROMPT_LENGTH" \
                    --teacher_enable_thinking "$TEACHER_ENABLE_THINKING" \
                    --output_file "$stage2_prompts"
            fi

            # TRD/y_r is always a teacher rewrite. OPD uses external Qwen3-14B;
            # OPSD uses the student's frozen step-0 Base model and additionally
            # conditions that self-teacher on y*.
            local stage2_gen_model_path="${current_teacher_model_path:-$current_model_path}"
            echo "  [Stage 2] ($DISTILL_MODE) Using TEACHER model for y_r generation: $stage2_gen_model_path"
            echo "  [Stage 2] Generating ${PROMPT_MODE_TAG} responses..."
                if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ -z "$current_teacher_model_path" ]; then
                    ensure_resident_y_o_server
                    "$PYTHON_BIN" -m recipe.opd.resident_y_o_generate \
                        --manifest "$RESIDENT_YO_MANIFEST" \
                        --input "$stage2_prompts" \
                        --output "$current_data_path" \
                        --prompt_key prompt \
                        --model_path "$MODEL_PATH" \
                        --temperature "$ROLLOUT_TEMPERATURE" \
                        --top_p "$ROLLOUT_TOP_P" \
                        --max_tokens "$MAX_RESPONSE_LENGTH"
                    release_resident_y_o_after_generation
                else
                    run_teacher_generation "$PYTHON_BIN" -m verl.trainer.main_generation_server \
                    trainer.nnodes="${NNODES}" \
                    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                    actor_rollout_ref.model.path="${stage2_gen_model_path}" \
                    actor_rollout_ref.model.trust_remote_code=true \
                    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
                    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
                    actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K" \
                    actor_rollout_ref.rollout.prompt_length=${STAGE2_PROMPT_LENGTH} \
                    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
                    actor_rollout_ref.rollout.max_model_len="${STAGE2_ROLLOUT_MAX_MODEL_LEN}" \
                    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
                    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
                    actor_rollout_ref.rollout.max_num_batched_tokens="${STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
                    actor_rollout_ref.rollout.name=vllm \
                    actor_rollout_ref.rollout.n=1 \
                        data.train_files="['${stage2_prompts}']" \
                        data.prompt_key=prompt \
                        +data.output_path="${current_data_path}"
                fi
            fi
        fi

        # Cached y_r data can bypass stage 1 generation. Ensure the managed
        # student server still releases its Ray GPU actors before teacher
        # scoring or KL training starts.
        if [ "$RESIDENT_YO_RELEASE_AFTER_STAGE1" = "true" ]; then
            stop_managed_resident_y_o_server
        fi

        # ---- P1.1: score y_1 and keep reward>=threshold (optionally also stage1_reward==0) ----
        # Runs for forward KL regardless of whether data was just generated, found on disk,
        # or reused via DATA_PATH override, so existing rewrite_all parquets can be recycled.
        if [ "$FORWARD_FILTER_STAGE2" = "true" ]; then
            local filter_suffix="filtered"
            [ "$FORWARD_FILTER_REQUIRE_STAGE1_FAILED" = "true" ] && filter_suffix="filtered_s1fail"
            local filtered_path="${current_data_path%.parquet}_${filter_suffix}.parquet"
            if file_exists_and_nonempty "$filtered_path"; then
                echo "  [Stage 2 filter] already filtered: $filtered_path"
            else
                echo "  [Stage 2 filter] scoring y_1 and keeping reward>=${FORWARD_FILTER_THRESHOLD}${FORWARD_FILTER_REQUIRE_STAGE1_FAILED:+ (stage1 failures only)} ..."
                local extra_flags=""
                [ "$FORWARD_FILTER_REQUIRE_STAGE1_FAILED" = "true" ] && extra_flags="--require_stage1_failed"
                "$PYTHON_BIN" -m recipe.opd.dataset.filter_stage2_by_reward \
                    --input  "$current_data_path" \
                    --output "$filtered_path" \
                    --threshold "$FORWARD_FILTER_THRESHOLD" \
                    $extra_flags
            fi
            current_data_path="$filtered_path"
        fi

        # ---- T4 prerequisite: stage2 parquet's extra_info.reward must equal
        # the stage1 base-model reward on the same problem (joined by
        # extra_info.index), otherwise per-difficulty bucketing degenerates
        # into "everything = hard". score_stage1_reward.py already populated
        # stage1; backfill propagates it into stage2.
        if [ "$LOG_DIFFICULTY_BUCKETS" = "true" ]; then
            local stage1_for_backfill="$current_gen_results_dir/${TASK_INTERMEDIATE_PREFIX}1${Y_O_ROLLOUT_TAG}_responses.parquet"
            if [ ! -f "$stage1_for_backfill" ]; then
                echo "  [T4 backfill] WARNING: stage1 parquet not found at $stage1_for_backfill;"
                echo "                training will fail at dataset init unless DATA_PATH points to a"
                echo "                parquet whose extra_info.reward is already populated with stage1 reward."
            else
                local backfilled_path="${current_data_path%.parquet}_s1reward.parquet"
                if file_exists_and_nonempty "$backfilled_path"; then
                    echo "  [T4 backfill] already backfilled: $backfilled_path"
                else
                    echo "  [T4 backfill] joining stage1 reward into stage2 parquet for difficulty bucketing..."
                    "$PYTHON_BIN" -m recipe.opd.dataset.backfill_stage2_reward_from_stage1 \
                        --stage1_parquet "$stage1_for_backfill" \
                        --stage2_parquet "$current_data_path" \
                        --output "$backfilled_path"
                fi
                current_data_path="$backfilled_path"
            fi
        fi
    else
        if file_exists_and_nonempty "$current_data_path"; then
            echo "Found existing Stage 1 data: $current_data_path"
        elif reuse_ms1_responses_if_available stage1 "$current_data_path"; then
            echo "  [step1 reuse] Stage 1 data ready: $current_data_path"
        else
            echo "Stage 1 data not found. Generating..."

            local stage1_prompts="$current_gen_results_dir/${TASK_INTERMEDIATE_PREFIX}1_prompts.parquet"

            prepare_stage1_prompts "$stage1_prompts" "$pipeline_batch_start" "$pipeline_current_batch_size"

            generate_stage1_y_o_responses \
                "$stage1_prompts" \
                "$current_data_path" \
                "$current_model_path" \
                "${current_teacher_model_path:-}" \
                "$pipeline_batch_start" \
                "$pipeline_current_batch_size" \
                "$rollout_lora_adapter_path"
        fi

        release_resident_y_o_after_generation


        # Score stage1 responses only when a downstream feature needs
        # extra_info.reward. Code scoring executes tests, so it is intentionally
        # skipped for the common direct training path.
        if should_score_stage1; then
            echo "  [Stage 1 score] Ensuring extra_info.reward is populated..."
            "$PYTHON_BIN" -m recipe.opd.dataset.score_stage1_reward \
                --parquet "$current_data_path"
        else
            echo "  [Stage 1 score] Skipping; no enabled downstream feature needs extra_info.reward."
        fi
    fi

    echo ""
    echo "Training data ready: $current_data_path"

    cat > "$current_output_dir/training_config.yaml" << EOF
task: $TASK
pipeline_epoch: $epoch
pipeline_total_epochs: $TOTAL_EPOCHS
multi_step: $MULTI_STEP
pipeline_auto_chunk_size: $PIPELINE_AUTO_CHUNK_SIZE
pipeline_step_mode: $PIPELINE_STEP_MODE
pipeline_max_chunk_size: $PIPELINE_MAX_CHUNK_SIZE
pipeline_balanced_remainder: $PIPELINE_BALANCED_REMAINDER
pipeline_balanced_increment: $PIPELINE_BALANCED_INCREMENT
pipeline_batch_index: ${pipeline_batch_index:-null}
pipeline_batch_start: ${pipeline_batch_start:-null}
pipeline_current_batch_size: ${pipeline_current_batch_size:-null}
pipeline_update_index: $update_index
pipeline_total_updates: $total_updates
pipeline_optimizer_step_offset: $wandb_global_step_offset
pipeline_total_optimizer_steps: ${PIPELINE_TOTAL_OPTIMIZER_STEPS:-0}
optimizer_eval_milestone_steps: ${OPTIMIZER_MILESTONE_STEPS:-null}
pipeline_multistep_tag: ${MULTISTEP_TAG:-null}
pipeline_keep_steps: ${PIPELINE_KEEP_STEPS:-null}
pipeline_keep_interval: ${PIPELINE_KEEP_INTERVAL:-null}
pipeline_local_keep_policy: ${PIPELINE_LOCAL_KEEP_POLICY}
pipeline_archive_pruned_mode: ${PIPELINE_ARCHIVE_PRUNED_MODE}
pipeline_archive_model_root: ${PIPELINE_ARCHIVE_MODEL_ROOT}
pipeline_archive_model_dir: ${PIPELINE_ARCHIVE_MODEL_DIR}
pipeline_archive_keep_mode: ${PIPELINE_ARCHIVE_KEEP_MODE}
pipeline_cleanup_batch_data: $PIPELINE_CLEANUP_BATCH_DATA
pipeline_store_resume_model_in_gen_results: $PIPELINE_STORE_RESUME_MODEL_IN_GEN_RESULTS
pipeline_cleanup_gen_results_on_complete: $PIPELINE_CLEANUP_GEN_RESULTS_ON_COMPLETE
pipeline_ephemeral_models: $PIPELINE_EPHEMERAL_MODELS
resident_student_rollout: $RESIDENT_STUDENT_ROLLOUT
pipeline_auto_resume: $PIPELINE_AUTO_RESUME
pipeline_resume_mode: $PIPELINE_RESUME_MODE
full_stage1_prompts: ${FULL_STAGE1_PROMPTS:-null}
run_eval_after_training: $run_eval_this_update
trainer_total_epochs: $TRAIN_EPOCHS_PER_ROUND
kl_type: $KL_TYPE
kl_method: $KL_METHOD
beta: $BETA
kl_token_clip: $KL_TOKEN_CLIP
top_k: $TOP_K
temperature: $TEMPERATURE
rollout_temperature: $ROLLOUT_TEMPERATURE
rollout_top_p: $ROLLOUT_TOP_P
rollout_top_k: $ROLLOUT_TOP_K
prompt_contract_version: $PROMPT_CONTRACT_VERSION
prompt_mode: $PROMPT_MODE_TAG
y_mode: $Y_MODE
y_o_rollout_mode: $Y_O_ROLLOUT_MODE
y_o_rollout_tag: ${Y_O_ROLLOUT_TAG:-}
trajectory_model_name: ${TRAJECTORY_MODEL_NAME:-null}
trajectory_model_path: ${TRAJECTORY_MODEL_PATH:-null}
teacher_trajectory_conditioning: ${TEACHER_TRAJECTORY_CONDITIONING:-}
teacher_trajectory_prompt_path: ${TEACHER_TRAJECTORY_PROMPT_PATH:-null}
teacher_trajectory_prompt_length: $TEACHER_TRAJECTORY_PROMPT_LENGTH
teacher_trajectory_rollout_max_model_len: $TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN
precomputed_y_o_trajectory_path: ${PRECOMPUTED_Y_O_TRAJECTORY_PATH:-null}
teacher_trajectory_cache_mode: $TEACHER_TRAJECTORY_CACHE_MODE
teacher_trajectory_cache_root: $TEACHER_TRAJECTORY_CACHE_ROOT
${SKD_ROLLOUT_METADATA_YAML}
use_initial_response: $USE_INITIAL_RESPONSE
forward_stage2_mode: $FORWARD_STAGE2_MODE
base_model_name: $MODEL_NAME
model_run_name: $MODEL_RUN_NAME
results_model_key: $RESULTS_MODEL_KEY
results_file: $RESULTS_FILE
gen_results_run_id: $GEN_RESULTS_RUN_ID
gen_results_run_id_file: $GEN_RESULTS_RUN_ID_FILE
gen_results_signature: $GEN_RESULTS_RUN_SIGNATURE
gen_results_signature_file: $GEN_RESULTS_RUN_SIGNATURE_FILE
gen_results_signature_details_file: $GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE
gen_results_local_signature_file: $GEN_RESULTS_LOCAL_SIGNATURE_FILE
gen_results_local_signature_details_file: $GEN_RESULTS_LOCAL_SIGNATURE_DETAILS_FILE
gen_results_base_dir: $GEN_RESULTS_BASE_DIR
gen_results_metadata_file: $GEN_RESULTS_METADATA_FILE
student_model_path: $current_model_path
teacher_model_path: ${current_teacher_model_path:-$current_model_path}
teacher_enable_thinking: $TEACHER_ENABLE_THINKING
teacher_thinking_name_tag: $TEACHER_THINKING_NAME_TAG
teacher_supervision_render_mode: $TEACHER_SUPERVISION_RENDER_MODE
teacher_rollout_render_mode: $TEACHER_ROLLOUT_RENDER_MODE
teacher_prompt_render_contract: $TEACHER_PROMPT_RENDER_CONTRACT
teacher_chat_template_token_buffer: $TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER
tuning_mode: $TUNING_MODE_TAG
use_lora: $USE_LORA
lora_rank: $LORA_RANK
lora_alpha: $LORA_ALPHA
learning_rate: $LEARNING_RATE
train_batch_size: $TRAIN_BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
gradient_accumulation_source: $GRADIENT_ACCUMULATION_SOURCE
base_prompt_length: $BASE_PROMPT_LENGTH
max_response_length: $MAX_RESPONSE_LENGTH
expert_solution_prompt_length: $EXPERT_SOLUTION_PROMPT_LENGTH
teacher_prompt_length: $MAX_PROMPT_LENGTH
stage2_prompt_length: $STAGE2_PROMPT_LENGTH
stage1_rollout_max_model_len: $STAGE1_ROLLOUT_MAX_MODEL_LEN
stage2_rollout_max_model_len: $STAGE2_ROLLOUT_MAX_MODEL_LEN
teacher_trajectory_max_num_batched_tokens: $TEACHER_TRAJECTORY_MAX_NUM_BATCHED_TOKENS
rollout_chat_template_token_buffer: $ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER
rollout_max_num_seqs: $ROLLOUT_MAX_NUM_SEQS
stage1_rollout_max_num_batched_tokens: $STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS
stage2_rollout_max_num_batched_tokens: $STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS
max_length: $MAX_LENGTH
warmup_ratio: $WARMUP_RATIO
weight_decay: $WEIGHT_DECAY
num_workers: $NUM_WORKERS
fsdp_strategy: $FSDP_STRATEGY
fsdp_size: $FSDP_SIZE
ulysses_sequence_parallel_size: $SP_SIZE
max_token_len_per_gpu: $MAX_TOKEN_LEN_PER_GPU
use_dynamic_bsz: $USE_DYNAMIC_BSZ
micro_batch_size_per_gpu: $MICRO_BATCH_SIZE_PER_GPU
use_torch_compile: $USE_TORCH_COMPILE
param_offload: $PARAM_OFFLOAD
optimizer_offload: $OPTIMIZER_OFFLOAD
offload_policy: $OFFLOAD_POLICY
data_path: $current_data_path
corrected_responses_path: ${CORRECTED_RESPONSES_PATH:-null}
max_samples: ${MAX_SAMPLES:-null}
output_dir: $current_output_dir
model_save_dir: $current_model_save_dir
gen_results_dir: $current_gen_results_dir
wandb_project: $WANDB_PROJECT
wandb_run_name: $current_wandb_run_name
wandb_run_id: $WANDB_RUN_ID
wandb_run_id_file: $WANDB_RUN_ID_FILE
wandb_run_identity: $WANDB_RUN_IDENTITY
wandb_global_step_offset: $wandb_global_step_offset
wandb_total_training_steps: ${wandb_total_training_steps:-null}
wandb_mode: $WANDB_MODE
eval_fractions: $EVAL_FRACTIONS
is_eval_milestone: $is_milestone_update
save_merged_model: $save_merged_this_update
EOF

    export PYTHONPATH="$VERL_ROOT:$PYTHONPATH"

    local cmd="$PYTHON_BIN -m torch.distributed.run \
        --nproc-per-node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node-rank=\${NODE_RANK} \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        $RECIPE_DIR/run_training.py \
        --nnodes $NNODES \
        --n_gpus_per_node $NGPUS_PER_NODE \
        --task $TASK \
        --distill_mode $DISTILL_MODE \
        --kl_type $KL_TYPE \
        --kl_method $KL_METHOD \
        --kl_token_clip $KL_TOKEN_CLIP \
        --beta $BETA \
        --temperature $TEMPERATURE \
        --student_model_path $student_model_path_for_train \
        ${current_teacher_model_path:+--teacher_model_path $current_teacher_model_path} \
        --teacher_enable_thinking $TEACHER_ENABLE_THINKING \
        --base_model_name $MODEL_NAME \
        --use_lora $USE_LORA \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --learning_rate $LEARNING_RATE \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --total_epochs $TRAIN_EPOCHS_PER_ROUND \
        --max_length $MAX_LENGTH \
        --warmup_steps_ratio $WARMUP_RATIO \
        --weight_decay $WEIGHT_DECAY \
        --min_lr_ratio 0.1 \
        --data_path $current_data_path \
        ${CORRECTED_RESPONSES_PATH:+--corrected_responses_path $CORRECTED_RESPONSES_PATH} \
        --num_workers $NUM_WORKERS \
        --fsdp_strategy $FSDP_STRATEGY \
        --fsdp_size $FSDP_SIZE \
        --ulysses_sequence_parallel_size $SP_SIZE \
        --max_token_len_per_gpu $MAX_TOKEN_LEN_PER_GPU \
        --use_dynamic_bsz $USE_DYNAMIC_BSZ \
        --micro_batch_size_per_gpu $MICRO_BATCH_SIZE_PER_GPU \
        --use_torch_compile $USE_TORCH_COMPILE \
        --param_offload $PARAM_OFFLOAD \
        --optimizer_offload $OPTIMIZER_OFFLOAD \
        --offload_policy $OFFLOAD_POLICY \
        --epoch_index $epoch \
        --output_dir $current_output_dir \
        --model_save_dir $current_model_save_dir \
        --gen_results_dir $current_gen_results_dir \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name $current_wandb_run_name \
        --wandb_run_id $WANDB_RUN_ID \
        --wandb_run_id_file $WANDB_RUN_ID_FILE \
        --wandb_run_identity $WANDB_RUN_IDENTITY \
        --wandb_global_step_offset $wandb_global_step_offset \
        $wandb_total_training_steps_arg \
        --save_merged_model false \
        --save_steps $SAVE_STEPS \
        --max_ckpt_to_keep $KEEP_LAST_N_CHECKPOINTS \
        $resume_checkpoint_arg \
        --resident_rollout_manifest $RESIDENT_YO_MANIFEST \
        --sync_resident_rollout $sync_resident_rollout_this_update \
        --async_hf_export false \
        --run_eval_after_training false \
        --eval_datasets $EVAL_DATASETS \
        --eval_datasets_dir $EVAL_DATASETS_DIR \
        --eval_fractions $EVAL_FRACTIONS \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        --use_initial_response $USE_INITIAL_RESPONSE \
        --prompt_truncation $PROMPT_TRUNCATION \
        --grad_cosine_interval $GRAD_COSINE_INTERVAL \
        --log_difficulty_buckets $LOG_DIFFICULTY_BUCKETS \
        --top_k $TOP_K \
        ${CORRECTION_TOKEN_PHRASES:+--correction_token_phrases \"$CORRECTION_TOKEN_PHRASES\"} \
        ${CORRECTION_TOKEN_IDS:+--correction_token_ids \"$CORRECTION_TOKEN_IDS\"}"

    echo "Running command:"
    echo "$cmd"
    echo ""

    local use_slurm_torchrun="$USE_SLURM_TORCHRUN"
    if [ "$use_slurm_torchrun" = "auto" ]; then
        if [ "$NNODES" -gt 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
            use_slurm_torchrun="true"
        else
            use_slurm_torchrun="false"
        fi
    fi

    if [ "$use_slurm_torchrun" = "true" ]; then
        local torchrun_cmd_file="$current_output_dir/logs/torchrun_${epoch}_${pipeline_batch_index:-0}.sh"
        cat > "$torchrun_cmd_file" <<EOF_TORCHRUN
#!/bin/bash
set -e
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:\${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_MODE="\${WANDB_MODE:-online}"
export NODE_RANK="\${SLURM_PROCID:-$NODE_RANK}"
$cmd
EOF_TORCHRUN
        chmod +x "$torchrun_cmd_file"
        echo "Launching torchrun through Slurm: $NNODES nodes x $NGPUS_PER_NODE GPUs/node"
        echo "Torchrun command file: $torchrun_cmd_file"
        local srun_job_arg=()
        if [ -n "${SLURM_JOB_ID:-}" ]; then
            srun_job_arg=(--jobid="$SLURM_JOB_ID")
        fi
        (
            srun "${srun_job_arg[@]}" --overlap \
                --cpu-bind=none \
                --nodes="$NNODES" \
                --ntasks="$NNODES" \
                --ntasks-per-node=1 \
                --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
                --gres="gpu:${SLURM_GPU_TYPE}:${NGPUS_PER_NODE}" \
                --kill-on-bad-exit=1 \
                bash "$torchrun_cmd_file"
        ) 2>&1 | tee "$current_output_dir/logs/training_$(date +%Y%m%d_%H%M%S).log"
    else
        (
            export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
            export WANDB_MODE="${WANDB_MODE:-online}"
            eval "$cmd"
        ) 2>&1 | tee "$current_output_dir/logs/training_$(date +%Y%m%d_%H%M%S).log"
    fi

    # The torchrun process, all FSDP ranks, the teacher, and optimizer must be
    # gone before a full-model merge starts. The trainer only writes the FSDP
    # checkpoint; this parent shell owns every HF export.
    if [ "$save_merged_this_update" = "true" ] || [ "$run_eval_this_update" = "true" ]; then
        if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
            if [ -z "$RESIDENT_YO_MANIFEST_USER_VALUE" ]; then
                # Sleep mode releases GPU memory but can retain a full model in
                # host RAM. Stop the managed server before loading merge inputs.
                stop_managed_resident_y_o_server
            else
                echo "WARNING: externally managed resident rollout remains alive during HF export." >&2
                echo "         Stop it externally if host RAM is constrained." >&2
            fi
        fi
        export_latest_fsdp_checkpoint_after_training "$current_model_save_dir"
    fi

    if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$sync_resident_rollout_this_update" = "true" ]; then
        latest_resident_ckpt="$(find_latest_fsdp_checkpoint "$current_model_save_dir")"
        if [ -n "$latest_resident_ckpt" ]; then
            mark_resident_y_o_synced "$latest_resident_ckpt"
        else
            echo "WARNING: resident rollout sync completed but no FSDP checkpoint was found in $current_model_save_dir" >&2
        fi
    fi

    cleanup_pipeline_batch_data "$current_gen_results_dir"

    if [ "$run_eval_this_update" = "true" ]; then
        if [ -n "$pipeline_batch_index" ] && [ "$total_updates" -eq 1 ]; then
            # ms=1 means one full-dataset rollout/training round.  Evaluate the
            # optimizer checkpoints selected by 0.25/0.5/0.75/1.0 rather than
            # collapsing all four fractions onto the lone pipeline update.
            run_single_rollout_optimizer_milestone_evals \
                "$current_model_save_dir" \
                "$current_output_dir" \
                "$PIPELINE_TOTAL_OPTIMIZER_STEPS"
        else
            run_post_training_eval_if_needed "$current_model_save_dir" "$current_output_dir" "$update_index" "$total_updates"
        fi
        cleanup_ephemeral_hf_export "$current_model_save_dir/hf_merged"
    fi

    echo ""
    echo "Epoch ${epoch} complete"
    echo "  Logs & Config: $current_output_dir"
    echo "  Model Checkpoints: $current_model_save_dir"
    echo "  Gen Results: $current_gen_results_dir"
    if [ "$run_eval_this_update" = "true" ]; then
        echo "  Eval Results: $RESULTS_FILE"
    fi
    if [ "$save_merged_this_update" = "true" ]; then
        if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
            echo "  Merged Model: consumed by eval and deleted"
        else
            echo "  Merged Model: $current_model_save_dir/hf_merged"
        fi
    else
        echo "  Temporary FSDP Checkpoint: $current_model_save_dir/global_step_*"
    fi
}


eval_results_complete() {
    local eval_model_name="$1"
    local eval_output_dir="$2"
    "$PYTHON_BIN" - "$TASK" "$RESULTS_FILE" "$eval_model_name" "$EVAL_DATASETS" "$PASS_K" "$eval_output_dir" "${CODE_EVAL_SIGNATURE:-}" <<'PYEVALCHECK'
import json
import os
import sys
from pathlib import Path

task, results_file, model_name, datasets_raw, pass_k_raw, output_dir, code_eval_signature = sys.argv[1:]
pass_k = int(pass_k_raw or 0)
datasets = [d.strip() for chunk in datasets_raw.replace(',', ' ').split() for d in [chunk.strip()] if d]
root = Path(output_dir)
missing = []

def has_number(entry, key):
    value = entry.get(key)
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def has_file(pattern):
    return any(path.is_file() and path.stat().st_size > 0 for path in root.glob(pattern))

try:
    with open(results_file) as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}
entry = results.get(model_name)
if not isinstance(entry, dict):
    missing.append(f"results entry missing: {model_name}")
    entry = {}

if task == "code":
    if not code_eval_signature:
        missing.append("code evaluation signature is missing")
    for ds in datasets:
        normalized = ds.replace('+', '_plus')
        if normalized in {"humaneval_plus", "humaneval"}:
            canonical = "humaneval_plus"
            evalplus_name = "humaneval"
            if not (has_number(entry, f"{canonical}_avg{pass_k}") and has_number(entry, f"{canonical}_pass{pass_k}")):
                missing.append(f"missing JSON metrics: {canonical}_avg{pass_k}/pass{pass_k}")
            result_pattern = f"evalplus/{code_eval_signature}/{evalplus_name}/*eval_results*.json"
            if not has_file(result_pattern):
                missing.append(f"missing EvalPlus result file: {result_pattern}")
        elif normalized in {"mbpp_plus", "mbpp"}:
            canonical = "mbpp_plus"
            evalplus_name = "mbpp"
            if not (has_number(entry, f"{canonical}_avg{pass_k}") and has_number(entry, f"{canonical}_pass{pass_k}")):
                missing.append(f"missing JSON metrics: {canonical}_avg{pass_k}/pass{pass_k}")
            result_pattern = f"evalplus/{code_eval_signature}/{evalplus_name}/*eval_results*.json"
            if not has_file(result_pattern):
                missing.append(f"missing EvalPlus result file: {result_pattern}")
        elif normalized in {"livecodebench_v6", "lcb_v6", "livecodebench"}:
            canonical = "livecodebench_v6"
            if not (has_number(entry, f"{canonical}_avg{pass_k}") and has_number(entry, f"{canonical}_pass{pass_k}")):
                missing.append(f"missing JSON metrics: {canonical}_avg{pass_k}/pass{pass_k}")
            result_pattern = f"livecodebench/{code_eval_signature}/**/*_eval_all.json"
            if not has_file(result_pattern):
                missing.append(f"missing LiveCodeBench eval file: {result_pattern}")
        else:
            missing.append(f"unknown code eval dataset: {ds}")
else:
    for ds in datasets:
        prefix = f"openai/{ds}" if ds == "gsm8k" else ds
        if pass_k == 16:
            required = [
                f"{prefix}_avg_pass1_generation_pass_16",
                f"{prefix}_pass16_generation_pass_16",
            ]
            for key in required:
                if not has_number(entry, key):
                    missing.append(f"missing JSON metric: {key}")
            parquet = root / f"{ds}_pass16_generation.parquet"
            if not (parquet.is_file() and parquet.stat().st_size > 0):
                missing.append(f"missing math generation parquet: {parquet}")
        else:
            if not any(k.startswith(f"{prefix}_") for k in entry):
                missing.append(f"missing JSON metrics for dataset: {ds}")

if missing:
    print("Eval incomplete:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)
print(f"Eval complete for {model_name}")
PYEVALCHECK
}

eval_missing_datasets() {
    local eval_model_name="$1"
    local eval_output_dir="$2"
    "$PYTHON_BIN" - "$TASK" "$RESULTS_FILE" "$eval_model_name" "$EVAL_DATASETS" "$PASS_K" "$eval_output_dir" "${CODE_EVAL_SIGNATURE:-}" <<'PYMISSCHECK'
import json
import sys
from pathlib import Path

task, results_file, model_name, datasets_raw, pass_k_raw, output_dir, code_eval_signature = sys.argv[1:]
pass_k = int(pass_k_raw or 0)
datasets = [d.strip() for chunk in datasets_raw.replace(',', ' ').split() for d in [chunk.strip()] if d]
root = Path(output_dir)

def has_number(entry, key):
    value = entry.get(key)
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def has_file(pattern):
    return any(path.is_file() and path.stat().st_size > 0 for path in root.glob(pattern))

try:
    with open(results_file) as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}
entry = results.get(model_name)
if not isinstance(entry, dict):
    entry = {}

missing = []
if task == "code":
    for ds in datasets:
        normalized = ds.replace('+', '_plus')
        if normalized in {"humaneval_plus", "humaneval"}:
            result_pattern = f"evalplus/{code_eval_signature}/humaneval/*eval_results*.json"
            if not (code_eval_signature and has_number(entry, f"humaneval_plus_avg{pass_k}") and has_number(entry, f"humaneval_plus_pass{pass_k}") and has_file(result_pattern)):
                missing.append("humaneval_plus")
        elif normalized in {"mbpp_plus", "mbpp"}:
            result_pattern = f"evalplus/{code_eval_signature}/mbpp/*eval_results*.json"
            if not (code_eval_signature and has_number(entry, f"mbpp_plus_avg{pass_k}") and has_number(entry, f"mbpp_plus_pass{pass_k}") and has_file(result_pattern)):
                missing.append("mbpp_plus")
        elif normalized in {"livecodebench_v6", "lcb_v6", "livecodebench"}:
            result_pattern = f"livecodebench/{code_eval_signature}/**/*_eval_all.json"
            if not (code_eval_signature and has_number(entry, f"livecodebench_v6_avg{pass_k}") and has_number(entry, f"livecodebench_v6_pass{pass_k}") and has_file(result_pattern)):
                missing.append("livecodebench_v6")
        else:
            missing.append(ds)
else:
    for ds in datasets:
        prefix = f"openai/{ds}" if ds == "gsm8k" else ds
        if pass_k == 16:
            required = [
                f"{prefix}_avg_pass1_generation_pass_16",
                f"{prefix}_pass16_generation_pass_16",
            ]
            parquet = root / f"{ds}_pass16_generation.parquet"
            if not (all(has_number(entry, key) for key in required) and parquet.is_file() and parquet.stat().st_size > 0):
                missing.append(ds)
        elif not any(k.startswith(f"{prefix}_") for k in entry):
            missing.append(ds)
print(" ".join(missing))
PYMISSCHECK
}

log_existing_eval_metrics_to_wandb() {
    local eval_output_dir="$1"
    local global_step="$2"
    local milestone_fraction="$3"
    local metrics_file="$eval_output_dir/metrics.json"
    local logged_marker="$eval_output_dir/.wandb_logged_step_${global_step}"

    [ -n "$WANDB_RUN_ID" ] || return 0
    [ -s "$metrics_file" ] || return 1
    [ -f "$logged_marker" ] && return 0

    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_ID="$WANDB_RUN_ID" \
    WANDB_RESUME="$WANDB_RESUME" \
    WANDB_GLOBAL_STEP="$global_step" \
    EVAL_MILESTONE_FRACTION="$milestone_fraction" \
        "$PYTHON_BIN" recipe/math_evaluation/log_metrics_wandb.py --metrics_file "$metrics_file"
    touch "$logged_marker"
}

run_post_training_eval_if_needed() {
    local final_model_save_dir="$1"
    local final_output_dir="$2"
    local update_index="${3:-}"
    local total_updates="${4:-}"
    local eval_global_step_override="${5:-}"
    local eval_milestone_fraction_override="${6:-}"
    local eval_model_path_override="${7:-}"
    local eval_model_name_override="${8:-}"
    local eval_output_dir_override="${9:-}"
    local eval_global_step="${update_index:-0}"
    local eval_milestone_fraction="1.0"
    if [ -n "$update_index" ] && [ -n "$total_updates" ] && [ "$total_updates" -gt 0 ]; then
        eval_milestone_fraction="$(pipeline_milestone_fraction "$update_index" "$total_updates")"
        eval_global_step="$(pipeline_optimizer_global_step_after_update "$update_index" "$total_updates")"
    else
        local final_checkpoint final_checkpoint_step
        final_checkpoint="$(find_latest_fsdp_checkpoint "$final_model_save_dir")"
        final_checkpoint_step="${final_checkpoint##*/global_step_}"
        if [ -n "$final_checkpoint" ] && [[ "$final_checkpoint_step" =~ ^[0-9]+$ ]]; then
            eval_global_step="$final_checkpoint_step"
        fi
    fi
    if [ -n "$eval_global_step_override" ]; then
        eval_global_step="$eval_global_step_override"
    fi
    if [ -n "$eval_milestone_fraction_override" ]; then
        eval_milestone_fraction="$eval_milestone_fraction_override"
    fi

    [ "$RUN_EVAL_AFTER_TRAINING" = "true" ] || return 0

    local eval_model_path="${eval_model_path_override:-$final_model_save_dir/hf_merged}"
    if [ "$eval_model_path" = "$final_model_save_dir/hf_merged" ] && ! hf_export_complete "$eval_model_path"; then
        export_latest_fsdp_checkpoint_after_training "$final_model_save_dir"
    fi
    if ! hf_export_complete "$eval_model_path"; then
        echo "ERROR: eval requested but merged model is missing or incomplete at $eval_model_path"
        exit 1
    fi

    local _eval_model_name="$RESULTS_MODEL_KEY"
    if [ "$MULTI_STEP" -gt 0 ] && [ -n "$update_index" ] && [ -n "$total_updates" ]; then
        _eval_model_name="${_eval_model_name}_step$(format_pipeline_batch_id "$update_index")of$(format_pipeline_batch_id "$total_updates")"
    fi
    local _eval_output_dir="gen_results/eval/${TASK}/${_eval_model_name}"
    if [ "$MULTI_STEP" -eq 0 ]; then
        _eval_model_name="${EVAL_MODEL_NAME:-$_eval_model_name}"
        _eval_output_dir="${EVAL_OUTPUT_DIR:-$_eval_output_dir}"
    fi
    if [ -n "$eval_model_name_override" ]; then
        _eval_model_name="$eval_model_name_override"
    fi
    if [ -n "$eval_output_dir_override" ]; then
        _eval_output_dir="$eval_output_dir_override"
    fi

    if eval_results_complete "$_eval_model_name" "$_eval_output_dir"; then
        if log_existing_eval_metrics_to_wandb \
            "$_eval_output_dir" "$eval_global_step" "$eval_milestone_fraction"; then
            echo "Evaluation already complete, skipping eval"
            echo "  Model:  $_eval_model_name"
            echo "  Output: $_eval_output_dir"
            return 0
        fi
        echo "Evaluation generations are complete but metrics.json is missing; rebuilding aggregate metrics."
    fi

    if [ "$TASK" = "math" ]; then
        local _missing_ds=""
        local IFS_BAK="$IFS"
        IFS=','
        for _ds in $EVAL_DATASETS; do
            local _ds_file="$EVAL_DATASETS_DIR/${_ds}/${_ds}_test.parquet"
            if ! file_exists_and_nonempty "$_ds_file"; then
                _missing_ds="$_missing_ds $_ds"
            fi
        done
        IFS="$IFS_BAK"
        if [ -n "$_missing_ds" ]; then
            echo "ERROR: missing eval dataset parquet under $EVAL_DATASETS_DIR:$_missing_ds"
            echo "       Expected layout: \$EVAL_DATASETS_DIR/<name>/<name>_test.parquet"
            exit 1
        fi
    fi

    echo ""
    echo "=========================================="
    echo "Waiting for GPU memory release before eval"
    echo "=========================================="
    local _wait_max=60
    local _waited=0
    while [ "$_waited" -lt "$_wait_max" ]; do
        local _used_mb
        _used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                   | awk '{s+=$1} END{print s+0}')
        local _threshold_mb=$((NGPUS_PER_NODE * 1024))
        if [ "$_used_mb" -lt "$_threshold_mb" ]; then
            echo "  GPUs released (total ${_used_mb} MB used, threshold ${_threshold_mb} MB)"
            break
        fi
        echo "  GPUs still busy (total ${_used_mb} MB used), waiting..."
        sleep 5
        _waited=$((_waited + 5))
    done
    if [ "$_waited" -ge "$_wait_max" ]; then
        echo "WARNING: GPU memory still high after ${_wait_max}s — eval may OOM."
    fi

    local _datasets_space
    _datasets_space=$(eval_missing_datasets "$_eval_model_name" "$_eval_output_dir")
    if [ -z "$_datasets_space" ]; then
        echo "Evaluation became complete before benchmark launch, skipping eval"
        echo "  Model:  $_eval_model_name"
        echo "  Output: $_eval_output_dir"
        return 0
    fi

    local EVAL_RECIPE_DIR="recipe/math_evaluation"
    local EVAL_BENCHMARK_SCRIPT="benchmark_kl_model.sh"
    if [ "$TASK" = "code" ]; then
        EVAL_RECIPE_DIR="recipe/code_evaluation"
        EVAL_BENCHMARK_SCRIPT="benchmark_code_model.sh"
    fi

    echo ""
    echo "=========================================="
    echo "Running evaluation via $EVAL_BENCHMARK_SCRIPT"
    echo "  Task:     $TASK"
    echo "  Model:    $eval_model_path"
    echo "  Eval TP:  $EVAL_GEN_TP"
    echo "  Step:     $eval_global_step (${eval_milestone_fraction})"
    echo "  Datasets: $_datasets_space"
    echo "  Output:   $_eval_output_dir"
    echo "=========================================="
    mkdir -p "$final_output_dir/logs"
    (
        cd "$VERL_ROOT"
        unset PYTORCH_CUDA_ALLOC_CONF
        NGPUS_PER_NODE="$NGPUS_PER_NODE" \
        NNODES="$NNODES" \
        GEN_TP="$EVAL_GEN_TP" \
        EVAL_DATASETS_DIR="$EVAL_DATASETS_DIR" \
        DATASETS="$_datasets_space" \
        PASS_K="$PASS_K" \
        EVAL_BASE_MODEL_NAME="$RESULTS_BASE_MODEL_NAME" \
        EVAL_MODEL_NAME="$_eval_model_name" \
        EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-$RESULTS_FILE}" \
        EVAL_OUTPUT_DIR="$_eval_output_dir" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_RUN_ID="$WANDB_RUN_ID" \
        WANDB_RUN_ID_FILE="$WANDB_RUN_ID_FILE" \
        WANDB_RUN_IDENTITY="$WANDB_RUN_IDENTITY" \
        WANDB_RESUME="$WANDB_RESUME" \
        WANDB_GLOBAL_STEP="$eval_global_step" \
        EVAL_GLOBAL_STEP="$eval_global_step" \
        EVAL_MILESTONE_FRACTION="$eval_milestone_fraction" \
        bash "$EVAL_RECIPE_DIR/$EVAL_BENCHMARK_SCRIPT" "$eval_model_path"
    ) 2>&1 | tee "$final_output_dir/logs/eval_$(date +%Y%m%d_%H%M%S).log"
    if [ -s "$_eval_output_dir/metrics.json" ]; then
        touch "$_eval_output_dir/.wandb_logged_step_${eval_global_step}"
    fi
}

run_single_rollout_optimizer_milestone_evals() {
    local model_save_dir="$1"
    local output_dir="$2"
    local total_optimizer_steps="$3"
    local milestone_step milestone_fraction checkpoint_path eval_model_path
    local padded_step padded_total eval_model_name eval_output_dir export_root
    local export_lora_rank=0
    local export_lora_alpha=0
    local -a optimizer_milestones=()

    [ "$RUN_EVAL_AFTER_TRAINING" = "true" ] || return 0
    [ -n "$OPTIMIZER_MILESTONE_STEPS" ] || return 0

    export_root="$model_save_dir/optimizer_eval"
    if [ "$USE_LORA" = "true" ]; then
        export_lora_rank="$LORA_RANK"
        export_lora_alpha="$LORA_ALPHA"
    fi
    IFS=',' read -r -a optimizer_milestones <<< "$OPTIMIZER_MILESTONE_STEPS"
    for milestone_step in "${optimizer_milestones[@]}"; do
        [ -n "$milestone_step" ] || continue
        milestone_fraction="$(pipeline_milestone_fraction "$milestone_step" "$total_optimizer_steps")"
        printf -v padded_step '%05d' "$milestone_step"
        printf -v padded_total '%05d' "$total_optimizer_steps"
        eval_model_name="${RESULTS_MODEL_KEY}_optstep${padded_step}of${padded_total}"
        eval_output_dir="gen_results/eval/${TASK}/${eval_model_name}"

        # Retention removes non-final model shards after their metrics are
        # complete.  A resumed run must be able to reuse those results without
        # demanding that the already-consumed checkpoint still exists.
        if eval_results_complete "$eval_model_name" "$eval_output_dir" && \
           log_existing_eval_metrics_to_wandb \
               "$eval_output_dir" "$milestone_step" "$milestone_fraction"; then
            echo "Optimizer milestone $milestone_step evaluation already complete"
            continue
        fi

        if [ "$milestone_step" -eq "$total_optimizer_steps" ]; then
            eval_model_path="$model_save_dir/hf_merged"
        else
            checkpoint_path="$model_save_dir/global_step_${milestone_step}"
            eval_model_path="$export_root/step${padded_step}/hf_merged"
            if ! hf_export_complete "$eval_model_path"; then
                if ! pipeline_fsdp_checkpoint_complete "$checkpoint_path"; then
                    echo "ERROR: optimizer milestone checkpoint is missing or incomplete: $checkpoint_path" >&2
                    return 1
                fi
                echo "Exporting optimizer milestone $milestone_step for evaluation"
                "$PYTHON_BIN" -m recipe.opd.export_checkpoint \
                    --local-dir "$checkpoint_path" \
                    --base-model "$MODEL_PATH" \
                    --target-dir "$eval_model_path" \
                    --lora-rank "$export_lora_rank" \
                    --lora-alpha "$export_lora_alpha" \
                    --trust-remote-code
            fi
        fi

        run_post_training_eval_if_needed \
            "$model_save_dir" \
            "$output_dir" \
            1 \
            1 \
            "$milestone_step" \
            "$milestone_fraction" \
            "$eval_model_path" \
            "$eval_model_name" \
            "$eval_output_dir"
        cleanup_ephemeral_hf_export "$eval_model_path"
    done
}

backfill_completed_pipeline_milestone_eval() {
    local update="$1"
    local total_updates="$2"
    local batches_per_epoch="$3"
    local position epoch batch eval_model_dir eval_model_path

    [ "$RUN_EVAL_AFTER_TRAINING" = "true" ] || return 0
    pipeline_should_keep_update "$update" "$total_updates" || return 0

    if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
        local padded_update padded_total eval_model_name eval_output_dir eval_global_step milestone_fraction
        printf -v padded_update '%05d' "$update"
        printf -v padded_total '%05d' "$total_updates"
        eval_model_name="${RESULTS_MODEL_KEY}_step${padded_update}of${padded_total}"
        eval_output_dir="gen_results/eval/${TASK}/${eval_model_name}"
        eval_global_step="$(pipeline_optimizer_global_step_after_update "$update" "$total_updates")"
        milestone_fraction="$(pipeline_milestone_fraction "$update" "$total_updates")"
        if eval_results_complete "$eval_model_name" "$eval_output_dir" && \
           log_existing_eval_metrics_to_wandb \
               "$eval_output_dir" "$eval_global_step" "$milestone_fraction"; then
            echo "Ephemeral milestone step $update evaluation already complete"
            return 0
        fi
        echo "ERROR: ephemeral milestone step $update is committed but its evaluation results are incomplete." >&2
        echo "       Its temporary model was intentionally deleted after evaluation." >&2
        return 1
    fi

    position="$(update_position_from_global_step "$update" "$batches_per_epoch")"
    epoch="${position%% *}"
    batch="${position##* }"
    eval_model_dir="$(model_dir_for_global_update "$update" "$batches_per_epoch")"
    if ! hf_export_complete "$eval_model_dir/hf_merged"; then
        eval_model_path="$(read_pipeline_done_model_path_for_step "$update" || true)"
        if [ "${eval_model_path##*/}" = "hf_merged" ]; then
            eval_model_dir="$(dirname "$eval_model_path")"
        fi
    fi
    if hf_export_complete "$eval_model_dir/hf_merged"; then
        run_post_training_eval_if_needed \
            "$eval_model_dir" \
            "$(update_output_dir "$epoch" "$batch")" \
            "$update" \
            "$total_updates"
    else
        echo "ERROR: milestone step $update is marked complete but its strict HF artifact is unavailable." >&2
        echo "       Refusing to skip a reproducibility/eval milestone." >&2
        return 1
    fi
}

format_duration_seconds() {
    local total_seconds="${1:-0}"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf '%02dh:%02dm:%02ds' "$hours" "$minutes" "$seconds"
}

ntfy_send() {
    local title="$1"
    local tags="$2"
    local message="$3"

    if [ "${NTFY_ENABLED:-true}" != "true" ]; then
        return 0
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "WARNING: ntfy notification skipped because curl is not available" >&2
        return 0
    fi

    curl -fsS \
        --retry 2 \
        --connect-timeout 5 \
        --max-time "${NTFY_TIMEOUT_SECONDS:-10}" \
        -H "Title: $title" \
        -H "Tags: $tags" \
        --data-binary "$message" \
        "$NTFY_URL" >/dev/null 2>&1 || \
        echo "WARNING: failed to send ntfy notification to $NTFY_URL" >&2
}

notify_training_start() {
    NTFY_JOB_STARTED="true"
    NTFY_JOB_START_TS=$(date +%s)
    local host
    host=$(hostname 2>/dev/null || echo unknown)
    local message
    message=$(cat << EOF
Status: started
Host: $host
Run: $MODEL_RUN_NAME
Model: $MODEL_NAME
Teacher: $TEACHER_MODEL_NAME
Mode: $DISTILL_FAMILY $KL_TYPE/$KL_METHOD $Y_MODE
Multi-step: $MULTI_STEP
Eval: $RUN_EVAL_AFTER_TRAINING
EOF
)
    ntfy_send "OPD training started: $MODEL_NAME" "rocket" "$message"
}

notify_training_exit() {
    local status="$1"
    if [ "${NTFY_JOB_STARTED:-false}" != "true" ]; then
        return 0
    fi

    local end_ts
    end_ts=$(date +%s)
    local duration
    duration=$(format_duration_seconds $((end_ts - NTFY_JOB_START_TS)))
    local host
    host=$(hostname 2>/dev/null || echo unknown)
    local state title tags
    if [ "$status" -eq 0 ]; then
        state="finished"
        title="OPD training finished: $MODEL_NAME"
        tags="white_check_mark"
    else
        state="failed"
        title="OPD training failed: $MODEL_NAME"
        tags="warning"
    fi

    local message
    message=$(cat << EOF
Status: $state
Exit code: $status
Duration: $duration
Host: $host
Run: $MODEL_RUN_NAME
Model: $MODEL_NAME
Teacher: $TEACHER_MODEL_NAME
Mode: $DISTILL_FAMILY $KL_TYPE/$KL_METHOD $Y_MODE
Multi-step: $MULTI_STEP
Eval: $RUN_EVAL_AFTER_TRAINING
EOF
)
    ntfy_send "$title" "$tags" "$message"
}

on_training_exit() {
    local status="$?"
    stop_managed_resident_y_o_server || true
    notify_training_exit "$status"
    exit "$status"
}

# =============================================================================
# Main
# =============================================================================

trap on_training_exit EXIT

print_base_configuration
notify_training_start

if [ "$TOTAL_EPOCHS" -gt 1 ] && [ -n "$DATA_PATH" ]; then
    echo "WARNING: DATA_PATH is manually set and will be reused for every epoch: $DATA_PATH"
fi

if [ "$TOTAL_EPOCHS" -gt 1 ] && [ -n "$CORRECTED_RESPONSES_PATH" ]; then
    echo "WARNING: CORRECTED_RESPONSES_PATH is manually set and will be reused for every epoch: $CORRECTED_RESPONSES_PATH"
fi

mkdir -p "$GEN_RESULTS_BASE_DIR"
if [ ! -s "$GEN_RESULTS_METADATA_FILE" ]; then
    cat > "$GEN_RESULTS_METADATA_FILE" << EOF
run_date: $RUN_DATE
gen_results_run_id: $GEN_RESULTS_RUN_ID
gen_results_run_id_file: $GEN_RESULTS_RUN_ID_FILE
gen_results_signature: $GEN_RESULTS_RUN_SIGNATURE
gen_results_signature_file: $GEN_RESULTS_RUN_SIGNATURE_FILE
gen_results_signature_details_file: $GEN_RESULTS_RUN_SIGNATURE_DETAILS_FILE
gen_results_base_dir: $GEN_RESULTS_BASE_DIR
model_save_base_dir: $MODEL_SAVE_BASE_DIR
output_base_dir: $OUTPUT_BASE_DIR
results_file: $RESULTS_FILE
results_model_key: $RESULTS_MODEL_KEY
model_run_name: $MODEL_RUN_NAME
run_descriptor: $RUN_DESCRIPTOR
experiment_tag: $EXPERIMENT_TAG
distill_family: $DISTILL_FAMILY
student_model_name: $MODEL_NAME
student_model_path: $MODEL_PATH
teacher_model_name: $TEACHER_MODEL_NAME
teacher_model_path: ${TEACHER_MODEL_PATH:-$MODEL_PATH}
teacher_enable_thinking: $TEACHER_ENABLE_THINKING
teacher_thinking_name_tag: $TEACHER_THINKING_NAME_TAG
teacher_supervision_render_mode: $TEACHER_SUPERVISION_RENDER_MODE
teacher_rollout_render_mode: $TEACHER_ROLLOUT_RENDER_MODE
teacher_prompt_render_contract: $TEACHER_PROMPT_RENDER_CONTRACT
teacher_chat_template_token_buffer: $TEACHER_CHAT_TEMPLATE_TOKEN_BUFFER
distill_mode: $DISTILL_MODE
prompt_mode: $PROMPT_MODE_TAG
y_mode: $Y_MODE
y_o_rollout_mode: $Y_O_ROLLOUT_MODE
y_o_rollout_tag: ${Y_O_ROLLOUT_TAG:-}
trajectory_model_name: ${TRAJECTORY_MODEL_NAME:-}
trajectory_model_path: ${TRAJECTORY_MODEL_PATH:-}
teacher_trajectory_conditioning: ${TEACHER_TRAJECTORY_CONDITIONING:-}
teacher_trajectory_prompt_path: ${TEACHER_TRAJECTORY_PROMPT_PATH:-}
teacher_trajectory_prompt_length: $TEACHER_TRAJECTORY_PROMPT_LENGTH
teacher_trajectory_rollout_max_model_len: $TEACHER_TRAJECTORY_ROLLOUT_MAX_MODEL_LEN
precomputed_y_o_trajectory_path: ${PRECOMPUTED_Y_O_TRAJECTORY_PATH:-}
teacher_trajectory_cache_mode: $TEACHER_TRAJECTORY_CACHE_MODE
teacher_trajectory_cache_root: $TEACHER_TRAJECTORY_CACHE_ROOT
${SKD_ROLLOUT_METADATA_YAML}
kl_type: $KL_TYPE
kl_method: $KL_METHOD
teacher_training_prompt: $TEACHER_TRAINING_PROMPT
use_initial_response: $USE_INITIAL_RESPONSE
kl_token_clip: $KL_TOKEN_CLIP
beta: $BETA
top_k: $TOP_K
forward_stage2_mode: $FORWARD_STAGE2_MODE
forward_filter_stage2: $FORWARD_FILTER_STAGE2
forward_filter_threshold: $FORWARD_FILTER_THRESHOLD
forward_filter_require_stage1_failed: $FORWARD_FILTER_REQUIRE_STAGE1_FAILED
multi_step: $MULTI_STEP
pipeline_auto_chunk_size: $PIPELINE_AUTO_CHUNK_SIZE
pipeline_step_mode: $PIPELINE_STEP_MODE
pipeline_max_chunk_size: $PIPELINE_MAX_CHUNK_SIZE
pipeline_balanced_remainder: $PIPELINE_BALANCED_REMAINDER
pipeline_balanced_increment: $PIPELINE_BALANCED_INCREMENT
pipeline_total_steps: ${PIPELINE_TOTAL_STEPS:-0}
pipeline_total_optimizer_steps: ${PIPELINE_TOTAL_OPTIMIZER_STEPS:-0}
optimizer_eval_milestone_steps: ${OPTIMIZER_MILESTONE_STEPS:-null}
pipeline_batches_per_epoch: ${PIPELINE_BATCHES_PER_EPOCH:-$TOTAL_EPOCHS}
pipeline_auto_resume: $PIPELINE_AUTO_RESUME
pipeline_resume_mode: $PIPELINE_RESUME_MODE
pipeline_keep_steps: ${PIPELINE_KEEP_STEPS:-}
pipeline_keep_interval: ${PIPELINE_KEEP_INTERVAL:-}
pipeline_local_keep_policy: ${PIPELINE_LOCAL_KEEP_POLICY}
pipeline_archive_pruned_mode: ${PIPELINE_ARCHIVE_PRUNED_MODE}
pipeline_archive_model_root: ${PIPELINE_ARCHIVE_MODEL_ROOT}
pipeline_archive_model_dir: ${PIPELINE_ARCHIVE_MODEL_DIR}
pipeline_archive_keep_mode: ${PIPELINE_ARCHIVE_KEEP_MODE}
pipeline_cleanup_batch_data: $PIPELINE_CLEANUP_BATCH_DATA
pipeline_temp_model_dir: $PIPELINE_TEMP_MODEL_DIR
pipeline_ephemeral_models: $PIPELINE_EPHEMERAL_MODELS
resident_student_rollout: $RESIDENT_STUDENT_ROLLOUT
pipeline_final_alias_dir: $(pipeline_final_alias_dir)
total_train_samples: ${TOTAL_TRAIN_SAMPLES:-}
pipeline_dropped_samples: ${PIPELINE_DROPPED_SAMPLES:-0}
base_prompt_length: $BASE_PROMPT_LENGTH
max_prompt_length: $MAX_PROMPT_LENGTH
expert_solution_prompt_length: $EXPERT_SOLUTION_PROMPT_LENGTH
stage2_prompt_length: $STAGE2_PROMPT_LENGTH
max_response_length: $MAX_RESPONSE_LENGTH
max_length: $MAX_LENGTH
max_token_len_per_gpu: $MAX_TOKEN_LEN_PER_GPU
stage1_rollout_max_model_len: $STAGE1_ROLLOUT_MAX_MODEL_LEN
stage2_rollout_max_model_len: $STAGE2_ROLLOUT_MAX_MODEL_LEN
teacher_trajectory_max_num_batched_tokens: $TEACHER_TRAJECTORY_MAX_NUM_BATCHED_TOKENS
rollout_chat_template_token_buffer: $ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER
rollout_gpu_memory_utilization: $ROLLOUT_GPU_MEMORY_UTILIZATION
rollout_max_num_seqs: $ROLLOUT_MAX_NUM_SEQS
stage1_rollout_max_num_batched_tokens: $STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS
stage2_rollout_max_num_batched_tokens: $STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS
temperature: $TEMPERATURE
rollout_temperature: $ROLLOUT_TEMPERATURE
rollout_top_p: $ROLLOUT_TOP_P
rollout_top_k: $ROLLOUT_TOP_K
prompt_contract_version: $PROMPT_CONTRACT_VERSION
max_samples: ${MAX_SAMPLES:-all}
data_path: ${DATA_PATH:-}
train_data_path: $TRAIN_DATA_PATH
corrected_responses_path: ${CORRECTED_RESPONSES_PATH:-}
tuning_mode: $TUNING_MODE_TAG
use_lora: $USE_LORA
lora_rank: $LORA_RANK
lora_alpha: $LORA_ALPHA
learning_rate: $LEARNING_RATE
train_batch_size: $TRAIN_BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
gradient_accumulation_source: $GRADIENT_ACCUMULATION_SOURCE
train_epochs_per_round: $TRAIN_EPOCHS_PER_ROUND
total_epochs: $TOTAL_EPOCHS
warmup_ratio: $WARMUP_RATIO
weight_decay: $WEIGHT_DECAY
num_workers: $NUM_WORKERS
nnodes: $NNODES
ngpus_per_node: $NGPUS_PER_NODE
gen_tp: $GEN_TP
fsdp_strategy: $FSDP_STRATEGY
fsdp_size: $FSDP_SIZE
ulysses_sequence_parallel_size: $SP_SIZE
use_dynamic_bsz: $USE_DYNAMIC_BSZ
micro_batch_size_per_gpu: $MICRO_BATCH_SIZE_PER_GPU
use_torch_compile: $USE_TORCH_COMPILE
param_offload: $PARAM_OFFLOAD
optimizer_offload: $OPTIMIZER_OFFLOAD
offload_policy: $OFFLOAD_POLICY
save_merged_model: $SAVE_MERGED_MODEL
save_steps: $SAVE_STEPS
keep_last_n_checkpoints: $KEEP_LAST_N_CHECKPOINTS
run_eval_after_training: $RUN_EVAL_AFTER_TRAINING
eval_datasets: $EVAL_DATASETS
eval_datasets_dir: $EVAL_DATASETS_DIR
pass_k: $PASS_K
wandb_project: $WANDB_PROJECT
wandb_run_name: $WANDB_RUN_NAME_BASE
wandb_run_id: $WANDB_RUN_ID
wandb_run_id_file: $WANDB_RUN_ID_FILE
wandb_run_identity: $WANDB_RUN_IDENTITY
wandb_resume: $WANDB_RESUME
wandb_mode: $WANDB_MODE
wandb_total_training_steps: ${PIPELINE_TOTAL_OPTIMIZER_STEPS:-0}
eval_fractions: $EVAL_FRACTIONS
EOF
fi
if ! grep -q '^wandb_run_id_file:' "$GEN_RESULTS_METADATA_FILE"; then
    {
        printf 'wandb_project: %s\n' "$WANDB_PROJECT"
        printf 'wandb_run_name: %s\n' "$WANDB_RUN_NAME_BASE"
        printf 'wandb_run_id: %s\n' "$WANDB_RUN_ID"
        printf 'wandb_run_id_file: %s\n' "$WANDB_RUN_ID_FILE"
        printf 'wandb_run_identity: %s\n' "$WANDB_RUN_IDENTITY"
    } >> "$GEN_RESULTS_METADATA_FILE"
fi
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$MODEL_SAVE_BASE_DIR"

if [ "$MULTI_STEP" -gt 0 ]; then
    TOTAL_PIPELINE_UPDATES="$PIPELINE_TOTAL_STEPS"

    echo ""
    echo "=========================================="
    echo "Offline Multi-Step Pipeline"
    echo "=========================================="
    echo "Dataset rows:         $TOTAL_TRAIN_SAMPLES"
    echo "Multi-step updates:   $MULTI_STEP"
    echo "Auto chunk size:      $PIPELINE_AUTO_CHUNK_SIZE"
    echo "Total policy updates: $TOTAL_PIPELINE_UPDATES"
    if [ "$PIPELINE_STEP_MODE" = "default_batch" ]; then
        echo "Final tail rows:      $PIPELINE_TAIL_BATCH_SIZE (retained)"
    else
        echo "Balanced partitions:  $PIPELINE_AUTO_CHUNK_SIZE..$PIPELINE_MAX_CHUNK_SIZE rows/update ($PIPELINE_BALANCED_REMAINDER larger partitions, increment=$PIPELINE_BALANCED_INCREMENT, none dropped)"
    fi
    echo "Optimizer steps:      planned total=$PIPELINE_TOTAL_OPTIMIZER_STEPS, full chunk=$PIPELINE_FULL_BATCH_OPTIMIZER_STEPS, tail=$PIPELINE_TAIL_OPTIMIZER_STEPS"
    echo "Eval checkpoints:     optimizer steps ${OPTIMIZER_MILESTONE_STEPS:-<none>}"
    echo "Multi-step tag:       $MULTISTEP_TAG"
    echo "Auto resume:          $PIPELINE_AUTO_RESUME"
    echo "Resume mode:          $PIPELINE_RESUME_MODE"
    if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
        echo "Model retention:      ephemeral, eval exports are deleted after use"
    else
        echo "Model keep policy:    local=${PIPELINE_LOCAL_KEEP_POLICY}, planned=${PIPELINE_KEEP_STEPS:-every ${PIPELINE_KEEP_INTERVAL} plus final}, archive=${PIPELINE_ARCHIVE_PRUNED_MODE}:${PIPELINE_ARCHIVE_MODEL_DIR}"
    fi
    echo "=========================================="

    if [ -z "$DATA_PATH" ]; then
        ensure_full_stage1_prompts
    else
        echo "Using pre-generated one-step DATA_PATH; stage1 prompt preparation is skipped."
    fi
    write_pipeline_checkpoint_plan "$TOTAL_PIPELINE_UPDATES" "$PIPELINE_BATCHES_PER_EPOCH"

    FINAL_PIPELINE_MODEL_DIR="$(update_model_save_dir "$TOTAL_EPOCHS" "$PIPELINE_BATCHES_PER_EPOCH")"
    if hf_export_complete "$FINAL_PIPELINE_MODEL_DIR/hf_merged"; then
        if [ "$PIPELINE_AUTO_RESUME" = "true" ]; then
            echo ""
            echo "=========================================="
            echo "Offline multi-step pipeline already complete, skipping all chunks"
            echo "  Found strict HF export: $FINAL_PIPELINE_MODEL_DIR/hf_merged"
            echo "=========================================="
            for ((COMPLETED_UPDATE = 1; COMPLETED_UPDATE < TOTAL_PIPELINE_UPDATES; COMPLETED_UPDATE++)); do
                backfill_completed_pipeline_milestone_eval \
                    "$COMPLETED_UPDATE" \
                    "$TOTAL_PIPELINE_UPDATES" \
                    "$PIPELINE_BATCHES_PER_EPOCH"
            done
            FINAL_MARKER_REFRESH="true"
            if read_pipeline_done_model_path_for_step "$TOTAL_PIPELINE_UPDATES" >/dev/null; then
                FINAL_MARKER_REFRESH="false"
            fi
            finalize_resumed_pipeline_update \
                "$TOTAL_PIPELINE_UPDATES" \
                "$TOTAL_PIPELINE_UPDATES" \
                "$PIPELINE_BATCHES_PER_EPOCH" \
                "$FINAL_MARKER_REFRESH"
        else
            echo "ERROR: final pipeline model already exists and PIPELINE_AUTO_RESUME=false: $FINAL_PIPELINE_MODEL_DIR/hf_merged"
            echo "       Use PIPELINE_RESUME_MODE=resume_matching to continue, or use fresh output dirs."
            exit 1
        fi
    else
    GLOBAL_UPDATE=0
    VALIDATED_PROGRESS_FRONTIER="$(
        pipeline_validated_completion_frontier "$TOTAL_PIPELINE_UPDATES" "$PIPELINE_BATCHES_PER_EPOCH"
    )"
    if [ "$VALIDATED_PROGRESS_FRONTIER" -gt 0 ]; then
        echo "Validated sequential pipeline completion through step $VALIDATED_PROGRESS_FRONTIER"
    fi
    PIPELINE_STOPPED_EARLY_UPDATE=""
    PIPELINE_STOPPED_EARLY_EPOCH=""
    PIPELINE_STOPPED_EARLY_BATCH=""
    for EPOCH in $(seq 1 "$TOTAL_EPOCHS"); do
        for PIPELINE_BATCH_INDEX in $(seq 1 "$PIPELINE_BATCHES_PER_EPOCH"); do
            GLOBAL_UPDATE=$((GLOBAL_UPDATE + 1))
            read -r BATCH_START PIPELINE_CURRENT_BATCH_SIZE < <(
                pipeline_partition_start_size "$PIPELINE_BATCH_INDEX"
            )
            UPDATE_MODEL_DIR="$(update_model_save_dir "$EPOCH" "$PIPELINE_BATCH_INDEX")"
            UPDATE_DONE_MARKER="$UPDATE_MODEL_DIR/hf_merged"
            UPDATE_PROGRESS_MARKER="$(pipeline_done_marker "$GLOBAL_UPDATE")"

            UPDATE_MARKER_DIRECT_VALID="false"
            UPDATE_MARKER_CHAIN_VALID="false"
            UPDATE_MARKER_STATUS=""
            UPDATE_HF_VALID="false"
            if pipeline_done_marker_structurally_valid "$GLOBAL_UPDATE"; then
                UPDATE_MARKER_STATUS="$(read_pipeline_done_marker_field "$UPDATE_PROGRESS_MARKER" status)"
                if pipeline_done_marker_semantically_valid "$GLOBAL_UPDATE" && \
                   pipeline_update_resume_checkpoint_complete \
                       "$GLOBAL_UPDATE" "$TOTAL_PIPELINE_UPDATES" "$PIPELINE_BATCHES_PER_EPOCH"; then
                    UPDATE_MARKER_DIRECT_VALID="true"
                elif [ "$GLOBAL_UPDATE" -le "$VALIDATED_PROGRESS_FRONTIER" ]; then
                    # Milestone exports are durable experiment artifacts and
                    # must remain directly valid for reproducibility/eval.
                    # Only artifacts intentionally rolled by retention may be
                    # covered by a later complete sequential checkpoint.
                    if pipeline_done_marker_can_use_frontier "$UPDATE_MARKER_STATUS"; then
                        UPDATE_MARKER_CHAIN_VALID="true"
                    fi
                fi
            fi
            UPDATE_RESUME_MODEL="$(pipeline_resume_model_dir "$GLOBAL_UPDATE")"
            if hf_export_complete "$UPDATE_DONE_MARKER" || hf_export_complete "$UPDATE_RESUME_MODEL"; then
                UPDATE_HF_VALID="true"
            fi

            if [ "$UPDATE_MARKER_DIRECT_VALID" = "true" ] || \
               [ "$UPDATE_MARKER_CHAIN_VALID" = "true" ] || \
               [ "$UPDATE_HF_VALID" = "true" ]; then
                if [ "$PIPELINE_AUTO_RESUME" = "true" ]; then
                    echo ""
                    echo "=========================================="
                    echo "Epoch ${EPOCH}/${TOTAL_EPOCHS} batch $(format_pipeline_batch_id "$PIPELINE_BATCH_INDEX") / step ${GLOBAL_UPDATE} — already complete, skipping"
                    if [ "$UPDATE_MARKER_DIRECT_VALID" = "true" ]; then
                        echo "  Found validated marker and artifact: $UPDATE_PROGRESS_MARKER"
                    elif [ "$UPDATE_MARKER_CHAIN_VALID" = "true" ]; then
                        echo "  Found validated marker covered by completion frontier $VALIDATED_PROGRESS_FRONTIER: $UPDATE_PROGRESS_MARKER"
                    else
                        echo "  Found strict HF export: $UPDATE_DONE_MARKER"
                    fi
                    echo "=========================================="
                    UPDATE_MARKER_REFRESH="false"
                    if [ "$UPDATE_MARKER_DIRECT_VALID" != "true" ] && \
                       [ "$UPDATE_MARKER_CHAIN_VALID" != "true" ]; then
                        UPDATE_MARKER_REFRESH="true"
                    fi
                    finalize_resumed_pipeline_update \
                        "$GLOBAL_UPDATE" \
                        "$TOTAL_PIPELINE_UPDATES" \
                        "$PIPELINE_BATCHES_PER_EPOCH" \
                        "$UPDATE_MARKER_REFRESH"
                    continue
                fi
                echo "ERROR: completed pipeline step already exists and PIPELINE_AUTO_RESUME=false: step ${GLOBAL_UPDATE}"
                if [ -f "$UPDATE_PROGRESS_MARKER" ]; then
                    echo "       Found: $UPDATE_PROGRESS_MARKER"
                else
                    echo "       Found strict HF export: $UPDATE_DONE_MARKER"
                fi
                echo "       Use PIPELINE_RESUME_MODE=resume_matching to continue, or use fresh output dirs."
                exit 1
            fi

            if [ -e "$UPDATE_PROGRESS_MARKER" ]; then
                RECORDED_STEP="$(read_pipeline_done_marker_field "$UPDATE_PROGRESS_MARKER" step 2>/dev/null || true)"
                RECORDED_STATUS="$(read_pipeline_done_marker_field "$UPDATE_PROGRESS_MARKER" status 2>/dev/null || true)"
                RECORDED_MODEL_PATH="$(read_pipeline_done_marker_field "$UPDATE_PROGRESS_MARKER" model_path 2>/dev/null || true)"
                echo "ERROR: refusing to resume from an invalid pipeline completion marker: $UPDATE_PROGRESS_MARKER" >&2
                echo "       expected_step=$GLOBAL_UPDATE recorded_step=${RECORDED_STEP:-missing}" >&2
                echo "       status=${RECORDED_STATUS:-missing} model_path=${RECORDED_MODEL_PATH:-missing}" >&2
                echo "       The marker fields or their corresponding HF/FSDP artifact failed validation." >&2
                exit 1
            fi

            if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ] || \
               [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
                CURRENT_MODEL_PATH="$MODEL_PATH"
            else
                CURRENT_MODEL_PATH="$(resolve_update_model_path "$EPOCH" "$PIPELINE_BATCH_INDEX" "$PIPELINE_BATCHES_PER_EPOCH")"
            fi
            CURRENT_TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH"
            run_epoch \
                "$EPOCH" \
                "$CURRENT_MODEL_PATH" \
                "$CURRENT_TEACHER_MODEL_PATH" \
                "$PIPELINE_BATCH_INDEX" \
                "$BATCH_START" \
                "$PIPELINE_CURRENT_BATCH_SIZE" \
                "$GLOBAL_UPDATE" \
                "$TOTAL_PIPELINE_UPDATES"
            mark_pipeline_update_done "$GLOBAL_UPDATE" "$TOTAL_PIPELINE_UPDATES" "$PIPELINE_BATCHES_PER_EPOCH"
            prune_pipeline_models "$GLOBAL_UPDATE" "$TOTAL_PIPELINE_UPDATES" "$PIPELINE_BATCHES_PER_EPOCH"
            prune_pipeline_temp_checkpoints "$GLOBAL_UPDATE" "$TOTAL_PIPELINE_UPDATES"
            if [ -n "$PIPELINE_STOP_AFTER_UPDATE" ] && [ "$GLOBAL_UPDATE" -ge "$PIPELINE_STOP_AFTER_UPDATE" ]; then
                PIPELINE_STOPPED_EARLY_UPDATE="$GLOBAL_UPDATE"
                PIPELINE_STOPPED_EARLY_EPOCH="$EPOCH"
                PIPELINE_STOPPED_EARLY_BATCH="$PIPELINE_BATCH_INDEX"
                echo ""
                echo "=========================================="
                echo "Reached PIPELINE_STOP_AFTER_UPDATE=$PIPELINE_STOP_AFTER_UPDATE; stopping pipeline after step $GLOBAL_UPDATE"
                echo "=========================================="
                break 2
            fi
        done
    done
    fi

    if [ -n "${PIPELINE_STOPPED_EARLY_UPDATE:-}" ]; then
        FINAL_OUTPUT_DIR="$(update_output_dir "$PIPELINE_STOPPED_EARLY_EPOCH" "$PIPELINE_STOPPED_EARLY_BATCH")"
        FINAL_MODEL_SAVE_DIR="$(update_model_save_dir "$PIPELINE_STOPPED_EARLY_EPOCH" "$PIPELINE_STOPPED_EARLY_BATCH")"
    else
        FINAL_OUTPUT_DIR="$(update_output_dir "$TOTAL_EPOCHS" "$PIPELINE_BATCHES_PER_EPOCH")"
        FINAL_MODEL_SAVE_DIR="$(update_model_save_dir "$TOTAL_EPOCHS" "$PIPELINE_BATCHES_PER_EPOCH")"
        if [ "$PIPELINE_EPHEMERAL_MODELS" != "true" ]; then
            link_pipeline_final_model_alias "$FINAL_MODEL_SAVE_DIR"
        fi
        cleanup_pipeline_gen_results_after_complete
    fi
else
    for EPOCH in $(seq 1 "$TOTAL_EPOCHS"); do
        EPOCH_MODEL_DIR="$(epoch_model_save_dir "$EPOCH")"
        if [ "$SAVE_MERGED_MODEL" = "true" ]; then
            EPOCH_DONE_MARKER="$EPOCH_MODEL_DIR/hf_merged"
        else
            EPOCH_DONE_MARKER="$EPOCH_MODEL_DIR/final/config.json"
        fi

        if { [ "$SAVE_MERGED_MODEL" = "true" ] && hf_export_complete "$EPOCH_DONE_MARKER"; } || \
           { [ "$SAVE_MERGED_MODEL" != "true" ] && [ -f "$EPOCH_DONE_MARKER" ]; }; then
            echo ""
            echo "=========================================="
            echo "Epoch ${EPOCH}/${TOTAL_EPOCHS} — already complete, skipping"
            echo "  Found: $EPOCH_DONE_MARKER"
            echo "=========================================="
            continue
        fi

        CURRENT_MODEL_PATH="$(resolve_epoch_model_path "$EPOCH")"
        CURRENT_TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH"
        run_epoch "$EPOCH" "$CURRENT_MODEL_PATH" "$CURRENT_TEACHER_MODEL_PATH"
    done

    FINAL_OUTPUT_DIR="$(epoch_output_dir "$TOTAL_EPOCHS")"
    FINAL_MODEL_SAVE_DIR="$(epoch_model_save_dir "$TOTAL_EPOCHS")"
fi

if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
    if [ "$MULTI_STEP" -gt 0 ]; then
        if [ "$PIPELINE_EPHEMERAL_MODELS" != "true" ] && \
           [ -z "${PIPELINE_STOPPED_EARLY_UPDATE:-}" ] && [ "$TOTAL_PIPELINE_UPDATES" -gt 1 ]; then
            run_post_training_eval_if_needed "$FINAL_MODEL_SAVE_DIR" "$FINAL_OUTPUT_DIR" "$TOTAL_PIPELINE_UPDATES" "$TOTAL_PIPELINE_UPDATES"
        fi
    else
        run_post_training_eval_if_needed "$FINAL_MODEL_SAVE_DIR" "$FINAL_OUTPUT_DIR"
    fi
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Final Epoch: $TOTAL_EPOCHS"
if [ -n "$MULTISTEP_TAG" ]; then
    echo "Multi-step: $MULTISTEP_TAG"
fi
echo "Run Name: $MODEL_RUN_NAME"
echo "Result Key: $RESULTS_MODEL_KEY"
echo "Gen Results ID: $GEN_RESULTS_RUN_ID"
echo "Gen ID File: $GEN_RESULTS_RUN_ID_FILE"
echo "Logs & Config: $FINAL_OUTPUT_DIR"
if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
    echo "Model Artifacts: deleted after evaluation"
else
    echo "Model Checkpoints: $FINAL_MODEL_SAVE_DIR"
fi
echo "Gen Results Base: $GEN_RESULTS_BASE_DIR"
echo "Gen Metadata: $GEN_RESULTS_METADATA_FILE"
if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
    echo "Eval Results: $RESULTS_FILE"
fi
if [ "$PIPELINE_EPHEMERAL_MODELS" = "true" ]; then
    echo "Merged Model: temporary eval exports consumed and deleted"
elif [ "$SAVE_MERGED_MODEL" = "true" ]; then
    echo "Merged Model: $FINAL_MODEL_SAVE_DIR/hf_merged"
    if [ -e "$(pipeline_final_alias_dir)/hf_merged" ]; then
        echo "Final Alias: $(pipeline_final_alias_dir)/hf_merged"
    fi
    if [ "$TASK" = "code" ]; then
        echo "Benchmark: TASK=code bash recipe/code_evaluation/benchmark_code_model.sh $FINAL_MODEL_SAVE_DIR/hf_merged"
    else
        echo "Benchmark: bash recipe/math_evaluation/benchmark_kl_model.sh $FINAL_MODEL_SAVE_DIR/hf_merged"
    fi
else
    echo "FSDP Checkpoints: $FINAL_MODEL_SAVE_DIR/global_step_*"
fi
echo "Wandb Dashboard: https://wandb.ai/$WANDB_PROJECT"
