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

# =============================================================================
# Configuration
# =============================================================================
# Distillation Mode
DISTILL_MODE=${DISTILL_MODE:-"opsd"}   # opsd: teacher = student, teacher prompt embeds y* (expert solution).
                                        # opd : teacher ≠ student, teacher prompt has NO expert reference.
case "$DISTILL_MODE" in
    opsd|opd) ;;
    *) echo "ERROR: DISTILL_MODE must be opsd or opd (got: $DISTILL_MODE)" >&2; exit 1 ;;
esac

# KL Training Settings
KL_TYPE=${KL_TYPE:-"reverse"}          # reverse | forward | jsd (OPSD generalized JSD)
KL_METHOD=${KL_METHOD:-"monte_carlo"}  # monte_carlo or full_vocab (JSD at beta∈(0,1) requires full_vocab)
KL_TOKEN_CLIP=${KL_TOKEN_CLIP:-0.06}    # Per-token KL/JSD clip. OPSD 8B uses 0.06; 0 disables.
TEMPERATURE=${TEMPERATURE:-0.7}         # Softmax temperature
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
PROMPT_TRUNCATION=${PROMPT_TRUNCATION:-"true"}  # When prompt+response > MAX_LENGTH: false=cut response tail (loses gradient), true=cut **Your Initial Solution:** block in teacher prompt (preserves response).
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

# Top-K teacher local support matching (Fu et al. 2026, arXiv:2603.25562).
# When > 0 (and KL_METHOD=full_vocab), the per-position KL is replaced by a
# truncated KL on the top-K teacher-supported tokens, with both teacher and
# student renormalized inside the support. Paper default K=32. JSD is not
# supported with TOP_K because the mixture only makes sense over full vocab.
TOP_K=${TOP_K:-0}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-""}  # Empty means same as student
USE_LORA=${USE_LORA:-true}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}

# Training Settings
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
USER_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
GRADIENT_ACCUMULATION_SOURCE="default"
[ -n "$USER_GRADIENT_ACCUMULATION_STEPS" ] && GRADIENT_ACCUMULATION_SOURCE="user"
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}  # Outer pipeline epochs
TRAIN_EPOCHS_PER_ROUND=${TRAIN_EPOCHS_PER_ROUND:-1}  # Trainer epochs for each pipeline epoch
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.005}

# Data Settings
DATA_PATH=${DATA_PATH:-""}  # Optional override; reused across epochs if set
CORRECTED_RESPONSES_PATH=${CORRECTED_RESPONSES_PATH:-""}  # Optional legacy two-file mode
MAX_SAMPLES=${MAX_SAMPLES:-""}  # For testing, leave empty for full data
# Offline multi-step on-policy optimization: 0 keeps the historical one-step path.
# When >0, this is the number of policy updates. Chunk size is derived as
# floor(num_train_rows / MULTI_STEP), and tail rows are dropped.
MULTI_STEP=${MULTI_STEP:-0}
USER_PIPELINE_BATCH_SIZE="${PIPELINE_BATCH_SIZE:-}"
PIPELINE_AUTO_CHUNK_SIZE=0  # Internal auto-computed chunk size when MULTI_STEP>0.
PIPELINE_KEEP_STEPS=${PIPELINE_KEEP_STEPS:-""}      # Optional comma list, e.g. 0,8,16,24,32,39.
PIPELINE_KEEP_INTERVAL=${PIPELINE_KEEP_INTERVAL:-""}  # Default = ceil(MULTI_STEP / 5).
PIPELINE_CLEANUP_BATCH_DATA=${PIPELINE_CLEANUP_BATCH_DATA:-"true"}
PIPELINE_TEMP_MODEL_DIR=${PIPELINE_TEMP_MODEL_DIR:-""}
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
GEN_TP=${GEN_TP:-1}  # Tensor parallel for generation

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
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"/data/data/jiangli/models"}  # Base model save dir; epoch subdirs are appended
WANDB_PROJECT=${WANDB_PROJECT:-"verl-kl-training"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}  # Base wandb run name; _epochN is appended
export WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-"true"}  # Merge LoRA after training
SAVE_STEPS=${SAVE_STEPS:-100}                   # FSDP ckpt every N optimizer steps (crash recovery)
KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS:-1}  # Rolling window; per-epoch hf_merged is always preserved

# Experimental resident student rollout path. When enabled, y_o generation is served
# by a long-lived student vLLM server and KL training syncs updated LoRA/student
# weights back into it, so intermediate batches no longer need blocking HF merge.
USER_RESIDENT_STUDENT_ROLLOUT="${RESIDENT_STUDENT_ROLLOUT:-}"
RESIDENT_STUDENT_ROLLOUT=${RESIDENT_STUDENT_ROLLOUT:-"true"}
RESIDENT_YO_MANIFEST=${RESIDENT_YO_MANIFEST:-""}
RESIDENT_YO_AUTOSTART=${RESIDENT_YO_AUTOSTART:-"true"}
RESIDENT_YO_LOAD_FORMAT=${RESIDENT_YO_LOAD_FORMAT:-"auto"}
ASYNC_HF_EXPORT=${ASYNC_HF_EXPORT:-"false"}

# Ray GPU resources are scheduler leases, not CUDA memory. A sleeping resident
# y_o vLLM releases memory but still holds its Ray GPU actors, which prevents
# the OPD y_r teacher standalone vLLM from scheduling on the same allocation.
if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$Y_MODE" = "y_r" ] && [ "$DISTILL_MODE" = "opd" ]; then
    if [ -n "$USER_RESIDENT_STUDENT_ROLLOUT" ]; then
        echo "ERROR: RESIDENT_STUDENT_ROLLOUT=true is incompatible with Y_MODE=y_r + DISTILL_MODE=opd." >&2
        echo "       The resident student y_o vLLM keeps Ray GPU resources while sleeping," >&2
        echo "       so the teacher y_r standalone vLLM cannot acquire GPUs." >&2
        echo "       Use RESIDENT_STUDENT_ROLLOUT=false for this path." >&2
        exit 1
    fi
    echo "WARNING: disabling RESIDENT_STUDENT_ROLLOUT for Y_MODE=y_r + DISTILL_MODE=opd;" >&2
    echo "         teacher y_r generation needs standalone Ray GPU resources." >&2
    RESIDENT_STUDENT_ROLLOUT="false"
fi

# Evaluation Settings
RUN_EVAL_AFTER_TRAINING=${RUN_EVAL_AFTER_TRAINING:-"true"}
# EVAL_DATASETS=${EVAL_DATASETS:-"aime24,aime25,math500,hmmt25"} DEFAULT_DATASETS="math500 hmmt25 beyondaime amobench gsm8k"
EVAL_DATASETS=${EVAL_DATASETS:-"aime24,aime25,hmmt25,beyondaime,amobench"}
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-"/data/data/jiangli/huggingface/datasets"}
PASS_K=${PASS_K:-16}

# Paths
# Layout: recipe/opd/run/<this_script>.sh
#         recipe/opd/<run_training.py, run_eval_suite.py>
#         recipe/opd/generation/<stage1/2 prep, scoring>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$SCRIPT_DIR")"                           # recipe/opd
VERL_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"               # repo root
PIPELINE_DIR="$RECIPE_DIR/generation"                           # recipe/opd/generation

# Training data path (for prompts)
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"/data/data/jiangli/data/DeepScaleR-Cleaned"}

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
BASE_PROMPT_LENGTH=${BASE_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
EXPERT_SOLUTION_PROMPT_LENGTH=${EXPERT_SOLUTION_PROMPT_LENGTH:-3072}
STAGE1_PROMPT_LENGTH=$BASE_PROMPT_LENGTH

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

TRAIN_MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
if [ "$BASE_PROMPT_LENGTH" -gt "$TRAIN_MAX_PROMPT_LENGTH" ]; then
    TRAIN_MAX_PROMPT_LENGTH=$BASE_PROMPT_LENGTH
fi
DERIVED_MAX_LENGTH=$((TRAIN_MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
if [ -z "$MAX_LENGTH" ]; then
    MAX_LENGTH=$DERIVED_MAX_LENGTH
elif [ "$MAX_LENGTH" -lt "$DERIVED_MAX_LENGTH" ]; then
    echo "WARNING: MAX_LENGTH=$MAX_LENGTH is below derived no-clip budget $DERIVED_MAX_LENGTH; response or prompt truncation may occur." >&2
fi

ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER=${ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER:-4096}
STAGE1_ROLLOUT_MAX_MODEL_LEN=$((STAGE1_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER))
STAGE2_ROLLOUT_MAX_MODEL_LEN=$((STAGE2_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER))
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
if [ "$STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS" -lt "$STAGE1_ROLLOUT_MAX_MODEL_LEN" ]; then
    STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS=$STAGE1_ROLLOUT_MAX_MODEL_LEN
fi
STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
if [ "$STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS" -lt "$STAGE2_ROLLOUT_MAX_MODEL_LEN" ]; then
    STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS=$STAGE2_ROLLOUT_MAX_MODEL_LEN
fi
if [ -z "$MAX_TOKEN_LEN_PER_GPU" ]; then
    MAX_TOKEN_LEN_PER_GPU=49152
    if [ "$MAX_LENGTH" -gt "$MAX_TOKEN_LEN_PER_GPU" ]; then
        MAX_TOKEN_LEN_PER_GPU=$MAX_LENGTH
    fi
elif [ "$MAX_TOKEN_LEN_PER_GPU" -lt "$MAX_LENGTH" ]; then
    echo "WARNING: MAX_TOKEN_LEN_PER_GPU=$MAX_TOKEN_LEN_PER_GPU is below MAX_LENGTH=$MAX_LENGTH; dynamic batching may split or fail on long samples." >&2
fi
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
# Teacher model name — used to qualify stage2 file paths AND the experiment
# tag. For OPSD (teacher empty → defaults to student) we fall back to the
# student name so the qualifier is still present and consistent.
if [ -n "${TEACHER_MODEL:-}" ]; then
    TEACHER_MODEL_NAME="$TEACHER_MODEL"
elif [ -n "${TEACHER_MODEL_PATH:-}" ]; then
    TEACHER_MODEL_NAME="${TEACHER_MODEL_PATH##*/}"
else
    TEACHER_MODEL_NAME="${MODEL_NAME}"
fi

EXPERIMENT_TAG="kl_${KL_TYPE}_${KL_METHOD}_${PROMPT_MODE_TAG}_${CLIP_TAG}${BETA_TAG}${TOPK_TAG}${Y_O_FILTER_TAG}_${DISTILL_MODE}_${TEACHER_TRAINING_PROMPT}"
# For OPD, also include teacher name so different teachers on same student
# don't collide. (For OPSD teacher = student, redundant — skipped.)
if [ "$DISTILL_MODE" = "opd" ]; then
    EXPERIMENT_TAG="${EXPERIMENT_TAG}_teacher${TEACHER_MODEL_NAME}"
fi

# y_r path no longer reads FORWARD_STAGE2_MODE — y_r_prepare.py always processes
# every stage1 row. Reward-based filtering of y_r happens post-generation via
# FORWARD_FILTER_STAGE2=true (recipe/opd/dataset/filter_stage2_by_reward.py).

# OPD requires a distinct teacher (teacher ≠ student).
if [ "$DISTILL_MODE" = "opd" ] && [ -z "$TEACHER_MODEL_PATH" ]; then
    echo "ERROR: DISTILL_MODE=opd requires TEACHER_MODEL_PATH to be set (teacher must differ from student)."
    echo "       Leave it empty only for DISTILL_MODE=opsd (teacher = student)."
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

if [ -n "$USER_PIPELINE_BATCH_SIZE" ]; then
    echo "ERROR: PIPELINE_BATCH_SIZE is deprecated. Set MULTI_STEP=<num_policy_updates> instead."
    exit 1
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

if ! [[ "$MULTI_STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MULTI_STEP must be a non-negative integer (0 means one-step; got: $MULTI_STEP)."
    exit 1
fi

if [ "$MULTI_STEP" -gt 0 ]; then
    if [ "$SAVE_MERGED_MODEL" != "true" ]; then
        echo "ERROR: MULTI_STEP>0 requires SAVE_MERGED_MODEL=true so the next chunk can load the updated policy."
        exit 1
    fi
    if [ -n "$DATA_PATH" ]; then
        echo "ERROR: MULTI_STEP>0 is incompatible with DATA_PATH override; it must generate fresh on-policy data per chunk."
        exit 1
    fi
    if [ -n "$CORRECTED_RESPONSES_PATH" ]; then
        echo "ERROR: MULTI_STEP>0 is incompatible with CORRECTED_RESPONSES_PATH legacy override."
        exit 1
    fi
    if [ -n "$PIPELINE_KEEP_INTERVAL" ] && ! [[ "$PIPELINE_KEEP_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: PIPELINE_KEEP_INTERVAL must be a positive integer when set (got: $PIPELINE_KEEP_INTERVAL)."
        exit 1
    fi
fi

PIPELINE_TOTAL_STEPS=0
PIPELINE_BATCHES_PER_EPOCH=0
PIPELINE_DROPPED_SAMPLES=0
MULTISTEP_TAG=""
if [ "$MULTI_STEP" -gt 0 ]; then
    TOTAL_TRAIN_SAMPLES="$(python3 - "$TRAIN_DATA_PATH" "${MAX_SAMPLES:-}" <<'PYDATASETLEN_EARLY'
import sys
import datasets

path, max_samples = sys.argv[1], sys.argv[2]
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
    if [ "$MULTI_STEP" -gt "$TOTAL_TRAIN_SAMPLES" ]; then
        echo "ERROR: MULTI_STEP=$MULTI_STEP is larger than available samples ($TOTAL_TRAIN_SAMPLES)."
        exit 1
    fi
    if [ "$TOTAL_EPOCHS" -ne 1 ]; then
        echo "ERROR: MULTI_STEP>0 expects TOTAL_EPOCHS=1; use MULTI_STEP as the total number of policy updates."
        exit 1
    fi
    PIPELINE_TOTAL_STEPS="$MULTI_STEP"
    PIPELINE_BATCHES_PER_EPOCH="$MULTI_STEP"
    PIPELINE_AUTO_CHUNK_SIZE=$((TOTAL_TRAIN_SAMPLES / MULTI_STEP))
    PIPELINE_DROPPED_SAMPLES=$((TOTAL_TRAIN_SAMPLES - PIPELINE_BATCHES_PER_EPOCH * PIPELINE_AUTO_CHUNK_SIZE))
    if [ "$PIPELINE_AUTO_CHUNK_SIZE" -lt 1 ]; then
        echo "ERROR: computed pipeline chunk size is <1 (samples=$TOTAL_TRAIN_SAMPLES, MULTI_STEP=$MULTI_STEP)."
        exit 1
    fi
    if [ -z "$USER_GRADIENT_ACCUMULATION_STEPS" ]; then
        PIPELINE_GLOBAL_MICRO_BATCH=$((TRAIN_BATCH_SIZE * NGPUS_PER_NODE * NNODES))
        if [ "$PIPELINE_GLOBAL_MICRO_BATCH" -lt 1 ]; then
            echo "ERROR: invalid global micro batch size: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE NGPUS_PER_NODE=$NGPUS_PER_NODE NNODES=$NNODES."
            exit 1
        fi
        GRADIENT_ACCUMULATION_STEPS=$(((PIPELINE_AUTO_CHUNK_SIZE + PIPELINE_GLOBAL_MICRO_BATCH - 1) / PIPELINE_GLOBAL_MICRO_BATCH))
        GRADIENT_ACCUMULATION_SOURCE="auto_one_update_per_chunk"
    fi
    if [ -z "$PIPELINE_KEEP_INTERVAL" ]; then
        PIPELINE_KEEP_INTERVAL=$(((PIPELINE_TOTAL_STEPS + 4) / 5))
    fi
    MULTISTEP_TAG="ms${PIPELINE_TOTAL_STEPS}"
    EXPERIMENT_TAG="${EXPERIMENT_TAG}_${MULTISTEP_TAG}"
fi

if ! [[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GRADIENT_ACCUMULATION_STEPS must be a positive integer (got: $GRADIENT_ACCUMULATION_STEPS)."
    exit 1
fi

# Base directories and run naming
DISTILL_FAMILY="$(echo "$DISTILL_MODE" | tr '[:lower:]' '[:upper:]')"
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d-%H%M%S)}
OPTIMIZATION_STEP_TAG="${MULTISTEP_TAG:-ms1}"
RUN_DESCRIPTOR="${PROMPT_MODE_TAG}_kl_${KL_TYPE}_${KL_METHOD}_${CLIP_TAG}${BETA_TAG}${TOPK_TAG}${Y_O_FILTER_TAG}_${TEACHER_TRAINING_PROMPT}_${OPTIMIZATION_STEP_TAG}_${RUN_DATE}"
if [ "$DISTILL_MODE" = "opd" ]; then
    MODEL_RUN_NAME="teacher${TEACHER_MODEL_NAME}_${RUN_DESCRIPTOR}"
else
    MODEL_RUN_NAME="$RUN_DESCRIPTOR"
fi
RESULTS_MODEL_KEY="${MODEL_NAME}_${DISTILL_FAMILY}_${MODEL_RUN_NAME}"
RESULTS_BASE_MODEL_NAME="${DISTILL_FAMILY}/${MODEL_NAME}"
RESULTS_FILE="$VERL_ROOT/results/${DISTILL_FAMILY}/${MODEL_NAME}.json"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_BASE_DIR="$VERL_ROOT/outputs/${DISTILL_FAMILY}/${MODEL_NAME}/${MODEL_RUN_NAME}"
else
    OUTPUT_BASE_DIR="$OUTPUT_DIR"
fi

if [ -z "$MODEL_SAVE_DIR" ] || [ "$MODEL_SAVE_DIR" = "/data/data/jiangli/models" ]; then
    MODEL_SAVE_BASE_DIR="/scratch/l/luli/jiangli/ckpt/${DISTILL_FAMILY}/${MODEL_NAME}/${MODEL_RUN_NAME}"
else
    MODEL_SAVE_BASE_DIR="$MODEL_SAVE_DIR"
fi

RESIDENT_YO_MANIFEST_USER_VALUE="$RESIDENT_YO_MANIFEST"

# gen_results uses a short random run id by default. In fresh mode the id is
# intentionally date-free (gen_uid-*) because semantic identity is carried by
# the signature below, not by RUN_DATE. In resume_matching mode we scan existing
# gen_results metadata for the same signature, switch back to that run's
# model_save_base_dir/gen_results_base_dir, and continue from its checkpoints.
GEN_RESULTS_ROOT="${GEN_RESULTS_ROOT:-$VERL_ROOT/gen_results}"
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
    local meta sig mtime

    [ -d "$GEN_RESULTS_ROOT" ] || return 0
    while IFS= read -r meta; do
        sig="$(gen_results_metadata_value "$meta" gen_results_signature)"
        [ "$sig" = "$signature" ] || continue
        mtime="$(stat -c %Y "$meta" 2>/dev/null || printf '0')"
        if [ -z "$best" ] || [ "$mtime" -gt "$best_mtime" ]; then
            best="$meta"
            best_mtime="$mtime"
        fi
    done < <(find "$GEN_RESULTS_ROOT" -mindepth 2 -maxdepth 3 -type f -name run_metadata.yaml 2>/dev/null)

    [ -n "$best" ] && printf '%s\n' "$best"
}

compute_gen_results_signature() {
    # Guard resume against accidentally reusing a gen_results tree for a different
    # algorithm/data target. Operational knobs used for recovery, such as
    # TRAIN_BATCH_SIZE and KL_FULL_VOCAB_CHUNK_SIZE, are intentionally excluded.
    GEN_RESULTS_RUN_SIGNATURE_CONTENT="$(cat <<EOF
distill_family=$DISTILL_FAMILY
distill_mode=$DISTILL_MODE
model_name=$MODEL_NAME
model_path=$MODEL_PATH
teacher_model_name=$TEACHER_MODEL_NAME
teacher_model_path=${TEACHER_MODEL_PATH:-$MODEL_PATH}
prompt_mode=$PROMPT_MODE_TAG
y_mode=$Y_MODE
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
pipeline_auto_chunk_size=$PIPELINE_AUTO_CHUNK_SIZE
pipeline_total_steps=${PIPELINE_TOTAL_STEPS:-0}
pipeline_batches_per_epoch=${PIPELINE_BATCHES_PER_EPOCH:-$TOTAL_EPOCHS}
base_prompt_length=$BASE_PROMPT_LENGTH
max_prompt_length=$MAX_PROMPT_LENGTH
expert_solution_prompt_length=$EXPERT_SOLUTION_PROMPT_LENGTH
stage2_prompt_length=$STAGE2_PROMPT_LENGTH
max_response_length=$MAX_RESPONSE_LENGTH
max_length=$MAX_LENGTH
stage1_rollout_max_model_len=$STAGE1_ROLLOUT_MAX_MODEL_LEN
stage2_rollout_max_model_len=$STAGE2_ROLLOUT_MAX_MODEL_LEN
rollout_chat_template_token_buffer=$ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER
rollout_max_num_seqs=$ROLLOUT_MAX_NUM_SEQS
stage1_rollout_max_num_batched_tokens=$STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS
stage2_rollout_max_num_batched_tokens=$STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS
temperature=$TEMPERATURE
max_samples=${MAX_SAMPLES:-all}
data_path=${DATA_PATH:-}
train_data_path=$TRAIN_DATA_PATH
corrected_responses_path=${CORRECTED_RESPONSES_PATH:-}
EOF
)"
    GEN_RESULTS_RUN_SIGNATURE="$(printf '%s\n' "$GEN_RESULTS_RUN_SIGNATURE_CONTENT" | sha256sum | awk '{print $1}')"
}

compute_gen_results_signature

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
    MATCHED_GEN_RESULTS_METADATA="$(find_matching_gen_results_metadata "$GEN_RESULTS_RUN_SIGNATURE")"
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

GEN_RESULTS_RUN_PREFIX_DEFAULT="gen"
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
    GEN_RESULTS_RANDOM_UID="$(python3 - <<'PYGENUID'
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
FULL_STAGE1_PROMPTS="$GEN_RESULTS_BASE_DIR/deepscaleR_stage1_prompts.parquet"

if [ -s "$GEN_RESULTS_RUN_SIGNATURE_FILE" ]; then
    GEN_RESULTS_PERSISTED_SIGNATURE="$(tr -d '[:space:]' < "$GEN_RESULTS_RUN_SIGNATURE_FILE")"
    if [ "$GEN_RESULTS_PERSISTED_SIGNATURE" != "$GEN_RESULTS_RUN_SIGNATURE" ] && [ "${GEN_RESULTS_ALLOW_CONFIG_MISMATCH:-false}" != "true" ]; then
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
    # y_o: train on stage1 student rollouts (no qualifier — student rollouts
    #      depend only on the student model already in the path).
    if [ "$Y_MODE" = "y_r" ]; then
        echo "$data_dir/deepscaleR_stage2_${PROMPT_MODE_TAG}_${DISTILL_MODE}_${TEACHER_MODEL_NAME}_responses.parquet"
    else
        case "${FORWARD_STAGE2_MODE:-}" in
            "stage1_reward_0_only") echo "$data_dir/deepscaleR_stage1_responses_reward0.parquet" ;;
            "stage1_reward_1_only") echo "$data_dir/deepscaleR_stage1_responses_reward1.parquet" ;;
            *) echo "$data_dir/deepscaleR_stage1_responses.parquet" ;;
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
export WANDB_MODE="\${WANDB_MODE:-offline}"
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
            export WANDB_MODE="${WANDB_MODE:-offline}"
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

    local sync_cmd="torchrun \
        --nproc-per-node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node-rank=\${NODE_RANK} \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        $RECIPE_DIR/run_training.py \
        --sync_resident_rollout_only true \
        --nnodes $NNODES \
        --n_gpus_per_node $NGPUS_PER_NODE \
        --distill_mode $DISTILL_MODE \
        --kl_type $KL_TYPE \
        --kl_method $KL_METHOD \
        --kl_token_clip $KL_TOKEN_CLIP \
        --beta $BETA \
        --temperature $TEMPERATURE \
        --student_model_path $MODEL_PATH \
        ${TEACHER_MODEL_PATH:+--teacher_model_path $TEACHER_MODEL_PATH} \
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
        python3 -m recipe.opd.resident_y_o_server \
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
    python3 -m recipe.opd.resident_y_o_control \
        --manifest "$RESIDENT_YO_MANIFEST" \
        --action sleep
}

ensure_full_stage1_prompts() {
    if file_exists_and_nonempty "$FULL_STAGE1_PROMPTS"; then
        echo "  Full Stage 1 prompts already prepared: $FULL_STAGE1_PROMPTS"
        return
    fi

    echo "  [Stage 1 prompt cache] Preparing all prompts once: $FULL_STAGE1_PROMPTS"
    local args=(
        --input_path "$TRAIN_DATA_PATH"
        --output_file "$FULL_STAGE1_PROMPTS"
    )
    if [ -n "$MAX_SAMPLES" ]; then
        args+=(--max_samples "$MAX_SAMPLES")
    fi
    python3 "$PIPELINE_DIR/y_o_prepare.py" "${args[@]}"
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
        python3 - "$FULL_STAGE1_PROMPTS" "$output_file" "$batch_start" "$batch_size" <<'PYCHUNKPROMPTS'
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
    )
    if [ -n "$MAX_SAMPLES" ]; then
        args+=(--max_samples "$MAX_SAMPLES")
    fi
    python3 "$PIPELINE_DIR/y_o_prepare.py" "${args[@]}"
}

train_dataset_num_rows() {
    python3 - "$TRAIN_DATA_PATH" "${MAX_SAMPLES:-}" <<'PYDATASETLEN'
import sys
import datasets

path, max_samples = sys.argv[1], sys.argv[2]
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

pipeline_progress_dir() {
    echo "$MODEL_SAVE_BASE_DIR/pipeline_progress"
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
            if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$update" -lt "$total_updates" ]; then
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
    local model_dir marker keep_status

    if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$update" -lt "$total_updates" ]; then
        model_dir="$(find_latest_fsdp_checkpoint "$(pipeline_temp_model_save_dir "$update")")"
        [ -z "$model_dir" ] && model_dir="$(pipeline_temp_model_save_dir "$update")"
        keep_status="rolling_temp"
    else
        model_dir="$(model_dir_for_global_update "$update" "$batches_per_epoch")/hf_merged"
        keep_status="prunable"
        if pipeline_should_keep_update "$update" "$total_updates"; then
            keep_status="kept"
        fi
    fi
    marker="$(pipeline_done_marker "$update")"

    mkdir -p "$(dirname "$marker")"
    {
        printf "step=%s\n" "$update"
        printf "status=%s\n" "$keep_status"
        printf "model_path=%s\n" "$model_dir"
    } > "$marker"
}

prune_pipeline_models() {
    local current_update="$1"
    local total_updates="$2"
    local batches_per_epoch="$3"
    local update dir

    if [ "$current_update" -le 1 ]; then
        return
    fi
    for update in $(seq 1 $((current_update - 1))); do
        if pipeline_should_keep_update "$update" "$total_updates"; then
            continue
        fi
        dir="$(model_dir_for_global_update "$update" "$batches_per_epoch")"
        case "$dir" in
            "$MODEL_SAVE_BASE_DIR"/epoch*/ms*/batch*)
                if [ -d "$dir" ]; then
                    echo "Pruning non-kept pipeline model step $update: $dir"
                    rm -rf "$dir"
                fi
                ;;
            *)
                echo "WARNING: refusing to prune unexpected model dir: $dir" >&2
                ;;
        esac
    done
}

link_pipeline_final_model_alias() {
    local final_model_save_dir="$1"
    local alias_dir
    alias_dir="$(pipeline_final_alias_dir)"

    [ -d "$final_model_save_dir/hf_merged" ] || return 0
    mkdir -p "$(dirname "$alias_dir")"
    ln -sfn "$final_model_save_dir" "$alias_dir"
    echo "Final model alias: $alias_dir/hf_merged -> $final_model_save_dir/hf_merged"
}

prune_pipeline_temp_checkpoints() {
    local current_update="$1"
    local total_updates="$2"
    local keep_dir dir

    [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] || return
    [ -n "$PIPELINE_TEMP_MODEL_DIR" ] || return
    [ -d "$PIPELINE_TEMP_MODEL_DIR" ] || return

    if [ "$current_update" -ge "$total_updates" ]; then
        echo "Removing rolling temporary FSDP checkpoints after final save: $PIPELINE_TEMP_MODEL_DIR"
        rm -rf "$PIPELINE_TEMP_MODEL_DIR"
        return
    fi

    keep_dir="$(pipeline_temp_model_save_dir "$current_update")"
    for dir in "$PIPELINE_TEMP_MODEL_DIR"/step*; do
        [ -d "$dir" ] || continue
        case "$dir" in
            "$PIPELINE_TEMP_MODEL_DIR"/step*) ;;
            *) echo "WARNING: refusing to prune unexpected temp checkpoint dir: $dir" >&2; continue ;;
        esac
        if [ "$dir" = "$keep_dir" ]; then
            continue
        fi
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

    local prev_model_dir
    prev_model_dir="$(update_model_save_dir "$prev_epoch" "$prev_batch")"
    local prev_model_path="$prev_model_dir/hf_merged"

    if [ ! -d "$prev_model_path" ]; then
        echo "ERROR: Previous policy model not found: $prev_model_path"
        exit 1
    fi

    echo "$prev_model_path"
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
    echo "  Distill:  $DISTILL_MODE  (teacher_training_prompt=$TEACHER_TRAINING_PROMPT, use_initial_response=$USE_INITIAL_RESPONSE)"
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
        echo "  Auto Chunk Size: $PIPELINE_AUTO_CHUNK_SIZE samples/update (drop tail)"
        echo "  Multi-step Tag:  $MULTISTEP_TAG"
        echo "  Auto Resume:     $PIPELINE_AUTO_RESUME"
        echo "  Resume Mode:     $PIPELINE_RESUME_MODE"
        echo "  Keep Models:     ${PIPELINE_KEEP_STEPS:-every ${PIPELINE_KEEP_INTERVAL} plus final} (base step 0 is recorded, not copied)"
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
    echo "  Grad Accum:     $GRADIENT_ACCUMULATION_STEPS ($GRADIENT_ACCUMULATION_SOURCE)"
    echo "  FSDP:           $FSDP_STRATEGY (size=$FSDP_SIZE, sp=$SP_SIZE)"
    echo "  Token/GPU:      $MAX_TOKEN_LEN_PER_GPU"
    echo ""
    echo "Data Settings:"
    echo "  Train Data:     $TRAIN_DATA_PATH"
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
    echo "  Wandb:          $WANDB_PROJECT / $WANDB_RUN_NAME_BASE (mode=$WANDB_MODE)"
    echo ""
    echo "Evaluation:"
    echo "  Run After Training: $RUN_EVAL_AFTER_TRAINING"
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
    local current_wandb_run_name="${WANDB_RUN_NAME_BASE}_epoch${epoch}"
    if [ -n "$pipeline_batch_index" ]; then
        current_wandb_run_name="${current_wandb_run_name}_step$(format_pipeline_batch_id "$update_index")of$(format_pipeline_batch_id "$total_updates")_batch$(format_pipeline_batch_id "$pipeline_batch_index")"
    fi
    local current_data_path
    current_data_path="$(resolve_data_path_in_dir "$current_gen_results_dir")"

    local run_eval_this_update="$RUN_EVAL_AFTER_TRAINING"
    if [ -n "$pipeline_batch_index" ] && [ "$update_index" -lt "$total_updates" ]; then
        run_eval_this_update="false"
    fi

    local save_merged_this_update="$SAVE_MERGED_MODEL"
    local async_hf_export_this_update="$ASYNC_HF_EXPORT"
    local sync_resident_rollout_this_update="false"
    local resume_checkpoint_arg=""
    local prev_ckpt=""
    local student_model_path_for_train="$current_model_path"
    if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
        student_model_path_for_train="$MODEL_PATH"
        sync_resident_rollout_this_update="true"
        if [ -n "$pipeline_batch_index" ] && [ "$update_index" -lt "$total_updates" ]; then
            save_merged_this_update="false"
            current_model_save_dir="$(pipeline_temp_model_save_dir "$update_index")"
        fi
        if [ "$update_index" -gt 1 ]; then
            prev_ckpt="$(pipeline_checkpoint_for_update $((update_index - 1)) "$PIPELINE_BATCHES_PER_EPOCH")"
            if [ -z "$prev_ckpt" ]; then
                echo "ERROR: resident rollout mode needs previous FSDP checkpoint for update $((update_index - 1))" >&2
                echo "       Checked marker: $(pipeline_done_marker $((update_index - 1)))" >&2
                echo "       Checked temp dir: $(pipeline_temp_model_save_dir $((update_index - 1)))" >&2
                exit 1
            fi
            resume_checkpoint_arg="--resume_checkpoint_path $prev_ckpt --resume_checkpoint_mode initialize"
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

            local stage1_output="$current_gen_results_dir/deepscaleR_stage1_responses.parquet"
            local stage1_prompts="$current_gen_results_dir/deepscaleR_stage1_prompts.parquet"

            if file_exists_and_nonempty "$stage1_output"; then
                echo "  Stage 1 already done: $stage1_output"
            else
                echo "  [Stage 1] Generating initial responses..."
                prepare_stage1_prompts "$stage1_prompts" "$pipeline_batch_start" "$pipeline_current_batch_size"

                if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
                    ensure_resident_y_o_server
                    python3 -m recipe.opd.resident_y_o_generate \
                        --manifest "$RESIDENT_YO_MANIFEST" \
                        --input "$stage1_prompts" \
                        --output "$stage1_output" \
                        --prompt_key prompt \
                        --model_path "$MODEL_PATH" \
                        --temperature 0.6 \
                        --top_p 0.95 \
                        --max_tokens "$MAX_RESPONSE_LENGTH"
                else
                    python3 -m verl.trainer.main_generation_server \
                        trainer.nnodes="${NNODES}" \
                        trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                        actor_rollout_ref.model.path="${current_model_path}" \
                        actor_rollout_ref.model.trust_remote_code=true \
                        actor_rollout_ref.rollout.temperature=0.6 \
                        actor_rollout_ref.rollout.top_p=0.95 \
                        actor_rollout_ref.rollout.top_k=20 \
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
            fi

            sleep_resident_y_o_server

            # Score stage1 responses and backfill extra_info.reward.
            # Required by T4 (LOG_DIFFICULTY_BUCKETS=true) and downstream reward
            # filtering (FORWARD_FILTER_STAGE2=true). Idempotent: no-op if
            # reward field is already populated.
            if [ "${SCORE_STAGE1:-true}" = "true" ]; then
                echo "  [Stage 1 score] Ensuring extra_info.reward is populated..."
                python3 -m recipe.opd.dataset.score_stage1_reward \
                    --parquet "$stage1_output"
            fi

            local stage2_prompts="$current_gen_results_dir/deepscaleR_stage2_${PROMPT_MODE_TAG}_${DISTILL_MODE}_${TEACHER_MODEL_NAME}_prompts.parquet"
            if file_exists_and_nonempty "$stage2_prompts"; then
                echo "  Stage 2 prompts already prepared: $stage2_prompts"
            else
                echo "  [Stage 2] Generating prompts over all stage1 rows..."
                # Generation prompt is fixed (always refine variant) — y_r_prepare.py
                # hardcodes use_initial_response=True for both OPSD and OPD. The
                # training-side teacher conditioning (TEACHER_TRAINING_PROMPT) is
                # independent and goes to run_training.py, not here.
                python3 "$PIPELINE_DIR/y_r_prepare.py" \
                    --stage1_output "$stage1_output" \
                    --distill_mode "$DISTILL_MODE" \
                    --output_file "$stage2_prompts"
            fi

            # OPD: y_r must come from the TEACHER's distribution.
            # OPSD: teacher = current student, so resident student vLLM can serve stage2 too.
            local stage2_gen_model_path="$current_model_path"
            if [ "$DISTILL_MODE" = "opd" ]; then
                stage2_gen_model_path="$current_teacher_model_path"
                echo "  [Stage 2] (OPD) Using TEACHER model for y_r generation: $stage2_gen_model_path"
            fi
            echo "  [Stage 2] Generating ${PROMPT_MODE_TAG} responses..."
            if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ] && [ "$DISTILL_MODE" = "opsd" ]; then
                ensure_resident_y_o_server
                python3 -m recipe.opd.resident_y_o_generate \
                    --manifest "$RESIDENT_YO_MANIFEST" \
                    --input "$stage2_prompts" \
                    --output "$current_data_path" \
                    --prompt_key prompt \
                    --model_path "$MODEL_PATH" \
                    --temperature 0.6 \
                    --top_p 0.95 \
                    --max_tokens "$MAX_RESPONSE_LENGTH"
                sleep_resident_y_o_server
            else
                python3 -m verl.trainer.main_generation_server \
                    trainer.nnodes="${NNODES}" \
                    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                    actor_rollout_ref.model.path="${stage2_gen_model_path}" \
                    actor_rollout_ref.model.trust_remote_code=true \
                    actor_rollout_ref.rollout.temperature=0.6 \
                    actor_rollout_ref.rollout.top_p=0.95 \
                    actor_rollout_ref.rollout.top_k=20 \
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
                python3 -m recipe.opd.dataset.filter_stage2_by_reward \
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
            local stage1_for_backfill="$current_gen_results_dir/deepscaleR_stage1_responses.parquet"
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
                    python3 -m recipe.opd.dataset.backfill_stage2_reward_from_stage1 \
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
        else
            echo "Stage 1 data not found. Generating..."

            local stage1_prompts="$current_gen_results_dir/deepscaleR_stage1_prompts.parquet"

            prepare_stage1_prompts "$stage1_prompts" "$pipeline_batch_start" "$pipeline_current_batch_size"

            if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
                ensure_resident_y_o_server
                python3 -m recipe.opd.resident_y_o_generate \
                    --manifest "$RESIDENT_YO_MANIFEST" \
                    --input "$stage1_prompts" \
                    --output "$current_data_path" \
                    --prompt_key prompt \
                    --model_path "$MODEL_PATH" \
                    --temperature 0.6 \
                    --top_p 0.95 \
                    --max_tokens "$MAX_RESPONSE_LENGTH"
            else
                python3 -m verl.trainer.main_generation_server \
                    trainer.nnodes="${NNODES}" \
                    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                    actor_rollout_ref.model.path="${current_model_path}" \
                    actor_rollout_ref.model.trust_remote_code=true \
                    actor_rollout_ref.rollout.temperature=0.6 \
                    actor_rollout_ref.rollout.top_p=0.95 \
                    actor_rollout_ref.rollout.top_k=20 \
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
                    +data.output_path="${current_data_path}"
            fi
        fi

        sleep_resident_y_o_server

        # Score stage1 responses and backfill extra_info.reward.
        # Required by T4 (LOG_DIFFICULTY_BUCKETS=true), and harmless otherwise.
        # Idempotent: no-op if the reward field is already populated.
        if [ "${SCORE_STAGE1:-true}" = "true" ]; then
            echo "  [Stage 1 score] Ensuring extra_info.reward is populated..."
            python3 -m recipe.opd.dataset.score_stage1_reward \
                --parquet "$current_data_path"
        fi
    fi

    echo ""
    echo "Training data ready: $current_data_path"

    cat > "$current_output_dir/training_config.yaml" << EOF
pipeline_epoch: $epoch
pipeline_total_epochs: $TOTAL_EPOCHS
multi_step: $MULTI_STEP
pipeline_auto_chunk_size: $PIPELINE_AUTO_CHUNK_SIZE
pipeline_batch_index: ${pipeline_batch_index:-null}
pipeline_batch_start: ${pipeline_batch_start:-null}
pipeline_update_index: $update_index
pipeline_total_updates: $total_updates
pipeline_multistep_tag: ${MULTISTEP_TAG:-null}
pipeline_keep_steps: ${PIPELINE_KEEP_STEPS:-null}
pipeline_keep_interval: ${PIPELINE_KEEP_INTERVAL:-null}
pipeline_cleanup_batch_data: $PIPELINE_CLEANUP_BATCH_DATA
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
prompt_mode: $PROMPT_MODE_TAG
y_mode: $Y_MODE
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
wandb_mode: $WANDB_MODE
save_merged_model: $SAVE_MERGED_MODEL
EOF

    export PYTHONPATH="$VERL_ROOT:$PYTHONPATH"

    local cmd="torchrun \
        --nproc-per-node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node-rank=\${NODE_RANK} \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        $RECIPE_DIR/run_training.py \
        --nnodes $NNODES \
        --n_gpus_per_node $NGPUS_PER_NODE \
        --distill_mode $DISTILL_MODE \
        --kl_type $KL_TYPE \
        --kl_method $KL_METHOD \
        --kl_token_clip $KL_TOKEN_CLIP \
        --beta $BETA \
        --temperature $TEMPERATURE \
        --student_model_path $student_model_path_for_train \
        ${current_teacher_model_path:+--teacher_model_path $current_teacher_model_path} \
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
        --save_merged_model $save_merged_this_update \
        --save_steps $SAVE_STEPS \
        --max_ckpt_to_keep $KEEP_LAST_N_CHECKPOINTS \
        $resume_checkpoint_arg \
        --resident_rollout_manifest $RESIDENT_YO_MANIFEST \
        --sync_resident_rollout $sync_resident_rollout_this_update \
        --async_hf_export $async_hf_export_this_update \
        --run_eval_after_training $run_eval_this_update \
        --eval_datasets $EVAL_DATASETS \
        --eval_datasets_dir $EVAL_DATASETS_DIR \
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
export WANDB_MODE="\${WANDB_MODE:-offline}"
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
            export WANDB_MODE="${WANDB_MODE:-offline}"
            eval "$cmd"
        ) 2>&1 | tee "$current_output_dir/logs/training_$(date +%Y%m%d_%H%M%S).log"
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
        local eval_model_path="$current_model_save_dir/hf_merged"

        # --- 1. Verify merged model exists & is loadable ---
        if [ ! -d "$eval_model_path" ] || [ ! -f "$eval_model_path/config.json" ]; then
            echo "ERROR: eval requested but merged model is missing or incomplete at $eval_model_path"
            exit 1
        fi

        # --- 2. Verify each requested eval dataset parquet is on disk ---
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

        # --- 3. Wait for training GPU memory to be released ---
        # The training subprocess has already exited by this point, so memory
        # should be freed almost immediately. We still poll defensively for up
        # to 60s in case any zombie process is holding memory — vLLM will OOM
        # if the previous FSDP allocator hasn't released its blocks.
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
            # Threshold: < 1 GB per GPU on average (8 GPUs ≈ 8 GB total)
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

        # --- 4. Hand off to benchmark_kl_model.sh ---
        # benchmark_kl_model.sh derives MODEL_NAME / output_dir / results_file
        # from the model path. It expects DATASETS as space-separated and
        # writes to relative paths under the repo root, so we cd there first.
        local _datasets_space
        _datasets_space=$(echo "$EVAL_DATASETS" | tr ',' ' ')

        local MATH_EVAL_DIR="$VERL_ROOT/recipe/math_evaluation"

        echo ""
        echo "=========================================="
        echo "Running evaluation via benchmark_kl_model.sh"
        echo "  Model:    $eval_model_path"
        echo "  Datasets: $_datasets_space"
        echo "=========================================="
        local _eval_model_name="$RESULTS_MODEL_KEY"
        if [ -n "$pipeline_batch_index" ]; then
            _eval_model_name="${_eval_model_name}_step$(format_pipeline_batch_id "$update_index")of$(format_pipeline_batch_id "$total_updates")"
        fi
        local _eval_output_dir="$VERL_ROOT/gen_results/eval/${_eval_model_name}"

        (
            cd "$VERL_ROOT"
            unset PYTORCH_CUDA_ALLOC_CONF
            NGPUS_PER_NODE="$NGPUS_PER_NODE" \
            NNODES="$NNODES" \
            GEN_TP="$GEN_TP" \
            EVAL_DATASETS_DIR="$EVAL_DATASETS_DIR" \
            DATASETS="$_datasets_space" \
            PASS_K="$PASS_K" \
            EVAL_BASE_MODEL_NAME="$RESULTS_BASE_MODEL_NAME" \
            EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-$_eval_model_name}" \
            EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-$RESULTS_FILE}" \
            EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$_eval_output_dir}" \
            bash "$MATH_EVAL_DIR/benchmark_kl_model.sh" "$eval_model_path"
        ) 2>&1 | tee "$current_output_dir/logs/eval_$(date +%Y%m%d_%H%M%S).log"
    fi

    echo ""
    echo "Epoch ${epoch} complete"
    echo "  Logs & Config: $current_output_dir"
    echo "  Model Checkpoints: $current_model_save_dir"
    echo "  Gen Results: $current_gen_results_dir"
    if [ "$run_eval_this_update" = "true" ]; then
        echo "  Eval Results: $RESULTS_FILE"
    fi
    if [ "$SAVE_MERGED_MODEL" = "true" ]; then
        echo "  Merged Model: $current_model_save_dir/hf_merged"
    else
        echo "  FSDP Checkpoints: $current_model_save_dir/global_step_*"
    fi
}

# =============================================================================
# Main
# =============================================================================

print_base_configuration

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
distill_mode: $DISTILL_MODE
prompt_mode: $PROMPT_MODE_TAG
y_mode: $Y_MODE
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
pipeline_total_steps: ${PIPELINE_TOTAL_STEPS:-0}
pipeline_batches_per_epoch: ${PIPELINE_BATCHES_PER_EPOCH:-$TOTAL_EPOCHS}
pipeline_auto_resume: $PIPELINE_AUTO_RESUME
pipeline_resume_mode: $PIPELINE_RESUME_MODE
pipeline_keep_steps: ${PIPELINE_KEEP_STEPS:-}
pipeline_keep_interval: ${PIPELINE_KEEP_INTERVAL:-}
pipeline_cleanup_batch_data: $PIPELINE_CLEANUP_BATCH_DATA
pipeline_temp_model_dir: $PIPELINE_TEMP_MODEL_DIR
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
rollout_chat_template_token_buffer: $ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER
rollout_gpu_memory_utilization: $ROLLOUT_GPU_MEMORY_UTILIZATION
rollout_max_num_seqs: $ROLLOUT_MAX_NUM_SEQS
stage1_rollout_max_num_batched_tokens: $STAGE1_ROLLOUT_MAX_NUM_BATCHED_TOKENS
stage2_rollout_max_num_batched_tokens: $STAGE2_ROLLOUT_MAX_NUM_BATCHED_TOKENS
temperature: $TEMPERATURE
max_samples: ${MAX_SAMPLES:-all}
data_path: ${DATA_PATH:-}
train_data_path: $TRAIN_DATA_PATH
corrected_responses_path: ${CORRECTED_RESPONSES_PATH:-}
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
wandb_mode: $WANDB_MODE
EOF
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
    echo "Dropped tail rows:    $PIPELINE_DROPPED_SAMPLES"
    echo "Multi-step tag:       $MULTISTEP_TAG"
    echo "Auto resume:          $PIPELINE_AUTO_RESUME"
    echo "Resume mode:          $PIPELINE_RESUME_MODE"
    echo "Model keep policy:    ${PIPELINE_KEEP_STEPS:-every ${PIPELINE_KEEP_INTERVAL} plus final}"
    echo "=========================================="

    ensure_full_stage1_prompts
    write_pipeline_checkpoint_plan "$TOTAL_PIPELINE_UPDATES" "$PIPELINE_BATCHES_PER_EPOCH"

    FINAL_PIPELINE_MODEL_DIR="$(update_model_save_dir "$TOTAL_EPOCHS" "$PIPELINE_BATCHES_PER_EPOCH")"
    if [ -f "$FINAL_PIPELINE_MODEL_DIR/hf_merged/config.json" ]; then
        if [ "$PIPELINE_AUTO_RESUME" = "true" ]; then
            echo ""
            echo "=========================================="
            echo "Offline multi-step pipeline already complete, skipping all chunks"
            echo "  Found: $FINAL_PIPELINE_MODEL_DIR/hf_merged/config.json"
            echo "=========================================="
        else
            echo "ERROR: final pipeline model already exists and PIPELINE_AUTO_RESUME=false: $FINAL_PIPELINE_MODEL_DIR/hf_merged"
            echo "       Use PIPELINE_RESUME_MODE=resume_matching to continue, or use fresh output dirs."
            exit 1
        fi
    else
    GLOBAL_UPDATE=0
    PIPELINE_STOPPED_EARLY_UPDATE=""
    PIPELINE_STOPPED_EARLY_EPOCH=""
    PIPELINE_STOPPED_EARLY_BATCH=""
    for EPOCH in $(seq 1 "$TOTAL_EPOCHS"); do
        for PIPELINE_BATCH_INDEX in $(seq 1 "$PIPELINE_BATCHES_PER_EPOCH"); do
            GLOBAL_UPDATE=$((GLOBAL_UPDATE + 1))
            BATCH_START=$(((PIPELINE_BATCH_INDEX - 1) * PIPELINE_AUTO_CHUNK_SIZE))
            UPDATE_MODEL_DIR="$(update_model_save_dir "$EPOCH" "$PIPELINE_BATCH_INDEX")"
            UPDATE_DONE_MARKER="$UPDATE_MODEL_DIR/hf_merged/config.json"
            UPDATE_PROGRESS_MARKER="$(pipeline_done_marker "$GLOBAL_UPDATE")"

            if [ -f "$UPDATE_PROGRESS_MARKER" ] || [ -f "$UPDATE_DONE_MARKER" ]; then
                if [ "$PIPELINE_AUTO_RESUME" = "true" ]; then
                    echo ""
                    echo "=========================================="
                    echo "Epoch ${EPOCH}/${TOTAL_EPOCHS} batch $(format_pipeline_batch_id "$PIPELINE_BATCH_INDEX") / step ${GLOBAL_UPDATE} — already complete, skipping"
                    if [ -f "$UPDATE_PROGRESS_MARKER" ]; then
                        echo "  Found: $UPDATE_PROGRESS_MARKER"
                    else
                        echo "  Found: $UPDATE_DONE_MARKER"
                    fi
                    echo "=========================================="
                    continue
                fi
                echo "ERROR: completed pipeline step already exists and PIPELINE_AUTO_RESUME=false: step ${GLOBAL_UPDATE}"
                if [ -f "$UPDATE_PROGRESS_MARKER" ]; then
                    echo "       Found: $UPDATE_PROGRESS_MARKER"
                else
                    echo "       Found: $UPDATE_DONE_MARKER"
                fi
                echo "       Use PIPELINE_RESUME_MODE=resume_matching to continue, or use fresh output dirs."
                exit 1
            fi

            if [ "$RESIDENT_STUDENT_ROLLOUT" = "true" ]; then
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
                "$PIPELINE_AUTO_CHUNK_SIZE" \
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
        link_pipeline_final_model_alias "$FINAL_MODEL_SAVE_DIR"
    fi
else
    for EPOCH in $(seq 1 "$TOTAL_EPOCHS"); do
        EPOCH_MODEL_DIR="$(epoch_model_save_dir "$EPOCH")"
        if [ "$SAVE_MERGED_MODEL" = "true" ]; then
            EPOCH_DONE_MARKER="$EPOCH_MODEL_DIR/hf_merged/config.json"
        else
            EPOCH_DONE_MARKER="$EPOCH_MODEL_DIR/final/config.json"
        fi

        if [ -f "$EPOCH_DONE_MARKER" ]; then
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
echo "Model Checkpoints: $FINAL_MODEL_SAVE_DIR"
echo "Gen Results Base: $GEN_RESULTS_BASE_DIR"
echo "Gen Metadata: $GEN_RESULTS_METADATA_FILE"
if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
    echo "Eval Results: $RESULTS_FILE"
fi
if [ "$SAVE_MERGED_MODEL" = "true" ]; then
    echo "Merged Model: $FINAL_MODEL_SAVE_DIR/hf_merged"
    if [ -e "$(pipeline_final_alias_dir)/hf_merged" ]; then
        echo "Final Alias: $(pipeline_final_alias_dir)/hf_merged"
    fi
    echo "Benchmark: bash $VERL_ROOT/recipe/math_evaluation/benchmark_kl_model.sh $FINAL_MODEL_SAVE_DIR/hf_merged"
else
    echo "FSDP Checkpoints: $FINAL_MODEL_SAVE_DIR/global_step_*"
fi
echo "Wandb Dashboard: https://wandb.ai/$WANDB_PROJECT"
