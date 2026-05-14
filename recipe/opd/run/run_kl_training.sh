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
#   - Each epoch generates its own stage1/stage2 data under gen_results/${MODEL_NAME}/epochN/
#   - Each epoch trains from the current model and saves to an epoch-specific checkpoint dir
#   - Epoch N+1 loads the model saved by epoch N
# =============================================================================

set -e
set -o pipefail

# `expandable_segments:True` avoids a ~6GiB reserved-but-unallocated block
# during FSDP backward, but it is incompatible with vLLM's CuMemAllocator
# memory pool, so we only set it around the torchrun training command below
# (not exported here, otherwise stage1/stage2 generation and eval — which
# all spawn vLLM — crash with "Expandable segments are not compatible with
# memory pool").

# =============================================================================
# Configuration
# =============================================================================
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
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}  # Outer pipeline epochs
TRAIN_EPOCHS_PER_ROUND=${TRAIN_EPOCHS_PER_ROUND:-1}  # Trainer epochs for each pipeline epoch
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.005}

# Data Settings
DATA_PATH=${DATA_PATH:-""}  # Optional override; reused across epochs if set
CORRECTED_RESPONSES_PATH=${CORRECTED_RESPONSES_PATH:-""}  # Optional legacy two-file mode
MAX_SAMPLES=${MAX_SAMPLES:-""}  # For testing, leave empty for full data

# Distributed Training
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
GEN_TP=${GEN_TP:-1}  # Tensor parallel for generation

# verl FSDP Settings
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}
FSDP_SIZE=${FSDP_SIZE:--1}
SP_SIZE=${SP_SIZE:-1}
# TODO forward KL should be higher this is only for reverse KL
MAX_LENGTH=${MAX_LENGTH:-18432}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-49152}
STAGE2_PROMPT_LENGTH=${STAGE2_PROMPT_LENGTH:-20480}  # Forward-KL stage2 rewrite prompt length; correction mode uses longer prompts
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
SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-"true"}  # Merge LoRA after training
SAVE_STEPS=${SAVE_STEPS:-100}                   # FSDP ckpt every N optimizer steps (crash recovery)
KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS:-1}  # Rolling window; per-epoch hf_merged is always preserved

# Evaluation Settings
RUN_EVAL_AFTER_TRAINING=${RUN_EVAL_AFTER_TRAINING:-"true"}
# EVAL_DATASETS=${EVAL_DATASETS:-"aime24,aime25,math500,hmmt25"} DEFAULT_DATASETS="math500 hmmt25 beyondaime amobench gsm8k"
EVAL_DATASETS=${EVAL_DATASETS:-"aime24,aime25,hmmt25,beyondaime,amobench"}
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-"/data/data/jiangli/huggingface/datasets"}
PASS_K=${PASS_K:-1}

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

# Model name from the original base model, not from epoch checkpoints
MODEL_NAME="${MODEL_PATH##*/}"
if [ "$Y_MODE" = "y_r" ]; then
    USE_INITIAL_RESPONSE="true"
elif [ "$Y_MODE" = "y_o" ]; then
    USE_INITIAL_RESPONSE="false"
else
    echo "ERROR: Y_MODE must be one of: y_o, y_r (got: $Y_MODE)"
    exit 1
fi

# MAX_PROMPT_LENGTH / MAX_RESPONSE_LENGTH default by Y_MODE: y_r prompts embed
# the expert solution + initial response and need more headroom than y_o.
# MAX_LENGTH = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH unless explicitly overridden.
if [ "$Y_MODE" = "y_r" ]; then
    MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-24576}
else
    MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
fi
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
if [ -z "${MAX_LENGTH+x}" ]; then
    MAX_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
fi
# STAGE2_PROMPT_LENGTH (used by stage2 generation server only) tracks MAX_PROMPT_LENGTH
# for y_r runs since both refer to the same correction-prompt budget.
if [ "$Y_MODE" = "y_r" ]; then
    STAGE2_PROMPT_LENGTH=${STAGE2_PROMPT_LENGTH:-$MAX_PROMPT_LENGTH}
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
EXPERIMENT_TAG="kl_${KL_TYPE}_${KL_METHOD}_${PROMPT_MODE_TAG}_${CLIP_TAG}${BETA_TAG}${TOPK_TAG}${Y_O_FILTER_TAG}"

# y_r path no longer reads FORWARD_STAGE2_MODE — y_r_prepare.py always processes
# every stage1 row. Reward-based filtering of y_r happens post-generation via
# FORWARD_FILTER_STAGE2=true (recipe/opd/dataset/filter_stage2_by_reward.py).

if [ "$TOTAL_EPOCHS" -lt 1 ]; then
    echo "ERROR: TOTAL_EPOCHS must be >= 1"
    exit 1
fi

if [ "$TOTAL_EPOCHS" -gt 1 ] && [ "$SAVE_MERGED_MODEL" != "true" ]; then
    echo "ERROR: Multi-epoch training requires SAVE_MERGED_MODEL=true so epoch N+1 can load epoch N as HuggingFace weights."
    exit 1
fi

# Base directories
GEN_RESULTS_BASE_DIR="$VERL_ROOT/gen_results/${MODEL_NAME}"
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d)}

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_BASE_DIR="$VERL_ROOT/outputs/${MODEL_NAME}_${EXPERIMENT_TAG}_${RUN_DATE}"
else
    OUTPUT_BASE_DIR="$OUTPUT_DIR"
fi

if [ -z "$MODEL_SAVE_DIR" ] || [ "$MODEL_SAVE_DIR" = "/data/data/jiangli/models" ]; then
    MODEL_SAVE_BASE_DIR="/data/data/jiangli/models/${MODEL_NAME}_${EXPERIMENT_TAG}_${RUN_DATE}"
else
    MODEL_SAVE_BASE_DIR="$MODEL_SAVE_DIR"
fi

if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME_BASE="${EXPERIMENT_TAG}_${RUN_DATE}"
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

resolve_epoch_data_path() {
    local epoch="$1"

    if [ -n "$DATA_PATH" ]; then
        echo "$DATA_PATH"
        return
    fi

    local epoch_dir
    epoch_dir="$(epoch_gen_results_dir "$epoch")"

    # y_r: train on stage2 teacher rewrites (correction prompt with initial response).
    # y_o: train on stage1 student rollouts (works with any KL_TYPE: forward via jsd β=0,
    #      reverse, jsd with β>0). KL_TYPE only controls the loss math.
    if [ "$Y_MODE" = "y_r" ]; then
        echo "$epoch_dir/deepscaleR_stage2_${PROMPT_MODE_TAG}_responses.parquet"
    else
        case "${FORWARD_STAGE2_MODE:-}" in
            "stage1_reward_0_only") echo "$epoch_dir/deepscaleR_stage1_responses_reward0.parquet" ;;
            "stage1_reward_1_only") echo "$epoch_dir/deepscaleR_stage1_responses_reward1.parquet" ;;
            *) echo "$epoch_dir/deepscaleR_stage1_responses.parquet" ;;
        esac
    fi
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
    echo "  Y Mode:   $Y_MODE  (prompt_tag=$PROMPT_MODE_TAG, use_initial_response=$USE_INITIAL_RESPONSE)"
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
    echo "  Train Epochs/Round: $TRAIN_EPOCHS_PER_ROUND"
    echo "  LR:             $LEARNING_RATE"
    echo "  Batch:          $TRAIN_BATCH_SIZE"
    echo "  Max Len:        $MAX_LENGTH"
    echo "  Grad Accum:     $GRADIENT_ACCUMULATION_STEPS"
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
    echo ""
    echo "Output Base:"
    echo "  Output Dir:     $OUTPUT_BASE_DIR"
    echo "  Model Save:     $MODEL_SAVE_BASE_DIR"
    echo "  Gen Results:    $GEN_RESULTS_BASE_DIR"
    echo "  Wandb:          $WANDB_PROJECT / $WANDB_RUN_NAME_BASE"
    echo ""
    echo "Evaluation:"
    echo "  Run After Training: $RUN_EVAL_AFTER_TRAINING"
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
    local current_gen_results_dir
    current_gen_results_dir="$(epoch_gen_results_dir "$epoch")"
    local current_output_dir
    current_output_dir="$(epoch_output_dir "$epoch")"
    local current_model_save_dir
    current_model_save_dir="$(epoch_model_save_dir "$epoch")"
    local current_wandb_run_name="${WANDB_RUN_NAME_BASE}_epoch${epoch}"
    local current_data_path
    current_data_path="$(resolve_epoch_data_path "$epoch")"

    mkdir -p "$current_gen_results_dir"
    mkdir -p "$current_output_dir"
    mkdir -p "$current_output_dir/logs"
    mkdir -p "$current_model_save_dir"

    echo ""
    echo "=========================================="
    echo "Epoch ${epoch}/${TOTAL_EPOCHS}"
    echo "=========================================="
    echo "Student Model: $current_model_path"
    echo "Teacher Model: ${current_teacher_model_path:-<same as student>}"
    echo "Gen Results Dir: $current_gen_results_dir"
    echo "Output Dir: $current_output_dir"
    echo "Model Save Dir: $current_model_save_dir"
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
                python3 "$PIPELINE_DIR/y_o_prepare.py" \
                    --input_path "$TRAIN_DATA_PATH" \
                    --output_file "$stage1_prompts"

                python3 -m verl.trainer.main_generation_server \
                    trainer.nnodes="${NNODES}" \
                    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                    actor_rollout_ref.model.path="${current_model_path}" \
                    actor_rollout_ref.model.trust_remote_code=true \
                    actor_rollout_ref.rollout.temperature=0.6 \
                    actor_rollout_ref.rollout.top_p=0.95 \
                    actor_rollout_ref.rollout.top_k=20 \
                    actor_rollout_ref.rollout.prompt_length=4096 \
                    actor_rollout_ref.rollout.response_length=16384 \
                    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
                    actor_rollout_ref.rollout.max_num_seqs=64 \
                    actor_rollout_ref.rollout.name=vllm \
                    actor_rollout_ref.rollout.n=1 \
                    data.train_files="['${stage1_prompts}']" \
                    data.prompt_key=prompt \
                    +data.output_path="${stage1_output}"
            fi

            # Score stage1 responses and backfill extra_info.reward.
            # Required by T4 (LOG_DIFFICULTY_BUCKETS=true) and downstream reward
            # filtering (FORWARD_FILTER_STAGE2=true). Idempotent: no-op if
            # reward field is already populated.
            if [ "${SCORE_STAGE1:-true}" = "true" ]; then
                echo "  [Stage 1 score] Ensuring extra_info.reward is populated..."
                python3 -m recipe.opd.dataset.score_stage1_reward \
                    --parquet "$stage1_output"
            fi

            local stage2_prompts="$current_gen_results_dir/deepscaleR_stage2_${PROMPT_MODE_TAG}_prompts.parquet"
            if file_exists_and_nonempty "$stage2_prompts"; then
                echo "  Stage 2 prompts already prepared: $stage2_prompts"
            else
                echo "  [Stage 2] Generating prompts over all stage1 rows..."
                python3 "$PIPELINE_DIR/y_r_prepare.py" \
                    --stage1_output "$stage1_output" \
                    --use_initial_response "$USE_INITIAL_RESPONSE" \
                    --output_file "$stage2_prompts"
            fi

            echo "  [Stage 2] Generating ${PROMPT_MODE_TAG} responses..."
            python3 -m verl.trainer.main_generation_server \
                trainer.nnodes="${NNODES}" \
                trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                actor_rollout_ref.model.path="${current_model_path}" \
                actor_rollout_ref.model.trust_remote_code=true \
                actor_rollout_ref.rollout.temperature=0.6 \
                actor_rollout_ref.rollout.top_p=0.95 \
                actor_rollout_ref.rollout.top_k=20 \
                actor_rollout_ref.rollout.prompt_length=${STAGE2_PROMPT_LENGTH} \
                actor_rollout_ref.rollout.response_length=16384 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
                actor_rollout_ref.rollout.max_num_seqs=64 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.n=1 \
                data.train_files="['${stage2_prompts}']" \
                data.prompt_key=prompt \
                +data.output_path="${current_data_path}"
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

            python3 "$PIPELINE_DIR/y_o_prepare.py" \
                --input_path "$TRAIN_DATA_PATH" \
                --output_file "$stage1_prompts"

            python3 -m verl.trainer.main_generation_server \
                trainer.nnodes="${NNODES}" \
                trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
                actor_rollout_ref.model.path="${current_model_path}" \
                actor_rollout_ref.model.trust_remote_code=true \
                actor_rollout_ref.rollout.temperature=0.6 \
                actor_rollout_ref.rollout.top_p=0.95 \
                actor_rollout_ref.rollout.top_k=20 \
                actor_rollout_ref.rollout.prompt_length=4096 \
                actor_rollout_ref.rollout.response_length=16384 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
                actor_rollout_ref.rollout.max_num_seqs=64 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.n=1 \
                data.train_files="['${stage1_prompts}']" \
                data.prompt_key=prompt \
                +data.output_path="${current_data_path}"
        fi

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
student_model_path: $current_model_path
teacher_model_path: ${current_teacher_model_path:-$current_model_path}
use_lora: $USE_LORA
lora_rank: $LORA_RANK
lora_alpha: $LORA_ALPHA
learning_rate: $LEARNING_RATE
train_batch_size: $TRAIN_BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
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
save_merged_model: $SAVE_MERGED_MODEL
EOF

    export PYTHONPATH="$VERL_ROOT:$PYTHONPATH"

    local cmd="torchrun \
        --nproc-per-node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT \
        $RECIPE_DIR/run_training.py \
        --nnodes $NNODES \
        --n_gpus_per_node $NGPUS_PER_NODE \
        --kl_type $KL_TYPE \
        --kl_method $KL_METHOD \
        --kl_token_clip $KL_TOKEN_CLIP \
        --beta $BETA \
        --temperature $TEMPERATURE \
        --student_model_path $current_model_path \
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
        --save_merged_model $SAVE_MERGED_MODEL \
        --save_steps $SAVE_STEPS \
        --max_ckpt_to_keep $KEEP_LAST_N_CHECKPOINTS \
        --run_eval_after_training $RUN_EVAL_AFTER_TRAINING \
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

    (
        export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
        eval "$cmd"
    ) 2>&1 | tee "$current_output_dir/logs/training_$(date +%Y%m%d_%H%M%S).log"

    if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
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
        (
            cd "$VERL_ROOT"
            unset PYTORCH_CUDA_ALLOC_CONF
            NGPUS_PER_NODE="$NGPUS_PER_NODE" \
            NNODES="$NNODES" \
            GEN_TP="$GEN_TP" \
            EVAL_DATASETS_DIR="$EVAL_DATASETS_DIR" \
            DATASETS="$_datasets_space" \
            PASS_K="$PASS_K" \
            bash "$MATH_EVAL_DIR/benchmark_kl_model.sh" "$eval_model_path"
        ) 2>&1 | tee "$current_output_dir/logs/eval_$(date +%Y%m%d_%H%M%S).log"
    fi

    echo ""
    echo "Epoch ${epoch} complete"
    echo "  Logs & Config: $current_output_dir"
    echo "  Model Checkpoints: $current_model_save_dir"
    echo "  Gen Results: $current_gen_results_dir"
    if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
        echo "  Eval Results: results/${MODEL_NAME}/results.json"
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
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$MODEL_SAVE_BASE_DIR"

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

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Final Epoch: $TOTAL_EPOCHS"
echo "Logs & Config: $FINAL_OUTPUT_DIR"
echo "Model Checkpoints: $FINAL_MODEL_SAVE_DIR"
echo "Gen Results Base: $GEN_RESULTS_BASE_DIR"
if [ "$RUN_EVAL_AFTER_TRAINING" = "true" ]; then
    echo "Eval Results: results/${MODEL_NAME}/results.json"
fi
if [ "$SAVE_MERGED_MODEL" = "true" ]; then
    echo "Merged Model: $FINAL_MODEL_SAVE_DIR/hf_merged"
    echo "Benchmark: bash $VERL_ROOT/recipe/math_evaluation/benchmark_kl_model.sh $FINAL_MODEL_SAVE_DIR/hf_merged"
else
    echo "FSDP Checkpoints: $FINAL_MODEL_SAVE_DIR/global_step_*"
fi
echo "Wandb Dashboard: https://wandb.ai/$WANDB_PROJECT"
