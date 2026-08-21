#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/recipe/opd/run/hf_export_validation.sh"

export PYTHONPATH=".:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GRPO_DRY_RUN="${GRPO_DRY_RUN:-false}"
case "$GRPO_DRY_RUN" in
    false|true|print) ;;
    *) echo "ERROR: GRPO_DRY_RUN must be false, true, or print" >&2; exit 2 ;;
esac

TASK="${TASK:?TASK must be math or code}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_ALIAS="${MODEL_ALIAS:?MODEL_ALIAS is required}"

case "$TASK" in
    math|code) ;;
    *) echo "ERROR: TASK must be math or code, got: $TASK" >&2; exit 1 ;;
esac

detect_visible_gpu_count() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
        local count=0 gpu_id
        for gpu_id in ${CUDA_VISIBLE_DEVICES//,/ }; do
            [ -n "$gpu_id" ] && count=$((count + 1))
        done
        if [ "$count" -gt 0 ]; then
            printf '%s\n' "$count"
            return 0
        fi
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count="$(nvidia-smi -L 2>/dev/null | awk '/^GPU [0-9]+:/ { count++ } END { print count + 0 }')"
        if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
            printf '%s\n' "$count"
            return 0
        fi
    fi
    printf '8\n'
}

require_file() {
    local path="$1"
    local description="$2"
    if [ ! -f "$path" ]; then
        echo "ERROR: missing $description: $path" >&2
        echo "Run: bash recipe/opd/run/grpo/prepare/ready_to_train.sh" >&2
        exit 1
    fi
}

require_model() {
    local reference="$1"
    if [ -d "$reference" ]; then
        require_file "$reference/config.json" "model config"
        return
    fi
    case "$reference" in
        /*|./*|../*|model/*)
            echo "ERROR: missing local base model or config.json: $reference" >&2
            echo "Run: bash recipe/opd/run/grpo/prepare/ready_to_train.sh or set MODEL_PATH to a Hugging Face repo ID." >&2
            exit 1
            ;;
        */*)
            echo "Using Hugging Face model reference: $reference"
            ;;
        *)
            echo "ERROR: MODEL_PATH is neither a local model directory nor a Hugging Face repo ID: $reference" >&2
            exit 1
            ;;
    esac
}

latest_actor_checkpoint() {
    local ckpt_root="$1"
    local latest_file="$ckpt_root/latest_checkpointed_iteration.txt"
    local step_dir=""
    if [ -f "$latest_file" ]; then
        local step
        step="$(tr -d '[:space:]' < "$latest_file")"
        if [ -n "$step" ] && [ -d "$ckpt_root/global_step_${step}/actor" ]; then
            printf '%s\n' "$ckpt_root/global_step_${step}/actor"
            return 0
        fi
    fi
    step_dir="$(find "$ckpt_root" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1)"
    if [ -n "$step_dir" ] && [ -d "$step_dir/actor" ]; then
        printf '%s\n' "$step_dir/actor"
        return 0
    fi
    return 1
}

latest_hf_checkpoint() {
    local actor_checkpoint
    actor_checkpoint="$(latest_actor_checkpoint "$1")" || return 1
    local step_name="$(basename "$(dirname "$actor_checkpoint")")"
    local step="${step_name#global_step_}"
    if ! [[ "$step" =~ ^[0-9]+$ ]]; then
        echo "ERROR: cannot derive checkpoint step from: $actor_checkpoint" >&2
        return 1
    fi
    local target_dir="$MODELS_DIR/step_${step}"
    if ! hf_export_complete "$target_dir"; then
        "$PYTHON_BIN" -m recipe.opd.export_checkpoint \
            --local-dir "$actor_checkpoint" \
            --target-dir "$target_dir" \
            --base-model "$MODEL_PATH" \
            --lora-rank "$LORA_RANK" \
            --lora-alpha "$LORA_ALPHA" \
            --trust-remote-code >&2
    fi
    require_file "$target_dir/config.json" "strict merged HuggingFace config"
    require_file "$target_dir/opd_export.json" "strict export marker"
    if ! hf_export_complete "$target_dir"; then
        echo "ERROR: strict Hugging Face export is incomplete: $target_dir" >&2
        return 1
    fi
    printf '%s\n' "$target_dir"
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

NGPUS_PER_NODE="${NGPUS_PER_NODE:-$(detect_visible_gpu_count)}"
NNODES="${NNODES:-1}"
# Use every allocated GPU together in one tensor-parallel vLLM replica for
# both the training rollout and post-training evaluation.
GEN_TP="${GEN_TP:-$NGPUS_PER_NODE}"
EVAL_GEN_TP="${EVAL_GEN_TP:-$NGPUS_PER_NODE}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-64}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d.%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-trd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_ALIAS}_${TASK}_grpo_${TIMESTAMP}}"
RUN_ROOT="${RUN_ROOT:-outputs/$PROJECT_NAME/$EXPERIMENT_NAME}"
TRAIN_DIR="${TRAIN_DIR:-$RUN_ROOT/training}"
CKPTS_DIR="${CKPTS_DIR:-$RUN_ROOT/checkpoints}"
MODELS_DIR="${MODELS_DIR:-$RUN_ROOT/models}"

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$TRAIN_DIR/tensorboard_log}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-$TRAIN_DIR/metrics.jsonl}"

if [ "$TASK" = "math" ]; then
    TRAIN_FILE="${TRAIN_FILE:-data/train_dataset/openthoughts_math_30k_opsd/train_grpo.parquet}"
    TEST_FILE="${TEST_FILE:-data/eval_dataset/math/aime26/aime26_test.parquet}"
    EVAL_DATASETS="${EVAL_DATASETS:-aime25 aime26 hmmt26 amobench}"
    PASS_K="${PASS_K:-16}"
    REWARD_CONFIG=(
        reward.custom_reward_function.path=recipe/opd/run/grpo/math_reward.py
        reward.custom_reward_function.name=compute_score
    )
else
    TRAIN_FILE="${TRAIN_FILE:-data/train_dataset/taco/canonical/train_grpo.parquet}"
    TEST_FILE="${TEST_FILE:-$TRAIN_FILE}"
    EVAL_DATASETS="${EVAL_DATASETS:-humaneval_plus mbpp_plus livecodebench_v6}"
    PASS_K="${PASS_K:-16}"
    export CODE_GRPO_REWARD_CONTRACT="${CODE_GRPO_REWARD_CONTRACT:-deepcoder_binary_15_longest_v1}"
    export CODE_GRPO_MAX_TEST_CASES="${CODE_GRPO_MAX_TEST_CASES:-15}"
    export CODE_GRPO_EXEC_TIMEOUT_SECONDS="${CODE_GRPO_EXEC_TIMEOUT_SECONDS:-10}"
    if [ "$CODE_GRPO_REWARD_CONTRACT" != deepcoder_binary_15_longest_v1 ] || \
       [ "$CODE_GRPO_MAX_TEST_CASES" != 15 ] || \
       [ "$CODE_GRPO_EXEC_TIMEOUT_SECONDS" != 10 ]; then
        echo "ERROR: code GRPO requires deepcoder_binary_15_longest_v1 with 15 tests and timeout=10s" >&2
        exit 2
    fi
    REWARD_CONFIG=(
        reward.custom_reward_function.path=recipe/opd/run/grpo/code_reward.py
        reward.custom_reward_function.name=compute_score
    )
fi
export HF_HOME="${GRPO_HF_HOME:-data/eval_dataset/$TASK/huggingface_cache}"
export HF_DATASETS_CACHE="${GRPO_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${GRPO_HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE

if [ "$GRPO_DRY_RUN" != "print" ]; then
    require_model "$MODEL_PATH"
    require_file "$TRAIN_FILE" "GRPO training parquet"
    require_file "$TEST_FILE" "validation parquet"
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-18432}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MAX_TRAIN_DURATION_SECONDS="${MAX_TRAIN_DURATION_SECONDS:-31536000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
STOP_AT_STEP="${STOP_AT_STEP:-null}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
SAVE_FREQ="${SAVE_FREQ:--1}"
SAVE_AT_END="${SAVE_AT_END:-true}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"

case "$MAX_TRAIN_DURATION_SECONDS" in
    *[!0-9]*|"") echo "ERROR: MAX_TRAIN_DURATION_SECONDS must be a positive integer" >&2; exit 1 ;;
esac
if [ "$MAX_TRAIN_DURATION_SECONDS" -le 0 ]; then
    echo "ERROR: MAX_TRAIN_DURATION_SECONDS must be positive" >&2
    exit 1
fi

HYDRA_ARGS=(
    --config-path=config
    --config-name=ppo_trainer.yaml
)

TRAIN_OVERRIDES=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.prompt_key=prompt
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.max_prompt_length="$MAX_PROMPT_LENGTH"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=False
    data.truncation=error
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
    actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]'
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.tensor_model_parallel_size="$GEN_TP"
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN"
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    trainer.use_legacy_worker_impl=disable
    trainer.critic_warmup=0
    trainer.logger='["console","wandb","tensorboard","file"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.default_local_dir="$CKPTS_DIR"
    trainer.n_gpus_per_node="$NGPUS_PER_NODE"
    trainer.nnodes="$NNODES"
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq="$SAVE_FREQ"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
    trainer.stop_at_step="$STOP_AT_STEP"
    +trainer.max_train_duration_seconds="$MAX_TRAIN_DURATION_SECONDS"
    +trainer.save_at_end="$SAVE_AT_END"
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep=1
    actor_rollout_ref.rollout.agent.default_agent_loop=base_completion_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/opd/config/base_completion_agent.yaml
)

TRAIN_COMMAND=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    "${HYDRA_ARGS[@]}"
    "${TRAIN_OVERRIDES[@]}"
    "${REWARD_CONFIG[@]}"
    "$@"
)

if [ "$GRPO_DRY_RUN" != "false" ]; then
    EVAL_RESULTS_KIND="${TASK}_avg${PASS_K}_pass${PASS_K}"
    if [ "$GRPO_DRY_RUN" = "print" ]; then
        echo "Asset validation:   skipped (GRPO_DRY_RUN=print)"
    else
        echo "Asset validation:   passed"
    fi
    echo "Task:              $TASK"
    echo "Base model:        $MODEL_PATH"
    echo "Train parquet:     $TRAIN_FILE"
    echo "Validation file:   $TEST_FILE"
    echo "Training logs:     $TRAIN_DIR"
    echo "Trained model dir: $CKPTS_DIR"
    echo "Milestone models:  $MODELS_DIR/step_<N>"
    echo "Eval results JSON: ${EVAL_RESULTS_FILE:-results/$MODEL_ALIAS/${EXPERIMENT_NAME}_${EVAL_RESULTS_KIND}.json}"
    echo "Results model key: ${EVAL_MODEL_NAME:-$EXPERIMENT_NAME}"
    echo "Training time cap: ${MAX_TRAIN_DURATION_SECONDS}s"
    echo "Full/stop steps:   $TOTAL_TRAINING_STEPS/$STOP_AT_STEP"
    echo "Micro/dynamic:     $PPO_MICRO_BATCH_SIZE_PER_GPU/$USE_DYNAMIC_BSZ"
    echo "Save frequency:    $SAVE_FREQ (save_at_end=$SAVE_AT_END)"
    echo "Rollout memory:    $ROLLOUT_GPU_MEMORY_UTILIZATION"
    echo "Rollout TP:        $GEN_TP"
    echo "Rollout max len:   $ROLLOUT_MAX_MODEL_LEN"
    echo "Rollout max seqs:  $ROLLOUT_MAX_NUM_SEQS"
    echo "Eval TP:           $EVAL_GEN_TP"
    echo "Eval datasets:     $EVAL_DATASETS"
    echo "Eval pass_k:       $PASS_K"
    if [ "$TASK" = code ]; then
        echo "Code reward:       $CODE_GRPO_REWARD_CONTRACT"
        echo "Reward tests:      top $CODE_GRPO_MAX_TEST_CASES by input length; binary all-pass; timeout=${CODE_GRPO_EXEC_TIMEOUT_SECONDS}s"
    fi
    echo "Eval after train:  ${RUN_EVAL_AFTER_TRAINING:-true}"
    print_command "${TRAIN_COMMAND[@]}"
    exit 0
fi

mkdir -p "$TRAIN_DIR" "$CKPTS_DIR" "$MODELS_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
set -x
"${TRAIN_COMMAND[@]}" 2>&1 | tee "$TRAIN_DIR/train_step_${STOP_AT_STEP}.log"
set +x

require_checkpoint_step "$CKPTS_DIR" "$STOP_AT_STEP"

current_hf_model="$(latest_hf_checkpoint "$CKPTS_DIR")"
if [ -z "${EVAL_MODEL_PATH:-}" ]; then
    EVAL_MODEL_PATH="$current_hf_model"
fi
if [ -z "$EVAL_MODEL_PATH" ] || ! hf_export_complete "$EVAL_MODEL_PATH"; then
    echo "ERROR: no Hugging Face checkpoint found under $CKPTS_DIR" >&2
    exit 1
fi

RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
if [ "$RUN_EVAL_AFTER_TRAINING" != "true" ]; then
    latest_step="$(tr -d '[:space:]' < "$CKPTS_DIR/latest_checkpointed_iteration.txt")"
    prune_old_step_checkpoints "$CKPTS_DIR" "$latest_step"
    exit 0
fi

if [ "$TASK" = "math" ]; then
    export DATASETS="$EVAL_DATASETS"
    export PASS_K
    export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-data/eval_dataset/math}"
    export NGPUS_PER_NODE NNODES EVAL_MAX_NUM_SEQS EVAL_GPU_MEMORY_UTILIZATION
    export GEN_TP="$EVAL_GEN_TP"
    export EVAL_BASE_MODEL_NAME="$MODEL_ALIAS"
    export EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-$EXPERIMENT_NAME}"
    export EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-results/$MODEL_ALIAS/${EXPERIMENT_NAME}_math_avg${PASS_K}_pass${PASS_K}.json}"
    export EVAL_RESULTS_CSV_FILE="${EVAL_RESULTS_CSV_FILE:-${EVAL_RESULTS_FILE%.json}.csv}"
    export GEN_OUTPUT_BASE_DIR="${GEN_OUTPUT_BASE_DIR:-gen_results/eval/math/grpo}"
    bash recipe/opd/run/grpo/benchmark_math_local.sh "$EVAL_MODEL_PATH" 2>&1 | tee "$TRAIN_DIR/eval_step_${STOP_AT_STEP}.log"
else
    export DATASETS="$EVAL_DATASETS"
    export PASS_K
    export NGPUS_PER_NODE EVAL_MAX_NUM_SEQS EVAL_GPU_MEMORY_UTILIZATION
    export GEN_TP="$EVAL_GEN_TP"
    export EVAL_BASE_MODEL_NAME="$MODEL_ALIAS"
    export EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-$EXPERIMENT_NAME}"
    export EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-results/$MODEL_ALIAS/${EXPERIMENT_NAME}_code_avg${PASS_K}_pass${PASS_K}.json}"
    export EVAL_RESULTS_CSV_FILE="${EVAL_RESULTS_CSV_FILE:-${EVAL_RESULTS_FILE%.json}.csv}"
    export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-gen_results/eval/code/grpo/$EXPERIMENT_NAME}"
    bash recipe/opd/run/grpo/benchmark_code_local.sh "$EVAL_MODEL_PATH" 2>&1 | tee "$TRAIN_DIR/eval_step_${STOP_AT_STEP}.log"
fi

latest_step="$(tr -d '[:space:]' < "$CKPTS_DIR/latest_checkpointed_iteration.txt")"
prune_old_step_checkpoints "$CKPTS_DIR" "$latest_step"
