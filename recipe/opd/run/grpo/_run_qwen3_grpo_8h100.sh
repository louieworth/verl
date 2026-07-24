#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$VERL_ROOT"

export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

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
        [ "$count" -gt 0 ] && { printf '%s\n' "$count"; return 0; }
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
            printf '%s\n' "$count"
            return 0
        fi
    fi
    printf '8\n'
}

first_parquet_under() {
    local root="$1"
    local pattern="$2"
    [ -d "$root" ] || return 1
    find "$root" -maxdepth 3 -type f -name "$pattern" | sort | head -n 1
}

prepare_train_file_if_needed() {
    local task="$1"
    local input_path="$2"
    local output_file="$3"
    local data_source="$4"

    if [ -f "$output_file" ] && [ "${REBUILD_TRAIN_DATA:-false}" != "true" ]; then
        return 0
    fi
    [ "${PREPARE_TRAIN_DATA:-auto}" != "false" ] || return 1
    [ -n "$input_path" ] || return 1
    [ -e "$input_path" ] || return 1
    mkdir -p "$(dirname "$output_file")"
    if [ "$task" = "math" ]; then
        python3 recipe/opd/dataset/prepare_deepscaler_grpo.py \
            --input_path "$input_path" \
            --output_dir "$(dirname "$output_file")" \
            --train_file_name "$(basename "$output_file")" \
            --data_source "$data_source"
    else
        python3 recipe/opd/generation/y_o_prepare.py \
            --task "$task" \
            --input_path "$input_path" \
            --output_file "$output_file" \
            --data_source "$data_source"
    fi
    [ -f "$output_file" ]
}

require_file() {
    local path="$1"
    local var_name="$2"
    if [ ! -f "$path" ]; then
        echo "ERROR: $var_name does not point to a file: $path" >&2
        exit 1
    fi
}

latest_hf_checkpoint() {
    local ckpt_root="$1"
    local latest_file="$ckpt_root/latest_checkpointed_iteration.txt"
    local step_dir=""
    if [ -f "$latest_file" ]; then
        local step
        step="$(tr -d '[:space:]' < "$latest_file")"
        if [ -n "$step" ] && [ -d "$ckpt_root/global_step_${step}/actor/huggingface" ]; then
            printf '%s\n' "$ckpt_root/global_step_${step}/actor/huggingface"
            return 0
        fi
    fi
    step_dir="$(find "$ckpt_root" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    if [ -n "$step_dir" ] && [ -d "$step_dir/actor/huggingface" ]; then
        printf '%s\n' "$step_dir/actor/huggingface"
        return 0
    fi
    return 1
}

NGPUS_PER_NODE="${NGPUS_PER_NODE:-$(detect_visible_gpu_count)}"
NNODES="${NNODES:-1}"
GEN_TP="${GEN_TP:-2}"
EVAL_GEN_TP="${EVAL_GEN_TP:-$NGPUS_PER_NODE}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d.%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_qwen3_8h100}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_ALIAS}_${TASK}_grpo_${TIMESTAMP}}"
TRAIN_DIR="${TRAIN_DIR:-outputs/$PROJECT_NAME/$EXPERIMENT_NAME}"
CKPTS_DIR="${CKPTS_DIR:-$TRAIN_DIR/checkpoints}"
mkdir -p "$TRAIN_DIR" "$CKPTS_DIR"

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$TRAIN_DIR/tensorboard_log}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-$TRAIN_DIR/metrics.jsonl}"

if [ "$TASK" = "math" ]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data/data/jiangli/data/DeepScaleR-Cleaned}"
    if [ -z "${TRAIN_FILE:-}" ]; then
        PREPARED_TRAIN_FILE="${PREPARED_TRAIN_FILE:-$VERL_ROOT/gen_results/grpo_data/deepscaleR_train.parquet}"
        if prepare_train_file_if_needed math "$TRAIN_DATA_PATH" "$PREPARED_TRAIN_FILE" deepscaleR; then
            TRAIN_FILE="$PREPARED_TRAIN_FILE"
        fi
    fi
    TEST_FILE="${TEST_FILE:-${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}/aime24/aime24_test.parquet}"
    EVAL_DATASETS="${EVAL_DATASETS:-aime24 aime25 hmmt25 beyondaime amobench}"
    PASS_K="${PASS_K:-16}"
    REWARD_CONFIG=(
        reward.custom_reward_function.path=recipe/r1_ascend/deepscaler.py
        reward.custom_reward_function.name=compute_score
    )
else
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data/data/jiangli/huggingface/datasets/TACO}"
    if [ -z "${TRAIN_FILE:-}" ]; then
        PREPARED_TRAIN_FILE="${PREPARED_TRAIN_FILE:-$VERL_ROOT/gen_results/grpo_data/taco_train.parquet}"
        if prepare_train_file_if_needed code "$TRAIN_DATA_PATH" "$PREPARED_TRAIN_FILE" BAAI/TACO; then
            TRAIN_FILE="$PREPARED_TRAIN_FILE"
        fi
    fi
    TEST_FILE="${TEST_FILE:-$TRAIN_FILE}"
    EVAL_DATASETS="${EVAL_DATASETS:-humaneval_plus mbpp_plus livecodebench_v6}"
    PASS_K="${PASS_K:-16}"
    REWARD_CONFIG=(
        reward.custom_reward_function.path=recipe/opd/grpo_code_reward.py
        reward.custom_reward_function.name=compute_score
    )
fi

require_file "$TRAIN_FILE" TRAIN_FILE
require_file "$TEST_FILE" TEST_FILE

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
if [ "$TASK" = "code" ]; then
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
else
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
fi
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"

TRAIN_ARGS=(
    --config-path=config
    --config-name=ppo_trainer.yaml
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.prompt_key=prompt
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.max_prompt_length="$MAX_PROMPT_LENGTH"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
    actor_rollout_ref.actor.use_dynamic_bsz=True
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
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]'
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.tensor_model_parallel_size="$GEN_TP"
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    trainer.use_legacy_worker_impl=disable
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard","file"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.default_local_dir="$CKPTS_DIR"
    trainer.n_gpus_per_node="$NGPUS_PER_NODE"
    trainer.nnodes="$NNODES"
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq="$SAVE_FREQ"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep=2
)

python3 -m verl.trainer.main_ppo \
    "${REWARD_CONFIG[@]}" \
    "${TRAIN_ARGS[@]}" \
    "$@" 2>&1 | tee "$TRAIN_DIR/train.log"

RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"
[ "$RUN_EVAL_AFTER_TRAINING" = "true" ] || exit 0

if [ -z "${EVAL_MODEL_PATH:-}" ]; then
    EVAL_MODEL_PATH="$(latest_hf_checkpoint "$CKPTS_DIR" || true)"
fi
if [ -z "$EVAL_MODEL_PATH" ] || [ ! -d "$EVAL_MODEL_PATH" ]; then
    echo "ERROR: no Hugging Face checkpoint found under $CKPTS_DIR" >&2
    echo "The scripts save HF checkpoints via actor_rollout_ref.actor.checkpoint.save_contents." >&2
    exit 1
fi

if [ "$TASK" = "math" ]; then
    export DATASETS="$EVAL_DATASETS"
    export PASS_K
    export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-/data/data/jiangli/huggingface/datasets}"
    export EVAL_PROMPT_LENGTH="${EVAL_PROMPT_LENGTH:-4096}"
    export EVAL_RESPONSE_LENGTH="${EVAL_RESPONSE_LENGTH:-16384}"
    export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-$((EVAL_PROMPT_LENGTH + EVAL_RESPONSE_LENGTH))}"
    export NGPUS_PER_NODE NNODES
    export GEN_TP="$EVAL_GEN_TP"
    export EVAL_BASE_MODEL_NAME="$MODEL_ALIAS"
    export EVAL_MODEL_NAME="$EXPERIMENT_NAME"
    export EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-$VERL_ROOT/results/$MODEL_ALIAS/${EXPERIMENT_NAME}_math_avg16_pass16.json}"
    export EVAL_RESULTS_CSV_FILE="${EVAL_RESULTS_CSV_FILE:-${EVAL_RESULTS_FILE%.json}.csv}"
    export GEN_OUTPUT_BASE_DIR="${GEN_OUTPUT_BASE_DIR:-$VERL_ROOT/gen_results/eval/math/grpo}"
    bash "$VERL_ROOT/recipe/math_evaluation/benchmark_kl_model.sh" "$EVAL_MODEL_PATH" 2>&1 | tee "$TRAIN_DIR/eval.log"
else
    export DATASETS="$EVAL_DATASETS"
    export PASS_K
    export CODE_EVAL_MAX_PROMPT_TOKENS="${CODE_EVAL_MAX_PROMPT_TOKENS:-2048}"
    export CODE_EVAL_MAX_RESPONSE_TOKENS="${CODE_EVAL_MAX_RESPONSE_TOKENS:-16384}"
    export CODE_EVAL_MAX_MODEL_LEN="${CODE_EVAL_MAX_MODEL_LEN:-$((CODE_EVAL_MAX_PROMPT_TOKENS + CODE_EVAL_MAX_RESPONSE_TOKENS))}"
    export NGPUS_PER_NODE
    export GEN_TP="$EVAL_GEN_TP"
    export EVAL_BASE_MODEL_NAME="$MODEL_ALIAS"
    export EVAL_MODEL_NAME="$EXPERIMENT_NAME"
    export EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-$VERL_ROOT/results/$MODEL_ALIAS/${EXPERIMENT_NAME}_code_avg16_pass16.json}"
    export EVAL_RESULTS_CSV_FILE="${EVAL_RESULTS_CSV_FILE:-${EVAL_RESULTS_FILE%.json}.csv}"
    export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$VERL_ROOT/gen_results/eval/code/grpo/$EXPERIMENT_NAME}"
    bash "$VERL_ROOT/recipe/code_evaluation/benchmark_code_model.sh" "$EVAL_MODEL_PATH" 2>&1 | tee "$TRAIN_DIR/eval.log"
fi
