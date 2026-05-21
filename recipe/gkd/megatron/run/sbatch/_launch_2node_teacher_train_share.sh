#!/bin/bash
# Launcher for the 2-node "teacher + student-share" y_o layout:
#   node 0: teacher Qwen3-8B vLLM replicas on GPUs 0,1,2,3
#   node 1: student rollout vLLM on GPUs 0,1 + Megatron actor on GPUs 2,3
set -E
set -o pipefail

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$SBATCH_DIR")"
MEGATRON_DIR="$(dirname "$RUN_DIR")"
VERL_ROOT="$(dirname "$(dirname "$(dirname "$MEGATRON_DIR")")")"

cd "$VERL_ROOT"

is_true() { case "${1:-}" in true|True|1|yes|Yes|y|Y) return 0 ;; *) return 1 ;; esac; }
SRUN_JOB_ARGS=()
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SRUN_JOB_ARGS=(--jobid="$SLURM_JOB_ID")
fi
port_from_job() {
    local base="$1" offset="${2:-0}" job_id="${SLURM_JOB_ID:-0}"
    [[ ! "$job_id" =~ ^[0-9]+$ ]] && job_id=0
    printf '%s\n' $((base + (job_id % 10000) + offset))
}
wait_server_ready() {
    local server="$1" host="$2" port="$3" timeout="${4:-300}" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if python3 - "$host" "$port" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=2): pass
except OSError:
    sys.exit(1)
PY
        then
            echo "$server is reachable at $host:$port"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "ERROR: timed out waiting for $server at $host:$port" >&2
    return 1
}

DEFAULT_STUDENT_MODEL_PATH="/scratch/l/luli/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_TEACHER_MODEL_PATH="/scratch/l/luli/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_PYTHON_BIN="/scratch/l/luli/conda/envs/verl/bin/python"

export HF_HOME="${HF_HOME:-/scratch/l/luli/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
[[ "$PYTHON_BIN" = */* ]] && export PATH="$(dirname "$PYTHON_BIN"):$PATH"

export MODEL_PATH="${MODEL_PATH:-$DEFAULT_STUDENT_MODEL_PATH}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$DEFAULT_TEACHER_MODEL_PATH}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen3-8B}"

if [ -z "${SLURM_NODELIST:-}" ]; then
    echo "ERROR: SLURM_NODELIST not set; run under Slurm." >&2
    exit 1
fi
mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#ALLOCATED_NODES[@]}" -lt 2 ]; then
    echo "ERROR: two_node_teacher_train_share layout requires >= 2 allocated nodes, got ${#ALLOCATED_NODES[@]}" >&2
    exit 1
fi

export TEACHER_NODE="${TEACHER_NODE:-${ALLOCATED_NODES[0]}}"
export TRAINING_NODE="${TRAINING_NODE:-${ALLOCATED_NODES[1]}}"
export ROLLOUT_NODE="${ROLLOUT_NODE:-$TRAINING_NODE}"
export SHARED_NODE="${SHARED_NODE:-$TRAINING_NODE}"

export GPU_TYPE="${GPU_TYPE:-h100}"
export NODE_GPUS_TOTAL="${NODE_GPUS_TOTAL:-4}"
export TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-4}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-2}"
export TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-2}"

export TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
export TEACHER_REPLICAS="${TEACHER_REPLICAS:-$((TEACHER_GPUS_PER_NODE / TEACHER_TP_SIZE))}"
if (( TEACHER_REPLICAS * TEACHER_TP_SIZE > TEACHER_GPUS_PER_NODE )); then
    echo "ERROR: TEACHER_REPLICAS=$TEACHER_REPLICAS x TEACHER_TP_SIZE=$TEACHER_TP_SIZE exceeds TEACHER_GPUS_PER_NODE=$TEACHER_GPUS_PER_NODE" >&2
    exit 1
fi

export N_GPUS_PER_NODE="$TRAINING_GPUS_PER_NODE"
export NGPUS_PER_NODE="$TRAINING_GPUS_PER_NODE"
export RAY_GPU_COUNT="$((TRAINING_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE))"
# Ray actor workers are created before rollout workers. Listing actor devices
# first makes the intended physical split actor=2,3 and rollout=0,1.
export RAY_CUDA_VISIBLE_DEVICES="${RAY_CUDA_VISIBLE_DEVICES:-${TRAINING_CUDA_VISIBLE_DEVICES},${ROLLOUT_CUDA_VISIBLE_DEVICES}}"
export HYBRID_ENGINE="${HYBRID_ENGINE:-False}"

export OPD_JOB_WORK_DIR="${OPD_JOB_WORK_DIR:-/scratch/l/luli/openclaw/tmp/opd_megatron/${SLURM_JOB_ID:-manual}}"
export TEACHER_LOG_DIR="${TEACHER_LOG_DIR:-$OPD_JOB_WORK_DIR/teacher}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/opd_ray_${SLURM_JOB_ID:-manual}}"
RERUN_ENV_FILE="$OPD_JOB_WORK_DIR/rerun_env.sh"
mkdir -p "$OPD_JOB_WORK_DIR" "$TEACHER_LOG_DIR" "$RAY_TMPDIR"

export TEACHER_SERVER_HOST="${TEACHER_SERVER_HOST:-$TEACHER_NODE}"
export TEACHER_SERVER_PORT="${TEACHER_SERVER_PORT:-$(port_from_job 15000 0)}"
export PROXY_FRONTEND_PORT="$TEACHER_SERVER_PORT"
export PROXY_BACKEND_PORT="${PROXY_BACKEND_PORT:-$((TEACHER_SERVER_PORT + 1))}"

export RAY_PORT="${RAY_PORT:-$(port_from_job 30000 0)}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$((RAY_PORT + 1))}"
export RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"

resolve_run_script() {
    local s="${1:-opd/forward_y_o.sh}"
    [[ "$s" = /* ]] && printf '%s\n' "$s" || printf '%s\n' "$RUN_DIR/$s"
}
RUN_SCRIPT_PATH="$(resolve_run_script "${RUN_SCRIPT:-opd/forward_y_o.sh}")"
export N_LOGPROBS="${N_LOGPROBS:-full_vocab}"
export TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.90}"
export TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-}"
export VERL_VLLM_USE_SHM_WEIGHT_SYNC="${VERL_VLLM_USE_SHM_WEIGHT_SYNC:-1}"

echo "=========================================="
echo "OPD Megatron 2-node teacher/student-share layout"
echo "  job:              ${SLURM_JOB_ID:-manual}"
echo "  teacher node:     $TEACHER_NODE"
echo "    - teacher:      Qwen3-8B vLLM replicas=$TEACHER_REPLICAS TP=$TEACHER_TP_SIZE on GPU(s) $TEACHER_CUDA_VISIBLE_DEVICES"
echo "    - memory:       gpu_memory_utilization=$TEACHER_GPU_MEMORY_UTILIZATION max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-unset}"
echo "  student node:     $TRAINING_NODE"
echo "    - rollout:      Qwen3-1.7B vLLM on physical GPU(s) $ROLLOUT_CUDA_VISIBLE_DEVICES"
echo "    - actor:        Megatron actor on physical GPU(s) $TRAINING_CUDA_VISIBLE_DEVICES"
echo "    - ray sees:     $RAY_CUDA_VISIBLE_DEVICES ($RAY_GPU_COUNT GPUs)"
echo "    - weight sync:  VERL_VLLM_USE_SHM_WEIGHT_SYNC=$VERL_VLLM_USE_SHM_WEIGHT_SYNC"
echo "  teacher endpoint: $TEACHER_SERVER_HOST:$TEACHER_SERVER_PORT"
echo "  ray head/dashbd:  $TRAINING_NODE:$RAY_PORT / $RAY_DASHBOARD_PORT"
echo "  hybrid_engine:    $HYBRID_ENGINE"
echo "  work dir:         $OPD_JOB_WORK_DIR"
echo "=========================================="

echo "Starting teacher on $TEACHER_NODE GPU(s) $TEACHER_CUDA_VISIBLE_DEVICES ..."
srun "${SRUN_JOB_ARGS[@]}" --overlap --nodes=1 --ntasks=1 --nodelist="$TEACHER_NODE" --gres="gpu:${GPU_TYPE}:${NODE_GPUS_TOTAL}" \
    /usr/bin/env \
    CUDA_VISIBLE_DEVICES="$TEACHER_CUDA_VISIBLE_DEVICES" \
    CKPT_PATH="${CKPT_PATH:-$TEACHER_MODEL_PATH}" \
    TEACHER_LOG_DIR="$TEACHER_LOG_DIR" \
    TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
    TEACHER_TP_SIZE="$TEACHER_TP_SIZE" \
    TEACHER_REPLICAS="$TEACHER_REPLICAS" \
    N_LOGPROBS="$N_LOGPROBS" \
    TEACHER_GPU_MEMORY_UTILIZATION="$TEACHER_GPU_MEMORY_UTILIZATION" \
    TEACHER_MAX_NUM_BATCHED_TOKENS="$TEACHER_MAX_NUM_BATCHED_TOKENS" \
    TEACHER_SEQ_LEN="$TEACHER_SEQ_LEN" \
    PROXY_FRONTEND_PORT="$PROXY_FRONTEND_PORT" \
    PROXY_BACKEND_PORT="$PROXY_BACKEND_PORT" \
    bash -lc "bash '$MEGATRON_DIR/teacher/start_server.sh'; while true; do sleep 3600; done" &

wait_server_ready teacher "$TEACHER_SERVER_HOST" "$TEACHER_SERVER_PORT" 600 || {
    echo "teacher boot failed"
    sleep infinity
    exit 1
}
sleep 30

echo "Starting Ray head on $TRAINING_NODE covering GPUs $RAY_CUDA_VISIBLE_DEVICES ..."
srun "${SRUN_JOB_ARGS[@]}" --overlap --nodes=1 --ntasks=1 --nodelist="$TRAINING_NODE" --gres="gpu:${GPU_TYPE}:${NODE_GPUS_TOTAL}" \
    /usr/bin/env \
    CUDA_VISIBLE_DEVICES="$RAY_CUDA_VISIBLE_DEVICES" \
    bash -lc "
        ray stop --force >/dev/null 2>&1 || true
        ray start --head \
            --port='$RAY_PORT' \
            --dashboard-host=0.0.0.0 \
            --dashboard-port='$RAY_DASHBOARD_PORT' \
            --num-cpus='${SLURM_CPUS_PER_TASK:-48}' \
            --num-gpus='$RAY_GPU_COUNT' \
            --temp-dir='$RAY_TMPDIR' \
            --resources='{\"actor\": 1, \"rollout\": 1}'
        while true; do sleep 3600; done
    " &
sleep 15

cat > "$RERUN_ENV_FILE" <<EOF
# Source inside the held allocation to rerun training.
cd $VERL_ROOT
export HF_HOME=$HF_HOME
export TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
export PYTHON_BIN=$PYTHON_BIN
export PATH=$(dirname "$PYTHON_BIN"):\$PATH
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export STUDENT_MODEL=$STUDENT_MODEL
export TEACHER_MODEL_PATH=$TEACHER_MODEL_PATH
export TEACHER_MODEL=$TEACHER_MODEL
export TEACHER_SERVER_HOST=$TEACHER_SERVER_HOST
export TEACHER_SERVER_PORT=$TEACHER_SERVER_PORT
export PROXY_FRONTEND_PORT=$PROXY_FRONTEND_PORT
export PROXY_BACKEND_PORT=$PROXY_BACKEND_PORT
export RAY_ADDRESS=$RAY_ADDRESS
export RAY_PORT=$RAY_PORT
export RAY_DASHBOARD_PORT=$RAY_DASHBOARD_PORT
export RAY_TMPDIR=$RAY_TMPDIR
export OPD_JOB_WORK_DIR=$OPD_JOB_WORK_DIR
export DISTILL_MODE=$DISTILL_MODE
export KL_TYPE=$KL_TYPE
export Y_MODE=$Y_MODE
export TEACHER_TRAINING_PROMPT=$TEACHER_TRAINING_PROMPT
export KL_TOKEN_CLIP=$KL_TOKEN_CLIP
export TOP_K=$TOP_K
export OPTIMIZATION_MODE=$OPTIMIZATION_MODE
export SCHEDULER=$SCHEDULER
export TEACHER_CLIENT_TIMEOUT_MS=$TEACHER_CLIENT_TIMEOUT_MS
export TEACHER_CLIENT_RCVTIMEO_MS=$TEACHER_CLIENT_RCVTIMEO_MS
export EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE
export TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS
export USE_DYNAMIC_BSZ=$USE_DYNAMIC_BSZ
export AGENT_NUM_WORKERS=$AGENT_NUM_WORKERS
export DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS
export UPDATE_WEIGHTS_BUCKET_MEGABYTES=$UPDATE_WEIGHTS_BUCKET_MEGABYTES
export ROLLOUT_MAX_NUM_SEQS=$ROLLOUT_MAX_NUM_SEQS
export TEACHER_REPLICAS=$TEACHER_REPLICAS
export TEACHER_N_SERVER_WORKERS=$TEACHER_N_SERVER_WORKERS
export TEACHER_REQUEST_BATCH_SIZE=$TEACHER_REQUEST_BATCH_SIZE
export MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
export TEACHER_MAX_PROMPT_LENGTH=$TEACHER_MAX_PROMPT_LENGTH
export MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH
export STUDENT_ACTOR_MAX_TOKENS_PER_GPU=$STUDENT_ACTOR_MAX_TOKENS_PER_GPU
export TEACHER_SEQ_LEN=$TEACHER_SEQ_LEN
export TEACHER_GPU_MEMORY_UTILIZATION=$TEACHER_GPU_MEMORY_UTILIZATION
export TEACHER_MAX_NUM_BATCHED_TOKENS=$TEACHER_MAX_NUM_BATCHED_TOKENS
export N_LOGPROBS=$N_LOGPROBS
export SHARED_NODE=$SHARED_NODE
export TEACHER_NODE=$TEACHER_NODE
export ROLLOUT_NODE=$ROLLOUT_NODE
export TRAINING_NODE=$TRAINING_NODE
export TEACHER_CUDA_VISIBLE_DEVICES=$TEACHER_CUDA_VISIBLE_DEVICES
export ROLLOUT_CUDA_VISIBLE_DEVICES=$ROLLOUT_CUDA_VISIBLE_DEVICES
export TRAINING_CUDA_VISIBLE_DEVICES=$TRAINING_CUDA_VISIBLE_DEVICES
export RAY_CUDA_VISIBLE_DEVICES=$RAY_CUDA_VISIBLE_DEVICES
export TEACHER_GPUS_PER_NODE=$TEACHER_GPUS_PER_NODE
export ROLLOUT_GPUS_PER_NODE=$ROLLOUT_GPUS_PER_NODE
export TRAINING_GPUS_PER_NODE=$TRAINING_GPUS_PER_NODE
export NODE_GPUS_TOTAL=$NODE_GPUS_TOTAL
export N_GPUS_PER_NODE=$TRAINING_GPUS_PER_NODE
export NGPUS_PER_NODE=$TRAINING_GPUS_PER_NODE
export HYBRID_ENGINE=$HYBRID_ENGINE
export ROLLOUT_ENABLE_SLEEP_MODE=$ROLLOUT_ENABLE_SLEEP_MODE
export ROLLOUT_FREE_CACHE_ENGINE=$ROLLOUT_FREE_CACHE_ENGINE
export ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION
export ROLLOUT_ENABLE_CHUNKED_PREFILL=$ROLLOUT_ENABLE_CHUNKED_PREFILL
export ROLLOUT_ENABLE_PREFIX_CACHING=$ROLLOUT_ENABLE_PREFIX_CACHING
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
export VERL_VLLM_USE_SHM_WEIGHT_SYNC=$VERL_VLLM_USE_SHM_WEIGHT_SYNC
bash $RUN_SCRIPT_PATH
EOF
echo "Rerun environment written to $RERUN_ENV_FILE"

echo "Launching training driver on $TRAINING_NODE ..."
srun "${SRUN_JOB_ARGS[@]}" --overlap --nodes=1 --ntasks=1 --nodelist="$TRAINING_NODE" --gres="gpu:${GPU_TYPE}:${NODE_GPUS_TOTAL}" \
    /usr/bin/env \
    CUDA_VISIBLE_DEVICES="$RAY_CUDA_VISIBLE_DEVICES" \
    N_GPUS_PER_NODE="$TRAINING_GPUS_PER_NODE" \
    NGPUS_PER_NODE="$TRAINING_GPUS_PER_NODE" \
    ROLLOUT_GPUS_PER_NODE="$ROLLOUT_GPUS_PER_NODE" \
    HYBRID_ENGINE="$HYBRID_ENGINE" \
    SCHEDULER="$SCHEDULER" \
    TEACHER_CLIENT_TIMEOUT_MS="$TEACHER_CLIENT_TIMEOUT_MS" \
    TEACHER_CLIENT_RCVTIMEO_MS="$TEACHER_CLIENT_RCVTIMEO_MS" \
    EFFECTIVE_BATCH_SIZE="$EFFECTIVE_BATCH_SIZE" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
    AGENT_NUM_WORKERS="$AGENT_NUM_WORKERS" \
    DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
    UPDATE_WEIGHTS_BUCKET_MEGABYTES="$UPDATE_WEIGHTS_BUCKET_MEGABYTES" \
    ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS" \
    TEACHER_N_SERVER_WORKERS="$TEACHER_N_SERVER_WORKERS" \
    TEACHER_REQUEST_BATCH_SIZE="$TEACHER_REQUEST_BATCH_SIZE" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    TEACHER_MAX_PROMPT_LENGTH="$TEACHER_MAX_PROMPT_LENGTH" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    TEACHER_SEQ_LEN="$TEACHER_SEQ_LEN" \
    STUDENT_ACTOR_MAX_TOKENS_PER_GPU="$STUDENT_ACTOR_MAX_TOKENS_PER_GPU" \
    ROLLOUT_ENABLE_SLEEP_MODE="$ROLLOUT_ENABLE_SLEEP_MODE" \
    ROLLOUT_FREE_CACHE_ENGINE="$ROLLOUT_FREE_CACHE_ENGINE" \
    ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    ROLLOUT_ENABLE_CHUNKED_PREFILL="$ROLLOUT_ENABLE_CHUNKED_PREFILL" \
    ROLLOUT_ENABLE_PREFIX_CACHING="$ROLLOUT_ENABLE_PREFIX_CACHING" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    VERL_VLLM_USE_SHM_WEIGHT_SYNC="$VERL_VLLM_USE_SHM_WEIGHT_SYNC" \
    RAY_ADDRESS="$RAY_ADDRESS" \
    RAY_PORT="$RAY_PORT" \
    RAY_DASHBOARD_PORT="$RAY_DASHBOARD_PORT" \
    RAY_TMPDIR="$RAY_TMPDIR" \
    bash "$RUN_SCRIPT_PATH" "$@"

run_status=$?
echo "Training exited with status $run_status"
if [ "$run_status" -ne 0 ] && is_true "${KEEP_ALIVE_ON_FAILURE:-true}"; then
    echo "Keeping allocation alive."
    echo "Debug: srun --jobid=${SLURM_JOB_ID} --pty bash"
    echo "       source $RERUN_ENV_FILE"
    sleep infinity
fi
exit "$run_status"
