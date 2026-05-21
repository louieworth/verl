#!/bin/bash
# Launcher for the 3-node split layout:
#   node 0 (teacher_node):  teacher Qwen3-8B vLLM server (proxy + worker)
#   node 1 (rollout_node):  student vLLM rollout — Ray worker
#   node 2 (training_node): student Megatron actor — Ray head + driver
#
# Ray cluster spans rollout_node + training_node (training_node is head).
# If TEACHER_RAY_MANAGED=true, teacher_node also joins Ray with a custom
# "teacher" resource and a Ray supervisor actor starts the existing ZMQ/vLLM
# teacher server. The trainer still talks to TEACHER_SERVER_HOST:PORT.
#
# UNTESTED with the current verl OPD recipe. Known assumptions to verify:
#   1. recipe/gkd/megatron/megatron_workers.py: MegatronOnPolicyDistillRolloutWorker
#      must work with `hybrid_engine=False`. The `sync_rollout_weights()`
#      method is already gated on `not self.config.hybrid_engine` so the
#      broadcast group "actor_rollout" should kick in.
#   2. The hydra config has to set `actor_rollout_ref.hybrid_engine=False`
#      and likely needs separate resource pools for actor vs rollout. We pass
#      the override via the entrypoint command in forward_y_r.sh.
#   3. The student vLLM rollout server has to be reachable from the actor
#      node — Ray placement groups handle this when both nodes are in the
#      same Ray cluster.
set -E
set -o pipefail

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$SBATCH_DIR")"
MEGATRON_DIR="$(dirname "$RUN_DIR")"
VERL_ROOT="$(dirname "$(dirname "$(dirname "$MEGATRON_DIR")")")"

cd "$VERL_ROOT"

is_true() { case "${1:-}" in true|True|1|yes|Yes|y|Y) return 0 ;; *) return 1 ;; esac; }
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

# --- Node assignment ---
if [ -z "${SLURM_NODELIST:-}" ]; then
    echo "ERROR: SLURM_NODELIST not set; run under Slurm." >&2; exit 1
fi
mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#ALLOCATED_NODES[@]}" -lt 3 ]; then
    echo "ERROR: three_node layout requires >= 3 allocated nodes, got ${#ALLOCATED_NODES[@]}" >&2; exit 1
fi
export TEACHER_NNODES="${TEACHER_NNODES:-1}"
case "$TEACHER_NNODES" in
    ""|*[!0-9]*) echo "ERROR: TEACHER_NNODES must be a positive integer, got $TEACHER_NNODES" >&2; exit 1 ;;
esac
if (( TEACHER_NNODES < 1 || TEACHER_NNODES >= ${#ALLOCATED_NODES[@]} )); then
    echo "ERROR: TEACHER_NNODES=$TEACHER_NNODES must leave at least one node for student services in ${#ALLOCATED_NODES[@]} allocated nodes" >&2
    exit 1
fi
TEACHER_NODE_ARRAY=("${ALLOCATED_NODES[@]:0:TEACHER_NNODES}")
export TEACHER_NODE="${TEACHER_NODE:-${TEACHER_NODE_ARRAY[0]}}"
TEACHER_NODES_CSV="$(IFS=,; echo "${TEACHER_NODE_ARRAY[*]}")"
export TEACHER_NODES="${TEACHER_NODES:-$TEACHER_NODES_CSV}"
DEFAULT_ROLLOUT_NODE="${ALLOCATED_NODES[$TEACHER_NNODES]}"
DEFAULT_TRAINING_NODE="${ALLOCATED_NODES[$TEACHER_NNODES]}"
if (( TEACHER_NNODES == 1 )); then
    DEFAULT_TRAINING_NODE="${ALLOCATED_NODES[$((TEACHER_NNODES + 1))]}"
fi
export ROLLOUT_NODE="${ROLLOUT_NODE:-$DEFAULT_ROLLOUT_NODE}"
export TRAINING_NODE="${TRAINING_NODE:-$DEFAULT_TRAINING_NODE}"

export GPU_TYPE="${GPU_TYPE:-h100}"
export TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-4}"
export ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-4}"
export TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-4}"
if [ "$ROLLOUT_NODE" = "$TRAINING_NODE" ]; then
    export STUDENT_SHARED_NODE=true
    export RAY_HEAD_GPUS_PER_NODE="${RAY_HEAD_GPUS_PER_NODE:-$((ROLLOUT_GPUS_PER_NODE + TRAINING_GPUS_PER_NODE))}"
    if [ -z "${RAY_HEAD_CUDA_VISIBLE_DEVICES:-}" ]; then
        RAY_HEAD_CUDA_VISIBLE_DEVICES=""
        for ((gpu_idx = 0; gpu_idx < RAY_HEAD_GPUS_PER_NODE; gpu_idx++)); do
            if [ -z "$RAY_HEAD_CUDA_VISIBLE_DEVICES" ]; then
                RAY_HEAD_CUDA_VISIBLE_DEVICES="$gpu_idx"
            else
                RAY_HEAD_CUDA_VISIBLE_DEVICES="$RAY_HEAD_CUDA_VISIBLE_DEVICES,$gpu_idx"
            fi
        done
        export RAY_HEAD_CUDA_VISIBLE_DEVICES
    fi
else
    export STUDENT_SHARED_NODE=false
    export RAY_HEAD_GPUS_PER_NODE="${RAY_HEAD_GPUS_PER_NODE:-$TRAINING_GPUS_PER_NODE}"
    export RAY_HEAD_CUDA_VISIBLE_DEVICES="${RAY_HEAD_CUDA_VISIBLE_DEVICES:-$TRAINING_CUDA_VISIBLE_DEVICES}"
fi

TEACHER_TOTAL_GPUS=$((TEACHER_NNODES * TEACHER_GPUS_PER_NODE))
export TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-$TEACHER_GPUS_PER_NODE}"
case "$TEACHER_TP_SIZE" in
    ""|*[!0-9]*) echo "ERROR: TEACHER_TP_SIZE must be a positive integer, got $TEACHER_TP_SIZE" >&2; exit 1 ;;
esac
if (( TEACHER_TP_SIZE < 1 )); then
    echo "ERROR: TEACHER_TP_SIZE must be positive, got $TEACHER_TP_SIZE" >&2
    exit 1
fi
if [ "${TEACHER_REPLICAS:-auto}" = "auto" ]; then
    export TEACHER_REPLICAS="$((TEACHER_TOTAL_GPUS / TEACHER_TP_SIZE))"
else
    export TEACHER_REPLICAS
fi
if [ "${TEACHER_N_SERVER_WORKERS:-auto}" = "auto" ]; then
    export TEACHER_N_SERVER_WORKERS="$TEACHER_REPLICAS"
else
    export TEACHER_N_SERVER_WORKERS
fi
case "$TEACHER_REPLICAS" in
    ""|*[!0-9]*) echo "ERROR: TEACHER_REPLICAS must be a positive integer, got $TEACHER_REPLICAS" >&2; exit 1 ;;
esac
if (( TEACHER_REPLICAS < 1 )); then
    echo "ERROR: TEACHER_REPLICAS must be positive, got $TEACHER_REPLICAS" >&2
    exit 1
fi
if (( TEACHER_REPLICAS * TEACHER_TP_SIZE > TEACHER_TOTAL_GPUS )); then
    echo "ERROR: TEACHER_REPLICAS=$TEACHER_REPLICAS x TEACHER_TP_SIZE=$TEACHER_TP_SIZE exceeds teacher total GPUs=$TEACHER_TOTAL_GPUS" >&2
    echo "Hint: set TEACHER_REPLICAS=auto when changing TEACHER_TP_SIZE, or set TEACHER_REPLICAS=$((TEACHER_TOTAL_GPUS / TEACHER_TP_SIZE))." >&2
    exit 1
fi
if (( TEACHER_REPLICAS % TEACHER_NNODES != 0 )); then
    echo "ERROR: TEACHER_REPLICAS=$TEACHER_REPLICAS must divide evenly across TEACHER_NNODES=$TEACHER_NNODES" >&2
    exit 1
fi
TEACHER_REPLICAS_PER_NODE=$((TEACHER_REPLICAS / TEACHER_NNODES))
if (( TEACHER_REPLICAS_PER_NODE * TEACHER_TP_SIZE > TEACHER_GPUS_PER_NODE )); then
    echo "ERROR: per-node teacher replicas=$TEACHER_REPLICAS_PER_NODE x TP=$TEACHER_TP_SIZE exceeds TEACHER_GPUS_PER_NODE=$TEACHER_GPUS_PER_NODE" >&2
    exit 1
fi
export N_GPUS_PER_NODE="$TRAINING_GPUS_PER_NODE"
export NGPUS_PER_NODE="$TRAINING_GPUS_PER_NODE"

# Hybrid engine is OFF: actor and rollout each get their own resource pool.
export HYBRID_ENGINE="${HYBRID_ENGINE:-False}"

# Ports + work dir
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
export TEACHER_RAY_MANAGED="${TEACHER_RAY_MANAGED:-false}"
export TEACHER_RAY_ACTOR_NAME="${TEACHER_RAY_ACTOR_NAME:-opd_teacher_server_${SLURM_JOB_ID:-manual}}"

# --- Resolve run script ---
resolve_run_script() {
    local s="${1:-opd/forward_y_r.sh}"
    [[ "$s" = /* ]] && printf '%s\n' "$s" || printf '%s\n' "$RUN_DIR/$s"
}
RUN_SCRIPT_PATH="$(resolve_run_script "${RUN_SCRIPT:-opd/forward_y_r.sh}")"
export N_LOGPROBS="${N_LOGPROBS:-full_vocab}"
export TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}"
export TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-}"
export TEACHER_DISABLE_CUSTOM_ALL_REDUCE="${TEACHER_DISABLE_CUSTOM_ALL_REDUCE:-false}"
export TEACHER_WORKER_READY_TIMEOUT="${TEACHER_WORKER_READY_TIMEOUT:-900}"
export VERL_VLLM_USE_SHM_WEIGHT_SYNC="${VERL_VLLM_USE_SHM_WEIGHT_SYNC:-1}"

echo "=========================================="
echo "OPD Megatron 3-node split layout"
echo "  job:              ${SLURM_JOB_ID:-manual}"
echo "  teacher nodes:    $TEACHER_NODES   (Qwen3-8B vLLM)"
echo "  rollout node:     $ROLLOUT_NODE   (Qwen3-1.7B vLLM)"
echo "  training node:    $TRAINING_NODE  (Megatron actor)"
echo "  teacher endpoint: $TEACHER_SERVER_HOST:$TEACHER_SERVER_PORT"
echo "  teacher workers:  replicas=$TEACHER_REPLICAS (${TEACHER_REPLICAS_PER_NODE}/node) tp_size=$TEACHER_TP_SIZE client_workers=$TEACHER_N_SERVER_WORKERS ready_timeout=$TEACHER_WORKER_READY_TIMEOUT"
echo "  teacher ray:      managed=$TEACHER_RAY_MANAGED actor=$TEACHER_RAY_ACTOR_NAME"
echo "  teacher memory:   gpu_memory_utilization=$TEACHER_GPU_MEMORY_UTILIZATION n_logprobs=$N_LOGPROBS max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-unset} disable_custom_all_reduce=$TEACHER_DISABLE_CUSTOM_ALL_REDUCE"
echo "  ray head/dashbd:  $TRAINING_NODE:$RAY_PORT / $RAY_DASHBOARD_PORT head_gpus=$RAY_HEAD_GPUS_PER_NODE shared_student=$STUDENT_SHARED_NODE"
echo "  weight sync:      VERL_VLLM_USE_SHM_WEIGHT_SYNC=$VERL_VLLM_USE_SHM_WEIGHT_SYNC"
echo "  hybrid_engine:    $HYBRID_ENGINE"
echo "  work dir:         $OPD_JOB_WORK_DIR"
echo "=========================================="

start_teacher_with_srun() {
    TEACHER_SRUN_PIDS=()
    for teacher_node_idx in "${!TEACHER_NODE_ARRAY[@]}"; do
        teacher_node="${TEACHER_NODE_ARRAY[$teacher_node_idx]}"
        replica_offset=$((teacher_node_idx * TEACHER_REPLICAS_PER_NODE))
        start_proxy=false
        proxy_host="$TEACHER_SERVER_HOST"
        proxy_addr="$TEACHER_SERVER_HOST:$PROXY_BACKEND_PORT"
        if (( teacher_node_idx == 0 )); then
            start_proxy=true
            proxy_host=localhost
            proxy_addr="localhost:$PROXY_BACKEND_PORT"
        fi

        echo "Starting teacher server part on $teacher_node GPUs $TEACHER_CUDA_VISIBLE_DEVICES replicas=$TEACHER_REPLICAS_PER_NODE offset=$replica_offset proxy=$start_proxy ..."
        srun --overlap --nodes=1 --ntasks=1 --nodelist="$teacher_node" --gres="gpu:${GPU_TYPE}:${TEACHER_GPUS_PER_NODE}" \
            /usr/bin/env \
            PYTHON_BIN="$PYTHON_BIN" \
            CUDA_VISIBLE_DEVICES="$TEACHER_CUDA_VISIBLE_DEVICES" \
            CKPT_PATH="${CKPT_PATH:-$TEACHER_MODEL_PATH}" \
            TEACHER_LOG_DIR="$TEACHER_LOG_DIR" \
            TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
            TEACHER_TP_SIZE="$TEACHER_TP_SIZE" \
            TEACHER_REPLICAS="$TEACHER_REPLICAS_PER_NODE" \
            TEACHER_REPLICA_OFFSET="$replica_offset" \
            START_TEACHER_PROXY="$start_proxy" \
            PROXY_HOST="$proxy_host" \
            PROXY_ADDR="$proxy_addr" \
            N_LOGPROBS="$N_LOGPROBS" \
            TEACHER_GPU_MEMORY_UTILIZATION="$TEACHER_GPU_MEMORY_UTILIZATION" \
            TEACHER_MAX_NUM_BATCHED_TOKENS="$TEACHER_MAX_NUM_BATCHED_TOKENS" \
            TEACHER_DISABLE_CUSTOM_ALL_REDUCE="$TEACHER_DISABLE_CUSTOM_ALL_REDUCE" \
            TEACHER_WORKER_READY_TIMEOUT="$TEACHER_WORKER_READY_TIMEOUT" \
            TEACHER_SEQ_LEN="$TEACHER_SEQ_LEN" \
            PROXY_FRONTEND_PORT="$PROXY_FRONTEND_PORT" \
            PROXY_BACKEND_PORT="$PROXY_BACKEND_PORT" \
            bash -lc "bash '$MEGATRON_DIR/teacher/start_server.sh' && while true; do sleep 3600; done" &
        TEACHER_SRUN_PIDS+=("$!")

        if (( teacher_node_idx == 0 )); then
            wait_server_ready teacher "$TEACHER_SERVER_HOST" "$TEACHER_SERVER_PORT" 300 || {
                echo "teacher boot failed"; sleep infinity; exit 1
            }
        fi
    done

    wait_server_ready teacher "$TEACHER_SERVER_HOST" "$TEACHER_SERVER_PORT" 300 || {
        echo "teacher boot failed"; sleep infinity; exit 1
    }
    sleep 30
}

start_ray_head() {
    echo "Starting Ray head on $TRAINING_NODE ..."
    RAY_HEAD_RESOURCES_JSON='{"actor": 1}'
    if is_true "$STUDENT_SHARED_NODE"; then
        RAY_HEAD_RESOURCES_JSON='{"actor": 1, "rollout": 1}'
    fi
    srun --overlap --nodes=1 --ntasks=1 --nodelist="$TRAINING_NODE" --gres="gpu:${GPU_TYPE}:${RAY_HEAD_GPUS_PER_NODE}" \
        /usr/bin/env \
        CUDA_VISIBLE_DEVICES="$RAY_HEAD_CUDA_VISIBLE_DEVICES" \
        bash -lc "
            ray stop --force >/dev/null 2>&1 || true
            ray start --head \
                --port='$RAY_PORT' \
                --dashboard-host=0.0.0.0 \
                --dashboard-port='$RAY_DASHBOARD_PORT' \
                --num-cpus='${SLURM_CPUS_PER_TASK:-48}' \
                --num-gpus='$RAY_HEAD_GPUS_PER_NODE' \
                --temp-dir='$RAY_TMPDIR' \
                --resources='$RAY_HEAD_RESOURCES_JSON'
            while true; do sleep 3600; done
        " &
    RAY_HEAD_PID=$!
    sleep 10

    TRAINING_NODE_IP=$(getent hosts "$TRAINING_NODE" | awk '{print $1}')
    [ -z "$TRAINING_NODE_IP" ] && TRAINING_NODE_IP="$TRAINING_NODE"
}

start_rollout_ray_worker() {
    if [ "$ROLLOUT_NODE" = "$TRAINING_NODE" ]; then
        echo "Rollout resource is served by the Ray head on shared student node $TRAINING_NODE."
        return
    fi
    echo "Starting Ray worker on $ROLLOUT_NODE joining $TRAINING_NODE_IP:$RAY_PORT ..."
    srun --overlap --nodes=1 --ntasks=1 --nodelist="$ROLLOUT_NODE" --gres="gpu:${GPU_TYPE}:${ROLLOUT_GPUS_PER_NODE}" \
        /usr/bin/env \
        CUDA_VISIBLE_DEVICES="$ROLLOUT_CUDA_VISIBLE_DEVICES" \
        bash -lc "
            ray stop --force >/dev/null 2>&1 || true
            ray start --address='$TRAINING_NODE_IP:$RAY_PORT' \
                --num-cpus='${SLURM_CPUS_PER_TASK:-48}' \
                --num-gpus='$ROLLOUT_GPUS_PER_NODE' \
                --temp-dir='$RAY_TMPDIR' \
                --resources='{\"rollout\": 1}'
            while true; do sleep 3600; done
        " &
    ROLLOUT_RAY_PID=$!
    sleep 15
}

start_teacher_ray_worker() {
    if (( TEACHER_NNODES != 1 )); then
        echo "ERROR: TEACHER_RAY_MANAGED=true currently supports TEACHER_NNODES=1 only; use TEACHER_RAY_MANAGED=false for multi-node teacher." >&2
        exit 1
    fi
    echo "Starting Ray teacher worker on $TEACHER_NODE joining $TRAINING_NODE_IP:$RAY_PORT ..."
    srun --overlap --nodes=1 --ntasks=1 --nodelist="$TEACHER_NODE" --gres="gpu:${GPU_TYPE}:${TEACHER_GPUS_PER_NODE}" \
        /usr/bin/env \
        CUDA_VISIBLE_DEVICES="$TEACHER_CUDA_VISIBLE_DEVICES" \
        bash -lc "
            ray stop --force >/dev/null 2>&1 || true
            ray start --address='$TRAINING_NODE_IP:$RAY_PORT' \
                --num-cpus='${SLURM_CPUS_PER_TASK:-48}' \
                --num-gpus='$TEACHER_GPUS_PER_NODE' \
                --temp-dir='$RAY_TMPDIR' \
                --resources='{\"teacher\": 1}'
            while true; do sleep 3600; done
        " &
    TEACHER_RAY_PID=$!
    sleep 15
}

start_teacher_with_ray() {
    echo "Starting Ray-managed teacher supervisor actor $TEACHER_RAY_ACTOR_NAME ..."
    srun --overlap --nodes=1 --ntasks=1 --nodelist="$TRAINING_NODE" \
        /usr/bin/env \
        RAY_ADDRESS="$RAY_ADDRESS" \
        PATH="$(dirname "$PYTHON_BIN"):$PATH" \
        bash -lc "
            ray job submit --runtime-env='$MEGATRON_DIR/config/runtime_env.yaml' \
                --working-dir '$MEGATRON_DIR' \
                -- /usr/bin/env \
                PYTHON_BIN='$PYTHON_BIN' \
                PATH='$(dirname "$PYTHON_BIN"):\$PATH' \
                HF_HOME='$HF_HOME' \
                TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' \
                HF_DATASETS_CACHE='$HF_DATASETS_CACHE' \
                CKPT_PATH='${CKPT_PATH:-$TEACHER_MODEL_PATH}' \
                TEACHER_LOG_DIR='$TEACHER_LOG_DIR' \
                TEACHER_MODEL_PATH='$TEACHER_MODEL_PATH' \
                TEACHER_TP_SIZE='$TEACHER_TP_SIZE' \
                TEACHER_REPLICAS='$TEACHER_REPLICAS' \
                N_LOGPROBS='$N_LOGPROBS' \
                TEACHER_GPU_MEMORY_UTILIZATION='$TEACHER_GPU_MEMORY_UTILIZATION' \
                TEACHER_MAX_NUM_BATCHED_TOKENS='$TEACHER_MAX_NUM_BATCHED_TOKENS' \
                TEACHER_DISABLE_CUSTOM_ALL_REDUCE='$TEACHER_DISABLE_CUSTOM_ALL_REDUCE' \
                TEACHER_WORKER_READY_TIMEOUT='$TEACHER_WORKER_READY_TIMEOUT' \
                TEACHER_SEQ_LEN='$TEACHER_SEQ_LEN' \
                PROXY_FRONTEND_PORT='$PROXY_FRONTEND_PORT' \
                PROXY_BACKEND_PORT='$PROXY_BACKEND_PORT' \
                TEACHER_SERVER_HOST='$TEACHER_SERVER_HOST' \
                TEACHER_SERVER_PORT='$TEACHER_SERVER_PORT' \
                TEACHER_RAY_ACTOR_NAME='$TEACHER_RAY_ACTOR_NAME' \
                TEACHER_RAY_NUM_GPUS='$TEACHER_GPUS_PER_NODE' \
                '$PYTHON_BIN' -m teacher.ray_supervisor \
                    --actor-name '$TEACHER_RAY_ACTOR_NAME' \
                    --host '$TEACHER_SERVER_HOST' \
                    --port '$TEACHER_SERVER_PORT' \
                    --num-gpus '$TEACHER_GPUS_PER_NODE' \
                    --log-dir '$TEACHER_LOG_DIR'
        "

    wait_server_ready teacher "$TEACHER_SERVER_HOST" "$TEACHER_SERVER_PORT" 300 || {
        echo "teacher boot failed"; sleep infinity; exit 1
    }
}

if is_true "$TEACHER_RAY_MANAGED"; then
    start_ray_head
    start_rollout_ray_worker
    start_teacher_ray_worker
    start_teacher_with_ray
else
    start_teacher_with_srun
    start_ray_head
    start_rollout_ray_worker
fi

# --- 4. Write rerun_env.sh for debug ---
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
export ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE
export ACTOR_UPDATE_BATCH_SIZE=${ACTOR_UPDATE_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
export MAX_SAFE_TRAIN_BATCH=$MAX_SAFE_TRAIN_BATCH
export DP_WORLD_SIZE=$DP_WORLD_SIZE
export GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS
export AGENT_NUM_WORKERS=$AGENT_NUM_WORKERS
export TEACHER_INFLIGHT_SAMPLES=${TEACHER_INFLIGHT_SAMPLES:-0}
export MAX_POLICY_LAG=${MAX_POLICY_LAG:-0}
export USE_DYNAMIC_BSZ=$USE_DYNAMIC_BSZ
export MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
export TEACHER_MAX_PROMPT_LENGTH=$TEACHER_MAX_PROMPT_LENGTH
export MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH
export STUDENT_ACTOR_MAX_TOKENS_PER_GPU=$STUDENT_ACTOR_MAX_TOKENS_PER_GPU
export TEACHER_SEQ_LEN=$TEACHER_SEQ_LEN
export TEACHER_REPLICAS=$TEACHER_REPLICAS
export TEACHER_TP_SIZE=$TEACHER_TP_SIZE
export TEACHER_NNODES=$TEACHER_NNODES
export TEACHER_NODES=$TEACHER_NODES
export TEACHER_RAY_MANAGED=$TEACHER_RAY_MANAGED
export TEACHER_RAY_ACTOR_NAME=$TEACHER_RAY_ACTOR_NAME
export TEACHER_GPU_MEMORY_UTILIZATION=$TEACHER_GPU_MEMORY_UTILIZATION
export TEACHER_MAX_NUM_BATCHED_TOKENS=$TEACHER_MAX_NUM_BATCHED_TOKENS
export TEACHER_DISABLE_CUSTOM_ALL_REDUCE=$TEACHER_DISABLE_CUSTOM_ALL_REDUCE
export TEACHER_WORKER_READY_TIMEOUT=$TEACHER_WORKER_READY_TIMEOUT
export TEACHER_REQUEST_BATCH_SIZE=$TEACHER_REQUEST_BATCH_SIZE
export TEACHER_N_SERVER_WORKERS=$TEACHER_N_SERVER_WORKERS
export TEACHER_LOCAL_CHUNK_SIZE=$TEACHER_LOCAL_CHUNK_SIZE
export N_LOGPROBS=$N_LOGPROBS

export TEACHER_NODE=$TEACHER_NODE
export ROLLOUT_NODE=$ROLLOUT_NODE
export TRAINING_NODE=$TRAINING_NODE
export TEACHER_CUDA_VISIBLE_DEVICES=$TEACHER_CUDA_VISIBLE_DEVICES
export ROLLOUT_CUDA_VISIBLE_DEVICES=$ROLLOUT_CUDA_VISIBLE_DEVICES
export TRAINING_CUDA_VISIBLE_DEVICES=$TRAINING_CUDA_VISIBLE_DEVICES
export TEACHER_GPUS_PER_NODE=$TEACHER_GPUS_PER_NODE
export ROLLOUT_GPUS_PER_NODE=$ROLLOUT_GPUS_PER_NODE
export TRAINING_GPUS_PER_NODE=$TRAINING_GPUS_PER_NODE
export STUDENT_SHARED_NODE=$STUDENT_SHARED_NODE
export RAY_HEAD_GPUS_PER_NODE=$RAY_HEAD_GPUS_PER_NODE
export RAY_HEAD_CUDA_VISIBLE_DEVICES=$RAY_HEAD_CUDA_VISIBLE_DEVICES
export N_GPUS_PER_NODE=$TRAINING_GPUS_PER_NODE
export NGPUS_PER_NODE=$TRAINING_GPUS_PER_NODE

export HYBRID_ENGINE=$HYBRID_ENGINE
export ROLLOUT_ENABLE_SLEEP_MODE=$ROLLOUT_ENABLE_SLEEP_MODE
export ROLLOUT_FREE_CACHE_ENGINE=$ROLLOUT_FREE_CACHE_ENGINE
export ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION
export ROLLOUT_ENABLE_CHUNKED_PREFILL=$ROLLOUT_ENABLE_CHUNKED_PREFILL
export ROLLOUT_ENABLE_PREFIX_CACHING=$ROLLOUT_ENABLE_PREFIX_CACHING
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
export VERL_VLLM_USE_SHM_WEIGHT_SYNC=$VERL_VLLM_USE_SHM_WEIGHT_SYNC

# Add resource_pool override to force separate pools for actor vs rollout.
# (Verl will need this and hybrid_engine=False to pick non-hybrid path.)
bash $RUN_SCRIPT_PATH
EOF
echo "Rerun environment written to $RERUN_ENV_FILE"

# --- 5. Run the training driver on TRAINING_NODE (Ray head) ---
echo "Launching training driver on $TRAINING_NODE ..."
srun --overlap --nodes=1 --ntasks=1 --nodelist="$TRAINING_NODE" --gres="gpu:${GPU_TYPE}:${TRAINING_GPUS_PER_NODE}" \
    /usr/bin/env \
    CUDA_VISIBLE_DEVICES="$TRAINING_CUDA_VISIBLE_DEVICES" \
    N_GPUS_PER_NODE="$TRAINING_GPUS_PER_NODE" \
    NGPUS_PER_NODE="$TRAINING_GPUS_PER_NODE" \
    ROLLOUT_GPUS_PER_NODE="$ROLLOUT_GPUS_PER_NODE" \
    HYBRID_ENGINE="$HYBRID_ENGINE" \
    SCHEDULER="$SCHEDULER" \
    TEACHER_CLIENT_TIMEOUT_MS="$TEACHER_CLIENT_TIMEOUT_MS" \
    TEACHER_CLIENT_RCVTIMEO_MS="$TEACHER_CLIENT_RCVTIMEO_MS" \
    EFFECTIVE_BATCH_SIZE="$EFFECTIVE_BATCH_SIZE" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
    ACTOR_UPDATE_BATCH_SIZE="${ACTOR_UPDATE_BATCH_SIZE:-$TRAIN_BATCH_SIZE}" \
    MAX_SAFE_TRAIN_BATCH="$MAX_SAFE_TRAIN_BATCH" \
    DP_WORLD_SIZE="$DP_WORLD_SIZE" \
    GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
    AGENT_NUM_WORKERS="$AGENT_NUM_WORKERS" \
    TEACHER_INFLIGHT_SAMPLES="${TEACHER_INFLIGHT_SAMPLES:-0}" \
    MAX_POLICY_LAG="${MAX_POLICY_LAG:-0}" \
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
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    VERL_VLLM_USE_SHM_WEIGHT_SYNC="$VERL_VLLM_USE_SHM_WEIGHT_SYNC" \
    TEACHER_REQUEST_BATCH_SIZE="$TEACHER_REQUEST_BATCH_SIZE" \
    TEACHER_N_SERVER_WORKERS="$TEACHER_N_SERVER_WORKERS" \
    TEACHER_LOCAL_CHUNK_SIZE="$TEACHER_LOCAL_CHUNK_SIZE" \
    RAY_ADDRESS="$RAY_ADDRESS" \
    RAY_PORT="$RAY_PORT" \
    RAY_DASHBOARD_PORT="$RAY_DASHBOARD_PORT" \
    RAY_TMPDIR="$RAY_TMPDIR" \
    bash "$RUN_SCRIPT_PATH" "$@"

run_status=$?
echo "Training exited with status $run_status"
if is_true "${KEEP_ALIVE_ON_FAILURE:-true}"; then
    echo "Keeping allocation alive."
    echo "Debug: srun --jobid=${SLURM_JOB_ID} --pty bash"
    echo "       source $RERUN_ENV_FILE"
    sleep infinity
fi
exit "$run_status"
