#!/bin/bash
# Common launcher for offline OPD multi-step runs on Slurm GPU nodes.
# Resource shape is controlled by NNODES, NGPUS_PER_NODE, and SLURM_GPU_TYPE.

opd_is_true() {
    case "${1:-}" in true|True|1|yes|Yes|y|Y) return 0 ;; *) return 1 ;; esac
}

opd_port_from_job() {
    local base="$1" offset="${2:-0}" job_id="${SLURM_JOB_ID:-0}"
    [[ ! "$job_id" =~ ^[0-9]+$ ]] && job_id=0
    printf '%s\n' $((base + (job_id % 10000) + offset))
}

opd_host_ip() {
    local host="$1" ip=""
    ip=$(getent hosts "${host}.tamia.ecpia.ca" 2>/dev/null | awk '$1 !~ /^127\./ {print $1; exit}')
    if [ -z "$ip" ]; then
        ip=$(getent hosts "$host" 2>/dev/null | awk '$1 !~ /^127\./ {print $1; exit}')
    fi
    if [ -z "$ip" ]; then
        ip=$(ip -4 -o addr show eno8303 scope global 2>/dev/null | awk '{print $4; exit}' | cut -d/ -f1)
    fi
    if [ -z "$ip" ]; then
        ip=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4; exit}' | cut -d/ -f1)
    fi
    printf '%s
' "$ip"
}

opd_ntfy_endpoint() {
    if [ -n "${NTFY_URL:-}" ]; then
        printf '%s\n' "$NTFY_URL"
        return 0
    fi
    local topic="${NTFY_TOPIC:-mila_lijiang_2026}"
    local server="${NTFY_SERVER:-https://ntfy.sh}"
    case "$topic" in
        http://*|https://*) printf '%s\n' "$topic" ;;
        *) printf '%s/%s\n' "${server%/}" "$topic" ;;
    esac
}

opd_ntfy_send() {
    local title="$1" message="$2" tags="${3:-rocket}" priority="${4:-default}" endpoint=""
    endpoint="$(opd_ntfy_endpoint)"
    if [ -z "$endpoint" ]; then
        echo "[ntfy] NTFY_URL/NTFY_TOPIC not set; notification skipped."
        return 0
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "[ntfy] curl not found; start notification skipped." >&2
        return 0
    fi

    local auth_args=()
    if [ -n "${NTFY_TOKEN:-}" ]; then
        auth_args=(-H "Authorization: Bearer $NTFY_TOKEN")
    elif [ -n "${NTFY_USER:-}" ] || [ -n "${NTFY_PASSWORD:-}" ]; then
        auth_args=(-u "${NTFY_USER:-}:${NTFY_PASSWORD:-}")
    fi

    curl -fsS -m "${NTFY_TIMEOUT:-5}" \
        "${auth_args[@]}" \
        -H "Title: $title" \
        -H "Tags: ${NTFY_TAGS:-$tags}" \
        -H "Priority: ${NTFY_PRIORITY:-$priority}" \
        -d "$message" \
        "$endpoint" >/dev/null || echo "[ntfy] notification failed: $endpoint" >&2
}

opd_notify_started() {
    local node_list="${SLURM_NODELIST:-unknown}"
    if [ "${#ALLOCATED_NODES[@]}" -gt 0 ]; then
        node_list="${ALLOCATED_NODES[*]}"
    fi

    local title="OPD started: ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-unknown}"
    local message
    message=$(cat <<EOF_NTFY_START
job: ${SLURM_JOB_ID:-unknown}
name: ${SLURM_JOB_NAME:-unknown}
partition: ${SLURM_JOB_PARTITION:-unknown}
nodes: ${node_list}
gpus: ${NNODES:-?} x ${NGPUS_PER_NODE:-?} ${SLURM_GPU_TYPE:-gpu}
time_limit: ${SLURM_TIMELIMIT:-unknown}
run_script: ${RUN_SCRIPT:-unknown}
work_dir: ${OPD_JOB_WORK_DIR:-unknown}
EOF_NTFY_START
)
    opd_ntfy_send "$title" "$message" "rocket"
}

opd_elapsed_since_start() {
    local start="${OPD_JOB_START_EPOCH:-}" now total
    if [[ ! "$start" =~ ^[0-9]+$ ]]; then
        printf '%s
' "unknown"
        return 0
    fi
    now="$(date +%s 2>/dev/null || true)"
    if [[ ! "$now" =~ ^[0-9]+$ ]] || [ "$now" -lt "$start" ]; then
        printf '%s
' "unknown"
        return 0
    fi
    total=$((now - start))
    printf '%02d:%02d:%02d
' $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}

opd_notify_finished() {
    local status="${1:-0}" state="${2:-finished}" reason="${3:-}" node_list="${SLURM_NODELIST:-unknown}"
    if [ "${#ALLOCATED_NODES[@]}" -gt 0 ]; then
        node_list="${ALLOCATED_NODES[*]}"
    fi

    local title tags priority
    case "$state" in
        failed)
            title="OPD failed: ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-unknown}"
            tags="x"
            priority="high"
            ;;
        *)
            title="OPD finished: ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-unknown}"
            tags="white_check_mark"
            priority="default"
            ;;
    esac

    local message
    message=$(cat <<EOF_NTFY_FINISH
job: ${SLURM_JOB_ID:-unknown}
name: ${SLURM_JOB_NAME:-unknown}
state: ${state}
exit_status: ${status}
elapsed: $(opd_elapsed_since_start)
partition: ${SLURM_JOB_PARTITION:-unknown}
nodes: ${node_list}
run_script: ${RUN_SCRIPT:-unknown}
work_dir: ${OPD_JOB_WORK_DIR:-unknown}
${reason:+reason: ${reason}}
EOF_NTFY_FINISH
)
    opd_ntfy_send "$title" "$message" "$tags" "$priority"
}

opd_disable_failure_trap() {
    trap - EXIT ERR
}

opd_exit_on_failure() {
    local status="${1:-$?}"
    opd_disable_failure_trap

    if [ "$status" -eq 0 ]; then
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "OPD sbatch failed before normal completion"
    echo "  exit status: $status"
    echo "  exiting so Slurm releases the allocation"
    echo "=========================================="

    opd_notify_finished "$status" "failed" "sbatch failed before normal completion; allocation released" || true
    exit "$status"
}

opd_install_failure_trap() {
    set -E
    trap 'opd_exit_on_failure $?' EXIT
    trap 'opd_exit_on_failure $?' ERR
}

opd_wait_ray_cluster() {
    local expected_gpus="$1" timeout="${2:-240}" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if "$PYTHON_BIN" - "$RAY_ADDRESS" "$expected_gpus" <<'PYRAYCHECK'
import sys
import ray
address = sys.argv[1]
expected = float(sys.argv[2])
try:
    ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)
    resources = ray.cluster_resources()
    gpus = float(resources.get("GPU", 0))
    ray.shutdown()
except Exception as exc:
    print(f"ray not ready: {exc}", flush=True)
    sys.exit(1)
print(f"ray cluster resources: {resources}", flush=True)
sys.exit(0 if gpus >= expected else 1)
PYRAYCHECK
        then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "ERROR: Ray cluster did not expose ${expected_gpus} GPUs within ${timeout}s" >&2
    return 1
}

opd_write_rerun_env() {
    cat > "$RERUN_ENV_FILE" <<EOF_RERUN
# Source inside an allocation to rerun the same OPD job.
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
export RUN_DATE=$RUN_DATE
export OPD_JOB_WORK_DIR=$OPD_JOB_WORK_DIR
export TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR
export TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
export VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT
export USE_LORA=${USE_LORA:-true}
export LORA_RANK=${LORA_RANK:-64}
export LORA_ALPHA=${LORA_ALPHA:-128}
export KL_FULL_VOCAB_CHUNK_SIZE=${KL_FULL_VOCAB_CHUNK_SIZE:-}
export GEN_RESULTS_ROOT=${GEN_RESULTS_ROOT:-}
export GEN_RESULTS_RUN_ID=${GEN_RESULTS_RUN_ID:-}
export GEN_RESULTS_RUN_ID_FILE=${GEN_RESULTS_RUN_ID_FILE:-}
export GEN_RESULTS_BASE_DIR=${GEN_RESULTS_BASE_DIR:-}
export TRAIN_DATA_PATH=$TRAIN_DATA_PATH
export EVAL_DATASETS_DIR=$EVAL_DATASETS_DIR
export NNODES=$NNODES
export NGPUS_PER_NODE=$NGPUS_PER_NODE
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-48}
export SLURM_CPU_BIND=none
export USE_SLURM_TORCHRUN=$USE_SLURM_TORCHRUN
export SLURM_GPU_TYPE=$SLURM_GPU_TYPE
export RAY_ADDRESS=$RAY_ADDRESS
export RAY_TMPDIR=$RAY_TMPDIR
export MULTI_STEP=$MULTI_STEP
export TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
export PIPELINE_RESUME_MODE=$PIPELINE_RESUME_MODE
export PIPELINE_AUTO_RESUME=$PIPELINE_AUTO_RESUME
export PIPELINE_CLEANUP_BATCH_DATA=$PIPELINE_CLEANUP_BATCH_DATA
export PIPELINE_TEMP_MODEL_DIR=${PIPELINE_TEMP_MODEL_DIR:-}
export RUN_EVAL_AFTER_TRAINING=$RUN_EVAL_AFTER_TRAINING
export WANDB_MODE=${WANDB_MODE:-offline}
export ROLLOUT_MAX_NUM_SEQS=$ROLLOUT_MAX_NUM_SEQS
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
export ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION
export NTFY_URL=${NTFY_URL:-}
export NTFY_TOPIC=${NTFY_TOPIC:-mila_lijiang_2026}
export NTFY_SERVER=${NTFY_SERVER:-https://ntfy.sh}
export NTFY_TOKEN=${NTFY_TOKEN:-}
export NTFY_USER=${NTFY_USER:-}
export NTFY_PASSWORD=${NTFY_PASSWORD:-}
export NTFY_PRIORITY=${NTFY_PRIORITY:-default}
export NTFY_TAGS=${NTFY_TAGS:-rocket}
export NTFY_TIMEOUT=${NTFY_TIMEOUT:-5}
set +e
bash $RUN_SCRIPT
rerun_status=\$?
set -e

if [ -n "\${SLURM_JOB_ID:-}" ] && command -v scancel >/dev/null 2>&1; then
    if [ "\$rerun_status" -eq 0 ]; then
        echo "OPD rerun completed successfully; releasing allocation \$SLURM_JOB_ID."
    else
        echo "OPD rerun exited with status \$rerun_status; releasing allocation \$SLURM_JOB_ID."
    fi
    scancel "\$SLURM_JOB_ID" || true
fi

return "\$rerun_status" 2>/dev/null || exit "\$rerun_status"
EOF_RERUN
    echo "Rerun environment written to $RERUN_ENV_FILE"
}

opd_wait_local_pids() {
    local timeout="$1"
    shift || true
    local pids=("$@")
    local deadline=$((SECONDS + timeout))
    local pid alive

    while [ "$SECONDS" -lt "$deadline" ]; do
        alive=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" >/dev/null 2>&1; then
                alive=1
                break
            fi
        done
        [ "$alive" -eq 0 ] && return 0
        sleep 1
    done
    return 1
}

opd_stop_ray_cluster() {
    if [ "${#ALLOCATED_NODES[@]}" -eq 0 ]; then
        return 0
    fi
    echo "Stopping Ray cluster..."
    local shutdown_timeout="${RAY_SHUTDOWN_TIMEOUT:-30}"
    local stop_timeout="${RAY_STOP_TIMEOUT:-20}"
    local node stop_pids=()

    if [ "${#RAY_STEP_PIDS[@]}" -gt 0 ]; then
        echo "Stopping Ray launcher srun steps: ${RAY_STEP_PIDS[*]}"
        kill "${RAY_STEP_PIDS[@]}" >/dev/null 2>&1 || true
        if ! opd_wait_local_pids "$shutdown_timeout" "${RAY_STEP_PIDS[@]}"; then
            echo "Ray launcher steps did not exit within ${shutdown_timeout}s; sending SIGKILL."
            kill -KILL "${RAY_STEP_PIDS[@]}" >/dev/null 2>&1 || true
        fi
        wait "${RAY_STEP_PIDS[@]}" >/dev/null 2>&1 || true
        RAY_STEP_PIDS=()
    fi

    for node in "${ALLOCATED_NODES[@]}"; do
        srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" --gres="gpu:${SLURM_GPU_TYPE}:${NGPUS_PER_NODE}" \
            bash -lc "export PATH='$(dirname "$PYTHON_BIN")':\$PATH; ray stop --force >/dev/null 2>&1 || true" &
        stop_pids+=("$!")
    done
    if [ "${#stop_pids[@]}" -gt 0 ]; then
        if ! opd_wait_local_pids "$stop_timeout" "${stop_pids[@]}"; then
            echo "Ray stop commands exceeded ${stop_timeout}s; killing cleanup srun steps."
            kill -KILL "${stop_pids[@]}" >/dev/null 2>&1 || true
        fi
        wait "${stop_pids[@]}" >/dev/null 2>&1 || true
    fi
    echo "Ray cluster stopped."
}

opd_run_slurm_gpu() {
    set -E
    set -o pipefail
    opd_install_failure_trap

    if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "ERROR: this launcher must run under sbatch/salloc." >&2
        exit 1
    fi
    if [ -z "${RUN_SCRIPT:-}" ]; then
        echo "ERROR: RUN_SCRIPT must point to an OPD wrapper." >&2
        exit 1
    fi

    SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    VERL_ROOT="$(cd "$SBATCH_DIR/../../../../.." && pwd)"
    cd "$VERL_ROOT"

    DEFAULT_STUDENT_MODEL_PATH="/scratch/l/luli/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    DEFAULT_TEACHER_MODEL_PATH="/scratch/l/luli/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    DEFAULT_PYTHON_BIN="/scratch/l/luli/conda/envs/verl/bin/python"

    export HF_HOME="${HF_HOME:-/scratch/l/luli/hf}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
    export PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
    export PATH="$(dirname "$PYTHON_BIN"):$PATH"
    export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

    export MODEL_PATH="${MODEL_PATH:-$DEFAULT_STUDENT_MODEL_PATH}"
    export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
    export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-1.7B}"
    export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$DEFAULT_TEACHER_MODEL_PATH}"
    export TEACHER_MODEL="${TEACHER_MODEL:-Qwen3-8B}"

    if [ -z "${TRAIN_DATA_PATH:-}" ]; then
        if [ -d "/scratch/l/luli/data/DeepScaleR-Cleaned" ]; then
            export TRAIN_DATA_PATH="/scratch/l/luli/data/DeepScaleR-Cleaned"
        else
            export TRAIN_DATA_PATH="/data/data/jiangli/data/DeepScaleR-Cleaned"
        fi
    fi
    export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:-/scratch/l/luli/jiangli/datasets/eval}"

    export NNODES="${NNODES:-4}"
    export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
    export SLURM_GPU_TYPE="${SLURM_GPU_TYPE:-h100}"
    export USE_SLURM_TORCHRUN="${USE_SLURM_TORCHRUN:-true}"
    export MULTI_STEP="${MULTI_STEP:-40}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    export PIPELINE_RESUME_MODE="${PIPELINE_RESUME_MODE:-resume_matching}"
    export PIPELINE_AUTO_RESUME="${PIPELINE_AUTO_RESUME:-false}"
    export PIPELINE_CLEANUP_BATCH_DATA="${PIPELINE_CLEANUP_BATCH_DATA:-true}"
    export PIPELINE_TEMP_MODEL_DIR="${PIPELINE_TEMP_MODEL_DIR:-}"
    export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-false}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M%S)}"

    export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"

    mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "$SLURM_NODELIST")
    if [ "${#ALLOCATED_NODES[@]}" -lt "$NNODES" ]; then
        echo "ERROR: expected at least $NNODES nodes, got ${#ALLOCATED_NODES[@]} from SLURM_NODELIST=$SLURM_NODELIST" >&2
        exit 1
    fi
    ALLOCATED_NODES=("${ALLOCATED_NODES[@]:0:$NNODES}")
    HEAD_NODE="${ALLOCATED_NODES[0]}"
    HEAD_IP="$(opd_host_ip "$HEAD_NODE")"
    [ -z "$HEAD_IP" ] && HEAD_IP="$HEAD_NODE"

    export MASTER_ADDR="${MASTER_ADDR:-$HEAD_IP}"
    export MASTER_PORT="${MASTER_PORT:-$(opd_port_from_job 29500 0)}"
    export RAY_PORT="${RAY_PORT:-$(opd_port_from_job 20000 0)}"
    export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$(opd_port_from_job 30000 0)}"
    export RAY_ADDRESS="${RAY_ADDRESS:-$HEAD_IP:$RAY_PORT}"
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/opd_ray_${SLURM_JOB_ID}}"
    export OPD_JOB_WORK_DIR="${OPD_JOB_WORK_DIR:-/scratch/l/luli/openclaw/tmp/opd_offline/${SLURM_JOB_ID}}"
    RERUN_ENV_FILE="$OPD_JOB_WORK_DIR/rerun_env.sh"
    mkdir -p "$OPD_JOB_WORK_DIR"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$OPD_JOB_WORK_DIR/torchinductor}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OPD_JOB_WORK_DIR/triton}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$OPD_JOB_WORK_DIR/torch_extensions}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$OPD_JOB_WORK_DIR/vllm_cache}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT"

    echo "=========================================="
    echo "OPD offline multi-step on Slurm GPU nodes"
    echo "  job:           $SLURM_JOB_ID"
    echo "  nodes:         ${ALLOCATED_NODES[*]}"
    echo "  GPUs:          $NNODES x $NGPUS_PER_NODE $SLURM_GPU_TYPE = $((NNODES * NGPUS_PER_NODE))"
    echo "  run script:    $RUN_SCRIPT"
    echo "  master:        $MASTER_ADDR:$MASTER_PORT"
    echo "  ray:           $RAY_ADDRESS dashboard=$RAY_DASHBOARD_PORT"
    echo "  multi_step:    $MULTI_STEP"
    echo "  train batch:   $TRAIN_BATCH_SIZE per GPU"
    echo "  resume mode:   $PIPELINE_RESUME_MODE"
    echo "  auto resume:   $PIPELINE_AUTO_RESUME"
    [ -n "$PIPELINE_TEMP_MODEL_DIR" ] && echo "  temp fsdp:     $PIPELINE_TEMP_MODEL_DIR"
    echo "  eval after:    $RUN_EVAL_AFTER_TRAINING"
    echo "  wandb mode:    $WANDB_MODE"
    echo "  run date:      $RUN_DATE"
    echo "  torch cache:   $TORCHINDUCTOR_CACHE_DIR"
    echo "  vllm cache:    $VLLM_CACHE_ROOT"
    [ -n "${GEN_RESULTS_RUN_ID:-}" ] && echo "  gen run id:    $GEN_RESULTS_RUN_ID"
    [ -n "${GEN_RESULTS_BASE_DIR:-}" ] && echo "  gen dir:       $GEN_RESULTS_BASE_DIR"
    echo "  work dir:      $OPD_JOB_WORK_DIR"
    echo "=========================================="

    export OPD_JOB_START_EPOCH="${OPD_JOB_START_EPOCH:-$(date +%s)}"
    opd_notify_started

    echo "Starting Ray head on $HEAD_NODE ($HEAD_IP)..."
    srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" --gres="gpu:${SLURM_GPU_TYPE}:${NGPUS_PER_NODE}" \
        bash -lc "
            export PATH='$(dirname "$PYTHON_BIN")':\$PATH
            ray stop --force >/dev/null 2>&1 || true
            mkdir -p '$RAY_TMPDIR'
            ray start --head \
                --node-ip-address='$HEAD_IP' \
                --port='$RAY_PORT' \
                --dashboard-host=0.0.0.0 \
                --dashboard-port='$RAY_DASHBOARD_PORT' \
                --num-cpus='${SLURM_CPUS_PER_TASK:-48}' \
                --num-gpus='$NGPUS_PER_NODE' \
                --temp-dir='$RAY_TMPDIR'
            while true; do sleep 3600; done
        " &
    RAY_STEP_PIDS=($!)
    sleep 10

    for node in "${ALLOCATED_NODES[@]:1}"; do
        echo "Starting Ray worker on $node..."
        srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" --gres="gpu:${SLURM_GPU_TYPE}:${NGPUS_PER_NODE}" \
            bash -lc "
                export PATH='$(dirname "$PYTHON_BIN")':\$PATH
                ray stop --force >/dev/null 2>&1 || true
                mkdir -p '$RAY_TMPDIR'
                ray start \
                    --address='$HEAD_IP:$RAY_PORT' \
                    --num-cpus='${SLURM_CPUS_PER_TASK:-48}' \
                    --num-gpus='$NGPUS_PER_NODE' \
                    --temp-dir='$RAY_TMPDIR'
                while true; do sleep 3600; done
            " &
        RAY_STEP_PIDS+=("$!")
    done

    opd_wait_ray_cluster "$((NNODES * NGPUS_PER_NODE))" 300 || {
        local startup_status=$?
        echo "Ray cluster startup failed. Cleaning up and exiting so Slurm releases the allocation."
        opd_stop_ray_cluster || true
        opd_notify_finished "$startup_status" "failed" "Ray cluster startup failed; allocation released" || true
        opd_disable_failure_trap
        return "$startup_status"
    }
    opd_write_rerun_env

    set +e
    bash "$RUN_SCRIPT" "$@"
    run_status=$?
    set -e

    echo "OPD run exited with status $run_status"
    if [ "$run_status" -ne 0 ]; then
        echo "RUN_SCRIPT exited nonzero; cleaning up and exiting so Slurm releases the allocation."
    fi

    opd_stop_ray_cluster || true
    if [ "$run_status" -eq 0 ]; then
        opd_notify_finished "$run_status" "finished" "Ray cleanup completed" || true
    else
        opd_notify_finished "$run_status" "failed" "RUN_SCRIPT exited nonzero; allocation released" || true
    fi
    opd_disable_failure_trap
    return "$run_status"
}

opd_run_h100_4node() {
    opd_run_slurm_gpu "$@"
}
