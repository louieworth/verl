#!/bin/bash
# Common launcher for offline OPD multi-step runs on 4 H100 nodes.
# Resource shape: 4 nodes x 4 H100 = 16 GPUs total.

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

opd_disable_keepalive_trap() {
    trap - EXIT ERR
}

opd_keepalive_on_failure() {
    local status="${1:-$?}"
    opd_disable_keepalive_trap

    if [ "$status" -eq 0 ]; then
        return 0
    fi
    if ! opd_is_true "${KEEP_ALIVE_ON_FAILURE:-true}"; then
        exit "$status"
    fi
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        exit "$status"
    fi

    echo ""
    echo "=========================================="
    echo "OPD sbatch failed before normal completion"
    echo "  exit status: $status"
    echo "  keeping allocation alive for debugging"
    echo "=========================================="

    if [ -z "${OPD_JOB_WORK_DIR:-}" ]; then
        export OPD_JOB_WORK_DIR="/scratch/l/luli/openclaw/tmp/opd_offline/${SLURM_JOB_ID}"
    fi
    mkdir -p "$OPD_JOB_WORK_DIR" || true
    if [ -z "${RERUN_ENV_FILE:-}" ]; then
        RERUN_ENV_FILE="$OPD_JOB_WORK_DIR/rerun_env.sh"
    fi

    if declare -F opd_write_rerun_env >/dev/null 2>&1 && [ -n "${VERL_ROOT:-}" ] && [ -n "${RUN_SCRIPT:-}" ]; then
        opd_write_rerun_env || true
    else
        cat > "$RERUN_ENV_FILE" <<EOF_KEEPALIVE_RERUN
# Minimal rerun helper written after early sbatch failure.
cd ${VERL_ROOT:-/scratch/l/luli/src/verl}
export KEEP_ALIVE_ON_FAILURE=true
${RUN_SCRIPT:+bash $RUN_SCRIPT}
EOF_KEEPALIVE_RERUN
        echo "Minimal rerun environment written to $RERUN_ENV_FILE"
    fi

    echo "Debug shell: srun --jobid=${SLURM_JOB_ID} --overlap --pty bash"
    echo "Then: source $RERUN_ENV_FILE"
    sleep infinity
}

opd_install_keepalive_trap() {
    set -E
    trap 'opd_keepalive_on_failure $?' EXIT
    trap 'opd_keepalive_on_failure $?' ERR
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
# Source inside the held allocation to rerun the same OPD job.
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
bash $RUN_SCRIPT
EOF_RERUN
    echo "Rerun environment written to $RERUN_ENV_FILE"
}

opd_stop_ray_cluster() {
    if [ "${#ALLOCATED_NODES[@]}" -eq 0 ]; then
        return 0
    fi
    echo "Stopping Ray cluster..."
    for node in "${ALLOCATED_NODES[@]}"; do
        srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" --gres="gpu:${SLURM_GPU_TYPE}:${NGPUS_PER_NODE}" \
            bash -lc "export PATH='$(dirname "$PYTHON_BIN")':\$PATH; ray stop --force >/dev/null 2>&1 || true" &
    done
    wait || true
    if [ "${#RAY_STEP_PIDS[@]}" -gt 0 ]; then
        kill "${RAY_STEP_PIDS[@]}" >/dev/null 2>&1 || true
        wait "${RAY_STEP_PIDS[@]}" >/dev/null 2>&1 || true
    fi
}

opd_run_h100_4node() {
    set -E
    set -o pipefail
    opd_install_keepalive_trap

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
    echo "OPD offline multi-step on 4-node H100"
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
        echo "Ray cluster startup failed. Keeping allocation for debugging."
        opd_write_rerun_env
        sleep infinity
    }
    opd_write_rerun_env

    set +e
    bash "$RUN_SCRIPT" "$@"
    run_status=$?
    set -e

    echo "OPD run exited with status $run_status"
    if [ "$run_status" -ne 0 ] && opd_is_true "${KEEP_ALIVE_ON_FAILURE:-true}"; then
        echo "Keeping allocation alive for debugging."
        echo "Debug shell: srun --jobid=$SLURM_JOB_ID --overlap --pty bash"
        echo "Then: source $RERUN_ENV_FILE"
        sleep infinity
    fi

    opd_stop_ray_cluster || true
    opd_disable_keepalive_trap
    return "$run_status"
}
