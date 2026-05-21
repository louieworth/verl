#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PROXY_FRONTEND_PORT="${PROXY_FRONTEND_PORT:-15555}"
export PROXY_BACKEND_PORT="${PROXY_BACKEND_PORT:-15556}"
START_TEACHER_PROXY="${START_TEACHER_PROXY:-true}"
PROXY_HOST="${PROXY_HOST:-localhost}"
PROXY_ADDR="${PROXY_ADDR:-${PROXY_HOST}:${PROXY_BACKEND_PORT}}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEACHER_PATCH_VLLM_FULL_VOCAB_RAW="${OPD_PATCH_VLLM_FULL_VOCAB_RAW:-true}"
TEACHER_LOG_DIR="${TEACHER_LOG_DIR:-$SCRIPT_DIR}"
STOP_EXISTING_TEACHER_SERVER="${STOP_EXISTING_TEACHER_SERVER:-true}"

BACKEND="${BACKEND:-vllm}"
CKPT_PATH="${CKPT_PATH:-${TEACHER_MODEL_PATH:-/path/to/TEACHER_MODEL/}}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_REPLICAS="${TEACHER_REPLICAS:-1}"
N_LOGPROBS="${N_LOGPROBS:-full_vocab}"
TEACHER_SEQ_LEN="${TEACHER_SEQ_LEN:-3840}"
TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.7}"
TEACHER_ENABLE_PREFIX_CACHING="${TEACHER_ENABLE_PREFIX_CACHING:-}"
TEACHER_ENFORCE_EAGER="${TEACHER_ENFORCE_EAGER:-false}"
TEACHER_DISABLE_CUSTOM_ALL_REDUCE="${TEACHER_DISABLE_CUSTOM_ALL_REDUCE:-false}"
TEACHER_WORKER_READY_TIMEOUT="${TEACHER_WORKER_READY_TIMEOUT:-900}"
TEACHER_REPLICA_OFFSET="${TEACHER_REPLICA_OFFSET:-0}"

case "$TEACHER_TP_SIZE" in
    ""|*[!0-9]*) echo "ERROR: TEACHER_TP_SIZE must be a positive integer, got $TEACHER_TP_SIZE" >&2; exit 1 ;;
esac
case "$TEACHER_REPLICAS" in
    ""|*[!0-9]*) echo "ERROR: TEACHER_REPLICAS must be a positive integer, got $TEACHER_REPLICAS" >&2; exit 1 ;;
esac
case "$TEACHER_WORKER_READY_TIMEOUT" in
    ""|*[!0-9]*) echo "ERROR: TEACHER_WORKER_READY_TIMEOUT must be a non-negative integer, got $TEACHER_WORKER_READY_TIMEOUT" >&2; exit 1 ;;
esac
case "$TEACHER_REPLICA_OFFSET" in
    ""|*[!0-9]*) echo "ERROR: TEACHER_REPLICA_OFFSET must be a non-negative integer, got $TEACHER_REPLICA_OFFSET" >&2; exit 1 ;;
esac
if (( TEACHER_TP_SIZE < 1 )); then
    echo "ERROR: TEACHER_TP_SIZE must be positive, got $TEACHER_TP_SIZE" >&2
    exit 1
fi
if (( TEACHER_REPLICAS < 1 )); then
    echo "ERROR: TEACHER_REPLICAS must be positive, got $TEACHER_REPLICAS" >&2
    exit 1
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" ]; then
    IFS=',' read -r -a VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
else
    VISIBLE_DEVICES=()
    for ((i = 0; i < TEACHER_REPLICAS * TEACHER_TP_SIZE; i++)); do
        VISIBLE_DEVICES+=("$i")
    done
fi

GPU_COUNT="${#VISIBLE_DEVICES[@]}"
REQUIRED_GPUS=$((TEACHER_REPLICAS * TEACHER_TP_SIZE))
if (( REQUIRED_GPUS > GPU_COUNT )); then
    echo "ERROR: TEACHER_REPLICAS=$TEACHER_REPLICAS x TEACHER_TP_SIZE=$TEACHER_TP_SIZE needs $REQUIRED_GPUS GPUs, but CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' exposes $GPU_COUNT" >&2
    exit 1
fi

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: teacher checkpoint path does not exist: $CKPT_PATH" >&2
    echo "Set CKPT_PATH=/path/to/teacher or TEACHER_MODEL_PATH=/path/to/teacher." >&2
    exit 1
fi

mkdir -p "$TEACHER_LOG_DIR"

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        if "$PYTHON_BIN" - "$ip" "$port" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=2):
        pass
except OSError:
    sys.exit(1)
PY
        then
            break
        else
            sleep 1
        fi
    done
}

stop_started_teacher_processes() {
    for pid in "${worker_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [ -f "$TEACHER_LOG_DIR/proxy.pid" ]; then
        kill "$(cat "$TEACHER_LOG_DIR/proxy.pid")" 2>/dev/null || true
    fi
}

wait_teacher_workers_ready() {
    if (( TEACHER_WORKER_READY_TIMEOUT == 0 )); then
        echo "skip teacher worker readiness wait because TEACHER_WORKER_READY_TIMEOUT=0"
        return 0
    fi

    local deadline=$((SECONDS + TEACHER_WORKER_READY_TIMEOUT))
    while true; do
        local all_ready=1
        for ((replica_rank = 0; replica_rank < TEACHER_REPLICAS; replica_rank++)); do
            local pid="${worker_pids[$replica_rank]}"
            local log="${worker_logs[$replica_rank]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "ERROR: teacher worker replica=$replica_rank exited before ready; log=$log" >&2
                tail -n 80 "$log" 2>/dev/null || true
                stop_started_teacher_processes
                return 1
            fi
            if ! grep -q "worker started" "$log" 2>/dev/null; then
                all_ready=0
            fi
        done

        if (( all_ready == 1 )); then
            echo "all teacher workers are ready"
            return 0
        fi
        if (( SECONDS >= deadline )); then
            echo "ERROR: timed out waiting for teacher workers after ${TEACHER_WORKER_READY_TIMEOUT}s" >&2
            for ((replica_rank = 0; replica_rank < TEACHER_REPLICAS; replica_rank++)); do
                echo "----- worker replica=$replica_rank tail: ${worker_logs[$replica_rank]} -----" >&2
                tail -n 80 "${worker_logs[$replica_rank]}" 2>/dev/null || true
            done
            stop_started_teacher_processes
            return 1
        fi
        echo "wait teacher workers ready..."
        sleep 5
    done
}

case "$STOP_EXISTING_TEACHER_SERVER" in
    true|True|1|yes|Yes|y|Y)
        case "$START_TEACHER_PROXY" in
            true|True|1|yes|Yes|y|Y)
                pkill -9 -f "[p]ython .*proxy.py" 2>/dev/null || true
                ;;
        esac
        pkill -9 -f "[p]ython .*worker.py" 2>/dev/null || true
        pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
        pkill -9 -f "[V]LLM::Worker_TP" 2>/dev/null || true
        ;;
esac

case "$START_TEACHER_PROXY" in
    true|True|1|yes|Yes|y|Y)
        nohup "$PYTHON_BIN" proxy.py &> "$TEACHER_LOG_DIR/proxy.log" &
        echo "$!" > "$TEACHER_LOG_DIR/proxy.pid"

        wait_server_ready proxy localhost $PROXY_BACKEND_PORT

        echo "teacher proxy is ready"
        ;;
    *)
        wait_server_ready proxy "$PROXY_HOST" "$PROXY_BACKEND_PORT"
        echo "remote teacher worker will join proxy at $PROXY_ADDR"
        ;;
esac
echo "teacher workers: replicas=$TEACHER_REPLICAS tp_size=$TEACHER_TP_SIZE required_gpus=$REQUIRED_GPUS visible_devices=${VISIBLE_DEVICES[*]} replica_offset=$TEACHER_REPLICA_OFFSET"

worker_args=(
    --proxy-addr "$PROXY_ADDR"
    --backend "$BACKEND"
    --tp-size "$TEACHER_TP_SIZE"
    --seq-len "$TEACHER_SEQ_LEN"
    --n-logprobs "$N_LOGPROBS"
    --gpu-memory-utilization "$TEACHER_GPU_MEMORY_UTILIZATION"
)
if [ -n "$TEACHER_MAX_NUM_BATCHED_TOKENS" ]; then
    worker_args+=(--max-num-batched-tokens "$TEACHER_MAX_NUM_BATCHED_TOKENS")
fi
if [ -n "$TEACHER_ENABLE_PREFIX_CACHING" ]; then
    worker_args+=(--enable-prefix-caching "$TEACHER_ENABLE_PREFIX_CACHING")
fi
case "$TEACHER_ENFORCE_EAGER" in
    true|True|1|yes|Yes|y|Y)
        worker_args+=(--enforce-eager)
        ;;
esac
case "$TEACHER_DISABLE_CUSTOM_ALL_REDUCE" in
    true|True|1|yes|Yes|y|Y)
        worker_args+=(--disable-custom-all-reduce)
        ;;
esac
worker_args+=(--ckpt-path "$CKPT_PATH")

worker_pids=()
worker_logs=()
for ((replica_rank = 0; replica_rank < TEACHER_REPLICAS; replica_rank++)); do
    global_replica_rank=$((TEACHER_REPLICA_OFFSET + replica_rank))
    replica_devices=""
    for ((tp_rank = 0; tp_rank < TEACHER_TP_SIZE; tp_rank++)); do
        device_index=$((replica_rank * TEACHER_TP_SIZE + tp_rank))
        device="${VISIBLE_DEVICES[$device_index]}"
        device="${device//[[:space:]]/}"
        if [ -z "$replica_devices" ]; then
            replica_devices="$device"
        else
            replica_devices="$replica_devices,$device"
        fi
    done

    worker_log="$TEACHER_LOG_DIR/worker.${global_replica_rank}.log"
    if (( TEACHER_REPLICAS == 1 && TEACHER_REPLICA_OFFSET == 0 )); then
        worker_log="$TEACHER_LOG_DIR/worker.log"
    fi

    echo "start teacher worker replica=$global_replica_rank cuda_visible_devices=$replica_devices log=$worker_log"
    CUDA_VISIBLE_DEVICES="$replica_devices" \
    OPD_PATCH_VLLM_FULL_VOCAB_RAW="$TEACHER_PATCH_VLLM_FULL_VOCAB_RAW" \
        nohup "$PYTHON_BIN" worker.py "${worker_args[@]}" &> "$worker_log" &
    worker_pids+=("$!")
    worker_logs+=("$worker_log")
    echo "$!" > "$TEACHER_LOG_DIR/worker.${global_replica_rank}.pid"
done
printf '%s\n' "${worker_pids[@]}" > "$TEACHER_LOG_DIR/worker.pid"
if (( TEACHER_REPLICAS > 1 )); then
    {
        echo "teacher multi-replica worker logs:"
        for ((replica_rank = 0; replica_rank < TEACHER_REPLICAS; replica_rank++)); do
            global_replica_rank=$((TEACHER_REPLICA_OFFSET + replica_rank))
            echo "  replica $global_replica_rank: $TEACHER_LOG_DIR/worker.${global_replica_rank}.log"
        done
    } > "$TEACHER_LOG_DIR/worker.log"
fi

wait_teacher_workers_ready
echo "teacher server is ready"
