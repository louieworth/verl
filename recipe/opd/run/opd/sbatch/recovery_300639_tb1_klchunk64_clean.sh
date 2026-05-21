#!/usr/bin/env bash
set -euo pipefail

JOB_WORK_DIR="/scratch/l/luli/openclaw/tmp/opd_offline/300639"
TAG="${RECOVERY_TAG:-manual}"
FLAG="$JOB_WORK_DIR/recovery_tb1_klchunk64_clean_${TAG}.done"
CACHE_BASE="$JOB_WORK_DIR/cache_clean_${TAG}"
INNER_LOG="$JOB_WORK_DIR/recovery_tb1_klchunk64_clean_${TAG}_rank0.log"

echo "[recovery-wrapper] host=$(hostname) proc=${SLURM_PROCID:-unset} tag=$TAG start=$(date -Is)"

if [ "${SLURM_PROCID:-0}" = "0" ]; then
    exec > >(tee -a "$INNER_LOG") 2>&1
    set -x
    export HYDRA_FULL_ERROR=1
    export PYTHONFAULTHANDLER=1
    rm -f "$FLAG"
    trap 'touch "$FLAG"' EXIT

    export PYTHONUNBUFFERED=1
    export RAY_DEDUP_LOGS=0
    export XDG_CACHE_HOME="$CACHE_BASE/xdg"
    export VLLM_CACHE_ROOT="$CACHE_BASE/vllm"
    export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/inductor"
    export TRITON_CACHE_DIR="$CACHE_BASE/triton"
    mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

    echo "[recovery-wrapper] rank0 launching rerun_tb1_klchunk64.sh"
    set +e
    bash "$JOB_WORK_DIR/rerun_tb1_klchunk64.sh"
    status=$?
    set -e
    echo "[recovery-wrapper] rank0 finished status=$status end=$(date -Is)"
    exit "$status"
fi

while [ ! -f "$FLAG" ]; do
    sleep 5
done

echo "[recovery-wrapper] non-rank0 observed done flag end=$(date -Is)"
