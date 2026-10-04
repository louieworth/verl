#!/usr/bin/env bash
# Shared runtime for the single-node, four-H100 Slurm jobs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

variant="${1:?Expected all, positive, negative, all_no_weight, positive_no_weight, negative_no_weight, or sft}"
shift
case "${variant}" in
  all) train_script=recipe/dpo/run/run_single_wise_click_hist_all.sh ;;
  positive) train_script=recipe/dpo/run/run_single_wise_click_hist_positive_only.sh ;;
  negative) train_script=recipe/dpo/run/run_single_wise_click_hist_negative_only.sh ;;
  all_no_weight) train_script=recipe/dpo/run/run_single_wise_click_hist_all_no_weight.sh ;;
  positive_no_weight) train_script=recipe/dpo/run/run_single_wise_click_hist_positive_only_no_weight.sh ;;
  negative_no_weight) train_script=recipe/dpo/run/run_single_wise_click_hist_negative_only_no_weight.sh ;;
  sft) train_script=recipe/dpo/run/run_pens_sft.sh ;;
  *) echo "Unknown PeNS Slurm experiment: ${variant}" >&2; exit 1 ;;
esac

dry_run="${PENS_SLURM_DRY_RUN:-false}"
if [[ -z "${SLURM_JOB_ID:-}" && "${dry_run}" != true ]]; then
  echo "Submit the .sbatch file with sbatch, or set PENS_SLURM_DRY_RUN=true to inspect it." >&2
  exit 1
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" != 1 ]]; then
  echo "These launchers require exactly one Slurm node." >&2
  exit 1
fi

# Preserve Slurm's CUDA mask. A four-GPU node uses 0,1,2,3 when no mask is set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#visible_gpus[@]} != 4 )); then
  echo "Expected four allocated GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 1
fi

# Keep Ray sockets and compiler caches on job-local storage.
job_tmp_base="${SLURM_TMPDIR:-/tmp}"
export TMPDIR="${job_tmp_base}/pens_${USER:-luli}_${SLURM_JOB_ID:-dryrun}"
ray_storage="${TMPDIR}/ray"
# Ray adds session/socket names; a long SLURM_TMPDIR can exceed AF_UNIX's
# 107-byte limit. Keep data on job-local storage through a short /tmp symlink.
ray_short_path="/tmp/pr_${UID}_${SLURM_JOB_ID:-${BASHPID}}"
if [[ -e "${ray_short_path}" && ! -L "${ray_short_path}" ]]; then
  echo "Ray short path already exists and is not a symlink: ${ray_short_path}" >&2
  exit 1
fi
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
export VLLM_CACHE_ROOT="${TMPDIR}/vllm"
mkdir -p "${TMPDIR}" "${ray_storage}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
ln -sfn "${ray_storage}" "${ray_short_path}"
export RAY_TMPDIR="${ray_short_path}"
export PENS_EVAL_RAY_TMPDIR="${ray_short_path}"

# common.sh loads the repository .env without printing its credentials.
source "${SCRIPT_DIR}/../common.sh"
# Compute nodes cannot reach W&B. Keep train/eval logs local, regardless of
# WANDB_MODE inherited from the submitter or loaded from .env.
export WANDB_MODE=offline
# Use verl's Python and command-line tools without relying on the submitter's env.
export PATH="$(dirname -- "${PENS_PYTHON_BIN}"):${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-32}"
unset RAY_ADDRESS
export PENS_EVAL_RAY_ADDRESS=local

export SINGLE_WISE_DPO_PYTHON_BIN="${PENS_PYTHON_BIN}"
export SINGLE_WISE_DPO_NNODES=1
export SINGLE_WISE_DPO_N_GPUS_PER_NODE=4
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_NNODES=1
export SINGLE_WISE_DPO_CHECKPOINT_EVAL_NGPUS_PER_NODE=4
export SFT_PYTHON_BIN="${PENS_PYTHON_BIN}"
export SFT_NNODES=1
export SFT_N_GPUS_PER_NODE=4
export SFT_CHECKPOINT_EVAL_NNODES=1
export SFT_CHECKPOINT_EVAL_NGPUS_PER_NODE=4

if [[ "${dry_run}" == true ]]; then
  export SINGLE_WISE_DPO_INTERLEAVED_DRY_RUN=true
  export SFT_DRY_RUN=true
fi

echo "Slurm job: ${SLURM_JOB_ID:-dryrun}; experiment: ${variant}; nodes: 1; GPUs: ${CUDA_VISIBLE_DEVICES}; CPUs: ${RAY_NUM_CPUS}"
echo "Python: ${PENS_PYTHON_BIN}; entrypoint: ${train_script}"
resource_args=()
if [[ "${variant}" != sft ]]; then
  resource_args+=("ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}")
  resource_args+=("+ray_kwargs.ray_init._temp_dir=${RAY_TMPDIR}")
fi
exec bash "${train_script}" "${resource_args[@]}" "$@"
