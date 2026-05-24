#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL="/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/epoch1/ms1/batch00001/hf_merged"
# DEFAULT_DEST="/data/data/jiangli/models/opd"
DEFAULT_DEST="/data/data/jiangli/models/opd/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/epoch1/ms1/batch00001"
DEFAULT_EC2_HOST="ec2-18-218-102-253.us-east-2.compute.amazonaws.com"
DEFAULT_EC2_KEY="${HOME}/.ssh/ut-mac.pem"

MODEL="${MODEL:-$DEFAULT_MODEL}"
DEST="${DEST:-$DEFAULT_DEST}"
EC2_HOST="${EC2_HOST:-$DEFAULT_EC2_HOST}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_KEY="${EC2_KEY:-$DEFAULT_EC2_KEY}"
SSH_PORT="${SSH_PORT:-22}"
DRY_RUN=0

usage() {
    cat <<EOF
Usage:
  $0 [--src <local-dir>] [--dest /data/data/jiangli/models] [--host <ec2-host>] [--key ~/.ssh/ut-mac.pem]

Options:
  --host HOST       EC2 public DNS/IP or login host. Default: ${DEFAULT_EC2_HOST}
  --user USER       SSH user. Default: ${EC2_USER}
  --key PATH        SSH private key path. Default: ${DEFAULT_EC2_KEY}. Can also use EC2_KEY env.
  --port PORT       SSH port. Default: ${SSH_PORT}
  --src PATH        Local source directory. Default: ${DEFAULT_MODEL}
  --model PATH      Alias for --src.
  --dest PATH       Remote base directory. The source basename is created under it. Default: ${DEFAULT_DEST}
  --dry-run         Print what rsync would do without transferring.
  -h, --help        Show this help.

Examples:
  $0
  $0 --key ~/.ssh/ut-mac.pem
  $0 --src /scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523 --dest /data/data/jiangli/models
  $0 --host amazon_8a100 --src /path/to/model_or_run_dir --dest /data/data/jiangli/models

Notes:
  - The source directory basename is created under the remote base directory.
  - rsync uses --partial so interrupted transfers can be resumed by running this again.
  - If you use an SSH config Host alias such as amazon_8a100, pass --host amazon_8a100.
  - The --key path must exist on the machine where this script runs.
  - Put ut-mac.pem at ~/.ssh/ut-mac.pem on this cluster and run: chmod 600 ~/.ssh/ut-mac.pem
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            EC2_HOST="${2:?missing value for --host}"
            shift 2
            ;;
        --user)
            EC2_USER="${2:?missing value for --user}"
            shift 2
            ;;
        --key)
            EC2_KEY="${2:?missing value for --key}"
            shift 2
            ;;
        --port)
            SSH_PORT="${2:?missing value for --port}"
            shift 2
            ;;
        --src|--model)
            MODEL="${2:?missing value for $1}"
            shift 2
            ;;
        --dest)
            DEST="${2:?missing value for --dest}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$EC2_HOST" ]]; then
    echo "Missing --host or EC2_HOST." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "$MODEL" ]]; then
    echo "Source directory does not exist: $MODEL" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required but was not found in PATH." >&2
    exit 1
fi

ssh_args=(
    ssh
    -p "$SSH_PORT"
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=10
    -o StrictHostKeyChecking=accept-new
)

if [[ -n "$EC2_KEY" ]]; then
    if [[ ! -f "$EC2_KEY" ]]; then
        echo "SSH key does not exist: $EC2_KEY" >&2
        exit 1
    fi
    ssh_args+=(-i "$EC2_KEY")
fi

remote="${EC2_USER}@${EC2_HOST}"
source_name="$(basename "${MODEL%/}")"
DEST="${DEST%/}/${source_name}"
printf -v quoted_dest '%q' "$DEST"

printf 'Local source: %s\n' "$MODEL"
printf 'Remote host:  %s\n' "$remote"
printf 'Remote dest:  %s\n' "$DEST"
du -sh "$MODEL"

echo "Creating remote destination..."
"${ssh_args[@]}" "$remote" "mkdir -p $quoted_dest"

rsync_rsh="$(printf '%q ' "${ssh_args[@]}")"
rsync_args=(
    -avh
    --partial
    --info=progress2
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    rsync_args+=(--dry-run)
fi

echo "Starting rsync..."
RSYNC_RSH="$rsync_rsh" rsync "${rsync_args[@]}" "$MODEL/" "${remote}:${DEST}/"

echo "Remote size after transfer:"
"${ssh_args[@]}" "$remote" "du -sh $quoted_dest && ls -lh $quoted_dest"
