#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-data/download_cache/huggingface}"
mkdir -p model/base model/teacher model/trained "$HF_HOME"

download_model() {
    local repo_id="$1"
    local destination="$2"
    if [ -s "$destination/config.json" ] && [ "${FORCE_DOWNLOAD:-false}" != "true" ]; then
        echo "Model already exists, skipping: $destination"
        return 0
    fi
    python3 - "$repo_id" "$destination" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, destination = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    local_dir=destination,
    resume_download=True,
)
PY
    [ -s "$destination/config.json" ] || {
        echo "ERROR: model download is incomplete: $destination" >&2
        exit 1
    }
}

download_model "Qwen/Qwen3-1.7B-Base" "model/base/Qwen3-1.7B-Base"
download_model "Qwen/Qwen3-4B-Base" "model/base/Qwen3-4B-Base"
download_model "Qwen/Qwen3-8B-Base" "model/base/Qwen3-8B-Base"
download_model "Qwen/Qwen3-14B" "model/teacher/Qwen3-14B"

echo "Base students are ready under model/base/; OPD teacher is under model/teacher/"
