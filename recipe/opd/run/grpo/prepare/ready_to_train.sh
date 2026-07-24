#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-all}"
case "$TARGET" in
    all)
        bash recipe/opd/run/grpo/prepare/download_models.sh
        bash recipe/opd/run/grpo/prepare/prepare_data.sh all
        bash recipe/opd/run/grpo/prepare/verify_assets.sh all
        ;;
    models)
        bash recipe/opd/run/grpo/prepare/download_models.sh
        bash recipe/opd/run/grpo/prepare/verify_assets.sh models
        ;;
    train-data|eval-data)
        bash recipe/opd/run/grpo/prepare/prepare_data.sh "$TARGET"
        bash recipe/opd/run/grpo/prepare/verify_assets.sh "$TARGET"
        ;;
    verify)
        bash recipe/opd/run/grpo/prepare/verify_assets.sh all
        ;;
    *)
        echo "Usage: bash recipe/opd/run/grpo/prepare/ready_to_train.sh [all|models|train-data|eval-data|verify]" >&2
        exit 2
        ;;
esac
