#!/usr/bin/env bash
# Thinking-mode variant of run_pens_personalized_eval.sh.
# Enables Qwen3 thinking mode in vLLM, gives the model a much larger response
# budget, and tags the result JSON key with thinkON + date.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_ENABLE_THINKING=true

# Cover 99.52% of prompts (<=8000 tokens); the ~99 outliers above 8000 are
# skipped via the 400-error handler in main_generation_server.py (they land in
# result.json as empty predictions, which the eval script already tolerates).
# Smaller max_model_len → more KV room → higher concurrency.
export GEN_PROMPT_LENGTH="${GEN_PROMPT_LENGTH:-8000}"
export GEN_RESPONSE_LENGTH="${GEN_RESPONSE_LENGTH:-4096}"
export GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-12288}"
export GEN_MAX_NUM_SEQS="${GEN_MAX_NUM_SEQS:-16}"

# OUTPUT_STEM and DATE_TAG are derived by the parent script once VLLM_ENABLE_THINKING
# is in scope, so we don't set them here — keeps the naming consistent and uses
# the real MODEL_KEY instead of a "pens" fallback.

exec bash "${SCRIPT_DIR}/run_pens_personalized_eval.sh" "$@"
