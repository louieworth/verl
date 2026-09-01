#!/usr/bin/env bash
# Validation shared by the path-driven launchers and canonical dispatcher.

opd_model_path_error() {
    echo "ERROR: $*" >&2
    return 2
}

opd_resolve_base_model_path() {
    local requested_path="$1"
    local default_hf_id="$2"
    local expected_hidden_size="$3"
    local expected_num_layers="$4"
    local label="$5"
    local resolved_path python_bin

    if [ "$requested_path" = "$default_hf_id" ]; then
        printf '%s\n' "$default_hf_id"
        return 0
    fi
    if [ ! -d "$requested_path" ]; then
        opd_model_path_error \
            "$label must be $default_hf_id or an existing local Qwen3 model directory (got: $requested_path)"
        return
    fi

    resolved_path="$(cd "$requested_path" && pwd -P)"
    if [ ! -f "$resolved_path/config.json" ]; then
        opd_model_path_error "$label local directory is missing config.json: $resolved_path"
        return
    fi
    if [ ! -f "$resolved_path/model.safetensors" ] && \
       [ ! -f "$resolved_path/model.safetensors.index.json" ] && \
       [ ! -f "$resolved_path/pytorch_model.bin" ] && \
       [ ! -f "$resolved_path/pytorch_model.bin.index.json" ] && \
       ! compgen -G "$resolved_path/model-*.safetensors" >/dev/null; then
        opd_model_path_error "$label local directory has no Hugging Face model weights: $resolved_path"
        return
    fi
    if [ ! -f "$resolved_path/tokenizer_config.json" ] || \
       { [ ! -f "$resolved_path/tokenizer.json" ] && [ ! -f "$resolved_path/vocab.json" ]; }; then
        opd_model_path_error "$label local directory is missing tokenizer assets: $resolved_path"
        return
    fi

    if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
        python_bin="$PYTHON_BIN"
    elif [ -x /data2/conda/envs/verl/bin/python ]; then
        python_bin=/data2/conda/envs/verl/bin/python
    elif command -v python3 >/dev/null 2>&1; then
        python_bin="$(command -v python3)"
    else
        opd_model_path_error "Python is required to validate local $label config.json"
        return
    fi

    if ! "$python_bin" - "$resolved_path/config.json" "$expected_hidden_size" "$expected_num_layers" "$label" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
expected_hidden_size = int(sys.argv[2])
expected_num_layers = int(sys.argv[3])
label = sys.argv[4]

try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"ERROR: cannot parse {label} config {config_path}: {exc}")

architectures = config.get("architectures") or []
actual = (
    config.get("model_type"),
    config.get("hidden_size"),
    config.get("num_hidden_layers"),
)
expected = ("qwen3", expected_hidden_size, expected_num_layers)
if actual != expected or "Qwen3ForCausalLM" not in architectures:
    raise SystemExit(
        f"ERROR: {label} does not match the expected Qwen3 size: "
        f"model_type/hidden_size/num_hidden_layers={actual}, expected={expected}"
    )

# Qwen3 variants share an architecture. Reject explicitly named Instruct
# checkpoints. Student callers must still use their documented Base repo;
# OPD's teacher caller instead uses the documented non-Base Qwen/Qwen3-14B repo.
provenance = f"{config_path.parent} {config.get('_name_or_path', '')}".lower()
if "instruct" in provenance:
    raise SystemExit(f"ERROR: {label} cannot be an Instruct checkpoint: {config_path.parent}")
PY
    then
        return 2
    fi

    printf '%s\n' "$resolved_path"
}

opd_resolve_teacher_model_path() {
    local requested_path="$1"
    local label="${2:-TEACHER_MODEL_PATH}"
    if [ -z "$requested_path" ]; then
        opd_model_path_error "$label cannot be empty"
        return 2
    fi

    if [ -d "$requested_path" ]; then
        (cd "$requested_path" && pwd -P)
    else
        printf '%s\n' "$requested_path"
    fi
}

opd_teacher_model_alias() {
    local model_path="$1"
    model_path="${model_path%/}"
    printf '%s\n' "${model_path##*/}"
}
