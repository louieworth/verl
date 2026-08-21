#!/usr/bin/env python3
"""Export an FSDP checkpoint as a strictly loadable full Hugging Face model.

``verl.model_merger`` preserves LoRA adapters separately.  Evaluation servers,
however, load the checkpoint root as a normal causal LM and would otherwise
ignore that adapter.  This wrapper records the LoRA metadata, invokes the
sharded merger, merges the adapter into the base weights, and rejects any
missing or unexpected model keys before marking the export complete.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EXPORT_SCHEMA = "opd_full_model_export/v1"
AT_FDCWD = -100
RENAME_EXCHANGE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True, help="FSDP checkpoint directory")
    parser.add_argument("--target-dir", required=True, help="Full HF model output directory")
    parser.add_argument("--base-model", required=True, help="Base config/tokenizer path or HF ID")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def ensure_lora_metadata(checkpoint_dir: Path, rank: int, alpha: int) -> None:
    if rank <= 0:
        return
    if alpha <= 0:
        raise ValueError("LoRA export requires a positive --lora-alpha")
    path = checkpoint_dir / "lora_train_meta.json"
    expected = {"r": rank, "lora_alpha": alpha, "task_type": "CAUSAL_LM"}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"LoRA metadata mismatch at {path}: {key}={existing.get(key)!r}, expected {value!r}"
                )
        return
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def strict_load(target_dir: Path, trust_remote_code: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        str(target_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        output_loading_info=True,
    )
    problems = {
        key: loading_info.get(key) or []
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    del model
    gc.collect()
    if problems:
        raise RuntimeError(f"Exported model failed strict load validation: {problems}")


def merge_lora_adapter(target_dir: Path, trust_remote_code: bool) -> bool:
    adapter_dir = target_dir / "lora_adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        return False

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        str(target_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    # Saving back into the same directory keeps tokenizer/config assets and the
    # adapter provenance while replacing the root weights with full weights.
    merged.save_pretrained(str(target_dir), safe_serialization=True)
    del merged, peft_model, base
    gc.collect()
    return True


def _path_exists(path: Path) -> bool:
    """Like lexists(), but keeps all path handling in pathlib call sites."""

    return os.path.lexists(path)


def _remove_path(path: Path) -> None:
    """Remove one known staging path without following directory symlinks."""

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _rename_exchange(left: Path, right: Path) -> None:
    """Atomically exchange two directory entries on Linux.

    ``os.replace`` cannot replace a non-empty directory.  renameat2 with
    RENAME_EXCHANGE lets a validated staging directory become the public
    target in one namespace operation while moving the previous target back
    to the private staging name.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic directory replacement requires renameat2(RENAME_EXCHANGE)")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(left),
        AT_FDCWD,
        os.fsencode(right),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{left} <-> {right}")


def atomic_publish_directory(staging_dir: Path, target_dir: Path) -> None:
    """Publish a validated staging directory without exposing partial files."""

    if _path_exists(target_dir):
        _rename_exchange(staging_dir, target_dir)
        return
    try:
        os.replace(staging_dir, target_dir)
    except OSError as exc:
        # A concurrent creator can make the target appear after lexists().
        # Exchange only for the two errors that represent that race; all other
        # failures (cross-device, permissions, etc.) must remain fatal.
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not _path_exists(target_dir):
            raise
        _rename_exchange(staging_dir, target_dir)


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.local_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"FSDP checkpoint does not exist: {checkpoint_dir}")
    if target_dir == checkpoint_dir:
        raise ValueError("--target-dir must differ from --local-dir")
    ensure_lora_metadata(checkpoint_dir, args.lora_rank, args.lora_alpha)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{target_dir.name}.staging.", dir=target_dir.parent))
    try:
        command = [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(checkpoint_dir),
            "--hf_model_config_path",
            args.base_model,
            "--target_dir",
            str(staging_dir),
        ]
        if args.trust_remote_code:
            command.append("--trust-remote-code")
        subprocess.run(command, check=True)

        lora_merged = merge_lora_adapter(staging_dir, args.trust_remote_code)
        if args.lora_rank > 0 and not lora_merged:
            raise RuntimeError(
                f"LoRA rank {args.lora_rank} was requested but no adapter was exported under {staging_dir}"
            )
        strict_load(staging_dir, args.trust_remote_code)

        marker = {
            "schema_version": EXPORT_SCHEMA,
            "checkpoint_dir": str(checkpoint_dir),
            "base_model": args.base_model,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_merged_into_root": lora_merged,
        }
        marker_path = staging_dir / "opd_export.json"
        temporary = marker_path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, marker_path)
        atomic_publish_directory(staging_dir, target_dir)
    finally:
        # After an exchange this path contains the previous committed target;
        # after any pre-publish failure it contains only private partial output.
        if _path_exists(staging_dir):
            try:
                _remove_path(staging_dir)
            except OSError as exc:
                print(f"WARNING: could not remove export staging path {staging_dir}: {exc}", file=sys.stderr)
    print(f"Strict full-model export complete: {target_dir}")


if __name__ == "__main__":
    main()
