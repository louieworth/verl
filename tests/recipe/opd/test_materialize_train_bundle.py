from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recipe.opd.script_code.materialize_train_bundle import (
    pack_bundle,
    restore_bundle,
    validate_bundle,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    source = tmp_path / "source"
    source.mkdir()
    payloads = {
        "grpo": b"grpo-contents-" * 7,
        "sft": b"sft-contents-" * 4,
        "distill": b"distill-contents-" * 5,
    }
    artifacts = {}
    for name, payload in payloads.items():
        path = source / f"train_{name}.parquet"
        path.write_bytes(payload)
        artifacts[name] = {
            "path": f"data/train_dataset/taco/canonical/{path.name}",
            "bytes": len(payload),
            "sha256": _sha(payload),
            "rows": 512,
        }
    (source / "manifest.json").write_text(json.dumps({"artifacts": artifacts}) + "\n")
    return source, payloads


def test_pack_and_restore_are_byte_exact(tmp_path: Path):
    source, payloads = _source(tmp_path)
    bundle = tmp_path / "bundle"
    restored = tmp_path / "restored"

    pack_bundle(source, bundle, chunk_bytes=17, overwrite=False)
    manifest, canonical_bytes = validate_bundle(bundle)
    assert all(
        chunk["bytes"] <= 17
        for artifact in manifest["artifacts"].values()
        for chunk in artifact["chunks"]
    )

    restore_bundle(bundle, restored)
    for name, payload in payloads.items():
        assert (restored / f"train_{name}.parquet").read_bytes() == payload
    assert (restored / "manifest.json").read_bytes() == canonical_bytes

    # Repeated materialization verifies and reuses the exact files.
    restore_bundle(bundle, restored)


def test_corrupt_chunk_is_rejected_before_restore(tmp_path: Path):
    source, _ = _source(tmp_path)
    bundle = tmp_path / "bundle"
    pack_bundle(source, bundle, chunk_bytes=17, overwrite=False)
    chunk = next((bundle / "chunks").iterdir())
    chunk.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="chunk"):
        restore_bundle(bundle, tmp_path / "restored")
