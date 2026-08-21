#!/usr/bin/env python3
"""Pack and restore the immutable canonical TACO training artifacts.

The Git bundle is a byte-for-byte transport format, not a second dataset
format.  Restoring it reproduces the canonical parquet files and manifest
exactly, so row order, padding, reward tests, and optimizer-step counts cannot
drift between preparation and training machines.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO


BUNDLE_SCHEMA_VERSION = "opd_taco_git_bundle/v1"
DEFAULT_CHUNK_BYTES = 40_000_000
ARTIFACT_NAMES = ("grpo", "sft", "distill")


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _artifact_source(source_dir: Path, metadata: dict) -> Path:
    source = source_dir / Path(str(metadata["path"])).name
    if not source.is_file():
        raise FileNotFoundError(f"missing canonical training artifact: {source}")
    return source


def pack_bundle(source_dir: Path, bundle_dir: Path, chunk_bytes: int, overwrite: bool) -> None:
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    source_dir = source_dir.resolve()
    bundle_dir = bundle_dir.resolve()
    canonical_manifest_path = source_dir / "manifest.json"
    canonical_manifest_bytes = canonical_manifest_path.read_bytes()
    canonical_manifest = json.loads(canonical_manifest_bytes)
    artifacts = canonical_manifest.get("artifacts") or {}
    if set(artifacts) != set(ARTIFACT_NAMES):
        raise ValueError("canonical manifest must contain exactly grpo/sft/distill artifacts")
    if bundle_dir.exists() and not overwrite:
        raise FileExistsError(f"bundle already exists (pass --overwrite): {bundle_dir}")

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_dir.name}.staging-", dir=bundle_dir.parent))
    chunks_dir = staging / "chunks"
    chunks_dir.mkdir()
    try:
        bundle_artifacts: dict[str, dict] = {}
        for name in ARTIFACT_NAMES:
            metadata = artifacts[name]
            source = _artifact_source(source_dir, metadata)
            expected_size = int(metadata["bytes"])
            expected_sha = str(metadata["sha256"])
            if source.stat().st_size != expected_size or sha256_file(source) != expected_sha:
                raise ValueError(f"canonical artifact does not match its manifest: {source}")

            chunk_records = []
            with source.open("rb") as input_stream:
                index = 0
                while True:
                    payload = input_stream.read(chunk_bytes)
                    if not payload:
                        break
                    filename = f"{source.name}.chunk-{index:04d}.bin"
                    chunk_path = chunks_dir / filename
                    chunk_path.write_bytes(payload)
                    chunk_records.append(
                        {
                            "path": f"chunks/{filename}",
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
                    index += 1
            if not chunk_records:
                raise ValueError(f"cannot bundle empty artifact: {source}")
            bundle_artifacts[name] = {
                "target": source.name,
                "bytes": expected_size,
                "sha256": expected_sha,
                "rows": int(metadata["rows"]),
                "chunks": chunk_records,
            }

        (staging / "canonical_manifest.json").write_bytes(canonical_manifest_bytes)
        bundle_manifest = {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "chunk_bytes": chunk_bytes,
            "canonical_manifest": {
                "path": "canonical_manifest.json",
                "bytes": len(canonical_manifest_bytes),
                "sha256": hashlib.sha256(canonical_manifest_bytes).hexdigest(),
            },
            "artifacts": bundle_artifacts,
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        backup = bundle_dir.with_name(f".{bundle_dir.name}.backup-{os.getpid()}")
        if bundle_dir.exists():
            os.replace(bundle_dir, backup)
        try:
            os.replace(staging, bundle_dir)
        except Exception:
            if backup.exists():
                os.replace(backup, bundle_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_bundle(bundle_dir: Path) -> tuple[dict, bytes]:
    bundle_dir = bundle_dir.resolve()
    bundle_manifest = _read_json(bundle_dir / "bundle_manifest.json")
    if bundle_manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported training bundle schema: {bundle_manifest.get('bundle_schema_version')!r}")
    chunk_bytes = int(bundle_manifest.get("chunk_bytes", 0))
    if chunk_bytes < 1:
        raise ValueError("invalid bundle chunk_bytes")

    canonical_record = bundle_manifest.get("canonical_manifest") or {}
    canonical_path = bundle_dir / str(canonical_record.get("path", ""))
    canonical_bytes = canonical_path.read_bytes()
    if len(canonical_bytes) != canonical_record.get("bytes") or hashlib.sha256(canonical_bytes).hexdigest() != canonical_record.get("sha256"):
        raise ValueError("canonical manifest copy does not match the bundle manifest")
    canonical_manifest = json.loads(canonical_bytes)
    canonical_artifacts = canonical_manifest.get("artifacts") or {}

    bundle_artifacts = bundle_manifest.get("artifacts") or {}
    if set(bundle_artifacts) != set(ARTIFACT_NAMES) or set(canonical_artifacts) != set(ARTIFACT_NAMES):
        raise ValueError("bundle must contain exactly grpo/sft/distill artifacts")
    for name in ARTIFACT_NAMES:
        artifact = bundle_artifacts[name]
        canonical = canonical_artifacts[name]
        if (
            artifact.get("target") != Path(str(canonical.get("path"))).name
            or artifact.get("bytes") != canonical.get("bytes")
            or artifact.get("sha256") != canonical.get("sha256")
            or artifact.get("rows") != canonical.get("rows")
        ):
            raise ValueError(f"bundle/canonical metadata mismatch for {name}")
        chunks = artifact.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"artifact {name} has no chunks")
        total = 0
        for index, record in enumerate(chunks):
            path = bundle_dir / str(record.get("path", ""))
            size = path.stat().st_size
            if size != record.get("bytes") or size > chunk_bytes:
                raise ValueError(f"invalid chunk size: {path}")
            if sha256_file(path) != record.get("sha256"):
                raise ValueError(f"chunk checksum mismatch: {path}")
            if index + 1 < len(chunks) and size != chunk_bytes:
                raise ValueError(f"non-final chunk is not exactly chunk_bytes: {path}")
            total += size
        if total != artifact.get("bytes"):
            raise ValueError(f"chunk byte total mismatch for {name}")
    return bundle_manifest, canonical_bytes


def _matches(path: Path, size: int, sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256_file(path) == sha256


def restore_bundle(bundle_dir: Path, output_dir: Path) -> None:
    bundle_dir = bundle_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".materialize.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        bundle_manifest, canonical_bytes = validate_bundle(bundle_dir)
        for name in ARTIFACT_NAMES:
            artifact = bundle_manifest["artifacts"][name]
            target = output_dir / artifact["target"]
            if _matches(target, int(artifact["bytes"]), str(artifact["sha256"])):
                print(f"[train bundle] verified existing {target}")
                continue

            fd, temporary_raw = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=output_dir)
            temporary = Path(temporary_raw)
            digest = hashlib.sha256()
            total = 0
            try:
                with os.fdopen(fd, "wb") as output_stream:
                    for record in artifact["chunks"]:
                        chunk_path = bundle_dir / record["path"]
                        with chunk_path.open("rb") as input_stream:
                            for payload in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                                output_stream.write(payload)
                                digest.update(payload)
                                total += len(payload)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                if total != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
                    raise ValueError(f"assembled artifact checksum mismatch for {name}")
                os.replace(temporary, target)
                print(f"[train bundle] restored {target} ({total} bytes)")
            finally:
                if temporary.exists():
                    temporary.unlink()

        manifest_target = output_dir / "manifest.json"
        if manifest_target.is_file() and manifest_target.read_bytes() == canonical_bytes:
            print(f"[train bundle] verified existing {manifest_target}")
        else:
            fd, temporary_raw = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=output_dir)
            temporary = Path(temporary_raw)
            try:
                with os.fdopen(fd, "wb") as output_stream:
                    output_stream.write(canonical_bytes)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                os.replace(temporary, manifest_target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            print(f"[train bundle] restored {manifest_target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack", help="split canonical artifacts into Git-safe chunks")
    pack.add_argument("--source-dir", type=Path, required=True)
    pack.add_argument("--bundle-dir", type=Path, required=True)
    pack.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    pack.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify", help="verify every bundled chunk")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    restore = subparsers.add_parser("restore", help="atomically restore canonical parquet files")
    restore.add_argument("--bundle-dir", type=Path, required=True)
    restore.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "pack":
        pack_bundle(args.source_dir, args.bundle_dir, args.chunk_bytes, args.overwrite)
        validate_bundle(args.bundle_dir)
        print(f"Git training bundle ready: {args.bundle_dir}")
    elif args.command == "verify":
        manifest, _ = validate_bundle(args.bundle_dir)
        count = sum(len(value["chunks"]) for value in manifest["artifacts"].values())
        print(f"Git training bundle verified: {count} chunks")
    else:
        restore_bundle(args.bundle_dir, args.output_dir)


if __name__ == "__main__":
    main()
