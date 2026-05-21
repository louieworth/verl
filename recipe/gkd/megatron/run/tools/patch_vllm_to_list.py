#!/usr/bin/env python3
"""Patch vLLM 0.11.0 GPUModelRunner._to_list to revert the buggy
pinned/event-sync path (the "short term mitigation" for vllm#22754,
PR #22760) back to the synchronous tolist() path.

The pinned/event path deterministically hits CUDA IMA in verl OPD
rollouts the moment the first batch of requests starts draining —
the failure is workload-proportional, not wallclock. Symptoms surface
either at gpu_model_runner._to_list / self.transfer_event.synchronize
or downstream as cublasGemmEx EXECUTION_FAILED, both with the same
root cause.

Cost: ~1.5% TTIT in disagg P/D serving (per the upstream issue).
Safe in this verl setup (TP=1, no KV disagg, hybrid_engine colocated).

Run once after `pip install vllm==0.11.0`:
    python recipe/gkd/megatron/run/tools/patch_vllm_to_list.py

The script is idempotent (skips when already patched) and writes a
.opd_backup_<date> next to the original on the first run.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import sys
from pathlib import Path

try:
    import vllm  # noqa: F401
except ImportError:
    print("ERROR: vllm is not importable in this Python.", file=sys.stderr)
    sys.exit(1)

import vllm.v1.worker.gpu_model_runner as _mod

TARGET = Path(_mod.__file__).resolve()

OLD = """    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[:sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()
"""

NEW = """    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # OPD patch (verl recipe/gkd/megatron/run/tools/patch_vllm_to_list.py):
        # revert PR #22760's pinned/event-sync path because it deterministically
        # CUDA-IMAs when the first batch of OPD rollouts starts draining
        # (workload-proportional; see vllm#22754). ~1.5% TTIT cost in disagg
        # P/D serving; safe in this setup (TP=1, no KV disagg, hybrid engine).
        return sampled_token_ids.tolist()
"""

MARKER = "# OPD patch (verl recipe/gkd/megatron/run/tools/patch_vllm_to_list.py):"


def main() -> int:
    src = TARGET.read_text()
    if MARKER in src:
        print(f"already patched: {TARGET}")
        return 0
    if OLD not in src:
        print(
            f"refusing to patch {TARGET}: expected vLLM 0.11.0 _to_list body not "
            "found verbatim. Inspect vllm version and re-evaluate before changing.",
            file=sys.stderr,
        )
        return 2

    stamp = _dt.date.today().isoformat()
    backup = TARGET.with_suffix(TARGET.suffix + f".opd_backup_{stamp}")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"backup: {backup}")

    TARGET.write_text(src.replace(OLD, NEW, 1))
    print(f"patched: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
