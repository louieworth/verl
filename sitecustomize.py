"""Local Python startup patches for this workspace.

This is intentionally narrow: it only patches vLLM's model registry so text-only
Qwen3.5 exports using `Qwen3_5ForCausalLM` can be resolved in subprocesses.
"""

from __future__ import annotations


def _patch_vllm_qwen3_5_registry() -> None:
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception:
        return

    if "Qwen3_5ForCausalLM" in ModelRegistry.models:
        return

    try:
        ModelRegistry.register_model(
            "Qwen3_5ForCausalLM",
            "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
        )
    except Exception:
        # Startup hooks must never break unrelated Python commands.
        return


def _patch_vllm_qwen3_5_weight_loader() -> None:
    try:
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
    except Exception:
        return

    original_load_weights = Qwen3_5ForCausalLMBase.load_weights
    if getattr(original_load_weights, "_pens_patched", False):
        return

    def patched_load_weights(self, weights):
        def renamed():
            for name, tensor in weights:
                if name.startswith("model.language_model."):
                    yield ("model." + name[len("model.language_model."):], tensor)
                else:
                    yield (name, tensor)

        return original_load_weights(self, renamed())

    patched_load_weights._pens_patched = True
    Qwen3_5ForCausalLMBase.load_weights = patched_load_weights


_patch_vllm_qwen3_5_registry()
_patch_vllm_qwen3_5_weight_loader()
