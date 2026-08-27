from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from verl.utils.checkpoint import fsdp_checkpoint_manager as checkpoint_module
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


class _TrackedState(dict):
    def __init__(self, name: str, events: list[str]):
        super().__init__(name=name)
        self.name = name
        self.events = events

    def __del__(self):
        self.events.append(f"released:{self.name}")


class _Config:
    name_or_path = ""

    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir, "config.json").write_text("{}\n", encoding="utf-8")


def _manager() -> FSDPCheckpointManager:
    manager = object.__new__(FSDPCheckpointManager)
    manager.rank = 0
    manager.world_size = 1
    manager.previous_global_step = None
    manager.previous_saved_paths = []
    manager.processing_class = None
    manager.trust_remote_code = False
    manager.checkpoint_save_contents = ["model", "optimizer", "extra"]
    manager.checkpoint_load_contents = ["model", "optimizer", "extra"]
    manager.ensure_checkpoint_capacity = lambda _max_keep: None
    manager.register_checkpoint = lambda _path, _max_keep: None
    manager.get_rng_state = lambda: {}
    manager.load_rng_state = lambda _state: None
    return manager


def _patch_checkpoint_runtime(monkeypatch) -> None:
    monkeypatch.setattr(checkpoint_module, "ShardedStateDictConfig", lambda **_kwargs: None)
    monkeypatch.setattr(checkpoint_module, "ShardedOptimStateDictConfig", lambda **_kwargs: None)
    monkeypatch.setattr(checkpoint_module, "get_fsdp_state_ctx", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(checkpoint_module, "fsdp_version", lambda _model: 2)
    monkeypatch.setattr(checkpoint_module.torch.distributed, "barrier", lambda: None)


def test_save_releases_model_shard_before_materializing_optimizer(monkeypatch, tmp_path: Path):
    events: list[str] = []
    manager = _manager()

    class Model:
        config = _Config()

        def state_dict(self):
            events.append("materialized:model")
            return _TrackedState("model", events)

        def can_generate(self):
            return False

    class Optimizer:
        def state_dict(self):
            assert "released:model" in events
            events.append("materialized:optimizer")
            return _TrackedState("optimizer", events)

    manager.model = Model()
    manager.optimizer = Optimizer()
    manager.lr_scheduler = SimpleNamespace(state_dict=lambda: {})
    _patch_checkpoint_runtime(monkeypatch)
    monkeypatch.setattr(
        checkpoint_module.torch,
        "save",
        lambda state, _path: events.append(f"saved:{state['name']}" if "name" in state else "saved:extra"),
    )

    manager.save_checkpoint(str(tmp_path / "checkpoint"), global_step=1, max_ckpt_to_keep=1)

    assert events.index("released:model") < events.index("materialized:optimizer")
    assert events.index("released:optimizer") < events.index("saved:extra")


def test_load_releases_model_shard_before_deserializing_optimizer(monkeypatch, tmp_path: Path):
    events: list[str] = []
    manager = _manager()

    class Model:
        def load_state_dict(self, state):
            events.append(f"loaded:{state['name']}")

    class Optimizer:
        def load_state_dict(self, state):
            events.append(f"loaded:{state['name']}")

    manager.model = Model()
    manager.optimizer = Optimizer()
    manager.lr_scheduler = SimpleNamespace(load_state_dict=lambda _state: events.append("loaded:scheduler"))
    _patch_checkpoint_runtime(monkeypatch)
    monkeypatch.setattr(checkpoint_module, "copy_to_local", lambda path: path)

    def fake_load(path, **_kwargs):
        if "model_world_size" in path:
            events.append("deserialized:model")
            return _TrackedState("model", events)
        if "optim_world_size" in path:
            assert "released:model" in events
            events.append("deserialized:optimizer")
            return _TrackedState("optimizer", events)
        return {"rng": {}, "lr_scheduler": {}}

    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    manager.load_checkpoint(str(tmp_path / "checkpoint"))

    assert events.index("released:model") < events.index("deserialized:optimizer")
    assert "released:optimizer" in events
    assert "loaded:scheduler" in events
