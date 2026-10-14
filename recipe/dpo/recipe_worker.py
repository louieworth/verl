from recipe.dpo.recipe_actor import RecipeDPOActor

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker


class RecipeDPOWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps=None):
        from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

        rng_state = BaseCheckpointManager.get_rng_state()
        try:
            return await super().update_weights(global_steps=global_steps)
        finally:
            BaseCheckpointManager.load_rng_state(rng_state)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_rollout and self._is_lora and self.config.rollout.name == "vllm":
            # Keep base weights during sleep; only the live adapter needs syncing.
            self.rollout.sleep_level = 1
        if self._is_actor:
            self.actor = RecipeDPOActor(
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )
