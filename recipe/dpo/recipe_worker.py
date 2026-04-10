from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import Dispatch, register

from recipe.dpo.recipe_actor import RecipeDPOActor


class RecipeDPOWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = RecipeDPOActor(
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )
