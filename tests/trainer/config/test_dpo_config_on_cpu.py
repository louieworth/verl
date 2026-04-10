# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


class TestDPOConfig(unittest.TestCase):
    def test_recipe_dpo_config_compose(self):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.abspath("recipe/dpo/config")):
            cfg = compose(config_name="dpo_trainer")

        self.assertEqual(cfg.algorithm.dpo_beta, 0.1)
        self.assertFalse(cfg.critic.enable)
        self.assertEqual(cfg.data.prompt_key, "prompt")
        self.assertFalse(cfg.algorithm.reference_free)

    def test_recipe_ipo_config_compose(self):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.abspath("recipe/dpo/config")):
            cfg = compose(config_name="dpo_ipo")

        self.assertEqual(cfg.algorithm.dpo_loss_type, "ipo")
        self.assertFalse(cfg.algorithm.reference_free)

    def test_recipe_simpo_config_compose(self):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.abspath("recipe/dpo/config")):
            cfg = compose(config_name="dpo_simpo")

        self.assertEqual(cfg.algorithm.dpo_loss_type, "simpo")
        self.assertTrue(cfg.algorithm.reference_free)
        self.assertEqual(cfg.algorithm.simpo_gamma, 0.5)

    def test_recipe_single_wise_config_compose(self):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.abspath("recipe/dpo/config")):
            cfg = compose(config_name="dpo_single_wise_dpo")

        self.assertEqual(cfg.algorithm.dpo_loss_type, "single_wise_dpo")
        self.assertFalse(cfg.algorithm.reference_free)
        self.assertEqual(cfg.data.prompt_truncation, "right")
        self.assertEqual(cfg.data.response_key, "response")


if __name__ == "__main__":
    unittest.main()
