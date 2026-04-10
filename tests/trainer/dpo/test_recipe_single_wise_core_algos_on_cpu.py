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

import unittest

import torch

from recipe.dpo.batching import build_single_wise_dpo_update_proto
from recipe.dpo.core_algos import (
    compute_single_wise_dpo_loss,
    is_pointwise_dpo_loss,
    is_single_wise_dpo_loss,
    requires_reference_model,
)
from verl import DataProto


class TestRecipeSingleWiseCoreAlgos(unittest.TestCase):
    def test_single_wise_loss_matches_bce_with_logits(self):
        policy_logps = torch.tensor([2.0, 0.5, -0.2])
        reference_logps = torch.tensor([1.0, 0.0, 0.3])
        labels = torch.tensor([1.0, 0.0, 1.0])
        beta = 0.25

        loss, stats = compute_single_wise_dpo_loss(
            policy_logps=policy_logps,
            reference_logps=reference_logps,
            labels=labels,
            beta=beta,
        )

        rewards = beta * (policy_logps - reference_logps)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(rewards, labels)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertTrue(is_single_wise_dpo_loss("single_wise_dpo"))
        self.assertTrue(is_pointwise_dpo_loss("single_wise_dpo"))
        self.assertTrue(requires_reference_model("single_wise_dpo", reference_free=False))

        signed_margin = torch.where(labels > 0.5, rewards, -rewards)
        expected_acc = (signed_margin > 0).float().mean().item()
        self.assertAlmostEqual(stats["accuracy"].item(), expected_acc, places=6)

    def test_single_wise_batching_helper(self):
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[0, 1, 1, 1, 1]]),
                "position_ids": torch.tensor([[0, 0, 1, 2, 3]]),
                "responses": torch.tensor([[4, 5]]),
                "response_mask": torch.tensor([[1, 1]]),
                "labels": torch.tensor([[-100, -100, -100, 4, 5]]),
                "label": torch.tensor([1.0]),
            }
        )

        update_batch = build_single_wise_dpo_update_proto(
            batch=batch,
            beta=0.1,
            reference_logps=torch.tensor([0.3]),
        )

        self.assertEqual(update_batch.meta_info["dpo_loss_type"], "single_wise_dpo")
        self.assertEqual(update_batch.meta_info["global_token_num"], 4)
        self.assertIn("reference_logps", update_batch.batch)
        self.assertIn("responses", update_batch.batch)
        self.assertIn("response_mask", update_batch.batch)


if __name__ == "__main__":
    unittest.main()
