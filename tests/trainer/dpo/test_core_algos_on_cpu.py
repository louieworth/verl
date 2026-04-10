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

from verl import DataProto
from verl.trainer.dpo import (
    build_dpo_update_proto,
    build_log_prob_batch,
    compute_dpo_loss,
    compute_sequence_log_probs,
    requires_reference_model,
    use_average_sequence_log_probs,
)


class TestDPOCoreAlgos(unittest.TestCase):
    def test_compute_dpo_loss_with_reference(self):
        loss, stats = compute_dpo_loss(
            policy_chosen_logps=torch.tensor([2.0, 1.5]),
            policy_rejected_logps=torch.tensor([0.5, 0.25]),
            reference_chosen_logps=torch.tensor([1.0, 0.9]),
            reference_rejected_logps=torch.tensor([0.75, 0.5]),
            beta=0.1,
            reference_free=False,
        )

        self.assertGreater(loss.item(), 0)
        self.assertGreater(stats["accuracy"].item(), 0.5)

    def test_compute_dpo_loss_reference_free(self):
        loss, stats = compute_dpo_loss(
            policy_chosen_logps=torch.tensor([1.2]),
            policy_rejected_logps=torch.tensor([0.2]),
            reference_chosen_logps=None,
            reference_rejected_logps=None,
            beta=0.2,
            reference_free=True,
        )

        self.assertGreater(loss.item(), 0)
        self.assertAlmostEqual(stats["reference_logratio"].item(), 0.0)

    def test_compute_dpo_loss_ipo(self):
        loss, stats = compute_dpo_loss(
            policy_chosen_logps=torch.tensor([0.6, 0.2]),
            policy_rejected_logps=torch.tensor([0.1, -0.1]),
            reference_chosen_logps=torch.tensor([0.3, 0.0]),
            reference_rejected_logps=torch.tensor([0.0, -0.2]),
            beta=0.5,
            loss_type="ipo",
            reference_free=False,
        )

        self.assertAlmostEqual(loss.item(), 0.725, places=6)
        self.assertAlmostEqual(stats["logits"].item(), 0.15, places=6)
        self.assertTrue(use_average_sequence_log_probs("ipo"))
        self.assertFalse(use_average_sequence_log_probs("sigmoid"))

    def test_compute_dpo_loss_simpo(self):
        loss, stats = compute_dpo_loss(
            policy_chosen_logps=torch.tensor([0.6, 0.2]),
            policy_rejected_logps=torch.tensor([0.1, -0.1]),
            reference_chosen_logps=None,
            reference_rejected_logps=None,
            beta=2.0,
            loss_type="simpo",
            reference_free=False,
            simpo_gamma=0.5,
        )

        expected_logits = torch.tensor([0.25, 0.05])
        expected_losses = -torch.nn.functional.logsigmoid(2.0 * expected_logits)
        self.assertAlmostEqual(loss.item(), expected_losses.mean().item(), places=6)
        self.assertAlmostEqual(stats["reference_logratio"].item(), 0.0, places=6)
        self.assertTrue(use_average_sequence_log_probs("simpo"))
        self.assertFalse(requires_reference_model("simpo", reference_free=False))

    def test_pair_batching_helpers(self):
        batch = DataProto.from_dict(
            tensors={
                "chosen_input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "chosen_attention_mask": torch.tensor([[0, 1, 1, 1, 1]]),
                "chosen_position_ids": torch.tensor([[0, 0, 1, 2, 3]]),
                "chosen_responses": torch.tensor([[4, 5]]),
                "chosen_response_mask": torch.tensor([[1, 1]]),
                "chosen_labels": torch.tensor([[-100, -100, -100, 4, 5]]),
                "rejected_input_ids": torch.tensor([[1, 2, 3, 6, 0]]),
                "rejected_attention_mask": torch.tensor([[0, 1, 1, 1, 0]]),
                "rejected_position_ids": torch.tensor([[0, 0, 1, 2, 2]]),
                "rejected_responses": torch.tensor([[6, 0]]),
                "rejected_response_mask": torch.tensor([[1, 0]]),
                "rejected_labels": torch.tensor([[-100, -100, -100, 6, -100]]),
            }
        )

        log_prob_batch, response_mask = build_log_prob_batch(batch)
        self.assertEqual(log_prob_batch.batch["input_ids"].shape, (2, 5))
        self.assertEqual(response_mask.shape, (2, 2))

        seq_log_probs = compute_sequence_log_probs(torch.tensor([[0.1, 0.2], [0.3, 0.4]]), response_mask)
        self.assertTrue(torch.allclose(seq_log_probs, torch.tensor([0.3, 0.3])))
        avg_seq_log_probs = compute_sequence_log_probs(
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]), response_mask, average_log_prob=True
        )
        self.assertTrue(torch.allclose(avg_seq_log_probs, torch.tensor([0.15, 0.3])))

        dpo_update = build_dpo_update_proto(
            batch=batch,
            beta=0.1,
            loss_type="sigmoid",
            simpo_gamma=0.5,
            reference_chosen_logps=torch.tensor([0.3]),
            reference_rejected_logps=torch.tensor([0.1]),
        )
        self.assertEqual(dpo_update.meta_info["global_token_num"], 7)
        self.assertIn("reference_chosen_logps", dpo_update.batch)


if __name__ == "__main__":
    unittest.main()
