# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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


import numpy as np
import torch

from verl.trainer.dpo.core_algos import compute_dpo_loss
from verl.trainer.dpo.core_algos import get_batch_logps


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_onlinedpo_pref(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes preferences between pairs of sequences based on summed rewards
    and returns a mask aligned with the interleaved batch.

    Assumes inputs are interleaved: [Resp1_Prompt0, Resp2_Prompt0, Resp1_Prompt1, Resp2_Prompt1, ...]

    Args:
        token_level_rewards: Tensor of shape [batch_size * 2, seq_len]
        response_mask: Tensor of shape [batch_size * 2, seq_len]

    Returns:
        torch.Tensor: A boolean mask of shape [batch_size * 2], where True indicates
                      the corresponding entry is the chosen response for its pair.
                      Example: [True, False, False, True, ...] means for prompt 0,
                               response 1 was chosen; for prompt 1, response 2 was chosen.
    """
    # print(f"---- [DEBUG] Inside compute_onlinedpo_pref ----")
    if token_level_rewards.shape[0] % 2 != 0 or response_mask.shape[0] % 2 != 0:
        raise ValueError(
            f"Input tensor batch dimension must be even for pair comparison, got shapes: "
            f"{token_level_rewards.shape}, {response_mask.shape}"
        )
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(f"Shape mismatch between rewards {token_level_rewards.shape} and mask {response_mask.shape}")

    # 1. Calculate Sequence Scores
    scores = (token_level_rewards * response_mask).sum(dim=-1)
    # print(f"  Calculated sequence scores shape: {scores.shape}") # [batch_size * 2]

    # 2. Reshape scores to group pairs: [batch_size, 2]
    try:
        score_pairs = scores.view(-1, 2)
    except RuntimeError as e:
        print(f"ERROR reshaping scores (shape {scores.shape}) into pairs: {e}")
        raise e
    print(f"  Reshaped score pairs shape: {score_pairs.shape}")  # [batch_size, 2]

    # 3. Compare scores to find which index (0 or 1) is the winner within each pair
    #    winner_indices[i] = 0 if score_pairs[i, 0] >= score_pairs[i, 1] else 1
    winner_indices = torch.argmax(score_pairs, dim=1)  # 0 if first is max, 1 if second is max
    # Handle ties explicitly if argmax behavior isn't guaranteed (usually picks first max)
    # Alternatively: winner_mask_original = score_pairs[:, 0] >= score_pairs[:, 1]
    # print(f"  Winner indices shape: {winner_indices.shape}") # [batch_size]
    # print(f"  Number where Response 2 (index 1) is preferred: {winner_indices.sum().item()}") # Counts number of 1s

    # 4. Create the final [batch_size * 2] mask
    num_pairs = score_pairs.shape[0]
    full_batch_size = num_pairs * 2
    # Create indices for the full batch [0, 1, 2, 3, ..., N*2-1]
    # full_indices = torch.arange(full_batch_size, device=scores.device)
    # Create indices corresponding to the winner within each pair's original index
    # E.g., if winner_indices is [0, 1, 0], pair_indices is [0, 1, 2]
    # winner_global_indices = (pair_indices * 2) + winner_indices -> [ (0*2)+0, (1*2)+1, (2*2)+0 ] -> [0, 3, 4]
    pair_indices = torch.arange(num_pairs, device=scores.device)
    winner_global_indices = (pair_indices * 2) + winner_indices

    # Create boolean mask - True at the winner's position
    output_preference_mask = torch.zeros(full_batch_size, dtype=torch.bool, device=scores.device)
    output_preference_mask[winner_global_indices] = True

    # print(f"  Output preference mask shape: {output_preference_mask.shape}") # Should be [batch_size * 2]
    # print(f"  Output mask True count (Chosen): {output_preference_mask.sum().item()}") # Should be batch_size
    # print(f"  Output mask False count (Rejected): {(~output_preference_mask).sum().item()}") # Should be batch_size
    # print(f"---- [DEBUG] Exiting compute_onlinedpo_pref ----")

    return output_preference_mask


def compute_online_dpo_loss(*args, **kwargs) -> torch.Tensor:
    loss, _ = compute_dpo_loss(*args, **kwargs)
    return loss
