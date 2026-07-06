# Copyright 2024 - Custom REINFORCE with Group-wise Baseline for Concept Enhancement
"""
REINFORCE with Group-wise Baseline - designed for concept enhancement training
Key features:
1. Use group-wise (per-question) baseline instead of batch-wide
2. Preserve relative advantages within group
3. Compatible with concept enhancement mechanism
"""

import torch
from collections import defaultdict
import verl.utils.torch_functional as verl_F


def compute_reinforce_group_baseline_advantage(token_level_rewards: torch.Tensor,
                                              eos_mask: torch.Tensor,
                                              index: torch.Tensor,
                                              gamma: float = 1.0):
    """
    Compute advantage for REINFORCE with group-wise baseline.
    Each group (same question, n=4 attempts) uses its own baseline.

    Args:
        token_level_rewards: shape (bs, response_length)
        eos_mask: shape (bs, response_length)
        index: shape (bs,) - group indices (same uid has same index)
        gamma: discount factor

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    with torch.no_grad():
        batch_size = token_level_rewards.shape[0]
        response_length = token_level_rewards.shape[1]

        # Step 1: Compute discounted returns (REINFORCE style)
        returns = torch.zeros_like(token_level_rewards)
        running_return = torch.zeros(batch_size, device=token_level_rewards.device)

        for t in reversed(range(response_length)):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        # Step 2: Compute per-sequence reward (total reward for each response)
        sequence_rewards = (returns * eos_mask).sum(dim=1) / (eos_mask.sum(dim=1) + 1e-8)

        # Step 3: Compute group-wise baseline (mean reward within each group)
        id2rewards = defaultdict(list)
        id2baseline = {}

        for i in range(batch_size):
            group_id = index[i].item() if hasattr(index[i], 'item') else index[i]
            id2rewards[group_id].append(sequence_rewards[i])

        # Calculate baseline for each group
        for group_id, rewards_list in id2rewards.items():
            if len(rewards_list) > 1:
                # Use mean as baseline for groups with multiple samples
                id2baseline[group_id] = torch.stack(rewards_list).mean()
            else:
                # For single sample groups, use 0 as baseline (no relative comparison)
                id2baseline[group_id] = torch.tensor(0.0, device=token_level_rewards.device)

        # Step 4: Compute advantages using group-wise baseline
        advantages = returns.clone()
        for i in range(batch_size):
            group_id = index[i].item() if hasattr(index[i], 'item') else index[i]
            baseline = id2baseline[group_id]

            # Subtract group baseline from returns to get advantages
            advantages[i] = (returns[i] - baseline) * eos_mask[i]

        # Step 5: Optional - normalize advantages within group (not globally!)
        # This preserves the signal that some groups are better than others
        for group_id in id2rewards.keys():
            group_indices = [i for i in range(batch_size)
                           if (index[i].item() if hasattr(index[i], 'item') else index[i]) == group_id]

            if len(group_indices) > 1:
                # Only normalize variance within group, keep relative differences
                group_advs = torch.stack([advantages[i] for i in group_indices])
                group_mask = torch.stack([eos_mask[i] for i in group_indices])

                # Compute group statistics
                group_var = verl_F.masked_var(group_advs, group_mask)
                if group_var > 0:
                    std = torch.sqrt(group_var + 1e-8)
                    for i in group_indices:
                        advantages[i] = advantages[i] / std

    return advantages, returns


def compute_reinforce_concept_aware_advantage(token_level_rewards: torch.Tensor,
                                             eos_mask: torch.Tensor,
                                             index: torch.Tensor,
                                             concept_enhanced_flags: dict = None,
                                             gamma: float = 1.0):
    """
    Enhanced version that's aware of concept enhancement.
    Gives bonus weight to concept-enhanced responses that succeed.

    Args:
        concept_enhanced_flags: dict mapping index to whether it was concept-enhanced
    """
    # First compute standard group-baseline advantages
    advantages, returns = compute_reinforce_group_baseline_advantage(
        token_level_rewards, eos_mask, index, gamma
    )

    if concept_enhanced_flags:
        with torch.no_grad():
            batch_size = advantages.shape[0]

            # Give bonus weight to successful concept-enhanced responses
            for i in range(batch_size):
                idx = index[i].item() if hasattr(index[i], 'item') else index[i]
                if concept_enhanced_flags.get(idx, False):
                    # Check if this concept-enhanced response succeeded
                    total_reward = token_level_rewards[i].sum().item()
                    if total_reward > 0:
                        # Amplify positive advantages for successful concept enhancement
                        # This encourages the model to learn from concept hints
                        advantages[i] = advantages[i] * 1.5

    return advantages, returns