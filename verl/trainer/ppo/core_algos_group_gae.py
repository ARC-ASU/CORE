# Group-Aware GAE - PPO+Critic advantage computation designed for Concept Enhancement
"""
Core innovation:
1. Use Critic (GAE) to reduce variance estimation
2. But use group-level statistics for normalization (rather than global)
3. Preserve relative differences between groups, so concept enhancement remains effective

Principle:
- GRPO completely does not use Critic, variance may be large
- Standard PPO uses Critic but global whitening destroys signal
- Group-Aware GAE: use Critic to reduce variance + group-level normalize to preserve signal
"""

import torch
from collections import defaultdict
import verl.utils.torch_functional as verl_F


def compute_group_aware_gae_advantage(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    gamma: float,
    lam: float
):
    """
    Group-Aware GAE: combine variance reduction with group-level baseline signal preservation

    Args:
        token_level_rewards: shape (bs, response_length)
        values: shape (bs, response_length) - Critic's prediction
        eos_mask: shape (bs, response_length)
        index: shape (bs,) - group ID (uid for the same question)
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    with torch.no_grad():
        batch_size = token_level_rewards.shape[0]
        gen_len = token_level_rewards.shape[-1]

        # Step 1: standard GAE calculation (using Critic to reduce variance)
        lastgaelam = torch.zeros(batch_size, device=token_level_rewards.device)
        advantages_reversed = []

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else torch.zeros(batch_size, device=values.device)
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            lastgaelam = lastgaelam * eos_mask[:, t]  # mask after EOS
            advantages_reversed.append(lastgaelam)

        advantages_raw = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages_raw + values

        # Step 2: group-level normalize (key innovation!)
        # do not do global whitening, but group-level normalize, preserve group-level differences
        advantages = torch.zeros_like(advantages_raw)

        # build group mapping
        id2indices = defaultdict(list)
        for i in range(batch_size):
            group_id = index[i].item() if hasattr(index[i], 'item') else index[i]
            id2indices[group_id].append(i)

        # normalize each group independently
        for group_id, group_indices in id2indices.items():
            if len(group_indices) == 1:
                # single sample group: do not normalize, keep original value
                i = group_indices[0]
                advantages[i] = advantages_raw[i]
            else:
                # multiple sample group: group-level normalize
                group_advs = torch.stack([advantages_raw[i] for i in group_indices])
                group_mask = torch.stack([eos_mask[i] for i in group_indices])

                # calculate group-level statistics
                group_mean = verl_F.masked_mean(group_advs, group_mask)
                group_var = verl_F.masked_var(group_advs, group_mask)
                group_std = torch.sqrt(group_var + 1e-8)

                # group-level standardization: only normalize variance, partially preserve mean signal
                # This reduces intra-group variance while preserving "this group is overall good/bad" signal
                for i in group_indices:
                    # Option 1: completely preserve mean (most aggressive)
                    # advantages[i] = advantages_raw[i] / group_std

                    # Option 2: partially center (recommended - balance variance reduction and signal preservation)
                    advantages[i] = (advantages_raw[i] - group_mean * 0.3) / group_std

                    # Option 3: GRPO-style completely center (most conservative)
                    # advantages[i] = (advantages_raw[i] - group_mean) / group_std

                # Apply mask
                for i in group_indices:
                    advantages[i] = advantages[i] * eos_mask[i]

    return advantages, returns


def compute_group_aware_gae_advantage_v2(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    gamma: float,
    lam: float,
    normalize_style: str = 'partial'  # 'none', 'partial', 'full'
):
    """
    V2 version: configurable normalize style

    normalize_style:
    - 'none': do not center, only normalize variance (most preserve signal)
    - 'partial': partially center (recommended - balance variance reduction and signal preservation)
    - 'full': completely center (similar to GRPO)
    """
    with torch.no_grad():
        batch_size = token_level_rewards.shape[0]
        gen_len = token_level_rewards.shape[-1]

        # GAE calculation
        lastgaelam = torch.zeros(batch_size, device=token_level_rewards.device)
        advantages_reversed = []

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else torch.zeros(batch_size, device=values.device)
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            lastgaelam = lastgaelam * eos_mask[:, t]
            advantages_reversed.append(lastgaelam)

        advantages_raw = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages_raw + values
        advantages = torch.zeros_like(advantages_raw)

        # group-level normalize
        id2indices = defaultdict(list)
        for i in range(batch_size):
            group_id = index[i].item() if hasattr(index[i], 'item') else index[i]
            id2indices[group_id].append(i)

        for group_id, group_indices in id2indices.items():
            if len(group_indices) == 1:
                i = group_indices[0]
                advantages[i] = advantages_raw[i]
            else:
                group_advs = torch.stack([advantages_raw[i] for i in group_indices])
                group_mask = torch.stack([eos_mask[i] for i in group_indices])

                group_mean = verl_F.masked_mean(group_advs, group_mask)
                group_var = verl_F.masked_var(group_advs, group_mask)
                group_std = torch.sqrt(group_var + 1e-8)

                for i in group_indices:
                    if normalize_style == 'none':
                        # only normalize variance, completely preserve mean
                        advantages[i] = advantages_raw[i] / group_std
                    elif normalize_style == 'partial':
                        # partially center (recommended)
                        advantages[i] = (advantages_raw[i] - group_mean * 0.3) / group_std
                    elif normalize_style == 'full':
                        # completely center (GRPO style)
                        advantages[i] = (advantages_raw[i] - group_mean) / group_std
                    else:
                        raise ValueError(f"Unknown normalize_style: {normalize_style}")

                    advantages[i] = advantages[i] * eos_mask[i]

    return advantages, returns