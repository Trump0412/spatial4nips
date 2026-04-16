"""Group-relative advantage estimation for GRPO."""

from __future__ import annotations

import torch
from typing import List


def compute_group_advantages(
    rewards: List[float],
    eps: float = 1e-8,
    mode: str = "zscore",
) -> List[float]:
    """
    Compute group-relative advantages from a list of rewards.

    Args:
        rewards: list of scalar rewards for one group (G responses).
        eps: numerical stability epsilon.
        mode: "zscore" (default) or "rank".

    Returns:
        List of advantage values, same length as rewards.
    """
    if len(rewards) == 0:
        return []

    r = torch.tensor(rewards, dtype=torch.float32)

    if mode == "zscore":
        mu = r.mean()
        sigma = r.std(unbiased=False)
        adv = (r - mu) / (sigma + eps)
    elif mode == "rank":
        # Rank-based: map ranks to [-1, 1]
        order = r.argsort()
        ranks = torch.zeros_like(r)
        ranks[order] = torch.arange(len(r), dtype=torch.float32)
        adv = (ranks / max(len(r) - 1, 1)) * 2 - 1
    else:
        raise ValueError(f"Unknown advantage mode: {mode}")

    return adv.tolist()


def compute_batch_advantages(
    rewards: List[List[float]],
    eps: float = 1e-8,
    mode: str = "zscore",
) -> List[List[float]]:
    """
    Compute advantages for a batch of groups.

    Args:
        rewards: list of groups, each group is a list of G rewards.

    Returns:
        List of groups of advantages.
    """
    return [compute_group_advantages(group, eps=eps, mode=mode) for group in rewards]
