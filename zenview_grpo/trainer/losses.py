"""Policy gradient losses for GRPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def compute_policy_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Clipped PPO/GRPO policy loss.

    Args:
        log_probs_new: (B,) log probs under current policy
        log_probs_old: (B,) log probs under old policy (detached)
        advantages:    (B,) group-relative advantages
        clip_range:    epsilon for clipping ratio
        mask:          (B,) optional boolean mask (1=valid)

    Returns:
        Scalar loss (to be minimized).
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss_unclipped = ratio * advantages
    loss_clipped = clipped * advantages
    loss = -torch.min(loss_unclipped, loss_clipped)

    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def compute_kl_loss(
    log_probs_new: torch.Tensor,
    log_probs_ref: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    KL divergence penalty: KL(pi_theta || pi_ref) approximated token-level.

    Args:
        log_probs_new: (B,) or (B, T) log probs under current policy
        log_probs_ref: (B,) or (B, T) log probs under reference policy
        mask:          optional mask

    Returns:
        Scalar KL estimate.
    """
    kl = log_probs_new - log_probs_ref  # approx KL per token
    if mask is not None:
        kl = kl * mask
        return kl.sum() / (mask.sum() + 1e-8)
    return kl.mean()


def compute_entropy(
    log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Entropy estimate from log probs."""
    entropy = -log_probs
    if mask is not None:
        entropy = entropy * mask
        return entropy.sum() / (mask.sum() + 1e-8)
    return entropy.mean()


def compute_sequence_log_prob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sequence sum of log probs over response tokens.

    Args:
        logits:        (B, T, V) model logits
        input_ids:     (B, T) token ids
        response_mask: (B, T) 1 for response tokens, 0 for prompt tokens

    Returns:
        (B,) sum of log probs over response tokens for each sequence.
    """
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]          # (B, T-1, V)
    shift_labels = input_ids[:, 1:]           # (B, T-1)
    shift_mask   = response_mask[:, 1:]       # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
    token_log_probs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)                              # (B, T-1)

    seq_log_probs = (token_log_probs * shift_mask).sum(dim=-1)  # (B,)
    return seq_log_probs
