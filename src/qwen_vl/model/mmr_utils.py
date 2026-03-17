"""Utility helpers for DA3-MMR interaction variants."""

from __future__ import annotations

from typing import Optional

import torch

from .msgf_utils import mean_pool_tokens


def build_monotonic_ids(num_frames: int, device: torch.device) -> torch.Tensor:
    if num_frames <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.arange(num_frames, device=device, dtype=torch.long)


def resolve_topk(max_topk: int, available: int) -> int:
    if max_topk <= 0 or available <= 0:
        return 0
    return min(int(max_topk), int(available))


def summarize_query_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return mean_pool_tokens(tokens)


def continuity_bias(
    query_frame_id: Optional[int],
    query_view_id: Optional[int],
    memory_frame_ids: Optional[torch.Tensor],
    memory_view_ids: Optional[torch.Tensor],
    *,
    use_temporal_continuity: bool,
    use_view_continuity: bool,
) -> torch.Tensor:
    if memory_frame_ids is None or memory_frame_ids.numel() == 0:
        if memory_view_ids is not None:
            return memory_view_ids.new_zeros((memory_view_ids.shape[0],), dtype=torch.float32)
        return torch.empty((0,), dtype=torch.float32)

    bias = memory_frame_ids.new_zeros((memory_frame_ids.shape[0],), dtype=torch.float32)

    if use_temporal_continuity and query_frame_id is not None:
        frame_gap = (memory_frame_ids - int(query_frame_id)).abs().to(torch.float32)
        bias = bias - 0.1 * frame_gap

    if use_view_continuity and query_view_id is not None and memory_view_ids is not None and memory_view_ids.numel() > 0:
        same_view = (memory_view_ids == int(query_view_id)).to(torch.float32)
        bias = bias + 0.05 * same_view

    return bias
