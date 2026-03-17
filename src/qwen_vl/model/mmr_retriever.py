"""Query-driven retrieval helpers for DA3-MMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .mmr_memory import FrameMemoryBank, RegionMemoryBank
from .mmr_utils import continuity_bias, resolve_topk
from .msgf_utils import l2_normalize, safe_topk


@dataclass
class RetrievedMemory:
    context: torch.Tensor
    available_frames: int
    available_regions: int
    frame_topk: int
    region_topk: int


def _similarity(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    if keys.numel() == 0:
        return keys.new_empty((0,))
    return torch.matmul(l2_normalize(keys), l2_normalize(query).squeeze(0))


class QueryDrivenRetriever:
    def __init__(
        self,
        *,
        frame_topk_max: int,
        region_topk_max: int,
        use_temporal_continuity: bool,
        use_view_continuity: bool,
        use_region_memory: bool,
    ):
        self.frame_topk_max = int(frame_topk_max)
        self.region_topk_max = int(region_topk_max)
        self.use_temporal_continuity = bool(use_temporal_continuity)
        self.use_view_continuity = bool(use_view_continuity)
        self.use_region_memory = bool(use_region_memory)

    def retrieve(
        self,
        query: torch.Tensor,
        frame_bank: FrameMemoryBank,
        region_bank: Optional[RegionMemoryBank] = None,
        *,
        query_frame_id: Optional[int] = None,
        query_view_id: Optional[int] = None,
    ) -> RetrievedMemory:
        frame_keys = frame_bank.summary_matrix()
        available_frames = frame_bank.available_frames
        frame_topk = resolve_topk(self.frame_topk_max, available_frames)
        if frame_keys.numel() == 0 or frame_topk == 0:
            empty = query.new_zeros((0, query.shape[-1]))
            return RetrievedMemory(empty, available_frames, 0, 0, 0)

        frame_scores = _similarity(query, frame_keys)
        frame_scores = frame_scores + continuity_bias(
            query_frame_id=query_frame_id,
            query_view_id=query_view_id,
            memory_frame_ids=frame_bank.frame_ids(),
            memory_view_ids=frame_bank.view_ids(),
            use_temporal_continuity=self.use_temporal_continuity,
            use_view_continuity=self.use_view_continuity,
        ).to(frame_scores.device, frame_scores.dtype)
        _, frame_indices = safe_topk(frame_scores, frame_topk)
        selected_frame_ids = {frame_bank.memories[idx].frame_id for idx in frame_indices.tolist()}
        frame_context = frame_keys[frame_indices] if frame_indices.numel() > 0 else frame_keys[:0]

        if not self.use_region_memory or region_bank is None:
            return RetrievedMemory(
                context=frame_context,
                available_frames=available_frames,
                available_regions=0,
                frame_topk=int(frame_indices.numel()),
                region_topk=0,
            )

        candidate_regions = region_bank.filter_by_frame_ids(selected_frame_ids)
        region_topk = resolve_topk(self.region_topk_max, int(candidate_regions.shape[0]))
        if candidate_regions.numel() == 0 or region_topk == 0:
            return RetrievedMemory(
                context=frame_context,
                available_frames=available_frames,
                available_regions=int(candidate_regions.shape[0]),
                frame_topk=int(frame_indices.numel()),
                region_topk=0,
            )

        region_scores = _similarity(query, candidate_regions)
        _, region_indices = safe_topk(region_scores, region_topk)
        region_context = candidate_regions[region_indices] if region_indices.numel() > 0 else candidate_regions[:0]
        context = torch.cat([frame_context, region_context], dim=0) if frame_context.numel() > 0 else region_context
        return RetrievedMemory(
            context=context,
            available_frames=available_frames,
            available_regions=int(candidate_regions.shape[0]),
            frame_topk=int(frame_indices.numel()),
            region_topk=int(region_indices.numel()),
        )
