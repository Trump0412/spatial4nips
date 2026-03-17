"""Memory containers for DA3-MMR interaction variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class FrameMemory:
    feat: torch.Tensor
    frame_id: int
    view_id: Optional[int] = None


@dataclass
class RegionMemory:
    feat: torch.Tensor
    frame_id: int
    region_id: int
    view_id: Optional[int] = None


class FrameMemoryBank:
    def __init__(self, memories: Sequence[FrameMemory], hidden_size: int, device: torch.device):
        self.memories = list(memories)
        self.hidden_size = int(hidden_size)
        self.device = device

    @classmethod
    def from_summaries(
        cls,
        summaries: Sequence[torch.Tensor],
        frame_ids: Optional[Sequence[int]] = None,
        view_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> "FrameMemoryBank":
        memories = []
        frame_ids = list(frame_ids) if frame_ids is not None else list(range(len(summaries)))
        if view_ids is None:
            view_ids = [None] * len(summaries)

        hidden_size = 0
        device = torch.device("cpu")
        for idx, summary in enumerate(summaries):
            if summary is None or summary.numel() == 0:
                continue
            hidden_size = summary.shape[-1]
            device = summary.device
            feat = summary if summary.dim() == 2 else summary.unsqueeze(0)
            memories.append(
                FrameMemory(
                    feat=feat,
                    frame_id=int(frame_ids[idx]),
                    view_id=None if view_ids[idx] is None else int(view_ids[idx]),
                )
            )
        return cls(memories=memories, hidden_size=hidden_size, device=device)

    def summary_matrix(self) -> torch.Tensor:
        if not self.memories:
            return torch.empty((0, self.hidden_size), device=self.device)
        return torch.cat([memory.feat for memory in self.memories], dim=0)

    def frame_ids(self) -> torch.Tensor:
        if not self.memories:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        return torch.tensor([memory.frame_id for memory in self.memories], device=self.device, dtype=torch.long)

    def view_ids(self) -> Optional[torch.Tensor]:
        if not self.memories:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        if all(memory.view_id is None for memory in self.memories):
            return None
        values = [(-1 if memory.view_id is None else memory.view_id) for memory in self.memories]
        return torch.tensor(values, device=self.device, dtype=torch.long)

    @property
    def available_frames(self) -> int:
        return len(self.memories)


class RegionMemoryBank:
    def __init__(self, memories: Sequence[RegionMemory], hidden_size: int, device: torch.device):
        self.memories = list(memories)
        self.hidden_size = int(hidden_size)
        self.device = device

    @classmethod
    def from_frame_atoms(
        cls,
        atoms_per_frame: Sequence[torch.Tensor],
        frame_ids: Optional[Sequence[int]] = None,
        view_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> "RegionMemoryBank":
        memories = []
        frame_ids = list(frame_ids) if frame_ids is not None else list(range(len(atoms_per_frame)))
        if view_ids is None:
            view_ids = [None] * len(atoms_per_frame)

        hidden_size = 0
        device = torch.device("cpu")
        for frame_idx, atoms in enumerate(atoms_per_frame):
            if atoms is None or atoms.numel() == 0:
                continue
            hidden_size = atoms.shape[-1]
            device = atoms.device
            for region_idx, atom in enumerate(atoms):
                memories.append(
                    RegionMemory(
                        feat=atom.unsqueeze(0),
                        frame_id=int(frame_ids[frame_idx]),
                        region_id=int(region_idx),
                        view_id=None if view_ids[frame_idx] is None else int(view_ids[frame_idx]),
                    )
                )
        return cls(memories=memories, hidden_size=hidden_size, device=device)

    def filter_by_frame_ids(self, selected_frame_ids: set[int]) -> torch.Tensor:
        if not self.memories or not selected_frame_ids:
            return torch.empty((0, self.hidden_size), device=self.device)
        selected = [memory.feat for memory in self.memories if memory.frame_id in selected_frame_ids]
        if not selected:
            return torch.empty((0, self.hidden_size), device=self.device)
        return torch.cat(selected, dim=0)

    @property
    def available_regions(self) -> int:
        return len(self.memories)
