"""Distributed training utilities."""

from __future__ import annotations

import os
import torch
import torch.distributed as dist


def init_distributed():
    """Initialize distributed training if environment variables are set."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return local_rank, world_size


def is_main_process(local_rank: int = None) -> bool:
    if local_rank is not None:
        return local_rank == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor
