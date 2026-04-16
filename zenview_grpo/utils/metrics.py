"""Metrics tracking and evaluation utilities."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self):
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if v is not None:
                self._sums[k] += float(v)
                self._counts[k] += 1

    def averages(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}

    def reset(self):
        self._sums.clear()
        self._counts.clear()


@torch.no_grad()
def evaluate_dataset(
    model,
    processor,
    dataloader,
    max_new_tokens: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run greedy evaluation over a dataloader.
    Returns accuracy metrics and error attribution stats.
    """
    from ..rewards.spatial_reward import compute_reward
    from ..rewards.parser import parse_response

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    tracker = MetricsTracker()
    error_attribution = {"frame_wrong": 0, "object_wrong": 0, "both_right_ans_wrong": 0, "total": 0}
    task_results: Dict[str, List[float]] = defaultdict(list)

    for batch in dataloader:
        inputs = batch["inputs"]
        samples = batch["samples"]

        inputs_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        prompt_len = inputs_on_device["input_ids"].shape[1]

        output_ids = model.generate(
            **inputs_on_device,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or 0,
        )
        new_tokens = output_ids[:, prompt_len:]
        decoded = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for sample, response in zip(samples, decoded):
            reward, rd = compute_reward(sample, response)
            parsed = parse_response(response)

            tracker.update({
                "reward_total": rd["reward_total"],
                "r_fmt": rd["r_fmt"],
                "r_ans": rd["r_ans"],
                "r_think_fmt": rd["r_think_fmt"],
                "r_acc": rd["r_acc"],
                "r_frame": rd["r_frame"],
                "r_object": rd["r_object"],
                "r_word": rd["r_word"],
                "think_complete": float(parsed.has_valid_think and parsed.explanation_non_empty),
            })

            task_type = sample.get("meta", {}).get("task_type", "unknown")
            task_results[task_type].append(rd["r_ans"])

            # Error attribution
            error_attribution["total"] += 1
            if rd["r_frame"] < 1.0:
                error_attribution["frame_wrong"] += 1
            if rd["r_object"] < 1.0:
                error_attribution["object_wrong"] += 1
            if rd["r_frame"] >= 1.0 and rd["r_object"] >= 1.0 and rd["r_ans"] < 1.0:
                error_attribution["both_right_ans_wrong"] += 1

    results = tracker.averages()

    # Per-task accuracy
    for task, accs in task_results.items():
        results[f"task_{task}_acc"] = sum(accs) / len(accs) if accs else 0.0

    # Error attribution rates
    total = max(error_attribution["total"], 1)
    results["error_frame_rate"] = error_attribution["frame_wrong"] / total
    results["error_object_rate"] = error_attribution["object_wrong"] / total
    results["error_spatial_only_rate"] = error_attribution["both_right_ans_wrong"] / total

    model.train()
    return results
