"""Group sampling utilities for GRPO."""

from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional, Tuple


@torch.no_grad()
def generate_group_responses(
    model,
    processor,
    inputs: Dict[str, torch.Tensor],
    group_size: int = 4,
    max_new_tokens: int = 384,
    temperature: float = 0.9,
    top_p: float = 0.95,
    do_sample: bool = True,
    pad_token_id: Optional[int] = None,
) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
    """
    For each prompt in the batch, generate group_size responses.

    Args:
        model:         the policy model (Qwen2-VL)
        processor:     the tokenizer/processor
        inputs:        dict from collator (input_ids, attention_mask, pixel_values, ...)
        group_size:    G responses per prompt
        ...

    Returns:
        responses:     list[batch] of list[G] decoded strings
        all_input_ids: (B*G, T_prompt+T_resp) full sequences
        response_mask: (B*G, T_prompt+T_resp) 1 for response tokens
    """
    device = next(model.parameters()).device
    batch_size = inputs["input_ids"].shape[0]
    prompt_len = inputs["input_ids"].shape[1]

    # Repeat each prompt G times: (B, ...) -> (B*G, ...)
    expanded = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            # Repeat along batch dim
            repeat_dims = [group_size] + [1] * (v.dim() - 1)
            # interleave: [p0,p0,...,p1,p1,...] via repeat_interleave
            expanded[k] = v.repeat_interleave(group_size, dim=0).to(device)
        else:
            expanded[k] = v

    pad_id = pad_token_id or processor.tokenizer.pad_token_id or 0

    output_ids = model.generate(
        **expanded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=pad_id,
    )  # (B*G, T_prompt + T_resp)

    # Build response mask: 1 for generated tokens, 0 for prompt
    response_mask = torch.zeros_like(output_ids)
    response_mask[:, prompt_len:] = 1
    # Also mask padding
    if pad_id is not None:
        pad_positions = (output_ids == pad_id)
        response_mask[pad_positions] = 0

    # Decode only the new tokens
    new_tokens = output_ids[:, prompt_len:]
    decoded = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Reshape to (B, G)
    responses = [
        decoded[i * group_size: (i + 1) * group_size]
        for i in range(batch_size)
    ]

    return responses, output_ids, response_mask
