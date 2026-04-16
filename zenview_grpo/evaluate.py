"""
ZenView GRPO Evaluation Script

Usage:
    python evaluate.py --config configs/zenview_grpo.yaml --eval_file data/val.jsonl
    python evaluate.py --config configs/zenview_grpo.yaml --eval_file data/val.jsonl --output_file results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from zenview_grpo.data.dataset import ZenViewGRPODataset
from zenview_grpo.data.collator import ZenViewCollator
from zenview_grpo.rewards.parser import parse_response
from zenview_grpo.rewards.spatial_reward import compute_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="Override model path")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model_path or cfg["model_name_or_path"]
    eval_file = args.eval_file or cfg.get("val_file") or cfg["train_file"]
    image_root = cfg.get("image_root", "./")
    lang = cfg.get("lang", "en")

    logger.info(f"Loading model: {model_path}")
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device

    dataset = ZenViewGRPODataset(
        data_file=eval_file,
        image_root=image_root,
        max_samples=args.max_samples,
        lang=lang,
    )
    logger.info(f"Eval samples: {len(dataset)}")

    collator = ZenViewCollator(processor, max_prompt_length=cfg.get("max_prompt_length", 2048))
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    all_results = []
    agg = defaultdict(list)
    error_attr = {"frame_wrong": 0, "object_wrong": 0, "both_right_ans_wrong": 0, "total": 0}

    for batch in loader:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch["inputs"].items()}
        samples = batch["samples"]
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or 0,
            )
        responses = processor.tokenizer.batch_decode(
            out_ids[:, prompt_len:], skip_special_tokens=True
        )

        for sample, response in zip(samples, responses):
            reward, rd = compute_reward(sample, response)
            parsed = parse_response(response)

            for k, v in rd.items():
                agg[k].append(v)
            agg["think_complete"].append(
                float(parsed.has_valid_think and parsed.explanation_non_empty)
            )

            task_type = sample.get("meta", {}).get("task_type", "unknown")
            agg[f"task_{task_type}_ans"].append(rd["r_ans"])

            error_attr["total"] += 1
            if rd["r_frame"] < 1.0:
                error_attr["frame_wrong"] += 1
            if rd["r_object"] < 1.0:
                error_attr["object_wrong"] += 1
            if rd["r_frame"] >= 1.0 and rd["r_object"] >= 1.0 and rd["r_ans"] < 1.0:
                error_attr["both_right_ans_wrong"] += 1

            all_results.append({
                "id": sample.get("id", ""),
                "question": sample.get("question", "")[:100],
                "response": response[:300],
                "reward_dict": rd,
                "parsed_frame": parsed.reference_frame,
                "parsed_object": parsed.target_object,
                "parsed_answer": parsed.answer,
            })

    # Summary
    total = max(error_attr["total"], 1)
    summary = {k: sum(v) / len(v) for k, v in agg.items() if v}
    summary["error_frame_rate"] = error_attr["frame_wrong"] / total
    summary["error_object_rate"] = error_attr["object_wrong"] / total
    summary["error_spatial_only_rate"] = error_attr["both_right_ans_wrong"] / total
    summary["total_samples"] = error_attr["total"]

    logger.info("=" * 60)
    logger.info("Evaluation Summary:")
    for k, v in sorted(summary.items()):
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info("=" * 60)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump({"summary": summary, "samples": all_results}, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
