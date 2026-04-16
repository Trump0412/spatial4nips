"""
ZenView GRPO Training Entry Point

Usage:
    python train_grpo.py --config configs/zenview_grpo.yaml
    torchrun --nproc_per_node=4 train_grpo.py --config configs/zenview_grpo.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from repo root or zenview_grpo dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from zenview_grpo.data.dataset import ZenViewGRPODataset
from zenview_grpo.data.collator import ZenViewCollator
from zenview_grpo.trainer.grpo_trainer import GRPOTrainer
from zenview_grpo.utils.distributed import init_distributed, is_main_process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    # Allow CLI overrides of any config key
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def load_config(args) -> dict:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    # CLI overrides
    for key in ("model_name_or_path", "train_file", "val_file", "output_dir",
                "num_train_epochs", "max_train_samples"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_processor(cfg: dict, local_rank: int):
    model_path = cfg["model_name_or_path"]
    logger.info(f"Loading model from {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    dtype = torch.bfloat16 if cfg.get("bf16") else (torch.float16 if cfg.get("fp16") else torch.float32)
    device_map = {"": local_rank} if torch.cuda.is_available() else "cpu"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.train()

    # Reference model (frozen)
    ref_model = None
    ref_path = cfg.get("reference_model_name_or_path")
    if ref_path and cfg.get("kl_coef", 0.0) > 0:
        logger.info(f"Loading reference model from {ref_path}")
        ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
            ref_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # Wrap with DDP if distributed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )

    return model, ref_model, processor


def main():
    args = parse_args()
    cfg = load_config(args)

    local_rank, world_size = init_distributed()
    set_seed(cfg.get("seed", 42))

    if is_main_process(local_rank):
        logger.info(f"Config: {cfg}")
        logger.info(f"World size: {world_size}")

    model, ref_model, processor = load_model_and_processor(cfg, local_rank)

    # Datasets
    lang = cfg.get("lang", "en")
    image_root = cfg.get("image_root", "./")
    max_samples = cfg.get("max_train_samples")

    train_dataset = ZenViewGRPODataset(
        data_file=cfg["train_file"],
        image_root=image_root,
        max_samples=max_samples,
        lang=lang,
    )
    logger.info(f"Train samples: {len(train_dataset)}")

    val_dataset = None
    if cfg.get("val_file"):
        val_dataset = ZenViewGRPODataset(
            data_file=cfg["val_file"],
            image_root=image_root,
            lang=lang,
        )
        logger.info(f"Val samples: {len(val_dataset)}")

    collator = ZenViewCollator(
        processor=processor,
        max_prompt_length=cfg.get("max_prompt_length", 2048),
    )

    if args.eval_only:
        from zenview_grpo.utils.metrics import evaluate_dataset
        from torch.utils.data import DataLoader
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        eval_ds = val_dataset or train_dataset
        loader = DataLoader(eval_ds, batch_size=1, collate_fn=collator)
        results = evaluate_dataset(model, processor, loader, device=device)
        logger.info(f"Eval results: {results}")
        return

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=cfg,
        local_rank=local_rank,
    )
    trainer.train()


if __name__ == "__main__":
    main()
