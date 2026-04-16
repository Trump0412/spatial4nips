"""ZenView GRPO Trainer."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from ..rewards.spatial_reward import batch_compute_rewards
from ..utils.metrics import MetricsTracker
from .advantage import compute_batch_advantages
from .losses import compute_kl_loss, compute_policy_loss, compute_sequence_log_prob
from .sampling import generate_group_responses

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for ZenView.

    Supports:
    - Single-machine multi-GPU via DistributedDataParallel
    - Reference model for KL penalty
    - Structured reward function
    - Logging, checkpointing, evaluation
    """

    def __init__(
        self,
        model,
        ref_model,
        processor,
        train_dataset,
        val_dataset,
        collator,
        config: Dict[str, Any],
        local_rank: int = 0,
    ):
        self.model = model
        self.ref_model = ref_model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collator = collator
        self.config = config
        self.local_rank = local_rank
        self.is_main = local_rank == 0

        self.group_size = config.get("group_size", 4)
        self.clip_range = config.get("ppo_clip_range", 0.2)
        self.kl_coef = config.get("kl_coef", 0.02)
        self.entropy_coef = config.get("entropy_coef", 0.0)
        self.advantage_norm = config.get("advantage_norm", "zscore")
        self.max_new_tokens = config.get("max_new_tokens", 384)
        self.temperature = config.get("temperature", 0.9)
        self.top_p = config.get("top_p", 0.95)
        self.logging_steps = config.get("logging_steps", 10)
        self.save_steps = config.get("save_steps", 200)
        self.eval_steps = config.get("eval_steps", 200)
        self.output_dir = config.get("output_dir", "outputs/zenview_grpo")
        self.sample_log_steps = config.get("sample_log_steps", 50)

        self.metrics = MetricsTracker()
        self.global_step = 0

        os.makedirs(self.output_dir, exist_ok=True)

        # Optimizer
        lr = config.get("learning_rate", 1e-6)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=config.get("weight_decay", 0.0),
        )

    def _make_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.config.get("per_device_train_batch_size", 1),
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.config.get("dataloader_num_workers", 2),
            pin_memory=True,
        )

    def train(self):
        num_epochs = self.config.get("num_train_epochs", 1)
        grad_accum = self.config.get("gradient_accumulation_steps", 8)
        train_loader = self._make_dataloader(self.train_dataset)

        total_steps = len(train_loader) * num_epochs // grad_accum
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.05))
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.model.train()
        if self.ref_model is not None:
            self.ref_model.eval()

        accum_loss = 0.0
        self.optimizer.zero_grad()

        for epoch in range(num_epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            for step, batch in enumerate(train_loader):
                loss, step_metrics = self._grpo_step(batch)
                loss = loss / grad_accum
                loss.backward()
                accum_loss += loss.item()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0),
                    )
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    self.metrics.update(step_metrics)
                    self.metrics.update({"loss": accum_loss})
                    accum_loss = 0.0

                    if self.is_main and self.global_step % self.logging_steps == 0:
                        self._log_metrics()

                    if self.is_main and self.global_step % self.save_steps == 0:
                        self._save_checkpoint()

                    if self.global_step % self.eval_steps == 0 and self.val_dataset:
                        self._evaluate()

        if self.is_main:
            self._save_checkpoint(final=True)

    def _grpo_step(self, batch: Dict[str, Any]):
        inputs = batch["inputs"]
        samples = batch["samples"]
        device = next(self.model.parameters()).device

        # Move inputs to device
        inputs_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # ── 1. Generate G responses per prompt ────────────────────────────────
        self.model.eval()
        with torch.no_grad():
            responses, output_ids, response_mask = generate_group_responses(
                self.model,
                self.processor,
                inputs_on_device,
                group_size=self.group_size,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        self.model.train()

        batch_size = len(samples)

        # ── 2. Compute rewards ────────────────────────────────────────────────
        # Flatten: (B*G,) responses paired with repeated samples
        flat_samples = [s for s in samples for _ in range(self.group_size)]
        flat_responses = [r for group in responses for r in group]
        flat_rewards, flat_reward_dicts = batch_compute_rewards(flat_samples, flat_responses)

        # Reshape to (B, G)
        grouped_rewards = [
            flat_rewards[i * self.group_size: (i + 1) * self.group_size]
            for i in range(batch_size)
        ]

        # ── 3. Compute advantages ─────────────────────────────────────────────
        grouped_advantages = compute_batch_advantages(
            grouped_rewards, mode=self.advantage_norm
        )
        flat_advantages = [a for group in grouped_advantages for a in group]
        advantages_tensor = torch.tensor(flat_advantages, dtype=torch.float32, device=device)

        # ── 4. Compute log probs under current policy ─────────────────────────
        with torch.cuda.amp.autocast(enabled=self.config.get("fp16", False) or self.config.get("bf16", True)):
            outputs = self.model(
                input_ids=output_ids.to(device),
                attention_mask=(output_ids != self.processor.tokenizer.pad_token_id).long().to(device),
                pixel_values=inputs_on_device.get("pixel_values"),
                image_grid_thw=inputs_on_device.get("image_grid_thw"),
            )
            logits = outputs.logits  # (B*G, T, V)

        log_probs_new = compute_sequence_log_prob(
            logits, output_ids.to(device), response_mask.to(device)
        )  # (B*G,)

        # ── 5. Compute old log probs (detached, from generation) ──────────────
        with torch.no_grad():
            old_outputs = self.model(
                input_ids=output_ids.to(device),
                attention_mask=(output_ids != self.processor.tokenizer.pad_token_id).long().to(device),
                pixel_values=inputs_on_device.get("pixel_values"),
                image_grid_thw=inputs_on_device.get("image_grid_thw"),
            )
        log_probs_old = compute_sequence_log_prob(
            old_outputs.logits.detach(), output_ids.to(device), response_mask.to(device)
        )

        # ── 6. Policy loss ────────────────────────────────────────────────────
        policy_loss = compute_policy_loss(
            log_probs_new, log_probs_old.detach(), advantages_tensor, clip_range=self.clip_range
        )

        total_loss = policy_loss

        # ── 7. KL penalty ─────────────────────────────────────────────────────
        kl_val = torch.tensor(0.0, device=device)
        if self.ref_model is not None and self.kl_coef > 0:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=output_ids.to(device),
                    attention_mask=(output_ids != self.processor.tokenizer.pad_token_id).long().to(device),
                    pixel_values=inputs_on_device.get("pixel_values"),
                    image_grid_thw=inputs_on_device.get("image_grid_thw"),
                )
            log_probs_ref = compute_sequence_log_prob(
                ref_outputs.logits.detach(), output_ids.to(device), response_mask.to(device)
            )
            kl_val = compute_kl_loss(log_probs_new, log_probs_ref.detach())
            total_loss = total_loss + self.kl_coef * kl_val

        # ── 8. Aggregate metrics ──────────────────────────────────────────────
        def _mean(key):
            vals = [d[key] for d in flat_reward_dicts]
            return sum(vals) / len(vals) if vals else 0.0

        resp_lens = [len(r.split()) for r in flat_responses]
        step_metrics = {
            "reward_total_mean": _mean("reward_total"),
            "r_fmt_mean": _mean("r_fmt"),
            "r_ans_mean": _mean("r_ans"),
            "r_think_fmt_mean": _mean("r_think_fmt"),
            "r_acc_mean": _mean("r_acc"),
            "r_word_mean": _mean("r_word"),
            "frame_acc_mean": _mean("r_frame"),
            "object_acc_mean": _mean("r_object"),
            "answer_acc_mean": _mean("r_ans"),
            "kl_mean": kl_val.item(),
            "response_len_mean": sum(resp_lens) / len(resp_lens),
        }

        # ── 9. Occasional sample logging ──────────────────────────────────────
        if self.is_main and self.global_step % self.sample_log_steps == 0:
            self._log_sample(samples[0], responses[0][0], flat_reward_dicts[0])

        return total_loss, step_metrics

    def _log_metrics(self):
        avg = self.metrics.averages()
        parts = [f"step={self.global_step}"]
        for k, v in avg.items():
            parts.append(f"{k}={v:.4f}")
        logger.info("  ".join(parts))
        self.metrics.reset()

    def _log_sample(self, sample, response, reward_dict):
        logger.info("=" * 60)
        logger.info(f"[Sample] Q: {sample.get('question', '')[:120]}")
        logger.info(f"[Response] {response[:300]}")
        logger.info(f"[Rewards] {reward_dict}")
        logger.info("=" * 60)

    def _save_checkpoint(self, final: bool = False):
        tag = "final" if final else f"step-{self.global_step}"
        save_path = os.path.join(self.output_dir, tag)
        os.makedirs(save_path, exist_ok=True)
        # Unwrap DDP if needed
        model_to_save = getattr(self.model, "module", self.model)
        model_to_save.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    @torch.no_grad()
    def _evaluate(self):
        if not self.val_dataset:
            return
        from ..utils.metrics import evaluate_dataset
        val_loader = self._make_dataloader(self.val_dataset, shuffle=False)
        results = evaluate_dataset(
            self.model, self.processor, val_loader,
            max_new_tokens=self.max_new_tokens,
            device=next(self.model.parameters()).device,
        )
        if self.is_main:
            logger.info(f"[Eval step={self.global_step}] {results}")
