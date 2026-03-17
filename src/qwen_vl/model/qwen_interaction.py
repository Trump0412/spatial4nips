"""Geometry interaction modules for VGGT and DA3-based variants."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import is_flash_attn_2_available

from .da3_adapter import DA3Projector
from .mmr_memory import FrameMemoryBank, RegionMemoryBank
from .mmr_retriever import QueryDrivenRetriever
from .mmr_utils import build_monotonic_ids, summarize_query_tokens
from .msgf_memory import BiDirectionalMemoryBank, HierarchicalMemoryBank, MemoryRefiner
from .msgf_utils import FrameLayout, compute_stage_ranges, infer_frame_layout, mean_pool_tokens, safe_topk, split_by_layout

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func
else:
    flash_attn_func = None


class QwenVGGTInteractionv1(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.geo_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.layer_idx = layer_idx

    def forward(self, image_hidden_states, vggt_features, **kwargs):
        q_input = self.input_layernorm(image_hidden_states)
        kv_input = self.geo_layernorm(vggt_features)

        q = self.q_proj(q_input).view(q_input.shape[0], q_input.shape[1], self.num_heads, self.head_dim)
        k = self.k_proj(kv_input).view(kv_input.shape[0], kv_input.shape[1], self.num_heads, self.head_dim)
        v = self.v_proj(kv_input).view(kv_input.shape[0], kv_input.shape[1], self.num_heads, self.head_dim)

        if flash_attn_func is not None:
            attn_output = flash_attn_func(q, k, v, causal=False)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(image_hidden_states.shape)
        attn_output = self.o_proj(attn_output)
        return torch.tanh(self.gate) * attn_output


class _FramewiseGeometryInteraction(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        use_spatial_bias: bool = True,
        use_importance_gate: bool = True,
        geo_learn_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = 0 if layer_idx is None else int(layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_spatial_bias = use_spatial_bias
        self.learnable_bias = geo_learn_bias
        self.use_importance_gate = use_importance_gate
        self.msgf_debug = bool(getattr(config, "msgf_debug", False))

        if hasattr(config, "vision_config"):
            self.pooling_stride = config.vision_config.spatial_merge_size
        else:
            self.pooling_stride = 2

        if kwargs.pop("depart_smi_token", False):
            self.pooling_stride *= kwargs.pop("smi_downsample_rate", 2)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.geo_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))

        if self.use_importance_gate:
            self.importance_net = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, 1),
                nn.Sigmoid(),
            )
        else:
            self.importance_net = None

        self.bias_gate = nn.Sequential(
            nn.Linear(config.hidden_size, max(config.num_attention_heads // 2, 1)),
            nn.Sigmoid(),
        )

        memory_heads = max(1, min(self.num_heads, 8))
        self.memory_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=memory_heads,
            batch_first=True,
        )
        self.memory_gate = nn.Parameter(torch.tensor(0.0))
        self.geo_projector = DA3Projector(config.hidden_size, config.hidden_size)
        self.query_mixer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def get_spatial_bias(self, h_feat: int, w_feat: int, device, dtype) -> torch.Tensor:
        y = torch.arange(h_feat, device=device, dtype=dtype)
        x = torch.arange(w_feat, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        dist_matrix = torch.cdist(coords, coords, p=2)
        diag_len = math.sqrt(h_feat**2 + w_feat**2) + 1e-6
        return -(dist_matrix / diag_len)

    def _build_layout(self, total_tokens: int, grid_thw: Optional[torch.Tensor]) -> FrameLayout:
        return infer_frame_layout(total_tokens=total_tokens, grid_thw=grid_thw, pooling_stride=self.pooling_stride)

    def _split_frames(self, hidden_states: torch.Tensor, layout: FrameLayout) -> List[torch.Tensor]:
        return split_by_layout(hidden_states, layout)

    def _build_attn_bias(
        self,
        q_input: torch.Tensor,
        k_input: torch.Tensor,
        frame_shape: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        q_len = q_input.shape[1]
        k_len = k_input.shape[1]
        attn_bias = None

        if self.use_spatial_bias and q_len == k_len and frame_shape[0] * frame_shape[1] == q_len:
            spatial_bias = self.get_spatial_bias(frame_shape[0], frame_shape[1], q_input.device, q_input.dtype)
            gate_score = self.bias_gate(k_input).transpose(1, 2).unsqueeze(-1)
            head_bias = torch.zeros(
                q_input.shape[0], self.num_heads, q_len, k_len, device=q_input.device, dtype=q_input.dtype
            )
            split_idx = max(self.num_heads // 2, 1)
            head_bias[:, :split_idx] = spatial_bias.unsqueeze(0).unsqueeze(0) * gate_score[:, :split_idx]
            attn_bias = head_bias

        if self.use_importance_gate and self.importance_net is not None:
            importance = self.importance_net(k_input)
            importance_logit = torch.log(importance + 0.1).view(k_input.shape[0], 1, 1, k_len)
            attn_bias = importance_logit if attn_bias is None else attn_bias + importance_logit

        return attn_bias

    def _frame_attention(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        frame_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if q_tokens.numel() == 0:
            return q_tokens
        if kv_tokens.numel() == 0:
            return torch.zeros_like(q_tokens)

        q_input = self.input_layernorm(q_tokens).unsqueeze(0)
        k_input = self.geo_layernorm(kv_tokens).unsqueeze(0)

        bsz = q_input.shape[0]
        q_len = q_input.shape[1]
        k_len = k_input.shape[1]

        q = self.q_proj(q_input).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_input).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(k_input).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_bias = self._build_attn_bias(q_input, k_input, frame_shape)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        output = output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        output = self.o_proj(output).squeeze(0)
        return torch.tanh(self.gate) * output

    def _local_frame_fusion(
        self,
        semantic_hidden: torch.Tensor,
        geo_hidden: torch.Tensor,
        grid_thw: Optional[torch.Tensor],
    ):
        image_layout = self._build_layout(semantic_hidden.shape[1], grid_thw)
        geo_layout = self._build_layout(geo_hidden.shape[1], grid_thw)
        image_frames = self._split_frames(semantic_hidden, image_layout)
        geo_frames = self._split_frames(geo_hidden, geo_layout)

        if len(image_frames) != len(geo_frames):
            image_layout = FrameLayout([semantic_hidden.shape[1]], [(semantic_hidden.shape[1], 1)])
            geo_layout = FrameLayout([geo_hidden.shape[1]], [(geo_hidden.shape[1], 1)])
            image_frames = self._split_frames(semantic_hidden, image_layout)
            geo_frames = self._split_frames(geo_hidden, geo_layout)

        outputs = []
        for frame_idx, image_tokens in enumerate(image_frames):
            geo_tokens = geo_frames[frame_idx] if frame_idx < len(geo_frames) else geo_frames[-1]
            frame_shape = image_layout.frame_shapes[min(frame_idx, len(image_layout.frame_shapes) - 1)]
            outputs.append(self._frame_attention(image_tokens, geo_tokens, frame_shape))

        if outputs:
            local_delta = torch.cat(outputs, dim=0).unsqueeze(0)
        else:
            local_delta = torch.zeros_like(semantic_hidden)
        return local_delta, image_layout, image_frames, geo_frames

    def _build_frame_query(
        self,
        frame_tokens: torch.Tensor,
        text_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        frame_summary = mean_pool_tokens(frame_tokens)
        if text_hidden_states is None or text_hidden_states.numel() == 0:
            return frame_summary

        if text_hidden_states.dim() == 3:
            text_hidden_states = text_hidden_states.squeeze(0)
        text_summary = mean_pool_tokens(text_hidden_states)
        return self.query_mixer(torch.cat([frame_summary, text_summary], dim=-1))

    def _select_frame_atoms(
        self,
        frame_tokens: torch.Tensor,
        geo_tokens: Optional[torch.Tensor],
        top_r: int,
    ) -> torch.Tensor:
        if frame_tokens.numel() == 0:
            return frame_tokens

        if self.use_importance_gate and self.importance_net is not None:
            scores = self.importance_net(self.input_layernorm(frame_tokens).unsqueeze(0)).squeeze(0).squeeze(-1)
        else:
            scores = frame_tokens.norm(dim=-1)
        _, token_indices = safe_topk(scores, top_r)
        if token_indices.numel() == 0:
            return frame_tokens[:0]

        atoms = frame_tokens[token_indices]
        if geo_tokens is not None and geo_tokens.numel() > 0:
            if geo_tokens.shape[0] == frame_tokens.shape[0]:
                geo_atoms = geo_tokens[token_indices.clamp(max=geo_tokens.shape[0] - 1)]
            else:
                geo_atoms = mean_pool_tokens(geo_tokens).expand_as(atoms)
            atoms = 0.5 * (atoms + self.geo_projector(geo_atoms))
        return atoms

    def _memory_update(self, frame_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens is None or context_tokens.numel() == 0:
            return torch.zeros_like(frame_tokens)

        query = self.memory_norm(frame_tokens).unsqueeze(0)
        context = self.memory_norm(context_tokens).unsqueeze(0)
        update, _ = self.memory_attn(query, context, context)
        return torch.tanh(self.memory_gate) * update.squeeze(0)

    def _log_memory_stats(self, tag: str, available_frames: int, available_atoms: int, frame_topk: int, atom_topk: int):
        if not self.msgf_debug:
            return
        print(
            f"[MSGF] layer={self.layer_idx} stage={tag} "
            f"available_frames={available_frames} available_atoms={available_atoms} "
            f"frame_topk={frame_topk} atom_topk={atom_topk}"
        )


class QwenVGGTInteractionv2(_FramewiseGeometryInteraction):
    def forward(self, semantic_hidden, vggt_features, grid_thw=None, **kwargs):
        local_delta, _, _, _ = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        return local_delta


class QwenVGGTInteractionv2Flash(QwenVGGTInteractionv2):
    pass


class QwenDA3SGFBaseline(QwenVGGTInteractionv2Flash):
    """DA3 + SGF baseline. Kept as a named alias for script switching."""


class QwenDA3MSGFBase(_FramewiseGeometryInteraction):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        self.stage_ranges = compute_stage_ranges(config.num_hidden_layers, "msgf", config)
        self.msgf_topr = int(getattr(config, "msgf_topr", 32))
        self.msgf_frame_topk_max = int(getattr(config, "msgf_frame_topk_max", 3))
        self.msgf_atom_topk_max = int(getattr(config, "msgf_atom_topk_max", 8))

    def forward(self, semantic_hidden, vggt_features, grid_thw=None, text_hidden_states=None, **kwargs):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if self.layer_idx <= self.stage_ranges.warmup_end:
            return local_delta

        frame_atoms = [
            self._select_frame_atoms(frame_tokens, geo_tokens, self.msgf_topr)
            for frame_tokens, geo_tokens in zip(fused_frames, geo_frames)
        ]
        bank = BiDirectionalMemoryBank.from_frame_atoms(frame_atoms)

        memory_updates = []
        frame_topk = 0
        atom_topk = 0
        for frame_idx, frame_tokens in enumerate(fused_frames):
            if self.stage_ranges.write_start <= self.layer_idx <= self.stage_ranges.write_end:
                context = frame_atoms[frame_idx] if frame_idx < len(frame_atoms) else frame_tokens[:0]
                frame_topk = 1 if context.numel() > 0 else 0
                atom_topk = int(context.shape[0]) if context.numel() > 0 else 0
            else:
                query = self._build_frame_query(frame_tokens, text_hidden_states)
                retrieved = bank.retrieve(query, self.msgf_frame_topk_max, self.msgf_atom_topk_max)
                context = retrieved.context
                frame_topk = max(frame_topk, retrieved.frame_topk)
                atom_topk = max(atom_topk, retrieved.atom_topk)
            memory_updates.append(self._memory_update(frame_tokens, context))

        self._log_memory_stats(
            tag="msgf",
            available_frames=len(frame_atoms),
            available_atoms=bank.total_atoms,
            frame_topk=frame_topk,
            atom_topk=atom_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)


class QwenDA3HMSGF(_FramewiseGeometryInteraction):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        self.stage_ranges = compute_stage_ranges(config.num_hidden_layers, "hmsgf", config)
        self.hmsgf_frame_topk_max = int(getattr(config, "hmsgf_frame_topk_max", 3))
        self.hmsgf_region_topr = int(getattr(config, "hmsgf_region_topr", 32))
        self.hmsgf_region_topk_max = int(getattr(config, "hmsgf_region_topk_max", 8))

    def forward(self, semantic_hidden, vggt_features, grid_thw=None, text_hidden_states=None, **kwargs):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if self.layer_idx <= self.stage_ranges.warmup_end:
            return local_delta

        region_atoms = [
            self._select_frame_atoms(frame_tokens, geo_tokens, self.hmsgf_region_topr)
            for frame_tokens, geo_tokens in zip(fused_frames, geo_frames)
        ]
        bank = HierarchicalMemoryBank.from_frame_atoms(region_atoms)

        memory_updates = []
        frame_topk = 0
        region_topk = 0
        for frame_idx, frame_tokens in enumerate(fused_frames):
            if self.stage_ranges.write_start <= self.layer_idx <= self.stage_ranges.write_end:
                own_atoms = region_atoms[frame_idx] if frame_idx < len(region_atoms) else frame_tokens[:0]
                own_summary = mean_pool_tokens(own_atoms) if own_atoms.numel() > 0 else frame_tokens[:1]
                context = torch.cat([own_summary, own_atoms], dim=0) if own_atoms.numel() > 0 else own_summary
                frame_topk = 1
                region_topk = max(region_topk, int(own_atoms.shape[0]))
            else:
                query = self._build_frame_query(frame_tokens, text_hidden_states)
                retrieved = bank.retrieve(query, self.hmsgf_frame_topk_max, self.hmsgf_region_topk_max)
                context = retrieved.context
                frame_topk = max(frame_topk, retrieved.frame_topk)
                region_topk = max(region_topk, retrieved.atom_topk)
            memory_updates.append(self._memory_update(frame_tokens, context))

        total_regions = sum(int(atoms.shape[0]) for atoms in region_atoms)
        self._log_memory_stats(
            tag="hmsgf",
            available_frames=len(region_atoms),
            available_atoms=total_regions,
            frame_topk=frame_topk,
            atom_topk=region_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)


class QwenDA3RMSGF(_FramewiseGeometryInteraction):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        self.stage_ranges = compute_stage_ranges(config.num_hidden_layers, "rmsgf", config)
        self.rmsgf_topr = int(getattr(config, "rmsgf_topr", 32))
        self.rmsgf_atom_topk_max = int(getattr(config, "rmsgf_atom_topk_max", 8))
        self.refiner = MemoryRefiner(
            hidden_size=config.hidden_size,
            use_gate=bool(getattr(config, "rmsgf_refine_gate", True)),
            residual=bool(getattr(config, "rmsgf_refine_residual", True)),
        )

    def forward(self, semantic_hidden, vggt_features, grid_thw=None, text_hidden_states=None, **kwargs):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if self.layer_idx <= self.stage_ranges.warmup_end:
            return local_delta

        init_atoms = [
            self._select_frame_atoms(frame_tokens, geo_tokens, self.rmsgf_topr)
            for frame_tokens, geo_tokens in zip(fused_frames, geo_frames)
        ]

        if self.stage_ranges.init_start <= self.layer_idx <= self.stage_ranges.init_end:
            memory_updates = [self._memory_update(frame_tokens, atoms) for frame_tokens, atoms in zip(fused_frames, init_atoms)]
            total_atoms = sum(int(atoms.shape[0]) for atoms in init_atoms)
            self._log_memory_stats("rmsgf_init", len(init_atoms), total_atoms, 1, self.rmsgf_topr)
            return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)

        refined_atoms = []
        for frame_tokens, geo_tokens, atoms in zip(fused_frames, geo_frames, init_atoms):
            if atoms.numel() == 0:
                refined_atoms.append(atoms)
                continue
            frame_summary = self._build_frame_query(frame_tokens, text_hidden_states)
            geo_summary = mean_pool_tokens(geo_tokens) if geo_tokens.numel() > 0 else frame_summary
            refined_atoms.append(self.refiner(atoms, frame_summary + geo_summary))

        bank = BiDirectionalMemoryBank.from_frame_atoms(refined_atoms)
        memory_updates = []
        atom_topk = 0
        for frame_tokens in fused_frames:
            query = self._build_frame_query(frame_tokens, text_hidden_states)
            retrieved = bank.retrieve(query, frame_topk_max=len(refined_atoms), atom_topk_max=self.rmsgf_atom_topk_max)
            atom_topk = max(atom_topk, retrieved.atom_topk)
            memory_updates.append(self._memory_update(frame_tokens, retrieved.context))

        self._log_memory_stats(
            tag="rmsgf_refine",
            available_frames=len(refined_atoms),
            available_atoms=bank.total_atoms,
            frame_topk=len(refined_atoms),
            atom_topk=atom_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)


class QwenDA3MMRInteraction(_FramewiseGeometryInteraction):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, use_importance_gate=False, **kwargs)
        self.stage_ranges = compute_stage_ranges(config.num_hidden_layers, "mmr", config)
        self.mmr_use_region_memory = bool(getattr(config, "mmr_use_region_memory", False))
        self.mmr_frame_topk_max = int(getattr(config, "mmr_frame_topk_max", 3))
        self.mmr_region_topk_max = int(getattr(config, "mmr_region_topk_max", 8))
        self.mmr_region_atoms_per_frame = int(getattr(config, "mmr_region_atoms_per_frame", 8))
        self.mmr_query_use_text = bool(getattr(config, "mmr_query_use_text", True))
        self.mmr_query_use_visual_summary = bool(getattr(config, "mmr_query_use_visual_summary", True))
        self.retriever = QueryDrivenRetriever(
            frame_topk_max=self.mmr_frame_topk_max,
            region_topk_max=self.mmr_region_topk_max,
            use_temporal_continuity=bool(getattr(config, "mmr_use_temporal_continuity", True)),
            use_view_continuity=bool(getattr(config, "mmr_use_view_continuity", True)),
            use_region_memory=self.mmr_use_region_memory,
        )

    def _write_frame_memory(
        self,
        fused_frames: Sequence[torch.Tensor],
        frame_ids: Sequence[int],
        view_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> FrameMemoryBank:
        summaries = []
        memory_frame_ids = []
        memory_view_ids = [] if view_ids is not None else None
        for idx, frame_tokens in enumerate(fused_frames):
            if frame_tokens.numel() == 0:
                continue
            summaries.append(mean_pool_tokens(frame_tokens))
            memory_frame_ids.append(int(frame_ids[idx]))
            if memory_view_ids is not None:
                memory_view_ids.append(view_ids[idx])
        return FrameMemoryBank.from_summaries(summaries, frame_ids=memory_frame_ids, view_ids=memory_view_ids)

    def _write_region_memory(
        self,
        fused_frames: Sequence[torch.Tensor],
        geo_frames: Sequence[torch.Tensor],
        frame_ids: Sequence[int],
        view_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> RegionMemoryBank:
        atoms_per_frame = []
        valid_frame_ids = []
        valid_view_ids = [] if view_ids is not None else None
        for idx, frame_tokens in enumerate(fused_frames):
            geo_tokens = geo_frames[idx] if idx < len(geo_frames) else frame_tokens[:0]
            if frame_tokens.numel() == 0:
                continue
            atoms_per_frame.append(self._select_frame_atoms(frame_tokens, geo_tokens, self.mmr_region_atoms_per_frame))
            valid_frame_ids.append(int(frame_ids[idx]))
            if valid_view_ids is not None:
                valid_view_ids.append(view_ids[idx])
        return RegionMemoryBank.from_frame_atoms(atoms_per_frame, frame_ids=valid_frame_ids, view_ids=valid_view_ids)

    def _build_mmr_query(
        self,
        frame_tokens: torch.Tensor,
        text_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        components = []
        if self.mmr_query_use_visual_summary:
            components.append(summarize_query_tokens(frame_tokens))
        if self.mmr_query_use_text and text_hidden_states is not None and text_hidden_states.numel() > 0:
            text_tokens = text_hidden_states.squeeze(0) if text_hidden_states.dim() == 3 else text_hidden_states
            components.append(summarize_query_tokens(text_tokens))

        if not components:
            components.append(summarize_query_tokens(frame_tokens))
        if len(components) == 1:
            return components[0]
        return self.query_mixer(torch.cat(components, dim=-1))

    def forward(
        self,
        semantic_hidden,
        da3_features,
        grid_thw=None,
        text_hidden_states=None,
        memory_bank=None,
        frame_ids=None,
        view_ids=None,
        **kwargs,
    ):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, da3_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if frame_ids is None:
            frame_ids_tensor = build_monotonic_ids(len(fused_frames), fused_hidden.device)
        else:
            frame_ids_tensor = frame_ids.to(fused_hidden.device)
        view_ids_tensor = view_ids.to(fused_hidden.device) if view_ids is not None else None
        frame_id_values = [int(x) for x in frame_ids_tensor.tolist()]
        view_id_values = [int(x) for x in view_ids_tensor.tolist()] if view_ids_tensor is not None else None

        frame_memory_bank = (
            memory_bank
            if isinstance(memory_bank, FrameMemoryBank)
            else self._write_frame_memory(fused_frames, frame_id_values, view_id_values)
        )
        region_memory_bank = (
            self._write_region_memory(fused_frames, geo_frames, frame_id_values, view_id_values)
            if self.mmr_use_region_memory
            else None
        )

        memory_updates = []
        frame_topk = 0
        region_topk = 0
        available_frames = frame_memory_bank.available_frames
        available_regions = region_memory_bank.available_regions if region_memory_bank is not None else 0

        for frame_idx, frame_tokens in enumerate(fused_frames):
            if frame_tokens.numel() == 0:
                memory_updates.append(frame_tokens)
                continue

            if self.stage_ranges.write_start <= self.layer_idx <= self.stage_ranges.write_end:
                context = mean_pool_tokens(frame_tokens)
                frame_topk = max(frame_topk, int(context.shape[0]))
            elif self.stage_ranges.read_start <= self.layer_idx <= self.stage_ranges.read_end:
                query = self._build_mmr_query(frame_tokens, text_hidden_states)
                retrieved = self.retriever.retrieve(
                    query,
                    frame_memory_bank,
                    region_memory_bank,
                    query_frame_id=int(frame_ids_tensor[frame_idx].item()) if frame_idx < frame_ids_tensor.numel() else frame_idx,
                    query_view_id=int(view_ids_tensor[frame_idx].item()) if view_ids_tensor is not None and frame_idx < view_ids_tensor.numel() else None,
                )
                context = retrieved.context
                frame_topk = max(frame_topk, retrieved.frame_topk)
                region_topk = max(region_topk, retrieved.region_topk)
                available_frames = max(available_frames, retrieved.available_frames)
                available_regions = max(available_regions, retrieved.available_regions)
            else:
                context = frame_tokens[:0]

            memory_updates.append(self._memory_update(frame_tokens, context))

        self._log_memory_stats(
            tag="mmr",
            available_frames=available_frames,
            available_atoms=available_regions,
            frame_topk=frame_topk,
            atom_topk=region_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)
