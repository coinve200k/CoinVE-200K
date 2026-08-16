"""Residual composite cross-attention modules.

The main DiT context remains a global single-instruction context built by
concatenating all per-region instructions. Per-instruction local contexts and
query masks are injected through runtime slots and only provide a masked residual
correction to the global attention output.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ResidualCompositeCrossAttention(nn.Module):
    def __init__(
        self,
        orig_cross_attn: nn.Module,
        layer_idx: int,
        residual_alpha_init: float = 1.0,
        learnable_residual_alpha: bool = False,
        residual_mode: str = "replace_delta",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if residual_mode not in {"replace_delta", "additive"}:
            raise ValueError(f"unknown residual_mode={residual_mode}")
        self.orig = orig_cross_attn
        self.layer_idx = int(layer_idx)
        self.dim = orig_cross_attn.dim
        self.num_heads = orig_cross_attn.num_heads
        self.head_dim = orig_cross_attn.head_dim
        self.has_image_input = orig_cross_attn.has_image_input
        self.residual_mode = residual_mode
        self.eps = float(eps)
        alpha = torch.tensor(float(residual_alpha_init), dtype=torch.float32)
        if learnable_residual_alpha:
            self.residual_alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("residual_alpha", alpha, persistent=True)
        self.current_local_contexts: Optional[torch.Tensor] = None
        self.current_q_mask: Optional[torch.Tensor] = None

    def _split_context(self, y: torch.Tensor) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        orig = self.orig
        if orig.has_image_input:
            if y.shape[1] < 257:
                raise RuntimeError(f"image-input cross-attn expected >=257 context tokens, got {y.shape}")
            return y[:, :257], y[:, 257:]
        return None, y

    def _attend_text(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        orig = self.orig
        k = orig.norm_k(orig.k(ctx))
        v = orig.v(ctx)
        return orig.attn(q, k, v)

    def _attend_image(self, q: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        orig = self.orig
        k_img = orig.norm_k_img(orig.k_img(img))
        v_img = orig.v_img(img)
        from ..wan_video_dit import flash_attention as _fa
        return _fa(q, k_img, v_img, num_heads=self.num_heads)

    def _attend_local_mix(
        self,
        q: torch.Tensor,
        local_contexts: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_contexts.dim() != 4:
            raise AssertionError(f"expected local_contexts [B,K,L,D], got {local_contexts.shape}")
        if q_mask.dim() != 4 or q_mask.shape[-1] != 1:
            raise AssertionError(f"expected q_mask [B,K,Lq,1], got {q_mask.shape}")
        B, K, _, _ = local_contexts.shape
        if q_mask.shape[0] != B or q_mask.shape[1] != K or q_mask.shape[2] != q.shape[1]:
            raise RuntimeError(
                f"q_mask shape {q_mask.shape} incompatible with local_contexts {local_contexts.shape} and q {q.shape}"
            )

        local_sum = None
        weight_sum = None
        for i in range(K):
            w_i = q_mask[:, i].to(device=q.device, dtype=q.dtype)
            _img_i, ctx_i = self._split_context(local_contexts[:, i])
            out_i = self._attend_text(q, ctx_i)
            weighted_i = out_i * w_i
            local_sum = weighted_i if local_sum is None else local_sum + weighted_i
            weight_sum = w_i if weight_sum is None else weight_sum + w_i

        if local_sum is None or weight_sum is None:
            raise RuntimeError("empty local residual contexts")
        local_mix = local_sum / weight_sum.clamp(min=self.eps).to(dtype=local_sum.dtype)
        coverage = weight_sum.clamp(0, 1).to(dtype=local_sum.dtype)
        return local_mix, coverage

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 3:
            raise AssertionError(f"residual attention expects global context [B,L,D], got {y.shape}")
        orig = self.orig
        q = orig.norm_q(orig.q(x))
        img, ctx = self._split_context(y)
        out = self._attend_text(q, ctx)

        local_contexts = self.current_local_contexts
        q_mask = self.current_q_mask
        if local_contexts is not None and q_mask is not None:
            local_contexts = local_contexts.to(device=x.device, dtype=y.dtype)
            q_mask = q_mask.to(device=x.device, dtype=x.dtype)
            local_mix, coverage = self._attend_local_mix(q, local_contexts, q_mask)
            alpha = self.residual_alpha.to(device=out.device, dtype=out.dtype)
            if self.residual_mode == "replace_delta":
                out = out + alpha * coverage * (local_mix - out)
            elif self.residual_mode == "additive":
                out = out + alpha * coverage * local_mix
            else:
                raise ValueError(f"unknown residual_mode={self.residual_mode}")

        if img is not None:
            out = out + self._attend_image(q, img)
        return orig.o(out)


class ResidualAttentionModule(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self._wrapped_layers: list[ResidualCompositeCrossAttention] = []
        self._last_mask_mean: Optional[torch.Tensor] = None
        self._last_coverage_mean: Optional[torch.Tensor] = None
        self._last_num_instructions: Optional[int] = None

    def register_layer(self, layer: ResidualCompositeCrossAttention) -> None:
        self._wrapped_layers.append(layer)
        if len(self._wrapped_layers) > self.num_layers:
            raise RuntimeError(f"too many wrapped layers ({len(self._wrapped_layers)} > {self.num_layers})")

    def set_runtime(
        self,
        local_contexts: torch.Tensor,
        mask_per_query: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if local_contexts.dim() == 3:
            local_contexts = local_contexts.unsqueeze(0)
        if local_contexts.dim() != 4:
            raise AssertionError(f"expected local_contexts [B,K,L,D] or [K,L,D], got {local_contexts.shape}")

        if mask_per_query.dim() == 3:
            if mask_per_query.shape[-1] == 1:
                mask = mask_per_query.unsqueeze(0)
            else:
                mask = mask_per_query.unsqueeze(-1)
        elif mask_per_query.dim() == 4 and mask_per_query.shape[-1] == 1:
            mask = mask_per_query
        else:
            raise AssertionError(f"expected mask_per_query [K,L,1], [B,K,L,1], or [B,K,L], got {mask_per_query.shape}")

        if mask.shape[0] != local_contexts.shape[0] or mask.shape[1] != local_contexts.shape[1]:
            raise RuntimeError(f"mask shape {mask.shape} incompatible with local_contexts {local_contexts.shape}")
        if device is None:
            device = local_contexts.device
        if dtype is None:
            dtype = local_contexts.dtype
        # Runtime slots are read by every checkpointed DiT block. If local_contexts
        # keeps the VLM graph, DeepSpeed activation checkpointing will try to
        # backprop through that same graph once per block during recompute.
        # Detach it here; VLM/learnable-query LoRA still trains through the
        # explicit global context path passed as the DiT context input.
        local_contexts = local_contexts.detach().to(device=device, dtype=dtype)
        mask = mask.detach().clamp(0, 1).to(device=device, dtype=dtype)

        self._last_mask_mean = mask.detach().float().mean()
        self._last_coverage_mean = mask.detach().float().sum(dim=1).clamp(0, 1).mean()
        self._last_num_instructions = int(mask.shape[1])
        for layer in self._wrapped_layers:
            layer.current_local_contexts = local_contexts
            layer.current_q_mask = mask.to(dtype=layer.orig.q.weight.dtype)

    @torch.no_grad()
    def clear(self) -> None:
        for layer in self._wrapped_layers:
            layer.current_local_contexts = None
            layer.current_q_mask = None

    @torch.no_grad()
    def stats(self) -> dict:
        alpha_vals = []
        for layer in self._wrapped_layers:
            alpha_vals.append(layer.residual_alpha.detach().float().reshape(()))
        alpha_mean = torch.stack(alpha_vals).mean().cpu().item() if alpha_vals else float("nan")
        return {
            "mask_mean": self._last_mask_mean.detach().float().cpu().item() if self._last_mask_mean is not None else float("nan"),
            "coverage_mean": self._last_coverage_mean.detach().float().cpu().item() if self._last_coverage_mean is not None else float("nan"),
            "alpha_mean": alpha_mean,
            "num_instructions": self._last_num_instructions,
        }

    @torch.no_grad()
    def export_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            f"layers.{i}.residual_alpha": layer.residual_alpha.detach().cpu()
            for i, layer in enumerate(self._wrapped_layers)
        }
