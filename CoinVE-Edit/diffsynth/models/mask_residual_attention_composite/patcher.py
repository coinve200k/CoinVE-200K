"""Inject residual composite cross-attention wrappers into a WanModel."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .modules import ResidualAttentionModule, ResidualCompositeCrossAttention


def inject_residual_attention_into_dit(
    dit: nn.Module,
    residual_alpha_init: float = 1.0,
    learnable_residual_alpha: bool = False,
    residual_mode: str = "replace_delta",
) -> ResidualAttentionModule:
    assert hasattr(dit, "blocks"), "expected `dit.blocks` (WanModel-style)"
    num_layers = len(dit.blocks)
    residual_module = ResidualAttentionModule(num_layers=num_layers)

    wrapped_count = 0
    for layer_idx, block in enumerate(dit.blocks):
        ca = getattr(block, "cross_attn", None)
        if ca is None:
            continue
        wrapper = ResidualCompositeCrossAttention(
            ca,
            layer_idx=layer_idx,
            residual_alpha_init=residual_alpha_init,
            learnable_residual_alpha=learnable_residual_alpha,
            residual_mode=residual_mode,
        )
        block.cross_attn = wrapper
        residual_module.register_layer(wrapper)
        wrapped_count += 1

    assert wrapped_count == num_layers, f"wrapped {wrapped_count} layers but dit has {num_layers} blocks"
    return residual_module


def build_query_mask(mask_latent: torch.Tensor) -> torch.Tensor:
    if mask_latent.dim() == 4:
        K, T_l, H_l, W_l = mask_latent.shape
        return mask_latent.reshape(K, T_l * H_l * W_l, 1)
    if mask_latent.dim() == 5:
        B, K, T_l, H_l, W_l = mask_latent.shape
        return mask_latent.reshape(B, K, T_l * H_l * W_l, 1)
    raise AssertionError(f"expected [K,T,H,W] or [B,K,T,H,W], got {mask_latent.shape}")


def set_residual_attention_from_masks(
    residual_module: ResidualAttentionModule,
    local_contexts: torch.Tensor,
    stage1_masks_pg: torch.Tensor,
    latent_thw: Tuple[int, int, int],
    interpolate_mode: str = "trilinear",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    from .utils import resample_mask_to_latent_grid

    if device is None:
        device = stage1_masks_pg.device
    masks = stage1_masks_pg.to(device=device)
    squeeze_b = False
    if masks.dim() == 5:
        B = masks.shape[0]
        if B != 1:
            raise NotImplementedError("residual composite attention currently supports batch_size=1")
        masks = masks.squeeze(0)
        squeeze_b = True
    elif masks.dim() != 4:
        raise AssertionError(f"expected masks [K,T,H,W] or [1,K,T,H,W], got {masks.shape}")

    m_lat = resample_mask_to_latent_grid(masks, latent_thw, mode=interpolate_mode)
    q_mask = build_query_mask(m_lat)
    if squeeze_b:
        q_mask = q_mask.unsqueeze(0)
    residual_module.set_runtime(
        local_contexts=local_contexts,
        mask_per_query=q_mask,
        device=device,
        dtype=dtype,
    )
