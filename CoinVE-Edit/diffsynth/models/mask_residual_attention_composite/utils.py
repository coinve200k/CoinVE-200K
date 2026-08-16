"""
Helpers for Stage-2 mask-bias / compose pipeline.

Two responsibilities:
  1. Read raw per-frame masks and resample to the stage-1 patch grid
     [T_v, H_v, W_v] with SOFT (continuous) values in [0, 1].
     This is the format that simulates the frozen MaskHead's output —
     stage-2 trains entirely against this, so we can drop in MaskHead.predict()
     at inference without touching the data pipeline.
  2. Resample stage-1 patch-grid masks to DiT latent grid for cross-attn bias.

Constants follow stage-1 (PATCH=16, SPATIAL_MERGE_SIZE=2, FRAME_FACTOR=2).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SPATIAL_MERGE_SIZE = 2
FRAME_FACTOR = 2


# -----------------------------------------------------------------------------
# Frame-sampling reproduction (mirrors qwen_vl_utils linspace+round)
# -----------------------------------------------------------------------------

def reproduce_sampled_indices(total_frames: int, n_sampled: int) -> List[int]:
    if total_frames <= 0 or n_sampled <= 0:
        return []
    return torch.linspace(0, total_frames - 1, n_sampled).round().long().tolist()


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def read_mask_uint8(path: str):
    """Decode a mask video/image to a single-channel uint8 [T, H, W]."""
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        m = np.asarray(Image.open(path).convert("L"))
        return m[None, :, :], None  # [1, H, W]
    import imageio
    r = imageio.get_reader(path)
    fps = r.get_meta_data().get("fps", None)
    frames = [np.asarray(f) for f in r]
    r.close()
    arr = np.stack(frames, axis=0)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr, fps


# -----------------------------------------------------------------------------
# Soft mask alignment to stage-1 patch grid
# -----------------------------------------------------------------------------

def align_mask_to_patch_grid_soft(
    mask_frames_uint8: np.ndarray,
    sampled_indices: List[int],
    h_resized: int,
    w_resized: int,
    T_v: int,
    H_v: int,
    W_v: int,
) -> torch.Tensor:
    """Down-sample a per-frame mask volume to the visual-token grid (soft).

    Steps:
      1. pick `sampled_indices` from the mask sequence
      2. /255 to float32 in [0, 1]  (NO binarization)
      3. BILINEAR resize each picked frame -> (h_resized, w_resized)
      4. BILINEAR resize each frame        -> (W_v, H_v)
      5. group every FRAME_FACTOR consecutive frames and take MEAN
         (vs. stage-1 which uses max) — gives smoother attention bias.

    Args
    ----
    mask_frames_uint8 : [T_full, H, W] uint8 (single channel; 255 == positive)
    sampled_indices   : len must be FRAME_FACTOR * T_v.

    Returns
    -------
    torch.FloatTensor of shape [T_v, H_v, W_v], values in [0, 1].
    """
    if mask_frames_uint8.ndim == 4 and mask_frames_uint8.shape[-1] in (3, 4):
        mask_frames_uint8 = mask_frames_uint8[..., 0]
    assert mask_frames_uint8.ndim == 3, f"expected [T,H,W], got {mask_frames_uint8.shape}"

    T_sampled = len(sampled_indices)
    if T_sampled != FRAME_FACTOR * T_v:
        raise ValueError(
            f"len(sampled_indices)={T_sampled} != FRAME_FACTOR*T_v="
            f"{FRAME_FACTOR}*{T_v}={FRAME_FACTOR * T_v}"
        )

    T_full = mask_frames_uint8.shape[0]
    idx = [min(max(i, 0), T_full - 1) for i in sampled_indices]
    sampled = mask_frames_uint8[idx].astype(np.float32) / 255.0   # [T_sampled, H, W]

    grid = np.zeros((T_sampled, H_v, W_v), dtype=np.float32)
    for t in range(T_sampled):
        m = Image.fromarray((sampled[t] * 255).astype(np.uint8), mode="L")
        m = m.resize((w_resized, h_resized), Image.BILINEAR)
        m = m.resize((W_v, H_v), Image.BILINEAR)
        grid[t] = np.asarray(m, dtype=np.float32) / 255.0

    merged = grid.reshape(T_v, FRAME_FACTOR, H_v, W_v).mean(axis=1)
    return torch.from_numpy(merged).float()


# -----------------------------------------------------------------------------
# Patch grid -> DiT latent grid (used by cross-attn at runtime)
# -----------------------------------------------------------------------------

def resample_mask_to_latent_grid(
    mask_pg: torch.Tensor,
    latent_thw: Tuple[int, int, int],
    mode: str = "trilinear",
) -> torch.Tensor:
    """Resample stage-1 patch-grid mask to DiT latent grid.

    Args
    ----
    mask_pg    : [N, T_v, H_v, W_v] soft mask in [0,1] (N = #instructions).
                 Or [T_v, H_v, W_v] for a single instruction.
    latent_thw : (T_l, H_l, W_l).
    mode       : 'trilinear' (default, soft) or 'nearest' (hard).

    Returns
    -------
    torch.FloatTensor of shape [N, T_l, H_l, W_l] (or [T_l, H_l, W_l] if input is 3D).
    """
    if mask_pg.dim() == 3:
        mask_pg_b = mask_pg.unsqueeze(0)               # [1, T_v, H_v, W_v]
        squeeze_n = True
    else:
        mask_pg_b = mask_pg                             # [N, T_v, H_v, W_v]
        squeeze_n = False
    # F.interpolate with mode='trilinear' needs 5D input [B, C, T, H, W]
    x = mask_pg_b.unsqueeze(1)                          # [N, 1, T_v, H_v, W_v]
    align_corners = False if mode == "trilinear" else None
    kwargs = dict(size=latent_thw, mode=mode)
    if align_corners is not None:
        kwargs["align_corners"] = align_corners
    y = F.interpolate(x, **kwargs).squeeze(1)           # [N, T_l, H_l, W_l]
    return y.squeeze(0) if squeeze_n else y
