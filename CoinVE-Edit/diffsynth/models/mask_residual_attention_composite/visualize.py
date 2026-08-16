from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from diffsynth import save_video
from diffsynth.models.mask_head.model import MaskHead
from .pipeline import (
    CaptureLastHidden,
    derive_grid_via_processor,
    postprocess_mask,
)


def scalar_int(sd: dict[str, torch.Tensor], key: str, default: int) -> int:
    return int(sd[key].item()) if key in sd else int(default)


def sub_state_dict(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _get_arg(args, name: str, default=None):
    return getattr(args, name, default)


def load_frozen_mask_head_from_checkpoint(args, pipe, device: torch.device) -> tuple[MaskHead, dict]:
    if not _get_arg(args, "mask_head_checkpoint", None):
        raise ValueError("--mask_head_checkpoint is required for pred-mask training")

    mask_sd_source = load_safetensors(args.mask_head_checkpoint)
    prefixed_mask_sd = sub_state_dict(mask_sd_source, "mask_head.")
    if prefixed_mask_sd:
        mask_sd = prefixed_mask_sd
    elif "proj_v.weight" in mask_sd_source and "query_embed" in mask_sd_source:
        mask_sd = {k: v for k, v in mask_sd_source.items() if not k.startswith("mask_config.")}
    else:
        mask_sd = {}
    if not mask_sd:
        raise RuntimeError(
            "No mask_head weights found. Provide a checkpoint containing mask_head.* "
            "or a raw MaskHead state_dict."
        )

    default_use_prompt_ctx = True if _get_arg(args, "mask_use_prompt_ctx", None) is None else bool(args.mask_use_prompt_ctx)
    mask_config = {
        "mask_head_checkpoint": args.mask_head_checkpoint,
        "mask_hidden": scalar_int(mask_sd_source, "mask_config.hidden", _get_arg(args, "mask_hidden", 256)),
        "mask_layers": scalar_int(mask_sd_source, "mask_config.layers", _get_arg(args, "mask_layers", 2)),
        "mask_heads": scalar_int(mask_sd_source, "mask_config.heads", _get_arg(args, "mask_heads", 8)),
        "mask_num_query": scalar_int(mask_sd_source, "mask_config.num_query", _get_arg(args, "mask_num_query", 4)),
        "mask_no_grid_pe": bool(scalar_int(mask_sd_source, "mask_config.no_grid_pe", int(_get_arg(args, "mask_no_grid_pe", False)))),
        "mask_max_t_v": scalar_int(mask_sd_source, "mask_config.max_t_v", _get_arg(args, "mask_max_t_v", 64)),
        "mask_max_h_v": scalar_int(mask_sd_source, "mask_config.max_h_v", _get_arg(args, "mask_max_h_v", 64)),
        "mask_max_w_v": scalar_int(mask_sd_source, "mask_config.max_w_v", _get_arg(args, "mask_max_w_v", 64)),
        "mask_use_prompt_ctx": bool(scalar_int(mask_sd_source, "mask_config.use_prompt_ctx", int(default_use_prompt_ctx))),
        "mask_ctx_dim": scalar_int(mask_sd_source, "mask_config.ctx_dim", _get_arg(args, "mask_ctx_dim", 5120)),
        "mask_postprocess": _get_arg(args, "mask_postprocess", "sigmoid"),
        "mask_threshold": float(_get_arg(args, "mask_threshold", 0.5)),
        "mask_floor": float(_get_arg(args, "mask_floor", 0.0)),
        "mask_dilate_px": int(_get_arg(args, "mask_dilate_px", 0)),
    }

    config = pipe.mllm.model.config
    vlm_dim = getattr(getattr(config, "text_config", config), "hidden_size")
    mask_head = MaskHead(
        vlm_dim=vlm_dim,
        hidden=mask_config["mask_hidden"],
        num_layers=mask_config["mask_layers"],
        num_heads=mask_config["mask_heads"],
        num_query=mask_config["mask_num_query"],
        use_grid_pe=not mask_config["mask_no_grid_pe"],
        max_t_v=mask_config["mask_max_t_v"],
        max_h_v=mask_config["mask_max_h_v"],
        max_w_v=mask_config["mask_max_w_v"],
        use_prompt_ctx=mask_config["mask_use_prompt_ctx"],
        ctx_dim=mask_config["mask_ctx_dim"],
    )
    res = mask_head.load_state_dict(mask_sd, strict=False)
    print(
        f"[predmask] mask_head loaded from {args.mask_head_checkpoint}: "
        f"keys={len(mask_sd)} missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
    )
    mask_head.to(device=device, dtype=torch.bfloat16).eval()
    for param in mask_head.parameters():
        param.requires_grad_(False)
    return mask_head, mask_config


@torch.no_grad()
def predict_composite_masks_and_contexts(
    pipe,
    mask_head: MaskHead,
    prompts: list[str],
    src_pil_full: list[Image.Image],
    num_video_queries: int,
    max_pixels_per_frame: Optional[int],
    mask_postprocess: str,
    mask_threshold: float,
    mask_floor: float,
    mask_dilate_px: int,
    device: torch.device,
):
    contexts = []
    masks = []
    grids = []
    video_pad_id = pipe.mllm.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    mh_dtype = next(mask_head.parameters()).dtype

    pipe.load_models_to_device(("mllm",))
    for prompt in prompts:
        cap = CaptureLastHidden(pipe.mllm.model)
        with cap:
            context = pipe.mllm(prompt, src_video=src_pil_full, ref_image=None)
        context = context.detach().to(device=device)
        contexts.append(context)

        visual_tokens = cap.last_hidden[:, cap.last_input_ids[0] == video_pad_id, :]
        T_v, H_v, W_v, _, _ = derive_grid_via_processor(
            prompt=prompt,
            src_pil_full=src_pil_full,
            processor=pipe.mllm.processor,
            num_video_queries=num_video_queries,
            max_pixels_per_frame=max_pixels_per_frame,
        )
        if T_v * H_v * W_v != visual_tokens.shape[1]:
            raise RuntimeError(
                f"grid {(T_v, H_v, W_v)} prod={T_v * H_v * W_v} "
                f"!= visual_tokens={visual_tokens.shape[1]}"
            )
        grids.append((T_v, H_v, W_v))

        if getattr(mask_head, "use_prompt_ctx", False):
            mask_logits = mask_head(
                visual_tokens.to(dtype=mh_dtype, device=device),
                T_v,
                H_v,
                W_v,
                ctx_features=context.to(dtype=mh_dtype, device=device),
            )
        else:
            mask_logits = mask_head(visual_tokens.to(dtype=mh_dtype, device=device), T_v, H_v, W_v)
        mask_pg = postprocess_mask(
            mask_logits,
            mode=mask_postprocess,
            threshold=mask_threshold,
            floor=mask_floor,
            dilate_px=mask_dilate_px,
        ).detach()
        masks.append(mask_pg[0].to(device=device))

    if any(grid != grids[0] for grid in grids):
        raise RuntimeError(f"per-instruction grids differ: {grids}")

    return torch.stack(contexts, dim=1), torch.stack(masks, dim=0), grids[0]


MASK_COLORS = np.asarray(
    [
        (255, 64, 64),
        (64, 192, 255),
        (64, 255, 128),
        (255, 208, 64),
        (192, 96, 255),
        (255, 128, 64),
        (96, 128, 255),
        (255, 96, 192),
    ],
    dtype=np.float32,
)


def overlay_mask_red(src_pil: Image.Image, mask01_np: np.ndarray, alpha: float = 0.5) -> Image.Image:
    if mask01_np.shape[:2] != (src_pil.height, src_pil.width):
        mask_pil = Image.fromarray((mask01_np * 255).clip(0, 255).astype(np.uint8), mode="L").resize(
            src_pil.size, Image.BILINEAR
        )
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0
    else:
        mask = mask01_np.astype(np.float32)
    src = np.asarray(src_pil.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(src)
    red[..., 0] = 255.0
    blend = (alpha * mask)[..., None]
    out = src * (1.0 - blend) + red * blend
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")


def overlay_masks_colored(src_pil: Image.Image, masks01_np: np.ndarray, alpha: float = 0.55) -> Image.Image:
    src = np.asarray(src_pil.convert("RGB"), dtype=np.float32)
    masks = np.asarray(masks01_np, dtype=np.float32)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[-2:] != (src_pil.height, src_pil.width):
        resized = []
        for mask in masks:
            mask_pil = Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8), mode="L").resize(
                src_pil.size, Image.BILINEAR
            )
            resized.append(np.asarray(mask_pil, dtype=np.float32) / 255.0)
        masks = np.stack(resized, axis=0)
    masks = masks.clip(0.0, 1.0)

    colors = MASK_COLORS[np.arange(masks.shape[0]) % len(MASK_COLORS)]
    weights = masks[..., None]
    denom = weights.sum(axis=0).clip(min=1e-6)
    color_mix = (weights * colors[:, None, None, :]).sum(axis=0) / denom
    opacity = alpha * masks.max(axis=0)[..., None]
    out = src * (1.0 - opacity) + color_mix * opacity
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")


def _mask_to_video_grid(stage1_masks: torch.Tensor, src_pil_full: list[Image.Image]) -> np.ndarray:
    t_dit = len(src_pil_full)
    H = src_pil_full[0].height
    W = src_pil_full[0].width
    return F.interpolate(
        stage1_masks.cpu().float().unsqueeze(0),
        size=(t_dit, H, W),
        mode="trilinear",
        align_corners=False,
    )[0].clamp(0, 1).numpy().astype(np.float32)


def save_pred_instruction_mask_videos(
    src_pil_full: list[Image.Image],
    stage1_masks: torch.Tensor,
    output_dir: str,
    base_name: str,
    fps: float,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    mask_sim = _mask_to_video_grid(stage1_masks, src_pil_full)
    mask_paths = []
    for k in range(stage1_masks.shape[0]):
        frames = [overlay_mask_red(src_pil_full[i].convert("RGB"), mask_sim[k, i]) for i in range(len(src_pil_full))]
        out_path = os.path.join(output_dir, f"{base_name}_mask{k}.mp4")
        save_video(frames, out_path, fps=fps if fps > 0 else 15, quality=6)
        mask_paths.append(out_path)
    return mask_paths


def save_pred_side_by_side(
    src_pil_full: list[Image.Image],
    stage1_masks: torch.Tensor,
    video,
    out_path: str,
    fps: float,
):
    t_dit = min(len(src_pil_full), len(video))
    mask_sim = _mask_to_video_grid(stage1_masks, src_pil_full)
    frames = []
    for fi in range(t_dit):
        src = src_pil_full[fi].convert("RGB")
        over = overlay_masks_colored(src, mask_sim[:, fi])
        gen = video[fi].convert("RGB") if hasattr(video[fi], "convert") else video[fi]
        if gen.size != src.size:
            gen = gen.resize(src.size, Image.LANCZOS)
        canvas = Image.new("RGB", (src.width * 3, src.height))
        canvas.paste(src, (0, 0))
        canvas.paste(over, (src.width, 0))
        canvas.paste(gen, (src.width * 2, 0))
        frames.append(canvas)
    save_video(frames, out_path, fps=fps if fps > 0 else 15, quality=6)


def sanitize_name(text: str, max_len: int = 60) -> str:
    keep = []
    for ch in str(text)[:max_len]:
        keep.append(ch if ch.isalnum() or ch in " _-." else "_")
    return "".join(keep).strip().replace(" ", "_") or "sample"


def make_eval_base_name(idx: int, prompts: list[str], src_video_path: str) -> str:
    src_stem = Path(src_video_path).stem
    return f"{idx}_K{len(prompts)}_{sanitize_name(prompts[0])}_{src_stem}"
