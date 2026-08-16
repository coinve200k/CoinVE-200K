from __future__ import annotations

from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors

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


def configure_mask_head_trainable(mask_head: MaskHead, train_parts: str) -> None:
    for param in mask_head.parameters():
        param.requires_grad_(False)
    parts = {x.strip() for x in str(train_parts or "").split(",") if x.strip()}
    if "all" in parts:
        for param in mask_head.parameters():
            param.requires_grad_(True)
        return

    prefixes: list[str] = []
    if "transformer" in parts:
        prefixes += ["blocks.", "final_q_to_v.", "norm_q_final.", "norm_v_final."]
    if "head" in parts:
        prefixes += ["mask_mlp."]
    if "query" in parts:
        prefixes += ["query_embed"]
    if "input" in parts:
        prefixes += ["proj_v.", "norm_v_in.", "proj_c.", "norm_c_in.", "type_emb_", "pe_"]

    for name, param in mask_head.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(True)


def load_mask_head_for_residual_training(
    args,
    pipe,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[MaskHead, dict]:
    mask_sd_source: dict[str, torch.Tensor] = {}
    mask_sd: dict[str, torch.Tensor] = {}
    ckpt = _get_arg(args, "mask_head_checkpoint", None)
    if ckpt:
        mask_sd_source = load_safetensors(ckpt)
        prefixed_mask_sd = sub_state_dict(mask_sd_source, "mask_head.")
        if prefixed_mask_sd:
            mask_sd = prefixed_mask_sd
        elif "proj_v.weight" in mask_sd_source and "query_embed" in mask_sd_source:
            mask_sd = {k: v for k, v in mask_sd_source.items() if not k.startswith("mask_config.")}

    default_use_prompt_ctx = True if _get_arg(args, "mask_use_prompt_ctx", None) is None else bool(args.mask_use_prompt_ctx)
    mask_config = {
        "mask_head_checkpoint": ckpt or "",
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
        "mask_train_parts": _get_arg(args, "mask_train_parts", "transformer,head"),
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
    if mask_sd:
        res = mask_head.load_state_dict(mask_sd, strict=False)
        print(
            f"[residual-predmask] mask_head loaded from {ckpt}: "
            f"keys={len(mask_sd)} missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
        )
    else:
        print("[residual-predmask] mask_head initialized from args (no checkpoint weights loaded)")
    mask_head.to(device=device, dtype=dtype)
    configure_mask_head_trainable(mask_head, mask_config["mask_train_parts"])
    n_train = sum(p.numel() for p in mask_head.parameters() if p.requires_grad)
    print(f"[residual-predmask] trainable mask_head params={n_train} parts={mask_config['mask_train_parts']}")
    return mask_head, mask_config


def run_mllm_with_mask_features(
    pipe,
    prompt: str,
    src_pil_full: list,
    ref_image=None,
    num_video_queries: int = 768,
    max_pixels_per_frame: Optional[int] = 262144,
):
    pipe.load_models_to_device(("mllm",))
    video_pad_id = pipe.mllm.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    cap = CaptureLastHidden(pipe.mllm.model)
    with cap:
        context = pipe.mllm(prompt, src_video=src_pil_full, ref_image=ref_image)
    if cap.last_hidden is None or cap.last_input_ids is None:
        raise RuntimeError("failed to capture MLLM hidden states")
    visual_tokens = cap.last_hidden[:, cap.last_input_ids[0] == video_pad_id, :].detach()
    T_v, H_v, W_v, _, _ = derive_grid_via_processor(
        prompt=prompt,
        src_pil_full=src_pil_full,
        processor=pipe.mllm.processor,
        num_video_queries=num_video_queries,
        max_pixels_per_frame=max_pixels_per_frame,
    )
    if T_v * H_v * W_v != visual_tokens.shape[1]:
        raise RuntimeError(
            f"grid {(T_v, H_v, W_v)} prod={T_v * H_v * W_v} != visual_tokens={visual_tokens.shape[1]}"
        )
    return context, visual_tokens, (T_v, H_v, W_v)


__all__ = [
    "configure_mask_head_trainable",
    "load_mask_head_for_residual_training",
    "postprocess_mask",
    "run_mllm_with_mask_features",
]
