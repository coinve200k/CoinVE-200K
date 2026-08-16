"""Stage-2 mask-bias inference building blocks.

Three things this module provides:

1. `build_inference_module(args, device)`:
     Constructs a `WanComposeTrainingModule` with `checkpoint=None`, then
     manually loads the merged stage-2 + mask-attn-insert ckpt with
     strict=False. The module's ckpt is self-contained (after the white-list
     fix in trainers/utils.py and the manual vae_condition merge into
     step-4000), so a single `--checkpoint` is enough.

     Returns (trainer, pipe, bias_module, mask_head).

2. `CaptureLastHidden`:
     Context manager that monkey-patches `pipe.mllm.model.forward` to record
     `output.hidden_states[-1]` and the `input_ids` of the last call. After
     `pipe.mllm(prompt, src_video=...)` finishes, `cap.last_hidden` and
     `cap.last_input_ids` hold the values needed to extract visual tokens
     at `<|video_pad|>` positions for MaskHead.

3. `run_pipe_skip_mllm`:
     A trimmed copy of `WanVideoPipeline.__call__` that skips the
     `WanVideoUnit_MLLMEmbedder` step and uses a precomputed `context` we
     already have from the captured mllm forward. Single mllm call total.

Plus two small helpers:
   - `derive_grid_via_processor`: rebuild (T_v, H_v, W_v) from the same
     processor inputs, mirrors ComposeCollator.
   - `postprocess_mask`: sigmoid / floor / dilate / binary on the patch-grid
     mask logits before feeding to set_bias_from_stage1_mask.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as load_safetensors

# Add repo root to path so this module can be imported by `infer_mask_bias.py`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffsynth.models.mask_head.model import MaskHead
from diffsynth.models.qwen_vl_utils import process_vision_info
from .utils import (
    FRAME_FACTOR,
    SPATIAL_MERGE_SIZE,
)


# -----------------------------------------------------------------------------
# 1. Builder: trainer + pipe + bias_module + mask_head, all on `device`.
# -----------------------------------------------------------------------------

def build_inference_module(args, device: torch.device):
    """Construct WanComposeTrainingModule with checkpoint=None, manually load
    the merged ckpt with strict=False, then load the MaskHead.

    Why this dance:
      - `WanComposeTrainingModule.__init__` runs the parent's
        `switch_pipe_to_training_mode` which loads `checkpoint` via
        `pipe.load_state_dict(strict=False)` BEFORE `inject_mask_bias_into_dit`
        wraps cross_attn into `MaskedCrossAttention(orig=...)`. That means
        any `cross_attn.orig.X.lora_*` keys in step-4000 would not match the
        still-bare cross_attn submodules at parent-load time and would be
        silently dropped.
      - So we pass `checkpoint=None` to skip that load, let the subclass
        inject the wrapper, THEN load the ckpt manually so wrapper-inside
        keys (`cross_attn.orig.*`) match.
    """
    # Lazy import: train_mask_attn_insert.py imports a bunch of training
    # machinery (accelerate, deepspeed, wandb), but it does define
    # `WanComposeTrainingModule` at module import time, so this works.
    from train_mask_attn_insert import WanComposeTrainingModule

    print("[build] constructing WanComposeTrainingModule (checkpoint=None) ...")
    trainer = WanComposeTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=getattr(args, "audio_processor_config", None),
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=None,
        freeze_lora_base_model=getattr(args, "freeze_lora_base_model", False),
        save_frozen_lora=False,
        dit_lora_base_model=args.dit_lora_base_model,
        dit_lora_target_modules=args.dit_lora_target_modules,
        dit_lora_rank=args.dit_lora_rank,
        use_gradient_checkpointing_offload=False,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        checkpoint=None,                     # ← critical: skip parent's load
        mllm_model=args.mllm_model,
        num_image_queries=args.num_image_queries,
        num_video_queries=args.num_video_queries,
        num_ref_queries=args.num_ref_queries,
        max_object_token=args.max_object_token,
        mllm_max_frame=args.mllm_max_frame,
        mllm_max_pixels_per_frame=args.mllm_max_pixels_per_frame,
        mllm_gradient_checkpointing=False,
        ref_pad_first=getattr(args, "ref_pad_first", False),
        skip_load_weights=False,
        alpha_init=args.alpha_init,
        learnable_alpha=False,
    )

    # Manual ckpt load — wrapper is now in place, `cross_attn.orig.*` keys match.
    print(f"[build] loading merged ckpt: {args.checkpoint}")
    sd = load_safetensors(args.checkpoint)
    # ckpt keys are saved without "pipe." prefix (remove_prefix_in_ckpt="pipe."
    # at training time); pipe.load_state_dict() expects the same.
    pipe = trainer.pipe
    res = pipe.load_state_dict(sd, strict=False)
    print(f"[build] pipe.load_state_dict: missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}")
    if res.unexpected_keys:
        print(f"[build] first 5 unexpected: {res.unexpected_keys[:5]}")
    # Sanity: after loading, the wrapped cross_attn should have non-init LoRA.
    # We don't hard-assert here; printing missing_keys is enough for inspection.

    # Freeze everything (inference only).
    trainer.eval()
    for p in trainer.parameters():
        p.requires_grad_(False)

    # Move to device.
    pipe.to(device)
    # WanVideoPipeline keeps `vae` outside parameters; ensure it's on device too.
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.to(device)
        pipe.vae.eval()
    pipe.device = device

    # Force bf16 dtype on conditional embedders. The patch_embedding nn.Conv3d
    # can end up with fp32 bias even when ckpt holds bf16, because the module
    # was constructed before dtype propagation and `load_state_dict` doesn't
    # cast. Without this, model_fn_wan_video crashes with
    # "Input type (BFloat16) and bias type (float) should be the same".
    for _name in ("vae_condition", "ref_vae_condition", "source_incontext_condition"):
        _mod = getattr(pipe, _name, None)
        if _mod is not None:
            _mod.to(dtype=torch.bfloat16)

    # MaskHead: independent module, separate ckpt.
    print(f"[build] constructing MaskHead and loading {args.mask_head_ckpt}")
    vlm_dim = pipe.mllm.model.config.text_config.hidden_size
    use_prompt_ctx = getattr(args, "mask_use_prompt_ctx", False)
    ctx_dim = getattr(args, "mask_ctx_dim", 5120)
    mask_head = MaskHead(
        vlm_dim=vlm_dim,
        hidden=args.mask_hidden,
        num_layers=args.mask_layers,
        num_heads=args.mask_heads,
        num_query=args.mask_num_query,
        use_grid_pe=not args.mask_no_grid_pe,
        max_t_v=args.mask_max_t_v,
        max_h_v=args.mask_max_h_v,
        max_w_v=args.mask_max_w_v,
        use_prompt_ctx=use_prompt_ctx,
        ctx_dim=ctx_dim,
    )
    mh_sd = load_safetensors(args.mask_head_ckpt)
    res_mh = mask_head.load_state_dict(mh_sd, strict=True)
    print(f"[build] MaskHead loaded ({len(mh_sd)} keys)")
    mask_head.to(device=device, dtype=torch.bfloat16)
    mask_head.eval()
    for p in mask_head.parameters():
        p.requires_grad_(False)

    return trainer, pipe, trainer.bias_module, mask_head


# -----------------------------------------------------------------------------
# 2. CaptureLastHidden context manager.
# -----------------------------------------------------------------------------

class CaptureLastHidden:
    """Monkey-patch `inner_model.forward` to record hidden_states[-1] and the
    input_ids tensor on the next forward.

    Use as:
        cap = CaptureLastHidden(pipe.mllm.model)
        with cap:
            context = pipe.mllm(prompt, src_video=src_video_pil)
        # cap.last_hidden : [1, L, D_vlm]
        # cap.last_input_ids : [1, L]
    """

    def __init__(self, inner_model: torch.nn.Module) -> None:
        self.m = inner_model
        self.last_hidden: Optional[torch.Tensor] = None
        self.last_input_ids: Optional[torch.Tensor] = None
        self._orig_forward: Optional[Callable] = None

    def __enter__(self):
        outer = self
        orig = self.m.forward
        self._orig_forward = orig

        def wrapped_forward(*args, **kwargs):
            # Force hidden states out regardless of caller.
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            out = orig(*args, **kwargs)
            # Record on every call; last call wins (we expect exactly one).
            outer.last_hidden = out.hidden_states[-1]
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            outer.last_input_ids = input_ids
            return out

        self.m.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_forward is not None:
            self.m.forward = self._orig_forward
        return False


# -----------------------------------------------------------------------------
# Helper: derive (T_v, H_v, W_v) and h_resized/w_resized from processor.
# -----------------------------------------------------------------------------

def derive_grid_via_processor(
    *,
    prompt: str,
    src_pil_full: list,
    processor: Any,
    num_video_queries: int,
    max_pixels_per_frame: Optional[int] = 262144,
    system_prompt: str = (
        "You will be given an image and instruction. Please describe the content "
        "of the image in detail based on instruction in your own words."
    ),
) -> tuple[int, int, int, int, int]:
    """Return (T_v, H_v, W_v, h_resized, w_resized) by replicating exactly
    what mllm_encoder.QwenVLMLLMEncoder.forward does (the video branch):

      video_data = {"type": "video", "video": src_pil_full,
                    "max_frames": <ignored by qwen_vl_utils for list inputs>,
                    "max_pixels": ...}
      → process_vision_info sees a *list* of frames; it samples to
        FPS_MAX_FRAMES=16 if len > 16, else uses len as-is.
        (qwen_vl_utils.py:421-424; the `max_frames` ele key is unused on
        this list-input branch.)

    Critically: we feed the FULL src_pil_full (T_dit frames) — same as
    mllm_encoder does — so that processor's internal sampling matches the
    mllm forward we already ran. Pre-sampling here would produce a
    different (T_v, H_v, W_v) than the mllm actually used, leading to
    `T_v*H_v*W_v != N_v` mismatch.
    """
    QUERY_TOKEN = "<|object_ref_start|>"
    video_data = {"type": "video", "video": src_pil_full}
    if max_pixels_per_frame is not None:
        video_data["max_pixels"] = max_pixels_per_frame
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                video_data,
                {"type": "text", "text": prompt},
                {"type": "text", "text": QUERY_TOKEN * num_video_queries},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # CRITICAL: replicate mllm_encoder.QwenVLMLLMEncoder.forward (line 166-173)
    # exactly — that code does NOT pass `return_video_kwargs=True` and does NOT
    # forward `video_kwargs` to the processor. Without those kwargs the
    # processor's `do_sample_frames` defaults to True, so it re-samples the
    # already-prepared frame tensor (using fps=2 / Qwen3-VL defaults), which
    # collapses 16 frames -> ~4. With them, do_sample_frames=False and
    # T_v matches the original sampled count. We must match the encoder's
    # behavior so visual_tokens.shape[1] == T_v * H_v * W_v.
    image_inputs, video_inputs = process_vision_info(messages)
    vid_tensor = video_inputs[0]
    n_sampled, _, h_resized, w_resized = vid_tensor.shape
    bf = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    t_pre, h_pre, w_pre = bf.video_grid_thw[0].tolist()
    T_v = t_pre
    H_v = h_pre // SPATIAL_MERGE_SIZE
    W_v = w_pre // SPATIAL_MERGE_SIZE
    return T_v, H_v, W_v, h_resized, w_resized


# -----------------------------------------------------------------------------
# Helper: postprocess MaskHead logits -> [N, T_v, H_v, W_v] in [0, 1].
# -----------------------------------------------------------------------------

def postprocess_mask(
    logits: torch.Tensor,
    *,
    mode: str = "sigmoid",
    threshold: float = 0.5,
    floor: float = 0.0,
    dilate_px: int = 0,
) -> torch.Tensor:
    """logits: [B, T_v, H_v, W_v] (raw from MaskHead).
    Returns [B, T_v, H_v, W_v] in [0, 1].

    mode: 'sigmoid' (soft), 'binary' (hard 0/1 above threshold), 'none'
          (raw logits — only useful if you want log directly; not recommended
          since set_bias_from_stage1_mask expects [0,1]).
    """
    if mode == "sigmoid":
        m = logits.float().sigmoid()
    elif mode == "binary":
        m = (logits.float().sigmoid() > threshold).float()
    elif mode == "none":
        m = logits.float()
    else:
        raise ValueError(f"unknown mask_postprocess mode: {mode}")

    if dilate_px > 0:
        # 2D dilation per (B, T_v) slice via max_pool2d.
        B, T_v, H_v, W_v = m.shape
        k = 2 * dilate_px + 1
        flat = m.reshape(B * T_v, 1, H_v, W_v)
        flat = F.max_pool2d(flat, kernel_size=k, stride=1, padding=dilate_px)
        m = flat.reshape(B, T_v, H_v, W_v)

    if floor > 0:
        m = m.clamp(min=floor, max=1.0)
    else:
        m = m.clamp(min=0.0, max=1.0)
    return m


# -----------------------------------------------------------------------------
# 3. Pipe runner that skips MLLMEmbedder.
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_pipe_skip_mllm(
    pipe,
    *,
    prompt: str,
    src_video,
    ref_image,
    num_frames: int,
    height: int,
    width: int,
    precomputed_context: torch.Tensor,
    num_inference_steps: int = 50,
    seed: int = 0,
    cfg_scale: float = 1.0,
    sigma_shift: float = 5.0,
    tiled: bool = True,
    tile_size: tuple = (30, 52),
    tile_stride: tuple = (15, 26),
    show_progress: bool = False,
):
    """Trimmed copy of `WanVideoPipeline.__call__` that skips MLLMEmbedder.

    Mirrors wan_video_mllm.py:230-311. Differences:
      - No CFG (cfg_scale=1.0), no negative prompt: simpler.
      - Skips `WanVideoUnit_MLLMEmbedder`; injects `precomputed_context` into
        inputs_shared after the other units run.
      - input_video=None (we noise from scratch; src_video only goes to the
        `source_input` path which fills `vae_source_input`).
    """
    from diffsynth.pipelines.wan_video_mllm import (
        WanVideoUnit_ShapeChecker,
        WanVideoUnit_NoiseInitializer,
        WanVideoUnit_InputVideoEmbedder,
        WanVideoUnit_MLLMEmbedder,
        WanVideoUnit_CfgMerger,
    )

    pipe.scheduler.set_timesteps(num_inference_steps, training=False, shift=sigma_shift)

    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": ""}
    inputs_shared = {
        "prompt": prompt,
        "input_video": None, "denoising_strength": 1.0,
        "src_video": src_video, "source_input": src_video, "ref_image": ref_image,
        "seed": seed, "rand_device": "cpu",
        "height": height, "width": width, "num_frames": num_frames,
        "cfg_scale": cfg_scale, "cfg_merge": False, "sigma_shift": sigma_shift,
        "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        "sliding_window_size": None, "sliding_window_stride": None,
    }

    # Run all units except MLLMEmbedder.
    for unit in pipe.units:
        if isinstance(unit, WanVideoUnit_MLLMEmbedder):
            continue
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    # Inject precomputed context (where MLLMEmbedder would have put it).
    inputs_shared["context"] = precomputed_context
    inputs_shared.pop("prompt", None)

    # Denoise loop.
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    timesteps_iter = pipe.scheduler.timesteps
    if show_progress:
        from tqdm import tqdm
        timesteps_iter = tqdm(timesteps_iter, desc="DiT denoising")
    for progress_id, t in enumerate(timesteps_iter):
        t = t.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(
            **models, **inputs_shared, **inputs_posi,
            timestep=t, scheduler=pipe.scheduler,
        )
        inputs_shared["latents"] = pipe.scheduler.step(
            noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"]
        )

    # Decode.
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        inputs_shared["latents"], device=pipe.device,
        tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
    )
    video = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    return video
