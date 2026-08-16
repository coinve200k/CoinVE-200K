"""Single-video residual-attention composite inference.

Usage:
    python infer_coinve_single.py \
        --src_video /path/to/src.mp4 \
        --prompts "make the sky red" "turn the car into a truck" \
        --composite_checkpoint /path/to/step-8000.safetensors \
        --output_dir /path/to/out

This is a thin wrapper around the building blocks in ``infer_coinve_multigpu``
that drops the multi-process / JSONL batch machinery and only runs one video
on one GPU. All model-shape / mask-head / tiling defaults are kept the same
as the batch script; only the frequently-changed knobs are exposed in the
companion shell wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse every helper from the bench inference script.
from infer_coinve_bench import (
    build_residual_inference_module,
    overlay_masks_colored,
    predict_residual_contexts_and_masks,
    quantize_4kp1,
    read_src_pil_resized,
    run_residual_with_pred_masks,
    save_side_by_side,
    sanitize_name,
)

from diffsynth import save_video


def save_side_by_side(src_pil_full: list[Image.Image], stage1_masks: torch.Tensor, video, out_path: str, fps: float):
    """Save a two-panel video: [source | edited]."""
    t_dit = min(len(src_pil_full), len(video))
    frames = []
    for fi in range(t_dit):
        src = src_pil_full[fi].convert("RGB")
        gen = video[fi].convert("RGB") if hasattr(video[fi], "convert") else video[fi]
        if gen.size != src.size:
            gen = gen.resize(src.size, Image.LANCZOS)
        canvas = Image.new("RGB", (src.width * 2, src.height))
        canvas.paste(src, (0, 0))
        canvas.paste(gen, (src.width, 0))
        frames.append(canvas)
    save_video(frames, out_path, fps=fps if fps > 0 else 15, quality=6)


def overlay_mask_red(src_pil: Image.Image, mask01_np: np.ndarray, alpha: float = 0.5) -> Image.Image:
    if mask01_np.shape[:2] != (src_pil.height, src_pil.width):
        mask_pil = Image.fromarray((mask01_np * 255).clip(0, 255).astype(np.uint8), mode="L").resize(src_pil.size, Image.BILINEAR)
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0
    else:
        mask = mask01_np.astype(np.float32)
    src = np.asarray(src_pil.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(src)
    red[..., 0] = 255.0
    blend = (alpha * mask)[..., None]
    out = src * (1.0 - blend) + red * blend
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")


def save_instruction_mask_videos(src_pil_full: list[Image.Image], stage1_masks: torch.Tensor, output_dir: str, base_name: str, fps: float) -> list[str]:
    from diffsynth import save_video
    os.makedirs(output_dir, exist_ok=True)
    t_dit = len(src_pil_full)
    H = src_pil_full[0].height
    W = src_pil_full[0].width
    mask_paths = []
    for k in range(stage1_masks.shape[0]):
        mask_sim = F.interpolate(
            stage1_masks[k:k + 1].cpu().float().unsqueeze(0),
            size=(t_dit, H, W),
            mode="trilinear",
            align_corners=False,
        )[0, 0].clamp(0, 1).numpy().astype(np.float32)
        frames = [overlay_mask_red(src_pil_full[i].convert("RGB"), mask_sim[i]) for i in range(t_dit)]
        out_path = os.path.join(output_dir, f"{base_name}_mask{k}.mp4")
        save_video(frames, out_path, fps=fps if fps > 0 else 15, quality=6)
        mask_paths.append(out_path)
    return mask_paths

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Silence tqdm in VAE encode/decode and save_video.
import diffsynth.models.wan_video_vae as _vae_mod
import diffsynth.data.video as _video_mod
_vae_mod.tqdm = lambda x, **kw: x
_video_mod.tqdm = lambda x, **kw: x


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-video CoinVE composite editing inference.")

    # --- frequently-changed knobs (exposed in the sh wrapper) ---
    p.add_argument("--src_video", required=True, help="Path to the source video.")
    p.add_argument("--prompts", nargs="+", required=True, help="One or more edit instructions.")
    p.add_argument("--composite_checkpoint", required=True, help="residual-attention composite checkpoint (.safetensors).")
    p.add_argument("--output_dir", required=True, help="Where to write the output video.")
    p.add_argument("--output_name", default=None, help="Output filename (without extension). Defaults to <src_stem>_edited.mp4.")
    p.add_argument("--eval_max_pixels", type=int, default=921600, help="Max pixels per frame for the source video resize.")
    p.add_argument("--eval_max_frame", type=int, default=49, help="Max number of frames to read from the source video.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--save_instruction_masks", action="store_true", default=False)
    p.add_argument("--no_side_by_side", action="store_true", default=False, help="Save only the edited video without src|mask|edit triptych.")
    p.add_argument("--show_progress", action="store_true", default=False)

    # --- model loading (hidden defaults, same as multigpu) ---
    p.add_argument("--model_paths", default=None)
    p.add_argument("--model_id_with_origin_paths", default="Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth")
    p.add_argument("--local_model_path", default="/host-ssd")
    p.add_argument("--audio_processor_config", default=None)
    p.add_argument("--mllm_model", default="/host-ssd/Qwen3-VL-8B-Instruct")
    p.add_argument("--trainable_models", default="mllm.image_queries,mllm.video_queries,mllm.connector,vae_condition")
    p.add_argument("--lora_base_model", default="mllm.model.model.language_model")
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj")
    p.add_argument("--lora_rank", type=int, default=256)
    p.add_argument("--freeze_lora_base_model", action="store_true", default=True)
    p.add_argument("--dit_lora_base_model", default="dit")
    p.add_argument("--dit_lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--dit_lora_rank", type=int, default=128)

    # --- mllm / mask-head shape (hidden defaults) ---
    p.add_argument("--num_image_queries", type=int, default=384)
    p.add_argument("--num_video_queries", type=int, default=768)
    p.add_argument("--num_ref_queries", type=int, default=1152)
    p.add_argument("--max_object_token", type=int, default=768)
    p.add_argument("--mllm_max_frame", type=int, default=10)
    p.add_argument("--mllm_max_pixels_per_frame", type=int, default=262144)
    p.add_argument("--num_video_queries_compose", type=int, default=768)
    p.add_argument("--mllm_max_pixels_per_frame_compose", type=int, default=262144)
    p.add_argument("--src_max_frames", type=int, default=49)
    p.add_argument("--extra_inputs", default="source_input")

    # --- generation (hidden defaults) ---
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--tile_size_h", type=int, default=30)
    p.add_argument("--tile_size_w", type=int, default=52)
    p.add_argument("--tile_stride_h", type=int, default=15)
    p.add_argument("--tile_stride_w", type=int, default=26)

    # --- residual attention (hidden defaults; alpha overrides ckpt if set) ---
    p.add_argument("--residual_alpha_init", type=float, default=0.1)
    p.add_argument("--learnable_residual_alpha", action="store_true", default=False)
    p.add_argument("--residual_mode", default="replace_delta", choices=["replace_delta", "additive"])
    p.add_argument("--residual_interpolate_mode", default="trilinear")
    p.add_argument("--global_prompt_format", default="concat", choices=["concat", "numbered"])
    p.add_argument("--global_prompt_separator", default=", ")
    p.add_argument("--global_prompt_prefix", default="")

    # --- mask head (hidden defaults) ---
    p.add_argument("--mask_head_checkpoint", default=None)
    p.add_argument("--mask_postprocess", default="sigmoid", choices=["sigmoid", "binary", "none"])
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--mask_floor", type=float, default=0.0)
    p.add_argument("--mask_dilate_px", type=int, default=0)
    p.add_argument("--mask_hidden", type=int, default=256)
    p.add_argument("--mask_layers", type=int, default=2)
    p.add_argument("--mask_heads", type=int, default=8)
    p.add_argument("--mask_num_query", type=int, default=4)
    p.add_argument("--mask_no_grid_pe", action="store_true", default=False)
    p.add_argument("--mask_max_t_v", type=int, default=64)
    p.add_argument("--mask_max_h_v", type=int, default=64)
    p.add_argument("--mask_max_w_v", type=int, default=64)
    p.add_argument("--mask_use_prompt_ctx", action="store_true", default=True)
    p.add_argument("--mask_no_prompt_ctx", dest="mask_use_prompt_ctx", action="store_false")
    p.add_argument("--mask_ctx_dim", type=int, default=5120)
    p.add_argument("--mask_train_parts", default="none")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.src_video):
        raise FileNotFoundError(args.src_video)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    prompts = [str(p).strip() for p in args.prompts if str(p).strip()]
    if not prompts:
        raise ValueError("at least one --prompts instruction is required")

    src_stem = Path(args.src_video).stem
    out_name = args.output_name or f"{src_stem}_edited"
    if not out_name.endswith(".mp4"):
        out_name += ".mp4"
    out_path = os.path.join(args.output_dir, out_name)

    print(f"[single] src={args.src_video}")
    print(f"[single] prompts ({len(prompts)}): {prompts}")
    print(f"[single] ckpt={args.composite_checkpoint}")
    print(f"[single] out={out_path}")
    print(f"[single] eval_max_pixels={args.eval_max_pixels} max_frames={args.eval_max_frame} seed={args.seed}")

    trainer, pipe, residual_module, mask_head = build_residual_inference_module(args, device)

    t0 = time.time()
    try:
        t_target = min(args.eval_max_frame, args.src_max_frames)
        src_pil_full, _, _, src_fps = read_src_pil_resized(args.src_video, t_target, args.eval_max_pixels)
        t_dit = quantize_4kp1(min(len(src_pil_full), args.src_max_frames))
        if t_dit < 5:
            raise RuntimeError(f"source video too short after quantization: t_dit={t_dit}")
        src_pil_full = src_pil_full[:t_dit]
        print(f"[single] loaded {len(src_pil_full)} frames @ {src_fps:.1f} fps, frame size={src_pil_full[0].size}")

        global_prompt, global_context, local_contexts, stage1_masks, grid = predict_residual_contexts_and_masks(
            pipe, mask_head, trainer, prompts, src_pil_full, args, device,
        )
        video = run_residual_with_pred_masks(
            pipe, residual_module, global_prompt, global_context,
            local_contexts, stage1_masks, src_pil_full, args, device,
        )

        if args.no_side_by_side:
            from diffsynth import save_video
            edited_frames = [f.convert("RGB") if hasattr(f, "convert") else f for f in video]
            save_video(edited_frames, out_path, fps=src_fps if src_fps > 0 else 15, quality=6)
        else:
            save_side_by_side(src_pil_full, stage1_masks, video, out_path, src_fps)

        mask_paths = []
        if args.save_instruction_masks:
            mask_paths = save_instruction_mask_videos(
                src_pil_full, stage1_masks,
                os.path.join(args.output_dir, "instruction_masks"),
                Path(out_name).stem, src_fps,
            )

        mask_means = [float(stage1_masks[k].detach().float().mean().cpu().item()) for k in range(stage1_masks.shape[0])]
        dt = time.time() - t0
        print(f"[single] saved {out_path} mask_mean={np.mean(mask_means):.3f} time={dt:.2f}s")

        meta = {
            "src": args.src_video,
            "prompts": prompts,
            "global_prompt": global_prompt,
            "K": len(prompts),
            "grid_thw": list(grid),
            "mask_means": mask_means,
            "residual_mode": args.residual_mode,
            "eval_max_pixels": args.eval_max_pixels,
            "eval_max_frame": args.eval_max_frame,
            "seed": args.seed,
            "out": out_path,
            "mask_videos": mask_paths,
            "time": dt,
        }
        meta_path = os.path.join(args.output_dir, f"{Path(out_name).stem}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[single] meta -> {meta_path}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            residual_module.clear()
        except Exception:
            pass
        torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    main()
