"""Multi-process residual-attention composite inference with predicted masks.

For each composite sample, this script:
  1. builds a global prompt by joining all instructions with a separator,
  2. runs MLLM once for the global prompt to get the DiT main context,
  3. runs MLLM once per instruction and predicts one mask per instruction,
  4. injects local contexts + predicted masks through residual attention,
  5. runs video generation and saves [source | masks | generated].
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from diffsynth import save_video
from diffsynth.models.mask_residual_attention_composite.dataset import ComposeDataset
from diffsynth.models.mask_residual_attention_composite.patcher import set_residual_attention_from_masks
from diffsynth.models.mask_residual_attention_composite.predmask import (
    postprocess_mask,
    run_mllm_with_mask_features,
)
from diffsynth.trainers.unified_dataset import ImageCropAndResize

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Silence tqdm progress bars from VAE encode/decode and save_video/save_frames.
import diffsynth.models.wan_video_vae as _vae_mod
import diffsynth.data.video as _video_mod
_vae_mod.tqdm = lambda x, **kw: x
_video_mod.tqdm = lambda x, **kw: x

NON_PIPE_PREFIXES = (
    "mask_head.",
    "mask_config.",
    "mask_predictor_config.",
    "residual_attention_module.",
    "residual_attention_config.",
)


# -----------------------------------------------------------------------------
# Args / distributed helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_json_path", default=None, help="Composite JSON path.")
    p.add_argument("--eval_dataset_file", default=None, help="Alias/fallback for --eval_json_path.")
    p.add_argument("--data_root", default="")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--composite_checkpoint", required=True, help="Checkpoint from residual attention composite training.")
    p.add_argument("--mask_head_checkpoint", default=None, help="Optional override mask head checkpoint. If unset, mask_head.* is loaded from --composite_checkpoint.")

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

    p.add_argument("--num_image_queries", type=int, default=384)
    p.add_argument("--num_video_queries", type=int, default=768)
    p.add_argument("--num_ref_queries", type=int, default=1152)
    p.add_argument("--max_object_token", type=int, default=768)
    p.add_argument("--mllm_max_frame", type=int, default=10)
    p.add_argument("--mllm_max_pixels_per_frame", type=int, default=262144)
    p.add_argument("--num_video_queries_compose", type=int, default=768)
    p.add_argument("--mllm_max_pixels_per_frame_compose", type=int, default=262144)
    p.add_argument("--src_max_frames", type=int, default=49)
    p.add_argument("--eval_max_frame", type=int, default=49)
    p.add_argument("--eval_max_pixels", type=int, default=921600)
    p.add_argument("--extra_inputs", default="source_input")

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--tile_size_h", type=int, default=30)
    p.add_argument("--tile_size_w", type=int, default=52)
    p.add_argument("--tile_stride_h", type=int, default=15)
    p.add_argument("--tile_stride_w", type=int, default=26)

    p.add_argument("--residual_alpha_init", type=float, default=0.1)
    p.add_argument("--learnable_residual_alpha", action="store_true", default=False)
    p.add_argument("--residual_mode", default="replace_delta", choices=["replace_delta", "additive"])
    p.add_argument("--residual_interpolate_mode", default="trilinear")
    p.add_argument("--global_prompt_format", default="concat", choices=["concat", "numbered"])
    p.add_argument("--global_prompt_separator", default=", ")
    p.add_argument("--global_prompt_prefix", default="")

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

    p.add_argument("--max_cases", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--show_progress", action="store_true", default=False)
    p.add_argument("--s3_upload_path", default=None, help="Optional s3://... prefix. Rank0 uploads output_dir after all ranks finish.")
    return p.parse_args()


def get_rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    return rank, world, local_rank


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket_and_key = s3_uri[5:]
    parts = bucket_and_key.split("/", 1)
    return parts[0], parts[1].rstrip("/") if len(parts) > 1 else ""


def resolve_checkpoint_path(path: str, cache_dir: str = "/host-ssd") -> str:
    if path.startswith("s3://"):
        import subprocess
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, os.path.basename(path))
        if not os.path.isfile(local_path) or os.path.getsize(local_path) == 0:
            subprocess.run(["aws", "s3", "cp", path, local_path], check=True)
        return local_path
    return path


def upload_dir_to_s3_and_remove(local_dir: str, s3_base_path: str) -> bool:
    import boto3
    import shutil

    bucket, prefix = parse_s3_uri(s3_base_path)
    local_dir_path = Path(local_dir)
    client = boto3.client("s3")
    ok = True
    for path in local_dir_path.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir_path).as_posix()
        s3_key = f"{prefix}/{rel}" if prefix else rel
        try:
            client.upload_file(str(path), bucket, s3_key)
            print(f"[s3] uploaded: {path} -> s3://{bucket}/{s3_key}")
        except Exception as exc:
            ok = False
            print(f"[s3][ERROR] upload failed for {path}: {exc}")
    if ok:
        shutil.rmtree(local_dir, ignore_errors=True)
        print(f"[s3] uploaded and removed local dir: {local_dir}")
    return ok


# -----------------------------------------------------------------------------
# Data helpers / visualization
# -----------------------------------------------------------------------------

def resolve_path(path: str, data_root: str) -> str:
    p = Path(path)
    if not p.is_absolute() and data_root:
        p = Path(data_root) / p
    return str(p)


def normalize_instruction(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def load_composite_items(path: str, data_root: str) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        raw_items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_items.append(json.loads(line))
    elif suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = [data]
        else:
            raise ValueError(f"unexpected JSON top-level type {type(data).__name__}: {path}")
    elif suffix in {".yaml", ".yml"}:
        raw_items = yaml.safe_load(p.read_text())
    elif p.suffix.lower() == ".csv":
        with p.open("r", newline="", encoding="utf-8") as f:
            raw_items = list(csv.DictReader(f))
    else:
        raise ValueError(f"unsupported eval file type: {path}")

    items = []
    for raw in raw_items:
        raw_id = raw.get("id")
        try:
            row = ComposeDataset._normalize_item(ComposeDataset, raw)
        except Exception:
            prompts = normalize_instruction(raw.get("instruction", raw.get("prompt", raw.get("prompts"))))
            src = raw.get("source_video_path", raw.get("src_video", raw.get("original_video", "")))
            if not prompts or not src:
                continue
            ops = raw.get("instruction_operation") or [""] * len(prompts)
            objs = raw.get("instruction_object") or [""] * len(prompts)
            if not isinstance(ops, list):
                ops = [ops] * len(prompts)
            if not isinstance(objs, list):
                objs = [objs] * len(prompts)
            row = {
                "id": raw_id,
                "src_video": str(src),
                "tgt_video": str(raw.get("edited_video_path", raw.get("tgt_video", ""))),
                "prompts": prompts,
                "mask_video_paths": normalize_instruction(raw.get("instruction_mask_video_paths", [])),
                "combined_mask_video_path": str(raw.get("combined_mask_video_path", "")),
                "instruction_operation": [str(x) for x in (ops + [""] * len(prompts))[:len(prompts)]],
                "instruction_object": [str(x) for x in (objs + [""] * len(prompts))[:len(prompts)]],
                "edit_type": str(raw.get("edit_type", "")),
                "original_video": str(raw.get("original_video", "")),
            }
        row.setdefault("id", raw_id)
        row.setdefault("edit_type", str(raw.get("edit_type", "")))
        row.setdefault("original_video", str(raw.get("original_video", "")))
        row["src_video"] = resolve_path(row["src_video"], data_root)
        if row.get("tgt_video"):
            row["tgt_video"] = resolve_path(row["tgt_video"], data_root)
        items.append(row)
    return items


def quantize_4kp1(n: int) -> int:
    return n - ((n - 1) % 4)


def read_src_pil_resized(src_path: str, t_target: int, max_pixels: int):
    reader = imageio.get_reader(src_path)
    fps = reader.get_meta_data().get("fps", 15)
    frames = []
    for i, frame in enumerate(reader):
        if i >= t_target:
            break
        frames.append(np.asarray(frame))
    reader.close()
    if not frames:
        raise RuntimeError(f"No frames read from {src_path}")
    arr = np.stack(frames, axis=0)
    resizer = ImageCropAndResize(
        height=None,
        width=None,
        max_pixels=max_pixels,
        height_division_factor=32,
        width_division_factor=32,
        min_pixels=None,
    )
    pil = [resizer(Image.fromarray(arr[i])) for i in range(arr.shape[0])]
    return pil, pil[0].size[1], pil[0].size[0], fps


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


def overlay_masks_colored(src_pil: Image.Image, masks01_np: np.ndarray, alpha: float = 0.55) -> Image.Image:
    src = np.asarray(src_pil.convert("RGB"), dtype=np.float32)
    masks = np.asarray(masks01_np, dtype=np.float32)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[-2:] != (src_pil.height, src_pil.width):
        resized = []
        for mask in masks:
            mask_pil = Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8), mode="L").resize(src_pil.size, Image.BILINEAR)
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


def sanitize_name(text: str, max_len: int = 60) -> str:
    keep = []
    for ch in str(text)[:max_len]:
        keep.append(ch if ch.isalnum() or ch in " _-." else "_")
    return "".join(keep).strip().replace(" ", "_") or "sample"


def scalar_int_any(sd: dict[str, torch.Tensor], keys: list[str], default: int) -> int:
    for key in keys:
        if key in sd:
            return int(sd[key].item())
    return int(default)


def scalar_float_any(sd: dict[str, torch.Tensor], keys: list[str], default: float) -> float:
    for key in keys:
        if key in sd:
            return float(sd[key].item())
    return float(default)


def sub_state_dict(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def build_residual_inference_module(args, device: torch.device):
    from diffsynth.trainers.modules import WanResidualAttentionPredMaskTrainingModule

    composite_checkpoint = resolve_checkpoint_path(args.composite_checkpoint)
    composite_sd = load_safetensors(composite_checkpoint)
    residual_alpha_init = args.residual_alpha_init
    if residual_alpha_init is None:
        residual_alpha_init = scalar_float_any(composite_sd, ["residual_attention_config.alpha_init"], 1.0)

    mask_args = argparse.Namespace(**vars(args))
    if mask_args.mask_head_checkpoint is None:
        mask_args.mask_head_checkpoint = composite_checkpoint

    print("[build] constructing WanResidualAttentionPredMaskTrainingModule (checkpoint=None) ...")
    trainer = WanResidualAttentionPredMaskTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        local_model_path=args.local_model_path,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=None,
        freeze_lora_base_model=args.freeze_lora_base_model,
        save_frozen_lora=False,
        dit_lora_base_model=args.dit_lora_base_model,
        dit_lora_target_modules=args.dit_lora_target_modules,
        dit_lora_rank=args.dit_lora_rank,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        checkpoint=None,
        mllm_model=args.mllm_model,
        num_image_queries=args.num_image_queries,
        num_video_queries=args.num_video_queries,
        num_ref_queries=args.num_ref_queries,
        max_object_token=args.max_object_token,
        mllm_max_frame=args.mllm_max_frame,
        mllm_max_pixels_per_frame=args.mllm_max_pixels_per_frame,
        mllm_gradient_checkpointing=False,
        skip_load_weights=False,
        mask_args=mask_args,
        residual_alpha_init=residual_alpha_init,
        learnable_residual_alpha=args.learnable_residual_alpha,
        residual_mode=args.residual_mode,
        residual_interpolate_mode=args.residual_interpolate_mode,
        global_prompt_format=args.global_prompt_format,
        global_prompt_separator=args.global_prompt_separator,
        global_prompt_prefix=args.global_prompt_prefix,
        lambda_mask=0.0,
        mask_loss_bce_weight=1.0,
        mask_loss_dice_weight=1.0,
        detach_pred_masks_for_diffusion=True,
        pred_num_video_queries=args.num_video_queries_compose,
        pred_max_pixels_per_frame=args.mllm_max_pixels_per_frame_compose,
    )

    pipe_sd = {k: v for k, v in composite_sd.items() if not k.startswith(NON_PIPE_PREFIXES)}
    model_sd = trainer.pipe.state_dict()
    skipped_shape = []
    filtered_pipe_sd = {}
    for k, v in pipe_sd.items():
        if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape):
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered_pipe_sd[k] = v
    if skipped_shape:
        print(f"[build] skipped shape-mismatched pipe keys: {len(skipped_shape)}")
        for k, src_shape, dst_shape in skipped_shape[:20]:
            print(f"[build]   skip {k}: ckpt={src_shape} model={dst_shape}")
    res_pipe = trainer.pipe.load_state_dict(filtered_pipe_sd, strict=False)
    print(f"[build] pipe.load_state_dict: keys={len(filtered_pipe_sd)} missing={len(res_pipe.missing_keys)} unexpected={len(res_pipe.unexpected_keys)}")

    residual_sd = sub_state_dict(composite_sd, "residual_attention_module.")
    if residual_sd:
        with torch.no_grad():
            for i, layer in enumerate(trainer.residual_module._wrapped_layers):
                key = f"layers.{i}.residual_alpha"
                if key in residual_sd:
                    layer.residual_alpha.copy_(residual_sd[key].to(device=layer.residual_alpha.device, dtype=layer.residual_alpha.dtype))
        print(f"[build] residual_attention_module loaded: keys={len(residual_sd)}")

    trainer.eval()
    for param in trainer.parameters():
        param.requires_grad_(False)

    pipe = trainer.pipe
    pipe.to(device)
    if pipe.vae is not None:
        pipe.vae.to(device).eval()
    pipe.device = device
    for name in ("vae_condition", "ref_vae_condition", "source_incontext_condition"):
        mod = getattr(pipe, name, None)
        if mod is not None:
            mod.to(dtype=torch.bfloat16)
    trainer.mask_head.to(device=device, dtype=torch.bfloat16).eval()
    return trainer, pipe, trainer.residual_module, trainer.mask_head


# -----------------------------------------------------------------------------
# Residual predmask inference
# -----------------------------------------------------------------------------

@torch.no_grad()
def predict_residual_contexts_and_masks(pipe, mask_head, trainer, prompts: list[str], src_pil_full: list[Image.Image], args, device: torch.device):
    global_prompt = trainer.build_global_prompt(prompts)
    pipe.load_models_to_device(("mllm",))
    global_context = pipe.mllm(global_prompt, src_video=src_pil_full, ref_image=None).detach().to(device=device)

    local_contexts = []
    masks = []
    grids = []
    mh_dtype = next(mask_head.parameters()).dtype
    for prompt in prompts:
        context, visual_tokens, grid = run_mllm_with_mask_features(
            pipe=pipe,
            prompt=prompt,
            src_pil_full=src_pil_full,
            ref_image=None,
            num_video_queries=args.num_video_queries_compose,
            max_pixels_per_frame=args.mllm_max_pixels_per_frame_compose,
        )
        context = context.detach().to(device=device)
        local_contexts.append(context)
        grids.append(grid)
        T_v, H_v, W_v = grid
        visual_for_mask = visual_tokens.detach().to(device=device, dtype=mh_dtype)
        if getattr(mask_head, "use_prompt_ctx", False):
            mask_logits = mask_head(
                visual_for_mask,
                T_v,
                H_v,
                W_v,
                ctx_features=context.detach().to(device=device, dtype=mh_dtype),
            )
        else:
            mask_logits = mask_head(visual_for_mask, T_v, H_v, W_v)
        mask_pg = postprocess_mask(
            mask_logits,
            mode=args.mask_postprocess,
            threshold=args.mask_threshold,
            floor=args.mask_floor,
            dilate_px=args.mask_dilate_px,
        ).detach()
        masks.append(mask_pg[0].to(device=device))

    if any(grid != grids[0] for grid in grids):
        raise RuntimeError(f"per-instruction grids differ: {grids}")
    return global_prompt, global_context, torch.stack(local_contexts, dim=1), torch.stack(masks, dim=0), grids[0]


@torch.no_grad()
def run_residual_with_pred_masks(
    pipe,
    residual_module,
    global_prompt: str,
    global_context: torch.Tensor,
    local_contexts: torch.Tensor,
    stage1_masks: torch.Tensor,
    src_video: list[Image.Image],
    args,
    device: torch.device,
):
    from diffsynth.pipelines.wan_video_mllm import WanVideoUnit_MLLMEmbedder

    H = src_video[0].size[1]
    W = src_video[0].size[0]
    num_frames = len(src_video)
    pipe.scheduler.set_timesteps(args.num_inference_steps, training=False, shift=args.sigma_shift)

    inputs_posi = {"prompt": global_prompt, "context": global_context}
    inputs_nega = {"negative_prompt": ""}
    inputs_shared = {
        "prompt": global_prompt,
        "input_video": None,
        "denoising_strength": 1.0,
        "src_video": src_video,
        "source_input": src_video,
        "ref_image": None,
        "seed": args.seed,
        "rand_device": "cpu",
        "height": H,
        "width": W,
        "num_frames": num_frames,
        "cfg_scale": args.cfg_scale,
        "cfg_merge": False,
        "sigma_shift": args.sigma_shift,
        "tiled": True,
        "tile_size": (args.tile_size_h, args.tile_size_w),
        "tile_stride": (args.tile_stride_h, args.tile_stride_w),
        "sliding_window_size": None,
        "sliding_window_stride": None,
    }

    for unit in pipe.units:
        if isinstance(unit, WanVideoUnit_MLLMEmbedder):
            continue
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    inputs_shared.pop("prompt", None)

    latent = inputs_shared["latents"]
    ps = pipe.dit.patch_size
    latent_thw = (latent.shape[-3] // ps[0], latent.shape[-2] // ps[1], latent.shape[-1] // ps[2])
    set_residual_attention_from_masks(
        residual_module,
        local_contexts=local_contexts,
        stage1_masks_pg=stage1_masks,
        latent_thw=latent_thw,
        interpolate_mode=args.residual_interpolate_mode,
        device=latent.device,
        dtype=latent.dtype,
    )

    try:
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        timesteps_iter = pipe.scheduler.timesteps
        if args.show_progress:
            from tqdm import tqdm
            timesteps_iter = tqdm(timesteps_iter, desc="DiT denoising")
        for progress_id, timestep in enumerate(timesteps_iter):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=timestep,
                scheduler=pipe.scheduler,
            )
            inputs_shared["latents"] = pipe.scheduler.step(
                noise_pred,
                pipe.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
        for unit in pipe.post_units:
            inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
        pipe.load_models_to_device(["vae"])
        video = pipe.vae.decode(
            inputs_shared["latents"],
            device=pipe.device,
            tiled=True,
            tile_size=(args.tile_size_h, args.tile_size_w),
            tile_stride=(args.tile_stride_h, args.tile_stride_w),
        )
        video = pipe.vae_output_to_video(video)
        pipe.load_models_to_device([])
        return video
    finally:
        residual_module.clear()


def save_side_by_side(src_pil_full: list[Image.Image], stage1_masks: torch.Tensor, video, out_path: str, fps: float):
    t_dit = min(len(src_pil_full), len(video))
    H = src_pil_full[0].height
    W = src_pil_full[0].width
    mask_sim = F.interpolate(
        stage1_masks.cpu().float().unsqueeze(0),
        size=(len(src_pil_full), H, W),
        mode="trilinear",
        align_corners=False,
    )[0].clamp(0, 1).numpy().astype(np.float32)

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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    eval_path = args.eval_json_path or args.eval_dataset_file
    if not eval_path:
        raise ValueError("provide --eval_json_path or --eval_dataset_file")

    rank, world, local_rank = get_rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    items = load_composite_items(eval_path, args.data_root)
    if args.max_cases is not None:
        items = items[:args.max_cases]
    local_items = [(i, item) for i, item in enumerate(items) if i % world == rank]
    print(f"[rank {rank}/{world} local_rank={local_rank}] total={len(items)} local={len(local_items)} device={device}")

    trainer, pipe, residual_module, mask_head = build_residual_inference_module(args, device)
    meta_log = []

    for local_i, (idx, item) in enumerate(local_items):
        t0 = time.time()
        prompts = item["prompts"]
        src_path = item["src_video"]
        out_name = f"{idx:04d}_K{len(prompts)}_{sanitize_name(prompts[0])}_{Path(src_path).stem}.mp4"
        out_path = os.path.join(args.output_dir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[rank {rank}] skip existing {out_path}")
            continue
        print(f"[rank {rank}] ({local_i + 1}/{len(local_items)}) idx={idx} K={len(prompts)} {prompts[0][:80]}")

        try:
            t_target = min(args.eval_max_frame, args.src_max_frames)
            src_pil_full, _, _, src_fps = read_src_pil_resized(src_path, t_target, args.eval_max_pixels)
            t_dit = quantize_4kp1(min(len(src_pil_full), args.src_max_frames))
            if t_dit < 5:
                print(f"[rank {rank}] skip short video idx={idx}: t_dit={t_dit}")
                continue
            src_pil_full = src_pil_full[:t_dit]

            global_prompt, global_context, local_contexts, stage1_masks, grid = predict_residual_contexts_and_masks(
                pipe,
                mask_head,
                trainer,
                prompts,
                src_pil_full,
                args,
                device,
            )
            video = run_residual_with_pred_masks(
                pipe,
                residual_module,
                global_prompt,
                global_context,
                local_contexts,
                stage1_masks,
                src_pil_full,
                args,
                device,
            )
            save_side_by_side(src_pil_full, stage1_masks, video, out_path, src_fps)

            # Save standalone target video named {id}.mp4 for evaluation
            item_id = item.get("id", idx)
            tgt_dir = os.path.join(args.output_dir, "tgt_videos")
            os.makedirs(tgt_dir, exist_ok=True)
            tgt_path = os.path.join(tgt_dir, f"{item_id}.mp4")
            tgt_frames = [f.convert("RGB") if hasattr(f, "convert") else f for f in video]
            save_video(tgt_frames, tgt_path, fps=src_fps if src_fps > 0 else 15, quality=9)

            mask_means = [float(stage1_masks[k].detach().float().mean().cpu().item()) for k in range(stage1_masks.shape[0])]
            mask_colors = [[int(x) for x in MASK_COLORS[k % len(MASK_COLORS)].tolist()] for k in range(stage1_masks.shape[0])]
            stats = residual_module.stats()
            meta = {
                "idx": idx,
                "rank": rank,
                "src": src_path,
                "prompts": prompts,
                "global_prompt": global_prompt,
                "instruction_operation": item.get("instruction_operation", []),
                "instruction_object": item.get("instruction_object", []),
                "K": len(prompts),
                "edit_type": item.get("edit_type", ""),
                "original_video": item.get("original_video", ""),
                "grid_thw": list(grid),
                "mask_means": mask_means,
                "mask_colors_rgb": mask_colors,
                "residual_stats": stats,
                "residual_mode": args.residual_mode,
                "global_prompt_separator": args.global_prompt_separator,
                "out": out_path,
                "tgt_video": tgt_path,
                "time": time.time() - t0,
            }
            meta_log.append(meta)
            print(f"[rank {rank}] saved {out_path} mask_mean={np.mean(mask_means):.3f} time={meta['time']:.2f}s")
        except Exception as exc:
            import traceback
            print(f"[rank {rank}] FAILED idx={idx}: {exc}")
            traceback.print_exc()
            try:
                residual_module.clear()
            except Exception:
                pass
            torch.cuda.empty_cache()

    meta_path = os.path.join(args.output_dir, f"meta_rank{rank}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_log, f, indent=2, ensure_ascii=False)
    print(f"[rank {rank}] done -> {meta_path}")

    if world > 1 and torch.distributed.is_available():
        try:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="gloo")
            torch.distributed.barrier()
        except Exception as exc:
            print(f"[rank {rank}] distributed barrier skipped/failed: {exc}")
    if args.s3_upload_path and rank == 0:
        upload_dir_to_s3_and_remove(args.output_dir, args.s3_upload_path)


if __name__ == "__main__":
    main()
