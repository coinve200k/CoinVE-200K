"""Training modules for WanVideo composite editing.

This module centralizes the trainable module classes so that both the training
script (``train_coinve.py``) and the inference
script (``infer_coinve_multigpu.py``) can import
them without depending on a ``train.py`` entry-point file.

Contents:
    - WanTrainingModule                                : base WanVideo training module
                                                      (loads pipeline + LoRA, builds
                                                      shared inputs, runs forward).
    - _dice_loss / mask_loss                           : BCE + Dice mask losses used by
                                                      the residual predmask module.
    - WanResidualAttentionPredMaskTrainingModule       : subclass that injects residual
                                                      attention + a mask head and
                                                      combines diffusion + mask loss.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from diffsynth.pipelines.wan_video_mllm import WanVideoPipeline, DEBUG
from diffsynth.trainers.utils import DiffusionTrainingModule
from diffsynth.models.mask_residual_attention_composite.patcher import (
    inject_residual_attention_into_dit,
    set_residual_attention_from_masks,
)
from diffsynth.models.mask_residual_attention_composite.predmask import (
    load_mask_head_for_residual_training,
    postprocess_mask,
    run_mllm_with_mask_features,
)


# -----------------------------------------------------------------------------
# WanTrainingModule
# -----------------------------------------------------------------------------

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, local_model_path=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        freeze_lora_base_model=False, save_frozen_lora=False,
        dit_lora_base_model=None, dit_lora_target_modules="q,k,v,o,ffn.0,ffn.2", dit_lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        checkpoint=None,
        num_image_queries=256,
        num_video_queries=512,
        num_ref_queries=768,
        max_object_token=768,
        mllm_model='Qwen/Qwen2.5-VL-3B-Instruct',
        mllm_max_frame=16,
        mllm_max_pixels_per_frame=512*512,
        mllm_gradient_checkpointing=False,
        ref_pad_first=False,
        skip_load_weights=False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False, local_model_path=local_model_path)
        print(model_configs)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config,
            num_image_queries=num_image_queries,
            num_video_queries=num_video_queries,
            num_ref_queries=num_ref_queries,
            max_object_token=max_object_token,
            mllm_model=mllm_model,
            mllm_max_frame=mllm_max_frame,
            mllm_max_pixels_per_frame=mllm_max_pixels_per_frame,
            mllm_gradient_checkpointing=mllm_gradient_checkpointing,
            ref_pad_first=ref_pad_first,
            skip_load_weights=skip_load_weights,
        )
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            dit_lora_base_model=dit_lora_base_model, dit_lora_target_modules=dit_lora_target_modules, dit_lora_rank=dit_lora_rank,
            enable_fp8_training=False, checkpoint=checkpoint,
            freeze_lora_base_model=freeze_lora_base_model, save_frozen_lora=save_frozen_lora,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary


    def _build_inputs_shared(self, data, vae=None):
        """Build the shared inputs dict from raw data (no GPU ops)."""
        inputs_shared = {
            "prompt": data["prompt"],
            "input_video": data["tgt_video"],
            "src_video": data["src_video"],
            "height": data["tgt_video"][0].size[1],
            "width": data["tgt_video"][0].size[0],
            "num_frames": len(data["tgt_video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "vae": vae
        }
        for extra_input in self.extra_inputs:
            if extra_input == "source_input":
                inputs_shared["source_input"] = data["src_video"]
            elif extra_input == "ref_image":
                if "ref_image" in data:
                    inputs_shared["ref_image"] = data["ref_image"]
                else:
                    inputs_shared["ref_image"] = None
        return inputs_shared

    def vae_preprocess(self, data, vae=None):
        """
        Run only the frozen-model units (ShapeChecker, NoiseInitializer, InputVideoEmbedder)
        for async prefetching. Does NOT run MLLM or CfgMerger.
        Returns (inputs_shared, inputs_posi, inputs_nega) after VAE encode.
        """
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = self._build_inputs_shared(data, vae)

        # Only run VAE-related units (skip MLLMEmbedder and CfgMerger)
        from diffsynth.pipelines.wan_video_mllm import (
            WanVideoUnit_ShapeChecker, WanVideoUnit_NoiseInitializer, WanVideoUnit_InputVideoEmbedder
        )
        _vae_unit_types = (WanVideoUnit_ShapeChecker, WanVideoUnit_NoiseInitializer, WanVideoUnit_InputVideoEmbedder)
        for unit in self.pipe.units:
            if isinstance(unit, _vae_unit_types):
                inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return inputs_shared, inputs_posi, inputs_nega

    def forward_preprocess(self, data, vae=None, prefetched_vae=None):
        """
        Full preprocess. If prefetched_vae is provided (from async prefetch),
        skip VAE units and only run MLLM + CfgMerger.
        """
        from diffsynth.pipelines.wan_video_mllm import (
            WanVideoUnit_ShapeChecker, WanVideoUnit_NoiseInitializer, WanVideoUnit_InputVideoEmbedder
        )
        _vae_unit_types = (WanVideoUnit_ShapeChecker, WanVideoUnit_NoiseInitializer, WanVideoUnit_InputVideoEmbedder)

        if prefetched_vae is not None:
            inputs_shared, inputs_posi, inputs_nega = prefetched_vae
        else:
            inputs_posi = {"prompt": data["prompt"]}
            inputs_nega = {}
            inputs_shared = self._build_inputs_shared(data, vae)

        # Run all units, but skip VAE units if already prefetched
        for unit in self.pipe.units:
            if prefetched_vae is not None and isinstance(unit, _vae_unit_types):
                continue  # already done in prefetch
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None, prefetched_vae=None, vae=None):
        if DEBUG: print("WanTrainingModule Raw Input", data.keys())
        if inputs is None: inputs = self.forward_preprocess(data, vae, prefetched_vae=prefetched_vae)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


# -----------------------------------------------------------------------------
# Mask loss
# -----------------------------------------------------------------------------

def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = logits.sigmoid().flatten(1)
    tgt = target.float().flatten(1)
    num = 2.0 * (pred * tgt).sum(dim=1) + eps
    den = pred.sum(dim=1) + tgt.sum(dim=1) + eps
    return (1.0 - num / den).mean()


def mask_loss(logits: torch.Tensor, target: torch.Tensor, bce_w: float = 1.0, dice_w: float = 1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    dice = _dice_loss(logits, target)
    return bce_w * bce + dice_w * dice, bce.detach(), dice.detach()


# -----------------------------------------------------------------------------
# WanResidualAttentionPredMaskTrainingModule
# -----------------------------------------------------------------------------

class WanResidualAttentionPredMaskTrainingModule(WanTrainingModule):
    def __init__(
        self,
        *args,
        mask_args=None,
        residual_alpha_init: float = 1.0,
        learnable_residual_alpha: bool = False,
        residual_mode: str = "replace_delta",
        residual_interpolate_mode: str = "trilinear",
        global_prompt_format: str = "concat",
        global_prompt_separator: str = ", ",
        global_prompt_prefix: str = "",
        lambda_mask: float = 1.0,
        mask_loss_bce_weight: float = 1.0,
        mask_loss_dice_weight: float = 1.0,
        detach_pred_masks_for_diffusion: bool = True,
        pred_num_video_queries: int = 768,
        pred_max_pixels_per_frame: Optional[int] = 262144,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.residual_module = inject_residual_attention_into_dit(
            self.pipe.dit,
            residual_alpha_init=residual_alpha_init,
            learnable_residual_alpha=learnable_residual_alpha,
            residual_mode=residual_mode,
        )
        self._residual_alpha_init = float(residual_alpha_init)
        self._learnable_residual_alpha = bool(learnable_residual_alpha)
        self._residual_mode = residual_mode
        self._residual_interpolate_mode = residual_interpolate_mode
        self._global_prompt_format = global_prompt_format
        self._global_prompt_separator = global_prompt_separator
        self._global_prompt_prefix = global_prompt_prefix
        self._lambda_mask = float(lambda_mask)
        self._mask_loss_bce_weight = float(mask_loss_bce_weight)
        self._mask_loss_dice_weight = float(mask_loss_dice_weight)
        self._detach_pred_masks_for_diffusion = bool(detach_pred_masks_for_diffusion)
        self._pred_num_video_queries = int(pred_num_video_queries)
        self._pred_max_pixels_per_frame = pred_max_pixels_per_frame

        if mask_args is None:
            raise ValueError("mask_args is required")
        self.mask_head, self.mask_predictor_config = load_mask_head_for_residual_training(
            mask_args, self.pipe, device=torch.device("cpu"), dtype=torch.bfloat16,
        )

        self._last_diff_loss = None
        self._last_mask_loss = None
        self._last_mask_bce = None
        self._last_mask_dice = None
        self._last_num_instructions = None

    def clear_runtime_slots(self):
        self.residual_module.clear()

    def build_global_prompt(self, prompts: list[str]) -> str:
        if self._global_prompt_format == "concat":
            body = self._global_prompt_separator.join(prompts)
        elif self._global_prompt_format == "numbered":
            body = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(prompts))
        else:
            raise ValueError(f"unknown global_prompt_format={self._global_prompt_format}")
        return f"{self._global_prompt_prefix}{body}" if self._global_prompt_prefix else body

    def _encode_context_and_mask_logits(self, prompt: str, data, mh_dtype: torch.dtype):
        ctx, visual_tokens, grid = run_mllm_with_mask_features(
            pipe=self.pipe,
            prompt=prompt,
            src_pil_full=data["src_video"],
            ref_image=data.get("ref_image", None),
            num_video_queries=self._pred_num_video_queries,
            max_pixels_per_frame=self._pred_max_pixels_per_frame,
        )
        T_v, H_v, W_v = grid
        visual_for_mask = visual_tokens.detach().to(device=ctx.device, dtype=mh_dtype)
        if getattr(self.mask_head, "use_prompt_ctx", False):
            ctx_for_mask = ctx.detach().to(device=ctx.device, dtype=mh_dtype)
            logits = self.mask_head(visual_for_mask, T_v, H_v, W_v, ctx_features=ctx_for_mask)
        else:
            logits = self.mask_head(visual_for_mask, T_v, H_v, W_v)
        return ctx, logits, grid

    def _match_mask_target_shape(self, target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if target.shape == logits.shape:
            return target
        target_5d = target.unsqueeze(1).float()
        resized = F.interpolate(target_5d, size=logits.shape[-3:], mode="trilinear", align_corners=False)
        return resized.squeeze(1)

    def forward(self, data, prefetched_vae=None, vae=None):
        if prefetched_vae is not None:
            inputs_shared, inputs_posi, _inputs_nega = prefetched_vae
        else:
            inputs_shared, inputs_posi, _inputs_nega = self.vae_preprocess(data, vae)
        inputs = {**inputs_shared, **inputs_posi}

        if "noise" in inputs and isinstance(inputs["noise"], torch.Tensor):
            inputs["noise"] = inputs["noise"].detach().requires_grad_(True)

        prompts = data.get("prompts", None)
        if prompts is None:
            prompts = [data["prompt"]]
        elif isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(p) for p in prompts]
        K = len(prompts)
        self._last_num_instructions = K

        latent = inputs["input_latents"]
        ps = self.pipe.dit.patch_size
        latent_thw = (
            latent.shape[-3] // ps[0],
            latent.shape[-2] // ps[1],
            latent.shape[-1] // ps[2],
        )

        mh_dtype = next(self.mask_head.parameters()).dtype
        global_prompt = self.build_global_prompt(prompts)
        self.pipe.load_models_to_device(("mllm",))
        global_ctx = self.pipe.mllm(
            global_prompt,
            src_video=data["src_video"],
            ref_image=data.get("ref_image", None),
        )
        inputs["context"] = global_ctx

        local_contexts = []
        mask_logits = []
        pred_masks = []
        first_grid = None
        for prompt in prompts:
            ctx_i, logits_i, grid_i = self._encode_context_and_mask_logits(prompt, data, mh_dtype)
            if first_grid is None:
                first_grid = grid_i
            elif grid_i != first_grid:
                raise RuntimeError(f"per-instruction grids differ: {first_grid} vs {grid_i}")
            local_contexts.append(ctx_i)
            mask_logits.append(logits_i[0])
            logits_for_attn = logits_i.detach() if self._detach_pred_masks_for_diffusion else logits_i
            mask_pg = postprocess_mask(
                logits_for_attn,
                mode=self.mask_predictor_config.get("mask_postprocess", "sigmoid"),
                threshold=float(self.mask_predictor_config.get("mask_threshold", 0.5)),
                floor=float(self.mask_predictor_config.get("mask_floor", 0.0)),
                dilate_px=int(self.mask_predictor_config.get("mask_dilate_px", 0)),
            )
            pred_masks.append(mask_pg[0].to(device=latent.device, dtype=latent.dtype))

        local_contexts_t = torch.stack(local_contexts, dim=1)
        pred_masks_t = torch.stack(pred_masks, dim=0)
        mask_logits_t = torch.stack(mask_logits, dim=0)

        target_masks = data["stage1_masks"].to(device=mask_logits_t.device, dtype=mask_logits_t.dtype)
        target_masks = self._match_mask_target_shape(target_masks, mask_logits_t)
        m_loss, m_bce, m_dice = mask_loss(
            mask_logits_t,
            target_masks,
            bce_w=self._mask_loss_bce_weight,
            dice_w=self._mask_loss_dice_weight,
        )

        set_residual_attention_from_masks(
            self.residual_module,
            local_contexts=local_contexts_t,
            stage1_masks_pg=pred_masks_t,
            latent_thw=latent_thw,
            interpolate_mode=self._residual_interpolate_mode,
            device=latent.device,
            dtype=latent.dtype,
        )

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        diff_loss = self.pipe.training_loss(**models, **inputs)
        if diff_loss.dim() != 0:
            diff_loss = diff_loss.mean()
        total_loss = diff_loss + self._lambda_mask * m_loss

        self._last_diff_loss = diff_loss.detach()
        self._last_mask_loss = m_loss.detach()
        self._last_mask_bce = m_bce.detach()
        self._last_mask_dice = m_dice.detach()
        return total_loss

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        full_state_dict = state_dict
        state_dict = super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        state_dict.update({
            f"residual_attention_module.{k}": v.detach().cpu()
            for k, v in self.residual_module.export_state_dict().items()
        })
        state_dict.update({
            k: v.detach().cpu()
            for k, v in full_state_dict.items()
            if k.startswith("mask_head.")
        })
        cfg = self.mask_predictor_config or {}
        int_keys = {
            "mask_hidden": "hidden",
            "mask_layers": "layers",
            "mask_heads": "heads",
            "mask_num_query": "num_query",
            "mask_no_grid_pe": "no_grid_pe",
            "mask_max_t_v": "max_t_v",
            "mask_max_h_v": "max_h_v",
            "mask_max_w_v": "max_w_v",
            "mask_use_prompt_ctx": "use_prompt_ctx",
            "mask_ctx_dim": "ctx_dim",
        }
        for key, out_key in int_keys.items():
            if key in cfg:
                state_dict[f"mask_predictor_config.{out_key}"] = torch.tensor(int(cfg[key]), dtype=torch.int64)
        state_dict["residual_attention_config.alpha_init"] = torch.tensor(self._residual_alpha_init, dtype=torch.float32)
        state_dict["residual_attention_config.learnable_alpha"] = torch.tensor(int(self._learnable_residual_alpha), dtype=torch.int64)
        state_dict["residual_attention_config.detach_pred_masks_for_diffusion"] = torch.tensor(int(self._detach_pred_masks_for_diffusion), dtype=torch.int64)
        return state_dict
