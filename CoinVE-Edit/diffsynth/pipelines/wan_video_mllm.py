import torch, warnings, glob, os, types
import numpy as np
from PIL import Image, ImageOps
from einops import repeat, rearrange
from typing import Optional, Union
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import random
from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d, ConditionalEmbedder
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.mllm_encoder import QwenVLMLLMEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader
import deepspeed

DEBUG = False

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, 
            time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        # ------ mllm encoder -----------
        self.mllm: QwenVLMLLMEncoder = None
        self.vae_condition: ConditionalEmbedder = None
        self.ref_vae_condition: ConditionalEmbedder = None
        # ------ mllm encoder -----------
        self.in_iteration_models = ("dit", "vae_condition", "ref_vae_condition")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_MLLMEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video
    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        if DEBUG: print(lora.keys())
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)
        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred = self.model_fn(**inputs, timestep=timestep, scheduler=self.scheduler)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = False,
        use_usp=False,
        checkpoint: str = None,
        num_image_queries = 256,
        num_video_queries = 512,
        num_ref_queries= 768,
        max_object_token = 768,
        mllm_model: str = 'Qwen/Qwen2.5-VL-3B-Instruct',
        mllm_max_frame: int = 16,
        mllm_max_pixels_per_frame: int = 512*512,
        mllm_gradient_checkpointing: bool = False,
        ref_pad_first: bool = False,
        skip_load_weights: bool = False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        # Download and load models
        model_manager = ModelManager()
        if skip_load_weights:
            print("[skip_load_weights=True] Skipping weight loading, using random initialization for debugging.")
        else:
            # print(model_configs)
            for model_config in model_configs:
                model_config.skip_download = True
                model_config.download_if_necessary(use_usp=use_usp)
                model_manager.load_model(
                    model_config.path,
                    device=model_config.offload_device or device,
                    torch_dtype=model_config.offload_dtype or torch_dtype
                )

        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if dit is None:
            # skip_load_weights mode: read config.json from the model directory to get WanModel kwargs
            import json
            print("[skip_load_weights=True] Inferring WanModel config from config.json...")
            # Find the directory containing the DiT safetensors files
            dit_dir = None
            for mc in model_configs:
                p = mc.path if isinstance(mc.path, list) else [mc.path]
                for f in p:
                    if f and f.endswith(".safetensors"):
                        dit_dir = os.path.dirname(f)
                        break
                if dit_dir:
                    break
            config_path = os.path.join(dit_dir, "config.json") if dit_dir else None
            if config_path and os.path.exists(config_path):
                with open(config_path, "r") as fp:
                    raw_cfg = json.load(fp)
            else:
                raw_cfg = {}
            # Map config.json fields to WanModel.__init__ kwargs, filling in known defaults
            model_type = raw_cfg.get("model_type", "")
            extra_kwargs = {
                "dim":        raw_cfg.get("dim", 3072),
                "in_dim":     raw_cfg.get("in_dim", 48),
                "ffn_dim":    raw_cfg.get("ffn_dim", 14336),
                "out_dim":    raw_cfg.get("out_dim", 48),
                "text_dim":   raw_cfg.get("text_dim", 4096),
                "freq_dim":   raw_cfg.get("freq_dim", 256),
                "eps":        raw_cfg.get("eps", 1e-6),
                "patch_size": raw_cfg.get("patch_size", [1, 2, 2]),
                "num_heads":  raw_cfg.get("num_heads", 24),
                "num_layers": raw_cfg.get("num_layers", 30),
                "has_image_input": raw_cfg.get("has_image_input", False),
            }
            # ti2v specific flags
            if model_type == "ti2v":
                extra_kwargs.update({
                    "seperated_timestep": True,
                    "require_clip_embedding": False,
                    "require_vae_embedding": False,
                    "fuse_vae_embedding_in_latents": True,
                })
            print(f"[skip_load_weights=True] Initializing WanModel with config: {extra_kwargs}")
            from ..models.utils import init_weights_on_device
            with init_weights_on_device():
                dit = WanModel(**extra_kwargs)
            # Materialize meta tensors to real random weights on the target device
            dit = dit.to_empty(device=device).to(dtype=torch_dtype)
            dit.eval()
        pipe.dit = dit
        pipe.mllm = QwenVLMLLMEncoder(
                    model_path=mllm_model,
                    device=device,
                    dit_dim=dit.dim,
                    num_image_queries=num_image_queries,
                    num_video_queries=num_video_queries,
                    num_ref_queries=num_ref_queries,
                    max_object_token=max_object_token,
                    max_frames=mllm_max_frame,
                    max_pixels_per_frame=mllm_max_pixels_per_frame,
                    gradient_checkpointing=mllm_gradient_checkpointing,
                    skip_load_weights=skip_load_weights,
                    )

        pipe.vae_condition = ConditionalEmbedder(in_dim=dit.in_dim, dim=dit.dim, patch_size=dit.patch_size, zero_init=True)
        pipe.ref_vae_condition = ConditionalEmbedder(in_dim=dit.in_dim, dim=dit.dim, patch_size=dit.patch_size, zero_init=True)
        pipe.ref_vae_condition.ref_pad_first = ref_pad_first
        # Size division factor
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        if pipe.vae is None and skip_load_weights:
            from ..models.wan_video_vae import WanVideoVAE38
            print("[skip_load_weights=True] Initializing WanVideoVAE38 with random weights.")
            pipe.vae = WanVideoVAE38()
            pipe.vae = pipe.vae.to(dtype=torch_dtype, device=device)
            print("[skip_load_weights=True] vae has been initialized")
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_video: Optional[list[Image.Image]] = None,
        src_video: Optional[list[Image.Image]] = None,
        source_input: Optional[list[Image.Image]] = None,
        ref_image: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "prompt": prompt,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "src_video": src_video, "source_input": source_input, "ref_image": ref_image,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        inputs_shared.pop("prompt")
        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            # Inference
            noise_pred = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, scheduler=self.scheduler)
            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
 
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vae"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vae):
        length = (num_frames - 1) // 4 + 1
        if getattr(pipe, "vae", None):
            vae = pipe.vae
        shape = (1, vae.model.z_dim, length, height // vae.upsampling_factor, width // vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}
    

def shrink_and_pad_centered(
    item,
    target_size,
    scale=0.75,      # < 1.0 guarantees extra space
    color="white"
):
    target_w, target_h = target_size
    # 1. Fit image inside target (keeps aspect ratio)
    fitted = ImageOps.contain(item, target_size)
    w, h = fitted.size
    # 2. Shrink further
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = fitted.resize((new_w, new_h), Image.BICUBIC)
    # 3. Center placement
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2

    canvas = Image.new(item.mode, target_size, color)
    canvas.paste(resized, (left, top))

    return canvas


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "input_video", "noise", "tiled", "tile_size", "tile_stride", "num_frames", "source_input", "ref_image", "vae"),
            # onload_model_names=("vae")
        )

    def process(self, pipe: WanVideoPipeline, height, width, input_video, noise, tiled, tile_size, tile_stride, num_frames, vae, source_input=None, ref_image=None):
        if getattr(pipe, "vae", None):
            vae = pipe.vae        
        if source_input is not None:
            if DEBUG: print("add reference vae", len(source_input))
            vae_source_input = pipe.preprocess_video([source_input[i] for i in range(min(num_frames,len(source_input)))])
            if DEBUG: print(vae_source_input[0].shape, tiled, tile_size, tile_stride)
            with torch.no_grad():
                vae_source_input = vae.encode(vae_source_input, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            if DEBUG: print(vae_source_input.size())
        else:
            vae_source_input = None
        if ref_image is not None:
            vae_ref_image = []
            for item in ref_image:
                target_size = (width, height)
                if DEBUG: print("add reference image", item.size, width, height)
                if pipe.scheduler.training:
                    item = shrink_and_pad_centered(item, target_size, scale=random.uniform(0.7, 1.0))
                else:
                    print("Inference Simple Pad")
                    item = ImageOps.pad(item, target_size, color="white", centering=(0.5, 0.5))
                if DEBUG: print("Resize reference image", item.size, width, height)
                item = pipe.preprocess_video([item])
                if DEBUG: print("Reference image size", item[0].shape)
                with torch.no_grad():
                    item = vae.encode(item, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                if DEBUG: print("Reference vae size", item.size())
                vae_ref_image.append(item)
        else:
            vae_ref_image = None
        if input_video is None:
            # if DEBUG: print("skip WanVideoUnit_InputVideoEmbedder")
            return {"latents": noise, "vae_source_input": vae_source_input, "vae_ref_image": vae_ref_image}
        # pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        with torch.no_grad():
            input_latents = vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents, "vae_source_input": vae_source_input, "vae_ref_image": vae_ref_image}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "vae_source_input": vae_source_input, "vae_ref_image": vae_ref_image}


class WanVideoUnit_MLLMEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("prompt", "src_video", "ref_image"),
            onload_model_names=("mllm")
        )
    
    def process(self, pipe: WanVideoPipeline, prompt, src_video, ref_image=None):
        if src_video is None or pipe.mllm is None:
            if DEBUG: print("skip mllm", prompt, src_video)
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if len(src_video) == 1:
            if DEBUG: print(len(src_video), "runing in image", prompt)
            prompt_emb = pipe.mllm(prompt, src_image=src_video, ref_image=ref_image)
        else:
            if DEBUG: 
                if ref_image is not None:
                    print(len(src_video), "runing in video with Ref", prompt)
                else:
                    print(len(src_video), "runing in video only", prompt)
            prompt_emb = pipe.mllm(prompt, src_video=src_video, ref_image=ref_image)
        if DEBUG: print(prompt_emb.size(), prompt, prompt_emb[0][:5,:5])
        return {"context": prompt_emb}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    vae_condition: ConditionalEmbedder = None,
    ref_vae_condition: ConditionalEmbedder = None,
    vae_source_input: Optional[torch.Tensor] = None,
    vae_ref_image: Optional[torch.Tensor] = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    scheduler: FlowMatchScheduler = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents"],
            batch_size=2 if cfg_merge else 1
        )
    # Timestep
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    # Camera control
    if vae_source_input is not None:
        if DEBUG: print("source cond before:", x.shape, vae_source_input.shape)
        vae_source_input = vae_condition(vae_source_input)
        x = dit.patchify(x, control_camera_latents_input)
        sigma = scheduler.get_sigma(timestep)
        if DEBUG: print("source cond after:", x.shape, vae_source_input.shape, sigma, timestep)
        x = x + vae_source_input * sigma
    else:
        x = dit.patchify(x, control_camera_latents_input)

    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    if vae_ref_image is not None:
        if len(vae_ref_image) > 1:
            vae_ref_image = torch.concat([vae_ref_image], dim=1)
        else:
            vae_ref_image = vae_ref_image[0]
        if DEBUG: print("vae_ref_image before patch embedding:", vae_ref_image.shape)
        vae_ref_image = ref_vae_condition(vae_ref_image)
        if DEBUG: print("vae_ref_image after patch embedding:", vae_ref_image.shape)
        ref_f, ref_h, ref_w = vae_ref_image.shape[2:]
        vae_ref_image = rearrange(vae_ref_image, 'b c f h w -> b (f h w) c').contiguous()
        if DEBUG: print("vae_ref_image / freqs reshape:", vae_ref_image.shape, freqs.shape)
        f += ref_f
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        f -= ref_f
        
        if ref_vae_condition.ref_pad_first:
            if DEBUG: print("seq_cat_pad_first vae_ref_image concat / freqs: ", vae_ref_image.shape, x.shape, freqs.shape)
            x = torch.concat([vae_ref_image, x], dim=1)
        else:
            if DEBUG: print("seq_cat_pad_last vae_ref_image concat / freqs: ", vae_ref_image.shape, x.shape, freqs.shape)
            x = torch.concat([x, vae_ref_image], dim=1)
        vae_ref_input_length = vae_ref_image.shape[1]
        if DEBUG: print("vae_ref_input_length: ", vae_ref_input_length, x.shape)

    # blocks
    
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        # Block
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            # x = torch.utils.checkpoint.checkpoint(
            #     create_custom_forward(block),
            #     x, context, t_mod, freqs,
            #     use_reentrant=False,
            # )
            x = deepspeed.checkpointing.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    if vae_ref_image is not None:
        if ref_vae_condition.ref_pad_first:
            if DEBUG: print("seq_cat_pad_first remove ref latents", x.shape, vae_ref_input_length)
            x = x[:, vae_ref_input_length:, :]
            if DEBUG: print("after", x.shape)
        else:
            if DEBUG: print("seq_cat_pad_last remove ref latents", x.shape, vae_ref_input_length)
            x = x[:, :-vae_ref_input_length, :]
            if DEBUG: print("after", x.shape)
    x = dit.unpatchify(x, (f, h, w))
    return x