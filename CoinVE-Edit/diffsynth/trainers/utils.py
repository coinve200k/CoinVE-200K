import imageio, os, torch, warnings, torchvision, argparse, json, time
import threading
from queue import Queue
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DeepSpeedPlugin

DEBUG = False  

class ImageDataset(torch.utils.data.Dataset):    
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat


def rgetattr(obj, attr):
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj

def rsetattr(obj, attr, value):
    parts = attr.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        # Optionally also save frozen lora_base_model params (lora_A/lora_B) even if not trained
        extra_save_names = set()
        if getattr(self, '_save_frozen_lora', False) and getattr(self, '_lora_base_model', None):
            lora_prefix = self._lora_base_model.replace('.', '/')
            for name in state_dict:
                # match pipe.<lora_base_model> prefix, keep lora_A/lora_B keys
                name_as_path = name.replace('.', '/')
                if lora_prefix in name_as_path and ('lora_A' in name or 'lora_B' in name):
                    extra_save_names.add(name)
        # Always save these modules regardless of trainable status
        _query_substrings = ('mllm.image_queries', 'mllm.video_queries', 'mllm.ref_queries',
                             'mllm.connector', 'mllm.ref_connector',
                             'vae_condition.', 'ref_vae_condition.',
                             'source_incontext_condition.')
        for name in state_dict:
            if any(qs in name for qs in _query_substrings):
                extra_save_names.add(name)
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names or name in extra_save_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False, local_model_path=None):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype, local_model_path=local_model_path) for i in model_id_with_origin_paths]
        return model_configs
    
    def mapping_mix_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, 
        lora_target_modules, 
        lora_rank,
        dit_lora_base_model=None, 
        dit_lora_target_modules=None, 
        dit_lora_rank=None,
        lora_checkpoint=None,
        enable_fp8_training=False,
        checkpoint: str = None,
        freeze_lora_base_model=False,
        save_frozen_lora=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                rgetattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            rsetattr(pipe, lora_base_model, model)
            if freeze_lora_base_model:
                rgetattr(pipe, lora_base_model).requires_grad_(False)
        self._save_frozen_lora = save_frozen_lora
        self._lora_base_model = lora_base_model
        if dit_lora_base_model is not None:
            model = self.add_lora_to_model(
                rgetattr(pipe, dit_lora_base_model),
                target_modules=dit_lora_target_modules.split(","),
                lora_rank=dit_lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            rsetattr(pipe, dit_lora_base_model, model)
            # inject_adapter_in_model resets requires_grad=False on all non-LoRA params.
            # Selectively re-enable any modules listed in trainable_models with a "*."
            # suffix wildcard, without calling freeze_except (which would re-freeze the
            # newly added LoRA params).
            if trainable_models is not None:
                # suffix_patterns: e.g. trainable_models entries like "*.foo" → ".foo"
                # named_parameters() returns "...foo.weight" / "...foo.bias",
                # so check the module portion (everything before the last ".weight"/".bias")
                _suffix_patterns = [m[1:] for m in trainable_models.split(",") if m.startswith("*.")]
                print(f"[restore_after_lora] suffix_patterns={_suffix_patterns}")
                _restored = 0
                for name, param in pipe.named_parameters():
                    # strip leaf param name (.weight/.bias) to get module name
                    module_name = name.rsplit(".", 1)[0] if "." in name else name
                    if any(module_name.endswith(s) for s in _suffix_patterns):
                        param.requires_grad_(True)
                        _restored += 1
                        if _restored <= 5:
                            print(f"[restore_after_lora] re-enabled grad: {name}")
                print(f"[restore_after_lora] total restored: {_restored}")
        if checkpoint is not None:
            state_dict = load_state_dict(checkpoint, torch_dtype=torch.bfloat16, device=pipe.mllm.model.device)
            if lora_base_model is not None:
                state_dict = self.mapping_mix_lora_state_dict(state_dict)
            
            new_state_dict = {}
            _model_sd = pipe.state_dict()
            for k, v in state_dict.items():
                if 'ref_queries' in k and k in _model_sd and v.shape != _model_sd[k].shape:
                    print(f"[load_ckpt] Skipped '{k}': ckpt shape {v.shape} != model shape {_model_sd[k].shape}")
                    continue
                new_state_dict[k] = v
            res = pipe.load_state_dict(new_state_dict, strict=False)
            if DEBUG: print(new_state_dict.keys())
            print(f"mllm model loaded: {checkpoint}, total {len(state_dict)} keys, {len(new_state_dict)} keys loaded, missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
            # del state_dict, new_state_dict

        # Initialize source_incontext_condition.patch_embedding from DiT's patch_embedding
        # if not already loaded from checkpoint (i.e. training from scratch).
        _dit_pe_sd = pipe.dit.patch_embedding.state_dict()
        _ce = getattr(pipe, "source_incontext_condition", None)
        if _ce is not None and _ce.patch_embedding.weight.shape == _dit_pe_sd["weight"].shape:
            _has_ckpt_weights = False
            if checkpoint is not None:
                _prefix = "source_incontext_condition.patch_embedding."
                _has_ckpt_weights = any(k.startswith(_prefix) for k in new_state_dict)
            if not _has_ckpt_weights:
                _ce.patch_embedding.load_state_dict(_dit_pe_sd)
                print(f"[init] source_incontext_condition.patch_embedding initialized from dit.patch_embedding (no ckpt weights found)")


class EMA:
    """Exponential Moving Average of trainable model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # Only track trainable parameters, store shadow in fp32 to avoid bf16 precision loss
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.float().clone()
        print(f"[EMA] Initialized with decay={decay}, tracking {len(self.shadow)} parameters (fp32 shadow)")

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters with exponential moving average."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data.float(), alpha=1 - self.decay)

    def apply(self, model):
        """Swap model params with EMA shadow params (for eval/save)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original params after eval/save."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.optimizer = None
        self.scheduler = None
        self.ema = None

    def set_optimizer_scheduler(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler


    def on_step_end(self, accelerator, model, save_steps=None, eval_steps=None, eval_fn=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            # Also save EMA checkpoint
            if self.ema is not None:
                self.save_ema_model(accelerator, model, f"step-{self.num_steps}_ema.safetensors")
    # Run evaluation after saving checkpoint
        if eval_steps is not None and eval_fn is not None and self.num_steps % eval_steps == 0:
            # Use EMA weights for evaluation if available
            if self.ema is not None:
                self.ema.apply(accelerator.unwrap_model(model))
            print(f"[Rank {accelerator.process_index}] [Step {self.num_steps}] Entering eval_fn...")
            eval_fn(accelerator, model, self.num_steps, self.output_path)
            print(f"[Rank {accelerator.process_index}] [Step {self.num_steps}] eval_fn done.")
            if self.ema is not None:
                self.ema.restore(accelerator.unwrap_model(model))


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        torch.distributed.destroy_process_group()


    def save_model(self, accelerator, model, file_name):
        # 1. Everyone must call this to participate in the ZeRO-3 gathering process
        accelerator.wait_for_everyone()
        full_state_dict = accelerator.get_state_dict(model)
        # 2. Only the main process handles the disk I/O and transformation
        if accelerator.is_main_process:
            # Use the gathered state_dict
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                full_state_dict, 
                remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            # Save the processed dict
            accelerator.save(state_dict, path, safe_serialization=True)
            del state_dict
        del full_state_dict
        # Save optimizer state — must be done outside is_main_process guard because
        # DeepSpeed ZeRO-2 shards optimizer state across all ranks. Each rank saves its
        # own shard named _optimizer_rankN.pt so it can be restored correctly on resume.
        if self.optimizer is not None:
            rank = accelerator.process_index
            # Save into a subfolder: optimizer/<step_name>/rank{rank}.pt
            step_name = file_name.replace(".safetensors", "")
            opt_dir = os.path.join(self.output_path, "optimizer", step_name)
            os.makedirs(opt_dir, exist_ok=True)
            opt_state = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "num_steps": self.num_steps,
            }
            opt_path = os.path.join(opt_dir, f"rank{rank}.pt")
            torch.save(opt_state, opt_path)
            if rank == 0:
                print(f"Optimizer state saved to: {opt_dir}/rank*.pt")
        torch.cuda.empty_cache()
        # 3. Optional: Ensure everyone waits until saving is done before moving to next iteration
        accelerator.wait_for_everyone()


    def save_ema_model(self, accelerator, model, file_name):
        """Save EMA weights as a separate checkpoint."""
        if self.ema is None:
            return
        unwrapped_model = accelerator.unwrap_model(model)
        # Apply EMA weights temporarily
        self.ema.apply(unwrapped_model)
        # Gather and save (same flow as save_model but skip optimizer)
        accelerator.wait_for_everyone()
        full_state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = unwrapped_model.export_trainable_state_dict(
                full_state_dict,
                remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            print(f"[EMA] Saved EMA checkpoint to: {path}")
            del state_dict
        del full_state_dict
        # Restore original weights
        self.ema.restore(unwrapped_model)
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


class MixDataloader:
    def __init__(self, dataloader, vid_dataloader, vid_ref_dataloader, num_epochs=1):
        self.dataloader = dataloader
        self.vid_dataloader = vid_dataloader
        self.vid_ref_dataloader = vid_ref_dataloader
        self.num_epochs = num_epochs
        max_iters = 10000000
        if dataloader:
            self.iter = iter(self.dataloader)
            print("Image dataloader", len(self.dataloader))
        else:
            print("No image dataloader is provided")
        if vid_dataloader:
            self.vid_iter = iter(self.vid_dataloader)
            print("Instuct Vid dataloader", len(self.vid_dataloader))
        else:
            print("No instruct vid dataloader is provided")
        if vid_ref_dataloader:
            print("Instuct Ref Vid dataloader", len(self.vid_ref_dataloader))
            self.vid_ref_iter = iter(self.vid_ref_dataloader)
        else:
            print("No instruct ref vid dataloader is provided")
        if dataloader and vid_dataloader and vid_ref_dataloader:
            self.sample_list = [0, 1, 2] * max_iters
            self.length = min(len(self.dataloader), len(self.vid_dataloader), len(self.vid_ref_dataloader)) * 3
        elif vid_dataloader and vid_ref_dataloader:
            self.sample_list = [1, 2] * max_iters
            self.length = min(len(self.vid_dataloader), len(self.vid_ref_dataloader)) * 2
        elif dataloader and vid_ref_dataloader:
            self.sample_list = [0, 2] * max_iters
            self.length = min(len(self.dataloader), len(self.vid_ref_dataloader)) * 2
        elif dataloader and vid_dataloader:
            self.sample_list = [0, 1] * max_iters
            self.length = min(len(self.dataloader), len(self.vid_dataloader)) * 2
        elif dataloader:
            self.sample_list = [0] * max_iters
            self.length = len(self.dataloader)
        elif vid_dataloader:
            self.sample_list = [1] * max_iters
            self.length = len(self.vid_dataloader)
        elif vid_ref_dataloader:
            self.sample_list = [2] * max_iters
            self.length = len(self.vid_ref_dataloader)
        else:
            raise ValueError("No dataloader is provided")
        print(self.sample_list[:10])

    _DATA_TYPE_NAMES = {0: "image", 1: "video", 2: "ref_video"}

    def __iter__(self):
        for sample_idx, i in enumerate(self.sample_list):
            if sample_idx >= (self.length * self.num_epochs):
                return
            if i == 0:
                try:
                    data = next(self.iter)
                except StopIteration:
                    self.iter = iter(self.dataloader)
                    data = next(self.iter)
            elif i == 1:
                try:
                    data = next(self.vid_iter)
                except StopIteration:
                    self.vid_iter = iter(self.vid_dataloader)
                    data = next(self.vid_iter)
            else:
                try:
                    data = next(self.vid_ref_iter)
                except StopIteration:
                    self.vid_ref_iter = iter(self.vid_ref_dataloader)
                    data = next(self.vid_ref_iter)
            data["_data_type"] = self._DATA_TYPE_NAMES[i]
            yield data
    
    def __len__(self):
        return self.length * self.num_epochs


class AsyncPrefetcher:
    """
    Prefetches the next batch from the dataloader in a background thread.
    
    Since DataLoader worker processes do the heavy CPU work (video decoding, image
    loading, resizing) in separate processes (bypassing GIL), the main thread only
    needs to receive the already-processed result from the worker queue.
    
    This prefetcher adds a buffer so that the next batch's DataLoader result is
    already waiting when the GPU finishes the current step's forward/backward pass.
    The key insight: DataLoader.__next__() internally does a queue.get() from worker
    processes. By calling it in a background thread, we allow the DataLoader workers
    to start producing the NEXT batch while the current batch is being processed on GPU.
    
    Additionally, if a preprocess_fn is provided (e.g., frozen VAE encode on a
    separate CUDA stream), it will be executed in the background thread after
    receiving data from the DataLoader.
    """
    def __init__(self, dataloader, preprocess_fn=None, maxsize=2, device=None):
        """
        Args:
            dataloader: iterable that yields raw data dicts
            preprocess_fn: optional callable(data) -> inputs dict. 
                          Can do GPU work on frozen models using a separate CUDA stream.
            maxsize: prefetch buffer size (number of batches to keep ready)
            device: target CUDA device for stream-based prefetch
        """
        self.dataloader = dataloader
        self.preprocess_fn = preprocess_fn
        self.queue = Queue(maxsize=maxsize)
        self.device = device
        self._thread = None

    def _worker(self):
        # Create a separate CUDA stream for preprocess if on GPU
        stream = None
        if self.device is not None and self.preprocess_fn is not None:
            stream = torch.cuda.Stream(device=self.device)
        try:
            for data in self.dataloader:
                try:
                    if self.preprocess_fn is not None:
                        if stream is not None:
                            with torch.cuda.stream(stream):
                                with torch.no_grad():
                                    inputs = self.preprocess_fn(data)
                            # Record event so main thread can synchronize
                            event = torch.cuda.Event()
                            event.record(stream)
                            self.queue.put((data, inputs, event))
                        else:
                            with torch.no_grad():
                                inputs = self.preprocess_fn(data)
                            self.queue.put((data, inputs, None))
                    else:
                        self.queue.put((data, None, None))
                except Exception as e:
                    print(f"[AsyncPrefetcher] preprocess error: {e}, skipping batch")
                    self.queue.put((data, None, None))
        except Exception as e:
            print(f"[AsyncPrefetcher] dataloader error: {e}")
        self.queue.put(None)  # sentinel

    def __iter__(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        while True:
            item = self.queue.get()
            if item is None:
                break
            data, inputs, event = item
            # Synchronize with the prefetch stream if needed
            if event is not None:
                event.synchronize()
            yield data, inputs

    def __len__(self):
        return len(self.dataloader)


def launch_mix_training_task(
    dataset: torch.utils.data.Dataset,
    vid_dataset: torch.utils.data.Dataset,
    vid_ref_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    eval_steps: int = None,
    eval_fn = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        eval_steps = getattr(args, "eval_steps", None)
    warmup_steps = getattr(args, "warmup_steps", 0)
    warmup_start_factor = getattr(args, "warmup_start_factor", 1e-6)
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    if warmup_steps > 0:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return warmup_start_factor + (1.0 - warmup_start_factor) * current_step / warmup_steps
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"LR scheduler: LinearWarmup({warmup_steps} steps, start_factor={warmup_start_factor}) -> ConstantLR")
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        print("LR scheduler: ConstantLR (no warmup)")
    model_logger.set_optimizer_scheduler(optimizer, scheduler)
    _opt_ckpt_path = None
    checkpoint = getattr(args, 'checkpoint', None)
    _dataloader_prefetch_factor = getattr(args, "dataloader_prefetch_factor", 2) if args is not None else 2
    _persistent_workers = num_workers > 0
    if dataset is None:
        dataloader = None
        print("Image dataset is None, skip training.")
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers,
            prefetch_factor=_dataloader_prefetch_factor if num_workers > 0 else None,
            persistent_workers=_persistent_workers,
        )
    if vid_dataset is None:
        vid_dataloader = None
        print("Video dataset is None, skip training.")
    else:
        vid_dataloader = torch.utils.data.DataLoader(
            vid_dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers,
            prefetch_factor=_dataloader_prefetch_factor if num_workers > 0 else None,
            persistent_workers=_persistent_workers,
        )
    if vid_ref_dataset is None:
        vid_ref_dataloader = None
        print("Video ref dataset is None, skip training.")
    else:
        vid_ref_dataloader = torch.utils.data.DataLoader(
            vid_ref_dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers,
            prefetch_factor=_dataloader_prefetch_factor if num_workers > 0 else None,
            persistent_workers=_persistent_workers,
        )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_plugin=DeepSpeedPlugin(),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters),
            InitProcessGroupKwargs(timeout=timedelta(seconds=5400))],
    )
    # Hack for deepspeed Zero3 run with different shape in different rank
    vae = getattr(model.pipe, "vae")
    delattr(model.pipe, "vae")
    # Note: scheduler is intentionally NOT passed to accelerator.prepare to avoid DeepSpeed
    # calling scheduler.step() internally during initialization (which would advance LR prematurely).
    # We manage the scheduler manually in the training loop.
    model, optimizer, dataloader, vid_dataloader, vid_ref_dataloader = accelerator.prepare(model, optimizer, dataloader, vid_dataloader, vid_ref_dataloader)
    # Auto-load optimizer/scheduler state after accelerator.prepare
    if checkpoint is not None:
        rank = accelerator.process_index
        ckpt_dir = os.path.dirname(checkpoint)
        step_name = os.path.basename(checkpoint).replace(".safetensors", "")
        _opt_ckpt_path_rank = os.path.join(ckpt_dir, "optimizer", step_name, f"rank{rank}.pt")
        _opt_ckpt_path_legacy_rank = checkpoint.replace(".safetensors", f"_optimizer_rank{rank}.pt")
        _opt_ckpt_path_legacy = checkpoint.replace(".safetensors", "_optimizer.pt")
        if os.path.isfile(_opt_ckpt_path_rank):
            _opt_ckpt_path = _opt_ckpt_path_rank
        elif os.path.isfile(_opt_ckpt_path_legacy_rank):
            _opt_ckpt_path = _opt_ckpt_path_legacy_rank
        elif os.path.isfile(_opt_ckpt_path_legacy):
            _opt_ckpt_path = _opt_ckpt_path_legacy
    if getattr(args, 'no_load_optimizer', False):
        if accelerator.is_main_process:
            print("[train] --no_load_optimizer set, skipping optimizer/scheduler restore.")
        _opt_ckpt_path = None
    if _opt_ckpt_path is not None and os.path.isfile(_opt_ckpt_path):
        print(f"[Rank {accelerator.process_index}] Loading optimizer/scheduler from {_opt_ckpt_path}")
        opt_state = torch.load(_opt_ckpt_path, map_location="cpu")
        _raw_opt = optimizer
        if hasattr(_raw_opt, 'optimizer'):  # accelerate wrapper
            _raw_opt = _raw_opt.optimizer
        if hasattr(_raw_opt, 'optimizer'):  # deepspeed wrapper
            _raw_opt = _raw_opt.optimizer
        _raw_opt.load_state_dict(opt_state["optimizer"])
        if opt_state.get("scheduler") is not None:
            scheduler.load_state_dict(opt_state["scheduler"])
        if opt_state.get("num_steps") is not None:
            model_logger.num_steps = opt_state["num_steps"]
        if accelerator.is_main_process:
            print(f"Resumed optimizer/scheduler from step {model_logger.num_steps}")
    elif _opt_ckpt_path is not None:
        if accelerator.is_main_process:
            print(f"No optimizer state found, starting optimizer fresh.")
    # Override learning rate after loading optimizer state (preserves momentum)
    _override_lr = getattr(args, 'override_lr', None)
    if _override_lr is not None:
        _raw_opt = optimizer
        if hasattr(_raw_opt, 'optimizer'):
            _raw_opt = _raw_opt.optimizer
        if hasattr(_raw_opt, 'optimizer'):
            _raw_opt = _raw_opt.optimizer
        for pg in _raw_opt.param_groups:
            pg['lr'] = _override_lr
        # Also update scheduler base_lrs so warmup targets the new lr
        scheduler.base_lrs = [_override_lr for _ in scheduler.base_lrs]
        if accelerator.is_main_process:
            print(f"[train] Overriding learning rate to {_override_lr}")
    vae = vae.to(accelerator.device)
    vae.eval()
    vae.requires_grad_(False)
    if accelerator.is_main_process:
        if DEBUG: print((sorted(model.trainable_param_names())))
        wandb.init(project=args.project_name, name=args.exp_name)
    dataloader = MixDataloader(dataloader, vid_dataloader, vid_ref_dataloader, num_epochs=num_epochs)

    # Wrap with AsyncPrefetcher: run frozen VAE encode in background CUDA stream
    # while the main thread does MLLM + DiT forward/backward.
    _prefetch_size = getattr(args, "prefetch_size", 2)
    _use_async_vae = getattr(args, "async_vae_prefetch", False)
    
    # Get the unwrapped training module for calling vae_preprocess
    _unwrapped_model = accelerator.unwrap_model(model)
    _m = _unwrapped_model
    while not hasattr(_m, 'pipe') and hasattr(_m, 'module'):
        _m = _m.module
    _unwrapped_model = _m

    # Ensure pipe.device is up-to-date after DeepSpeed moves models to GPU
    _unwrapped_model.pipe.device = accelerator.device

    if _use_async_vae and hasattr(_unwrapped_model, 'vae_preprocess'):
        def _vae_prefetch_fn(data):
            """Run frozen VAE encode (ShapeChecker + NoiseInitializer + InputVideoEmbedder) in background."""
            with torch.no_grad():
                return _unwrapped_model.vae_preprocess(data, vae=vae)
        prefetcher = AsyncPrefetcher(dataloader, preprocess_fn=_vae_prefetch_fn, maxsize=_prefetch_size, device=accelerator.device)
        if accelerator.is_main_process:
            print(f"[AsyncPrefetcher] enabled with VAE prefetch on background stream, buffer size={_prefetch_size}")
    else:
        prefetcher = AsyncPrefetcher(dataloader, preprocess_fn=None, maxsize=_prefetch_size, device=accelerator.device)
        if accelerator.is_main_process:
            print(f"[AsyncPrefetcher] enabled (data-only), buffer size={_prefetch_size}")

    # Run eval before training starts if requested
    eval_at_start = getattr(args, "eval_at_start", False)
    if eval_at_start and eval_fn is not None:
        print(f"[Rank {accelerator.process_index}] Running eval_fn at step 0 (before training)...")
        eval_fn(accelerator, model, 0, model_logger.output_path)
        print(f"[Rank {accelerator.process_index}] eval_fn at step 0 done.")
    _step_start_time = time.time()
    _iters_per_epoch = len(dataloader) // num_epochs
    _global_iter = 0

    for data, prefetched_vae in prefetcher:
        _global_iter += 1
        _cur_epoch = (_global_iter - 1) // _iters_per_epoch + 1 if _iters_per_epoch > 0 else 1
        _iter_in_epoch = (_global_iter - 1) % _iters_per_epoch + 1 if _iters_per_epoch > 0 else _global_iter
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            loss = model(data, prefetched_vae=prefetched_vae, vae=vae)
            accelerator.backward(loss)
            _max_grad_norm = getattr(args, 'max_grad_norm', None) or float('inf')
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=_max_grad_norm)
            optimizer.step()
            model_logger.on_step_end(accelerator, model, save_steps, eval_steps=eval_steps, eval_fn=eval_fn)
            scheduler.step()
        if accelerator.is_main_process:
            _step_time = time.time() - _step_start_time
            _step_start_time = time.time()
            _now = time.strftime('%Y-%m-%d %H:%M:%S')
            _progress = f"[{_now}] [step-{model_logger.num_steps}/{_iter_in_epoch}/{_iters_per_epoch}/epoch{_cur_epoch}]"
            _tgt = data.get("tgt_video", None)
            _num_frames = len(_tgt) if _tgt is not None else (data.get("video", None) is not None and len(data["video"])) or "N/A"
            _data_type = data.get("_data_type", "unknown")
            _loss_val = loss.item()
            _grad_norm_val = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            print(_progress, f"loss({_data_type}): ", _loss_val, "num_frames: ", _num_frames, "lr: ", scheduler.get_last_lr()[0], "grad_norm: ", _grad_norm_val, "step_time: ", f"{_step_time:.2f}s")
            _wandb_log = {
                "loss": _loss_val,
                f"loss_{_data_type}": _loss_val,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": _grad_norm_val,
                "step_time": _step_time,
                "data_type": _data_type,
            }
            wandb.log(_wandb_log)
    if save_steps is None:
        model_logger.on_epoch_end(accelerator, model, num_epochs)
    model_logger.on_training_end(accelerator, model, save_steps)


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default=None, help="Base path of the dataset.")
    parser.add_argument("--vid_dataset_metadata_path", type=str, default=None, help="Base path of the dataset.")
    parser.add_argument("--vid_ref_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--img_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--min_pixels", type=int, default=None, help="Minimum pixels per frame for training dataloader. If set, frames smaller than this area are upscaled (preserving aspect ratio) to approximately this area. Must be <= --max_pixels. Default None disables upscaling.")
    parser.add_argument("--img_min_pixels", type=int, default=None, help="Minimum pixels for image training data. If set, images smaller than this area are upscaled. Default None keeps original behavior.")
    parser.add_argument("--eval_max_pixels", type=int, default=None, help="Maximum number of pixels per frame during eval. Falls back to --max_pixels if not set.")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video (max when --rand_num_frames is enabled).")
    parser.add_argument("--rand_num_frames", action="store_true", default=False, help="Randomly sample num_frames per step from [min_num_frames, num_frames] with step rand_num_frames_step.")
    parser.add_argument("--min_num_frames", type=int, default=None, help="Minimum num_frames when --rand_num_frames is enabled (e.g. 49).")
    parser.add_argument("--rand_num_frames_step", type=int, default=8, help="Step size between candidate frame counts when --rand_num_frames is enabled (default: 8).")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--local_model_path", type=str, default=None, help="Local model base path for model_id lookup. Default: ./models")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Model ID with origin paths to the audio processor config, e.g., Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--dit_lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--dit_lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--dit_lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--freeze_lora_base_model", action="store_true", default=False, help="If set, lora_base_model LoRA parameters are frozen (not trained) after injection.")
    parser.add_argument("--save_frozen_lora", action="store_true", default=False, help="If set, frozen lora_base_model lora_A/lora_B weights are saved in ckpt even if not trained.")
    parser.add_argument("--project_name", type=str, default='diffsynth', help="Project name.")
    parser.add_argument("--exp_name", type=str, default='run', help="Experiment name.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint. If provided, the model will be loaded from this checkpoint.")
    parser.add_argument("--no_load_optimizer", default=False, action="store_true", help="Skip loading optimizer/scheduler state from checkpoint. Useful when fine-tuning on a different data distribution.")
    parser.add_argument("--override_lr", type=float, default=None, help="Override learning rate after loading optimizer state. Preserves Adam momentum/variance but changes lr. Also updates scheduler base_lrs.")
    parser.add_argument("--num_image_queries", type=int, default=256, help="Number of image queries.")
    parser.add_argument("--num_video_queries", type=int, default=512, help="Number of video queries.")
    parser.add_argument("--num_ref_queries", type=int, default=768, help="Number of reference queries.")
    parser.add_argument("--max_object_token", type=int, default=768, help="Maximum number of object tokens.")
    parser.add_argument("--mllm_model", type=str, default='Qwen/Qwen2.5-VL-3B-Instruct', help="Path to the MLLM model.")
    parser.add_argument("--mllm_gradient_checkpointing", type=bool, default=False, help="Whether to use gradient checkpointing for MLLM.")
    parser.add_argument("--mllm_max_frame", type=int, default=16, help="Maximum number of frames for MLLM.")
    parser.add_argument("--mllm_max_pixels_per_frame", type=int, default=None, help="Maximum number of pixels per frame for MLLM.")
    parser.add_argument("--ref_pad_first", type=bool, default=False, help="Pad reference video to the first frame.")
    # Evaluation parameters for eval_fn
    parser.add_argument("--eval_dataset_file", type=str, default=None, help="Path to the evaluation dataset YAML file")
    parser.add_argument("--eval_data_root", type=str, default=None, help="Root directory for evaluation data")
    parser.add_argument("--eval_max_frame", type=int, default=81, help="Maximum number of frames for evaluation")
    parser.add_argument("--eval_save_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--eval_steps", type=int, default=None, help="Number of training steps between evaluations")
    parser.add_argument("--eval_at_start", action="store_true", default=False, help="Run eval_fn once at step 0 before training starts.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of linear warmup steps. LR ramps from warmup_start_factor*lr to learning_rate over this many steps. 0 means no warmup.")
    parser.add_argument("--warmup_start_factor", type=float, default=1e-6, help="Initial LR factor at warmup start. Actual initial LR = learning_rate * warmup_start_factor. Default 1e-6 (near zero).")
    parser.add_argument("--eval_prompt_only", action="store_true", default=False, help="If set, ref_image will be None during evaluation (prompt-only mode).")
    parser.add_argument("--skip_load_weights", action="store_true", default=False, help="Skip loading weights from disk, use random initialization for debugging.")
    parser.add_argument("--ema_decay", type=float, default=None, help="EMA decay rate. If set, EMA weights are tracked and saved as separate checkpoints (e.g. 0.999 or 0.9999).")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm for clipping. If None, no clipping is applied (only grad_norm logging).")
    parser.add_argument("--prefetch_size", type=int, default=2, help="Async prefetcher buffer size (number of batches preprocessed ahead). Larger values overlap more but use more GPU memory.")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="PyTorch DataLoader prefetch_factor (batches per worker prefetched from disk). Larger values reduce IO stalls.")
    parser.add_argument("--async_vae_prefetch", action="store_true", default=False, help="Run frozen VAE encode in background CUDA stream to overlap with training.")
    return parser
