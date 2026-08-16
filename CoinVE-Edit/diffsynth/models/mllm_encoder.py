import torch
import torch.nn as nn
from transformers import AutoProcessor
from diffsynth.models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from diffsynth.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from diffsynth.models.qwen_vl_utils import process_vision_info
import deepspeed
DEBUG = False

class QueryVector(nn.Module):
    def __init__(self, num_query, dim, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_query, dim, dtype=dtype)*0.02)

    def forward(self):
        return self.q

class QwenVLMLLMEncoder(torch.nn.Module):
    def __init__(self, 
        model_path="Qwen/Qwen2.5-VL-3B-Instruct", 
        dtype=torch.bfloat16, 
        device="cuda", 
        dit_dim=3072,
        num_image_queries = 256,
        num_video_queries = 512,
        num_ref_queries = 512,
        max_object_token = 512,
        max_frames = 16,
        max_pixels_per_frame = 512*512,
        gradient_checkpointing = False,
        skip_load_weights = False,
        ):
        super().__init__()
        if skip_load_weights:
            print(f"[skip_load_weights=True] Skipping Qwen model loading from {model_path}, using random initialization.")
            # Build model from config only (no weights loaded)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)
            if "Qwen2.5" in model_path:
                self.model = Qwen2_5_VLForConditionalGeneration(config).to(dtype=dtype, device=device)
                hidden_size = config.hidden_size
            elif "Qwen3" in model_path:
                self.model = Qwen3VLForConditionalGeneration(config).to(dtype=dtype, device=device)
                hidden_size = config.text_config.hidden_size
            else:
                raise ValueError(f"model_path {model_path} is not supported.")
            self.processor = AutoProcessor.from_pretrained(model_path)
        elif "Qwen2.5" in model_path:
            print('Load qwen2.5 from: {}'.format(model_path))
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, device_map=device
            )
            hidden_size = self.model.config.hidden_size
            self.processor = AutoProcessor.from_pretrained(model_path)
        elif "Qwen3" in model_path:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, device_map=device
            )
            hidden_size = self.model.config.text_config.hidden_size
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            raise ValueError(f"model_path {model_path} is not supported.")
        self.num_image_queries = num_image_queries
        self.num_video_queries = num_video_queries
        self.num_ref_queries = num_ref_queries
        self.max_object_token = max_object_token
        self.model.model.num_image_queries = self.num_image_queries
        self.model.num_image_queries = self.num_image_queries
        self.model.model.num_video_queries = self.num_video_queries
        self.model.num_video_queries = self.num_video_queries
        self.model.model.num_ref_queries = self.num_ref_queries
        self.image_queries = QueryVector(self.num_image_queries, hidden_size, dtype=dtype, device=device)
        self.video_queries = QueryVector(self.num_video_queries, hidden_size, dtype=dtype, device=device)
        self.ref_queries = QueryVector(self.num_ref_queries, hidden_size, dtype=dtype, device=device)
        self.max_frames = max_frames
        self.max_pixels_per_frame = max_pixels_per_frame
        self.gradient_checkpointing = gradient_checkpointing
        self.connector = nn.Sequential(
            nn.Linear(hidden_size, dit_dim, dtype=dtype),
            nn.GELU(approximate='tanh'),
            nn.Linear(dit_dim, dit_dim, dtype=dtype),
        ).to(device)
        nn.init.zeros_(self.connector[2].weight)
        nn.init.zeros_(self.connector[2].bias)

        self.ref_connector = nn.Sequential(
            nn.Linear(hidden_size, dit_dim, dtype=dtype),
            nn.GELU(approximate='tanh'),
            nn.Linear(dit_dim, dit_dim, dtype=dtype),
        ).to(device)
        nn.init.zeros_(self.ref_connector[2].weight)
        nn.init.zeros_(self.ref_connector[2].bias)

        self.system_prompt = "You will be given an image and instruction. Please describe the content of the image in detail based on instruction in your own words."
        print(model_path,
            "max_frames:", self.max_frames,
            "max_pixels_per_frame:", self.max_pixels_per_frame,
            "Using Dual Connector:\n", self.connector, self.ref_connector,
            "\nSys Prompt:\n",self.system_prompt, 
            "\nNum Image Queries:", self.num_image_queries,
            "\nNum Video Queries:", self.num_video_queries,
            "\nNum Ref Queries:", self.num_ref_queries)
        
    def forward(
        self, 
        instruction,
        src_image=None,
        src_video=None,
        ref_image=None,
        **kwargs
    ):
        # <|object_ref_start|> is a simple hack for query placeholder
        is_video = src_video is not None
        system_prompt = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        if ref_image:
            instruction += " Use the reference input from last frame."
            num_queries = self.num_ref_queries
            input_query = self.ref_queries().to(self.model.device)
            if DEBUG: print("Ref Image + Instuction Video Edit Mode / Num Query:", num_queries)
            video_data = {"type": "video", "video": src_video, 'max_frames': self.max_frames}
            if self.max_pixels_per_frame:
                video_data['max_pixels'] = self.max_pixels_per_frame
            messages = [
                system_prompt,
                {"role": "user",
                "content": [
                    video_data,
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": ref_image[0], 'max_pixels': 28*28*self.max_object_token},
                    {"type": "text", "text": "<|object_ref_start|>"*(num_queries)},
                ]}
            ]
        elif is_video:
            num_queries = self.num_video_queries
            input_query = self.video_queries().to(self.model.device)
            if DEBUG: print("Instuction Video Edit Mode / Num Query:", num_queries)
            video_data = {"type": "video", "video": src_video, 'max_frames': self.max_frames}
            if self.max_pixels_per_frame:
                video_data['max_pixels'] = self.max_pixels_per_frame
            messages = [
                system_prompt,
                {"role": "user",
                "content": [
                    video_data,
                    {"type": "text", "text": instruction},
                    {"type": "text", "text": "<|object_ref_start|>"*(num_queries)},
                ]}
            ]
        else:
            num_queries = self.num_image_queries
            input_query = self.image_queries().to(self.model.device)
            if DEBUG: print("Image Edit Mode / Num Query:", num_queries)
            messages = [
                system_prompt,
                {"role": "user",
                    "content": [
                    {"type": "image", "image": src_image[0]},
                    {"type": "text", "text": instruction},
                    {"type": "text", "text": "<|object_ref_start|>"*num_queries},
                ]}
            ]
            
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        # print(inputs.pixel_values.dtype, self.model.visual.dtype)
        if self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(inputs, input_query, **forward_kwargs):
                    # 1. Run the base model
                    raw_out = module(
                        **inputs,
                        learnable_query=input_query,
                        output_attentions=False, 
                        output_hidden_states=True,
                        return_dict=True,
                        **forward_kwargs
                    )
                    return (raw_out.hidden_states,)
                return custom_forward
            checkpoint_output = deepspeed.checkpointing.checkpoint(
                create_custom_forward(self.model),
                inputs,
                input_query,
                **kwargs
            )
            #print(checkpoint_output[-1].shape)
            hidden_states = checkpoint_output[-1]
            #learnable_query_features = hidden_states[:,-input_query.shape[0]:,:]
            query_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
            query_mask = (inputs.input_ids[0] == query_token_id)
            learnable_query_features = hidden_states[:, query_mask, :]
        else:
            outputs = self.model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                learnable_query=input_query,
                **kwargs,
            )
            hidden_states = outputs.hidden_states[-1]
            #learnable_query_features = hidden_states[:,-input_query.shape[0]:,:]
            query_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
            query_mask = (inputs.input_ids[0] == query_token_id)
            learnable_query_features = hidden_states[:, query_mask, :]

        learnable_query_features = self.connector(learnable_query_features)
        if ref_image:
            vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            input_ids = inputs.input_ids[0]
            vision_start_indices = (input_ids == vision_start_id).nonzero(as_tuple=True)[-1]
            if len(vision_start_indices) > 0:
                last_vision_start = vision_start_indices[-1]
                remaining_ids = input_ids[last_vision_start:]
                end_relative_idx = (remaining_ids == vision_end_id).nonzero(as_tuple=True)[-1]
                if len(end_relative_idx) > 0:
                    last_vision_end = last_vision_start + end_relative_idx[0]
                    # print(last_vision_start, last_vision_end)
                    ref_image_features = hidden_states[:, last_vision_start + 1 : last_vision_end, :]
                    ref_image_features = self.ref_connector(ref_image_features)
                    learnable_query_features = torch.cat([ref_image_features, learnable_query_features], dim=1)
                    print(f"Ref Image Features Shape: {ref_image_features.shape} Learnable Query Features Shape: {learnable_query_features.shape}")
                else:
                    print("Warning: Found vision_start but no matching vision_end for Ref Image.")
            else:
                print("Warning: No vision_start tokens found, cannot extract Ref Image features.")
        return learnable_query_features

    @staticmethod
    def state_dict_converter():
        return QwenImageTextEncoderStateDictConverter()

class QwenImageTextEncoderStateDictConverter():
    def __init__(self):
        pass
    def from_diffusers(self, state_dict):
        state_dict_ = {}
        for k, v in state_dict.items():
            if k.startswith("visual."):
                k = "model." + k
            elif k.startswith("model."):
                k = k.replace("model.", "model.language_model.")
            state_dict_[k] = v
        return state_dict_

