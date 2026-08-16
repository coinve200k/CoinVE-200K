"""
Composite multi-instruction Stage-2 dataset + collator.

Each JSONL row is one composite editing sample with K instructions. The collator
produces one source/target video pair plus K per-instruction GT masks aligned to
the Qwen-VL patch grid:

  - src_video / tgt_video : list[PIL.Image]
  - prompts               : list[str] len=K
  - stage1_masks          : torch.FloatTensor [K, T_v, H_v, W_v] in [0,1]
  - stage1_grid_thw       : (T_v, H_v, W_v)

Qwen-VL itself is not loaded here; the processor is used only to derive the
training/inference-consistent video_grid_thw.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import (
    FRAME_FACTOR,
    SPATIAL_MERGE_SIZE,
    align_mask_to_patch_grid_soft,
    read_mask_uint8,
)

from diffsynth.trainers.unified_dataset import ImageCropAndResize


DEFAULT_SYSTEM_PROMPT = (
    "You will be given an image and instruction. Please describe the content of "
    "the image in detail based on instruction in your own words."
)
QUERY_TOKEN = "<|object_ref_start|>"


def _read_video_uint8(path: str):
    """Read a video as [T, H, W, C] uint8."""
    import imageio
    r = imageio.get_reader(path)
    fps = r.get_meta_data().get("fps", None)
    frames = [np.asarray(f) for f in r]
    r.close()
    return np.stack(frames, axis=0), fps


def build_messages_for_grid(
    instruction: str,
    pil_frames: List[Image.Image],
    num_video_queries: int,
    max_pixels_per_frame: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    video_data: Dict[str, Any] = {"type": "video", "video": pil_frames}
    if max_pixels_per_frame is not None:
        video_data["max_pixels"] = max_pixels_per_frame
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                video_data,
                {"type": "text", "text": instruction},
                {"type": "text", "text": QUERY_TOKEN * num_video_queries},
            ],
        },
    ]


class ComposeDataset(Dataset):
    """JSONL dataset for composite multi-instruction training."""

    def __init__(
        self,
        jsonl_path: str,
        max_instructions: Optional[int] = None,
        min_instructions: int = 1,
        **_: Any,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.max_instructions = max_instructions
        self.min_instructions = min_instructions
        self.rows: List[Dict[str, Any]] = []
        n_dropped = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                try:
                    row = self._normalize_item(item)
                except Exception as e:
                    n_dropped += 1
                    print(f"[CompositeDataset] drop line {line_no}: {e}")
                    continue
                n_inst = len(row["prompts"])
                if n_inst < min_instructions:
                    n_dropped += 1
                    continue
                if max_instructions is not None and n_inst > max_instructions:
                    row["prompts"] = row["prompts"][:max_instructions]
                    row["mask_video_paths"] = row["mask_video_paths"][:max_instructions]
                    row["instruction_operation"] = row["instruction_operation"][:max_instructions]
                    row["instruction_object"] = row["instruction_object"][:max_instructions]
                    row["is_style"] = row["is_style"][:max_instructions]
                self.rows.append(row)

        print(
            f"[CompositeDataset] loaded {len(self.rows)} rows from {jsonl_path} "
            f"(dropped={n_dropped}, min_instructions={min_instructions}, "
            f"max_instructions={max_instructions})"
        )

    @staticmethod
    def _as_list(value: Any, name: str) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
        raise ValueError(f"field '{name}' must be list or str")

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        prompts = [str(x).strip() for x in self._as_list(item.get("instruction"), "instruction")]
        raw_mask_paths = item.get("instruction_mask_video_paths", None)
        if raw_mask_paths is None:
            mask_paths = [""] * len(prompts)
        else:
            mask_paths = [str(x).strip() for x in self._as_list(
                raw_mask_paths, "instruction_mask_video_paths"
            )]
        if len(prompts) != len(mask_paths):
            raise ValueError(f"instruction/mask length mismatch: {len(prompts)} vs {len(mask_paths)}")
        if not item.get("source_video_path") or not item.get("edited_video_path"):
            raise ValueError("missing source_video_path or edited_video_path")
        if any(not p for p in prompts):
            raise ValueError("empty instruction")

        is_style = item.get("is_style", None)
        if is_style is None:
            category = str(item.get("category", "")).strip().lower()
            task = str(item.get("task", "")).strip().lower()
            is_style = 1 if category == "style_transfer" or task == "style_transfer" else 0
        if isinstance(is_style, list):
            style_flags = [float(x) for x in is_style]
        else:
            style_flags = [float(is_style)] * len(prompts)
        if len(style_flags) != len(prompts):
            style_flags = (style_flags + [0.0] * len(prompts))[:len(prompts)]
        if any((not m) and s <= 0.5 for m, s in zip(mask_paths, style_flags)):
            raise ValueError("empty mask path for non-style instruction")

        ops = item.get("instruction_operation") or [""] * len(prompts)
        objs = item.get("instruction_object") or [""] * len(prompts)
        if not isinstance(ops, list):
            ops = [ops] * len(prompts)
        if not isinstance(objs, list):
            objs = [objs] * len(prompts)
        if len(ops) != len(prompts):
            ops = (ops + [""] * len(prompts))[:len(prompts)]
        if len(objs) != len(prompts):
            objs = (objs + [""] * len(prompts))[:len(prompts)]

        return {
            "src_video": str(item["source_video_path"]),
            "tgt_video": str(item["edited_video_path"]),
            "prompts": prompts,
            "mask_video_paths": mask_paths,
            "combined_mask_video_path": str(item.get("combined_mask_video_path", "")),
            "instruction_operation": [str(x) for x in ops],
            "instruction_object": [str(x) for x in objs],
            "is_style": style_flags,
            "data_source": str(item.get("data_source", "composite_jsonl")),
            "gate_mode": str(item.get("gate_mode", "override_one")),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


@dataclass
class ComposeCollator:
    processor: Any
    process_vision_info: Callable
    num_video_queries: int = 768
    max_frames: int = 10
    max_pixels_per_frame: Optional[int] = 262144
    src_max_frames: Optional[int] = 49
    max_pixels: int = 360_000
    height_division_factor: int = 32
    width_division_factor: int = 32
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    _dataset: Optional[Any] = None
    _max_retries: int = 10

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) == 1, "ComposeCollator only supports batch_size=1 (resolution varies)"
        row = batch[0]
        for _retry in range(self._max_retries):
            try:
                return self._process_row(row)
            except Exception as e:
                import random as _rand
                print(f"[CompositeCollator] Skipping bad sample: {row.get('src_video', '?')} | Error: {e}")
                if self._dataset is not None and len(self._dataset) > 0:
                    row = self._dataset[_rand.randint(0, len(self._dataset) - 1)]
                else:
                    raise
        raise RuntimeError(f"[CompositeCollator] Failed after {self._max_retries} retries")

    def _process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if self.max_frames % FRAME_FACTOR != 0:
            raise ValueError(f"max_frames={self.max_frames} must be multiple of {FRAME_FACTOR}")

        prompts = row["prompts"]
        mask_paths = row["mask_video_paths"]
        if len(prompts) != len(mask_paths):
            raise ValueError(f"prompts/masks mismatch: {len(prompts)} vs {len(mask_paths)}")
        K = len(prompts)

        src_arr, _ = _read_video_uint8(row["src_video"])
        tgt_arr, _ = _read_video_uint8(row["tgt_video"])
        T_full = min(src_arr.shape[0], tgt_arr.shape[0])
        T_src = T_full if not self.src_max_frames else min(T_full, self.src_max_frames)
        T_dit = T_src - ((T_src - 1) % 4)
        if T_dit < 5:
            raise RuntimeError(f"video too short for VAE 4k+1 alignment: T_src={T_src}")
        src_window = src_arr[:T_dit]
        tgt_window = tgt_arr[:T_dit]

        n_vlm = min(self.max_frames, T_dit)
        if n_vlm % FRAME_FACTOR != 0:
            n_vlm = (n_vlm // FRAME_FACTOR) * FRAME_FACTOR
        if n_vlm == 0:
            raise RuntimeError(f"video too short for VLM sampling: T_dit={T_dit}")
        sampled_idx = torch.linspace(0, T_dit - 1, n_vlm).round().long().tolist()

        resizer = ImageCropAndResize(
            height=None,
            width=None,
            max_pixels=self.max_pixels,
            height_division_factor=self.height_division_factor,
            width_division_factor=self.width_division_factor,
            min_pixels=None,
        )
        src_pil_full = [resizer(Image.fromarray(src_window[i])) for i in range(T_dit)]
        tgt_pil_full = [resizer(Image.fromarray(tgt_window[i])) for i in range(T_dit)]
        src_pil_sampled = [src_pil_full[i] for i in sampled_idx]

        messages = build_messages_for_grid(
            instruction=prompts[0],
            pil_frames=src_pil_sampled,
            num_video_queries=self.num_video_queries,
            max_pixels_per_frame=self.max_pixels_per_frame,
            system_prompt=self.system_prompt,
        )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(
            messages, return_video_kwargs=True,
        )
        vid_tensor = video_inputs[0]
        n_sampled, _, h_resized, w_resized = vid_tensor.shape
        if n_sampled != n_vlm:
            raise RuntimeError(
                f"processor sampled {n_sampled} but we picked {n_vlm}; check src_max_frames / max_frames"
            )

        bf = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        t_pre, h_pre, w_pre = bf.video_grid_thw[0].tolist()
        T_v = t_pre
        H_v = h_pre // SPATIAL_MERGE_SIZE
        W_v = w_pre // SPATIAL_MERGE_SIZE

        style_flags = row.get("is_style", [0.0] * K)
        if isinstance(style_flags, torch.Tensor):
            style_flags = style_flags.detach().cpu().tolist()
        elif not isinstance(style_flags, list):
            style_flags = [float(style_flags)] * K
        if len(style_flags) != K:
            style_flags = (style_flags + [0.0] * K)[:K]

        masks = []
        for mask_path, is_style in zip(mask_paths, style_flags):
            if mask_path:
                mask_arr, _ = read_mask_uint8(mask_path)
                if mask_arr.shape[0] == 1:
                    mask_arr = np.broadcast_to(
                        mask_arr, (max(T_dit, 1),) + mask_arr.shape[1:]
                    ).copy()
                masks.append(
                    align_mask_to_patch_grid_soft(
                        mask_frames_uint8=mask_arr[:T_dit] if mask_arr.shape[0] >= T_dit else mask_arr,
                        sampled_indices=sampled_idx,
                        h_resized=h_resized,
                        w_resized=w_resized,
                        T_v=T_v,
                        H_v=H_v,
                        W_v=W_v,
                    )
                )
            elif float(is_style) > 0.5:
                masks.append(torch.ones((T_v, H_v, W_v), dtype=torch.float32))
            else:
                raise ValueError("empty mask path for non-style instruction")
        stage1_masks = torch.stack(masks, dim=0).float()
        is_style_tensor = torch.tensor(style_flags, dtype=torch.float32)

        return {
            "src_video": src_pil_full,
            "tgt_video": tgt_pil_full,
            "prompt": prompts[0],
            "_data_type": "video",
            "prompts": prompts,
            "stage1_masks": stage1_masks,
            "stage1_grid_thw": (T_v, H_v, W_v),
            "is_style": is_style_tensor,
            "data_source": row.get("data_source", "composite_jsonl"),
            "gate_mode": row.get("gate_mode", "override_one"),
            "meta": {
                "src_video_path": row["src_video"],
                "tgt_video_path": row["tgt_video"],
                "mask_video_paths": mask_paths,
                "combined_mask_video_path": row.get("combined_mask_video_path", ""),
                "instruction_operation": row.get("instruction_operation", []),
                "instruction_object": row.get("instruction_object", []),
                "data_source": row.get("data_source", "composite_jsonl"),
                "gate_mode": row.get("gate_mode", "override_one"),
                "sampled_indices": sampled_idx,
                "T_dit": T_dit,
                "n_vlm": n_vlm,
                "num_instructions": K,
            },
        }


CompositeJsonlDataset = ComposeDataset
CompositeCollator = ComposeCollator
