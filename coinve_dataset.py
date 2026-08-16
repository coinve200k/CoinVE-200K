"""Dataset loader for the extracted CoinVE-200K release.

This module provides a thin, self-contained `CoinVEDataset` that reads the
`metadata.jsonl` shipped with CoinVE-200K and returns loaded frames for the
source video, edited (target) video, per-instruction mask videos, and the
combined mask video. It reuses the robust video / mask loading primitives
(`LoadVideo`, `LoadMaskVideo`, `ImageCropAndResize`) from the CoinVE-Edit
training codebase (`diffsynth/trainers/unified_dataset.py`) so that
frame-count sampling, spatial resizing, and src/tgt/mask synchronisation
behave consistently with the training pipeline.

Place this file at the repository root (next to the ``CoinVE-Edit/`` folder)
so that the ``diffsynth.trainers.unified_dataset`` import resolves correctly,
or run it from any directory after adding the repo root to ``sys.path``.

Example
-------
    from coinve_dataset import CoinVEDataset

    ds = CoinVEDataset(
        base_path="/data/CoinVE-200K",        # extracted tar root
        metadata_path="/data/CoinVE-200K/metadata.jsonl",
        num_frames=81,
        max_pixels=1920 * 1080,
    )
    sample = ds[0]
    # sample["src_video"]   -> list[PIL.Image]  (length == num_frames chosen)
    # sample["tgt_video"]   -> list[PIL.Image]
    # sample["instruction_masks"] -> list[list[PIL.Image]]  (one per instruction)
    # sample["combined_mask"]     -> list[PIL.Image] | None
    # sample["instruction"]       -> list[str]
    # sample["instruction_operation"] -> list[str]
    # sample["instruction_object"]     -> list[str]
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

import torch

# Make the CoinVE-Edit subpackage importable when this file lives at the repo root.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_COINVE_EDIT = os.path.join(_REPO_ROOT, "CoinVE-Edit")
if _COINVE_EDIT not in sys.path:
    sys.path.insert(0, _COINVE_EDIT)

from diffsynth.trainers.unified_dataset import (
    ImageCropAndResize,
    LoadMaskVideo,
    LoadVideo,
    RouteByType,
    ToAbsolutePath,
)


class CoinVEDataset(torch.utils.data.Dataset):
    """Dataset over the CoinVE-200K `metadata.jsonl`.

    Parameters
    ----------
    base_path : str
        Root directory of the extracted CoinVE-200K dataset. Relative paths in
        `metadata.jsonl` (e.g. ``src_videos/src_video_000/foo.mp4``) are
        resolved against this directory.
    metadata_path : str, optional
        Path to `metadata.jsonl`. Defaults to ``<base_path>/metadata.jsonl``.
    num_frames : int
        Target number of frames to load for src / tgt / mask videos. The actual
        count may be smaller for short clips (kept consistent across src/tgt/mask
        via the frame-sync mechanism in `LoadVideo`).
    height, width : int | None
        Optional fixed spatial size. If both are ``None`` the frames are resized
        to fit within ``max_pixels`` while preserving aspect ratio.
    max_pixels : int
        Upper bound on frame area when ``height``/``width`` are ``None``.
    height_division_factor, width_division_factor : int
        Spatial dimensions are rounded down to multiples of these factors
        (required by the VAE).
    time_division_factor, time_division_remainder : int
        Frame count is constrained to ``n % time_division_factor == time_division_remainder``.
    rand_num_frames : bool
        If ``True``, randomly sample a target frame count per item (useful for
        training). If ``False``, always aim for ``num_frames``.
    min_num_frames, rand_num_frames_step : int
        Controls the lower bound and stride of random frame-count sampling.
    max_retry : int
        Number of retry attempts on per-item load failure (a different item is
        sampled on each retry).
    """

    def __init__(
        self,
        base_path: str,
        metadata_path: Optional[str] = None,
        num_frames: int = 81,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = 1920 * 1080,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        rand_num_frames: bool = False,
        min_num_frames: Optional[int] = None,
        rand_num_frames_step: int = 8,
        max_retry: int = 30,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path or os.path.join(base_path, "metadata.jsonl")
        self.max_retry = max_retry
        self.rows: List[Dict[str, Any]] = self._load_metadata(self.metadata_path)

        # Spatial processor shared by video and mask frames.
        frame_processor = ImageCropAndResize(
            height, width, max_pixels,
            height_division_factor, width_division_factor,
            min_pixels=None,
        )
        self._frame_processor = frame_processor

        # Video loader (src / tgt).
        self._video_loader = LoadVideo(
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            frame_processor=frame_processor,
            rand_num_frames=rand_num_frames,
            min_num_frames=min_num_frames,
            rand_num_frames_step=rand_num_frames_step,
        )

        # Mask loader (per-instruction masks + combined mask).
        self._mask_loader = LoadMaskVideo(
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            frame_processor=frame_processor,
            rand_num_frames=rand_num_frames,
            min_num_frames=min_num_frames,
            rand_num_frames_step=rand_num_frames_step,
        )

        # Operators that resolve a relative path under base_path and load it.
        self._video_op = RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> self._video_loader),
        ])
        self._mask_op = RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> self._mask_loader),
        ])

    # ------------------------------------------------------------------ metadata

    @staticmethod
    def _load_metadata(path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[CoinVEDataset] skip malformed line {line_no}: {e}")
        print(f"[CoinVEDataset] loaded {len(rows)} rows from {path}")
        return rows

    # ------------------------------------------------------------------ helpers

    def _resolve(self, rel_path: str) -> str:
        return os.path.join(self.base_path, rel_path)

    def _sync_frame_count(self, src_rel: str) -> Optional[int]:
        """Pre-sample a target frame count from the source video so that
        src/tgt/mask all load the same number of frames."""
        src_len = self._raw_frame_count(self._resolve(src_rel))
        if src_len <= 0:
            return None
        return self._video_loader.sample_target_num_frames(src_len)

    @staticmethod
    def _raw_frame_count(path: str) -> int:
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n if n > 0 else -1
        except Exception:
            return -1

    # ------------------------------------------------------------------ __getitem__

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        retry = 0
        while retry < self.max_retry:
            try:
                row = self.rows[idx % len(self.rows)]
                src_rel = row["source_video_path"]
                tgt_rel = row["edited_video_path"]

                # Pre-sample a shared frame count so src/tgt/mask stay in sync.
                override = self._sync_frame_count(src_rel)
                self._video_loader._override_num_frames = override
                self._mask_loader._override_num_frames = override

                src_frames = self._video_op(src_rel)
                tgt_frames = self._video_op(tgt_rel)

                # Per-instruction mask videos.
                instr_masks_rel: List[str] = row.get("instruction_mask_video_paths", []) or []
                instr_masks: List[List[Any]] = []
                for mrel in instr_masks_rel:
                    if not mrel:
                        instr_masks.append([])
                        continue
                    instr_masks.append(self._mask_op(mrel))

                # Combined mask (optional).
                combined_rel = row.get("combined_mask_video_path", "") or ""
                combined_frames = self._mask_op(combined_rel) if combined_rel else None

                # Reset override state after successful load.
                self._video_loader._override_num_frames = None
                self._mask_loader._override_num_frames = None

                return {
                    "src_video": src_frames,
                    "tgt_video": tgt_frames,
                    "instruction_masks": instr_masks,
                    "combined_mask": combined_frames,
                    "instruction": list(row.get("instruction", [])),
                    "instruction_operation": list(row.get("instruction_operation", [])),
                    "instruction_object": list(row.get("instruction_object", [])),
                    "source_video_path": src_rel,
                    "edited_video_path": tgt_rel,
                }
            except Exception as e:
                self._video_loader._override_num_frames = None
                self._mask_loader._override_num_frames = None
                print(f"[CoinVEDataset] error loading idx={idx}: {e}")
                retry += 1
                idx = random.randint(0, len(self.rows) - 1)
        raise RuntimeError(f"[CoinVEDataset] failed after {self.max_retry} retries")

    def __len__(self) -> int:
        return len(self.rows)


# Convenience function for ad-hoc inspection / debugging.
def build_default_dataset(
    base_path: str,
    metadata_path: Optional[str] = None,
    num_frames: int = 81,
    max_pixels: int = 1920 * 1080,
) -> CoinVEDataset:
    return CoinVEDataset(
        base_path=base_path,
        metadata_path=metadata_path,
        num_frames=num_frames,
        max_pixels=max_pixels,
    )
