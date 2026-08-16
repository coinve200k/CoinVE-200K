"""IO and sampling utilities shared by training/inference scripts.

Extracted from `train_mask_q_blending_composite_predmask.py` so that the
residual-attention composite path can reuse them without importing the
full q-blending training module.

Contents:
    - RatioShardSampler            : per-epoch no-replacement sampler with
                                      fixed source ratios.
    - upload_to_s3_and_cleanup     : upload a single file to S3, then delete it.
    - upload_dir_to_s3_and_cleanup : upload a directory tree to S3, then remove it.
"""
from __future__ import annotations

import os
import shutil

import torch
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
class RatioShardSampler(Sampler[int]):
    """No-replacement per-epoch sampler with fixed source ratios.

    Each source dataset is shuffled once at initialization. Consecutive epochs
    walk through non-overlapping shards of that fixed shuffled order, wrapping
    back to the beginning without reshuffling when a source is exhausted.
    """

    def __init__(self, dataset_lengths: list[int], ratios: list[float], seed: int = 0):
        if len(dataset_lengths) != len(ratios):
            raise ValueError("dataset_lengths and ratios must have the same length")
        if any(n <= 0 for n in dataset_lengths):
            raise ValueError("all dataset lengths must be positive")
        if any(r <= 0 for r in ratios):
            raise ValueError("all ratios must be positive")
        ratio_sum = float(sum(ratios))
        self.ratios = [float(r) / ratio_sum for r in ratios]
        self.dataset_lengths = [int(n) for n in dataset_lengths]
        self.offsets = []
        offset = 0
        for n in self.dataset_lengths:
            self.offsets.append(offset)
            offset += n
        epoch_total = min(n / r for n, r in zip(self.dataset_lengths, self.ratios))
        self.counts = [min(n, int(epoch_total * r)) for n, r in zip(self.dataset_lengths, self.ratios)]
        self.counts = [max(1, c) for c in self.counts]
        self.seed = int(seed)
        self.permutations = []
        for source_idx, n in enumerate(self.dataset_lengths):
            gen = torch.Generator()
            gen.manual_seed(self.seed + source_idx * 1000003)
            self.permutations.append(torch.randperm(n, generator=gen))
        self.epoch = 0

    def __len__(self) -> int:
        return int(sum(self.counts))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _source_indices(self, source_idx: int) -> list[int]:
        n = self.dataset_lengths[source_idx]
        count = self.counts[source_idx]
        start = (self.epoch * count) % n
        take = count
        chunks = []
        cur_start = start
        perm = self.permutations[source_idx]
        while take > 0:
            n_take = min(take, n - cur_start)
            chunks.append(perm[cur_start:cur_start + n_take])
            take -= n_take
            cur_start = 0
        local = torch.cat(chunks).tolist()
        base = self.offsets[source_idx]
        return [base + int(i) for i in local]

    def __iter__(self):
        indices = []
        for source_idx in range(len(self.dataset_lengths)):
            indices.extend(self._source_indices(source_idx))
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch * 104729)
        order = torch.randperm(len(indices), generator=gen).tolist()
        return iter([indices[i] for i in order])


# ---------------------------------------------------------------------------
# S3 upload utilities
# ---------------------------------------------------------------------------
def _parse_s3_uri(s3_uri: str):
    """Parse 's3://bucket/prefix' into (bucket, prefix)."""
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")


def upload_to_s3_and_cleanup(local_path: str, s3_base_path: str, relative_key: str):
    """Upload a local file to S3 and delete it on success.

    Args:
        local_path: absolute path to the local file.
        s3_base_path: S3 URI base, e.g. s3://bucket/prefix/
        relative_key: relative key under s3_base_path.
    """
    import boto3
    bucket, prefix = _parse_s3_uri(s3_base_path)
    s3_key = f"{prefix}/{relative_key}" if prefix else relative_key
    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, s3_key)
        os.remove(local_path)
        print(f"[s3] uploaded & removed: {local_path} -> s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"[s3][ERROR] upload failed for {local_path}: {e} (local file kept)")


def upload_dir_to_s3_and_cleanup(local_dir: str, s3_base_path: str, relative_prefix: str):
    """Upload all files in a local directory to S3 and remove the directory.

    Args:
        local_dir: absolute path to the local directory.
        s3_base_path: S3 URI base.
        relative_prefix: prefix under s3_base_path for the directory contents.
    """
    import boto3
    bucket, prefix = _parse_s3_uri(s3_base_path)
    s3 = boto3.client("s3")
    all_ok = True
    for root, _dirs, files in os.walk(local_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, local_dir)
            s3_key = f"{prefix}/{relative_prefix}/{rel}" if prefix else f"{relative_prefix}/{rel}"
            try:
                s3.upload_file(fpath, bucket, s3_key)
            except Exception as e:
                print(f"[s3][ERROR] upload failed for {fpath}: {e}")
                all_ok = False
    if all_ok:
        shutil.rmtree(local_dir, ignore_errors=True)
        print(f"[s3] uploaded & removed dir: {local_dir} -> s3://{bucket}/{prefix}/{relative_prefix}/")
    else:
        print(f"[s3] some uploads failed; local dir kept: {local_dir}")
