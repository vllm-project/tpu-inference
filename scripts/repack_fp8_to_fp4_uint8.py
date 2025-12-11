#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
from glob import glob

import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Ensure project modules are importable if needed
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tpu_inference.models.jax.utils.quantization.mxfp4_utils import FP4_LUT


def map_values_to_fp4_codes(x: torch.Tensor) -> torch.Tensor:
    # Return int64 codes in [0,15] matching FP4_LUT. Preserve -0.0 as index 8, +0.0 as index 0.
    x32 = x.to(torch.float32)
    lut = FP4_LUT.to(x32.device)  # (16,)
    # Nearest LUT index
    diffs = (x32.unsqueeze(-1) - lut)
    idx = torch.argmin(diffs.abs(), dim=-1)  # (...)
    # Disambiguate zeros: map +0.0 -> 0, -0.0 -> 8 if present
    zero_mask = (x32 == 0)
    if hasattr(torch, "signbit"):
        neg_zero_mask = zero_mask & torch.signbit(x32)
    else:
        # Fallback: treat all zeros as +0
        neg_zero_mask = torch.zeros_like(zero_mask, dtype=torch.bool)
    pos_zero_mask = zero_mask & (~neg_zero_mask)
    idx = idx.masked_fill(pos_zero_mask, 0)
    idx = idx.masked_fill(neg_zero_mask, 8)
    return idx.to(torch.int64)


def pack_codes_to_uint8(codes: torch.Tensor) -> torch.Tensor:
    # codes: (..., C) int64 in [0,15]; returns uint8 (..., C//2) with low-high nibble order
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"Last dim must be even to pack nibbles, got {codes.shape}")
    c = codes.view(*codes.shape[:-1], -1, 2)  # (..., bytes, 2)
    low = (c[..., 0] & 0x0F).to(torch.uint8)
    high = ((c[..., 1] & 0x0F).to(torch.uint8) << 4)
    return (low | high).contiguous()


def _is_mlp_key(k: str) -> bool:
    # Convert any MLP tensor regardless of routed/shared naming
    return ".mlp." in k


def process_shard(src_path: str, dst_path: str, dry_run: bool = False) -> tuple[int, int]:
    changed = 0
    total = 0
    state = load_safetensors(src_path)
    new_state: dict[str, torch.Tensor] = {}

    for k, t in tqdm(list(state.items()), desc=os.path.basename(src_path)):
        total += 1
        do_convert = False
        if hasattr(torch, "float8_e4m3fn") and t.dtype == torch.float8_e4m3fn:
            # Convert ANY MLP float8 tensor (dense or experts)
            do_convert = _is_mlp_key(k)
        if do_convert:
            # Map to FP4 nibble codes and pack
            codes = map_values_to_fp4_codes(t)
            try:
                packed = pack_codes_to_uint8(codes)
            except Exception as e:
                print(f"[WARN] Pack failed for {k} in {os.path.basename(src_path)}: {e}")
                new_state[k] = t
                continue
            if dry_run:
                print(f"[DRY] {k}: float8_e4m3fn {tuple(t.shape)} -> uint8 {tuple(packed.shape)}")
                new_state[k] = t
            else:
                new_state[k] = packed
                changed += 1
        else:
            new_state[k] = t

    if dry_run:
        print(f"[DRY] Would write {dst_path} (changed={changed}/{total})")
    else:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        save_safetensors(new_state, dst_path)
        print(f"Wrote {dst_path} (changed={changed}/{total})")
    return changed, total


def main():
    ap = argparse.ArgumentParser(description="Repack MLP FP8 tensors into uint8 FP4 using the MXFP4 FP4_LUT (nibble-packed).")
    ap.add_argument("--src_dir", required=True, help="Source checkpoint directory with .safetensors")
    ap.add_argument("--dst_dir", required=True, help="Destination directory for repacked checkpoint")
    ap.add_argument("--dry_run", action="store_true", help="Don't write files; print planned conversions")
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    # Mirror non-safetensors files/directories from src->dst first
    entries = sorted(os.listdir(args.src_dir))
    for name in tqdm(entries, desc="Copying non-safetensors", leave=False):
        src_path = os.path.join(args.src_dir, name)
        dst_path = os.path.join(args.dst_dir, name)
        if os.path.isdir(src_path):
            # Copy entire directory tree
            try:
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            except Exception as e:
                print(f"[WARN] Failed to copy dir {name}: {e}")
            continue
        # Files: skip .safetensors (they will be written shard-by-shard)
        if name.endswith('.safetensors'):
            continue
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"[WARN] Failed to copy file {name}: {e}")

    # Process each safetensors shard
    total_changed = 0
    total_tensors = 0
    shard_files = sorted(glob(os.path.join(args.src_dir, "*.safetensors")))
    for src_path in tqdm(shard_files, desc="Repacking shards"):
        dst_path = os.path.join(args.dst_dir, os.path.basename(src_path))
        ch, tot = process_shard(src_path, dst_path, dry_run=args.dry_run)
        total_changed += ch
        total_tensors += tot

    print(f"Done. Converted {total_changed}/{total_tensors} tensors.")


if __name__ == "__main__":
    main()
