#!/usr/bin/env python3
import argparse
import os
import sys
from glob import glob

import torch
from safetensors.torch import load_file as load_safetensors

# Ensure we can import project modules when run from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tpu_inference.models.jax.utils.quantization.mxfp4_utils import unpack_mxfp4


def _list_all_keys(dir_path: str):
    keys = []
    for f in sorted(glob(os.path.join(dir_path, "*.safetensors"))):
        try:
            state = load_safetensors(f)
        except Exception:
            continue
        keys.extend(list(state.keys()))
    return keys


def find_tensor(dir_path: str, key: str, allow_fuzzy: bool = True):
    """Search all .safetensors shards under dir_path for `key` and its scale.

    If not found and allow_fuzzy, tries substring suggestions.
    Returns (weight_tensor, scale_tensor_or_None, filename)
    """
    shard_files = sorted(glob(os.path.join(dir_path, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors found under {dir_path}")

    scale_key = key.replace(".weight", ".weight_scale_inv")
    for f in shard_files:
        try:
            state = load_safetensors(f)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
            continue
        if key in state:
            w = state[key]
            s = state.get(scale_key)
            return w, s, f
    if allow_fuzzy:
        # Try fuzzy: list candidates that contain layer/expert and suffix
        suffix = key.split("model.")[-1]
        tail = suffix.split(".", 2)[-1] if "." in suffix else suffix
        candidates = []
        for f in shard_files:
            try:
                state = load_safetensors(f)
            except Exception:
                continue
            for k in state.keys():
                if tail in k:
                    candidates.append(k)
        suggestions = "\n  - " + "\n  - ".join(sorted(set(candidates))[:50]) if candidates else " (none)"
        raise KeyError(
            f"Key {key} not found under {dir_path}.\n"
            f"Here are up to 50 matching tails containing '{tail}':{suggestions}\n"
            f"Tip: re-run with --param set to an exact key from the suggestions or use --list to dump all keys.")
    raise KeyError(f"Key {key} not found under {dir_path}")


def prepare_value(weight: torch.Tensor,
                  scale: torch.Tensor | None,
                  out_dtype: torch.dtype,
                  mode: str = "codes") -> torch.Tensor:
    """Return comparison tensor using DeepSeek unpack logic.

    - mode="codes":
        If weight is uint8 (packed FP4), unpack to FP4 code values (float32) and cast.
        If weight is float-like, cast to out_dtype.
    - mode="dequant":
        If unpacked codes available and float-like scale present, apply per-block scale over the last dim.
        Block size is inferred as codes_last_dim // scale_last_dim if divisible.
    """
    if weight.dtype == torch.uint8:
        # Unpack low/high nibbles to FP4 code values (float32) using default codebook
        codes = unpack_mxfp4(weight).to(torch.float32)
        if mode == "dequant" and scale is not None and scale.dtype in (
                torch.bfloat16, torch.float16, torch.float32):
            # Direct broadcast multiply; no block assumptions
            dq = codes * scale.to(torch.float32)
            return dq.to(out_dtype)
        return codes.to(out_dtype)
    # Handle FP8 values (already quantized values, not packed codes)
    if hasattr(torch, "float8_e4m3fn") and weight.dtype == torch.float8_e4m3fn:
        if mode == "dequant" and scale is not None and scale.dtype in (
                torch.bfloat16, torch.float16, torch.float32):
            dq = weight.to(torch.float32) * scale.to(torch.float32)
            return dq.to(out_dtype)
        return weight.to(out_dtype)

    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return weight.to(out_dtype)
    raise TypeError(
        f"Unsupported dtype combination: weight={weight.dtype}, scale={None if scale is None else scale.dtype}")


def compare_tensors(a: torch.Tensor, b: torch.Tensor, topk: int = 10):
    if a.shape != b.shape:
        print(f"Shape mismatch: A{a.shape} vs B{b.shape}")
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    diff = (a32 - b32).abs()
    a_norm = torch.linalg.vector_norm(a.float())
    b_norm = torch.linalg.vector_norm(b.float())
    l2_rel = (torch.linalg.vector_norm(diff.float()) /
              (a_norm + 1e-6)).item()
    print("-- Stats --")
    print(f"dtype A={a.dtype} B={b.dtype}")
    print(f"shape A={tuple(a.shape)} B={tuple(b.shape)}")
    print(f"mean |A|={a.abs().mean().item():.6g} |B|={b.abs().mean().item():.6g}")
    print(f"mean |A-B|={diff.mean().item():.6g} max |A-B|={diff.max().item():.6g}")
    print(f"rel L2(A,B)={l2_rel:.6g}; ||A||={a_norm.item():.6g} ||B||={b_norm.item():.6g}")

    # Sign flip analysis: same magnitude within tol, different sign
    tol = 1e-3
    same_mag = (a32.abs() - b32.abs()).abs() <= tol
    sign_flip = (a32.sign() != b32.sign()) & same_mag
    if sign_flip.any():
        cnt = int(sign_flip.sum().item())
        print(f"sign-flip count (same |mag| within {tol}): {cnt}")

    # Top-K differences
    flat_diff = diff.flatten()
    if flat_diff.numel() > 0:
        topk = min(topk, flat_diff.numel())
        vals, idxs = torch.topk(flat_diff, k=topk)
        print("-- Top diffs (value_a, value_b, |diff|, index) --")
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            multi_idx = list(torch.unravel_index(torch.tensor(idx), a32.shape))
            a_val = a32.flatten()[idx].item()
            b_val = b32.flatten()[idx].item()
            print(f"A={a_val:.6g} B={b_val:.6g} |d|={v:.6g} @ {tuple(multi_idx)}")


def main():
    p = argparse.ArgumentParser(description="Compare a single weight between pack/non-pack checkpoints using MXFP4 dequant logic.")
    p.add_argument("--pack_dir", default="/mnt/pd/checkpoints/deepseek-r1-fp4-mlp-256-pack/",
                   help="Directory with packed MXFP4 checkpoint (codes+scales)")
    p.add_argument("--base_dir", default="/mnt/pd/checkpoints/deepseek-r1-fp4-mlp-256/",
                   help="Directory with baseline checkpoint (may be packed or dequantized)")
    p.add_argument("--param", required=False,
                   help="Fully-qualified weight name, e.g., model.layers.0.mlp.experts.0.down_proj.weight")
    p.add_argument("--list", action="store_true", help="List all keys in both directories and exit")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                   help="Comparison dtype after dequantization")
    p.add_argument("--mode", default="codes", choices=["codes", "dequant"],
                   help="Compare unpacked FP4 codes (default) or dequantized values (requires float scales)")
    p.add_argument("--topk", type=int, default=10, help="Report top-K absolute differences")
    args = p.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.list:
        print("-- pack_dir keys --")
        for k in _list_all_keys(args.pack_dir)[:2000]:
            print(k)
        print("-- base_dir keys --")
        for k in _list_all_keys(args.base_dir)[:2000]:
            print(k)
        return

    if not args.param:
        print("Error: --param is required unless --list is provided", file=sys.stderr)
        sys.exit(2)

    print(f"Loading from pack_dir={args.pack_dir}")
    w_pack, s_pack, f_pack = find_tensor(args.pack_dir, args.param)
    print(f"  Found in {f_pack}; weight dtype={w_pack.dtype} scale dtype={None if s_pack is None else s_pack.dtype}")

    print(f"Loading from base_dir={args.base_dir}")
    w_base, s_base, f_base = find_tensor(args.base_dir, args.param)
    print(f"  Found in {f_base}; weight dtype={w_base.dtype} scale dtype={None if s_base is None else s_base.dtype}")

    A = prepare_value(w_pack, s_pack, out_dtype,
                      mode=args.mode)
    B = prepare_value(w_base, s_base, out_dtype,
                      mode=args.mode)

    compare_tensors(A, B, topk=args.topk)


if __name__ == "__main__":
    main()
