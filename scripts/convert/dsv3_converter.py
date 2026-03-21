# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert DeepSeek-V3 FP8 2D-subchannel [128,128] to 1D-subchannel [1,N].

Optionally quantize MoE expert weights to FP4 packed uint8 with --fp4.

Usage:
    python dsv3_converter.py \
        --input /path/to/DeepSeek-V3.1 \
        --output /path/to/DeepSeek-V3.1-1D-256

    python dsv3_converter.py \
        --input /path/to/DeepSeek-V3.1 \
        --output /path/to/DeepSeek-V3.1-FP4-MoE \
        --fp4
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import copy
import json
import logging
import math
import shutil
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save
from tqdm import tqdm

from tpu_inference.layers.common.quantization import quantize_tensor

FP8_MAX = float(ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max)  # 448.0
FP4_MAX = 6.0

log = logging.getLogger(__name__)


def torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8).numpy().view(
            ml_dtypes.float8_e4m3fn).reshape(t.shape)
    if t.dtype == torch.bfloat16:
        return t.float().numpy().astype(ml_dtypes.bfloat16)
    return t.numpy()


def numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
    if arr.dtype == ml_dtypes.float8_e4m3fn:
        raw = torch.from_numpy(arr.view(np.uint8).copy())
        return raw.view(torch.float8_e4m3fn)
    if arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(arr.astype(np.float32).copy()).bfloat16()
    return torch.from_numpy(np.ascontiguousarray(arr))


def dequantize_fp8_2d(weight_fp8, scale_inv, block_size=(128, 128)):
    """FP8 + 2D block scale → FP32."""
    M, N = weight_fp8.shape
    bh, bw = block_size

    w = weight_fp8.view(ml_dtypes.float8_e4m3fn).astype(np.float32)

    pad_m = (bh - M % bh) % bh
    pad_n = (bw - N % bw) % bw
    if pad_m or pad_n:
        w = np.pad(w, ((0, pad_m), (0, pad_n)))

    Mp, Np = w.shape
    nb_m, nb_n = Mp // bh, Np // bw

    blocked = w.reshape(nb_m, bh, nb_n, bw)
    sc = scale_inv[:nb_m, :nb_n].astype(np.float32)[:, None, :, None]

    return (blocked * sc).reshape(Mp, Np)[:M, :N]


def quantize_fp8_1d(weight_f32, block_size=256, scale_max=FP4_MAX):
    """FP32 → FP8 with 1D block scale. Uses FP4 max (6.0) by default."""
    M, N = weight_f32.shape

    pad_n = (block_size - N % block_size) % block_size
    if pad_n:
        weight_f32 = np.pad(weight_f32, ((0, 0), (0, pad_n)))

    Np = weight_f32.shape[1]
    nb_n = Np // block_size

    blocked = weight_f32.reshape(M, nb_n, block_size)
    abs_max = np.maximum(np.max(np.abs(blocked), axis=2, keepdims=True), 1e-12)
    scale_inv = (abs_max / scale_max).astype(np.float32)

    w = np.clip(blocked / scale_inv, -FP8_MAX, FP8_MAX)
    w = w.astype(ml_dtypes.float8_e4m3fn).reshape(M, Np)[:, :N].copy()

    s = scale_inv.reshape(M, nb_n)[:, :math.ceil(N / block_size)]
    return w, s


def _process_shard(shard_file,
                   input_path,
                   output_path,
                   weight_map,
                   block_size,
                   src_block_size,
                   output_fp4,
                   scale_max,
                   fp4_block_size=512):
    file_name = os.path.basename(shard_file)
    t0 = time.time()

    torch_dict = safetensors_load(shard_file)
    state = {k: torch_to_numpy(v) for k, v in torch_dict.items()}
    del torch_dict

    loaded_files = {file_name: state}

    def get_tensor(name):
        fn = weight_map[name]
        if fn not in loaded_files:
            td = safetensors_load(os.path.join(input_path, fn))
            loaded_files[fn] = {k: torch_to_numpy(v) for k, v in td.items()}
        return loaded_files[fn][name]

    out = {}
    wmap = {}
    converted = passthrough = 0

    for name, tensor in state.items():
        if name.endswith("_scale_inv"):
            continue

        scale_name = f"{name}_scale_inv"

        if scale_name in weight_map:
            try:
                scale = get_tensor(scale_name)
            except KeyError:
                out[name] = tensor
                wmap[name] = file_name
                passthrough += 1
                continue

            w_f32 = dequantize_fp8_2d(tensor, scale, src_block_size)

            # MoE experts → FP4 packed uint8, everything else → FP8 1D
            if output_fp4 and ".mlp.experts." in name:
                w_jax = jnp.array(w_f32, dtype=jnp.float32)
                w_fp4, s_fp4 = quantize_tensor(jnp.float4_e2m1fn,
                                               w_jax,
                                               axis=1,
                                               block_size=fp4_block_size)
                # pack 2 fp4 values per byte
                packed = jax.lax.bitcast_convert_type(
                    w_fp4.reshape(w_fp4.shape[:-1] + (-1, 2)), jnp.uint8)
                out[name] = torch.from_numpy(np.array(packed).copy())
                out[scale_name] = np.array(s_fp4)
            else:
                new_w, new_s = quantize_fp8_1d(w_f32, block_size, scale_max)
                out[name] = new_w
                out[scale_name] = new_s

            wmap[name] = file_name
            wmap[scale_name] = file_name
            converted += 1
        else:
            out[name] = tensor
            wmap[name] = file_name
            passthrough += 1

    torch_out = {
        k: v if isinstance(v, torch.Tensor) else numpy_to_torch(v)
        for k, v in out.items()
    }
    safetensors_save(torch_out, os.path.join(output_path, file_name))

    elapsed = time.time() - t0
    log.debug("[%s] %.1fs — converted=%d passthrough=%d", file_name, elapsed,
              converted, passthrough)

    return {
        "file_name": file_name,
        "converted": converted,
        "passthrough": passthrough,
        "weight_map": wmap
    }


def main(input_path,
         output_path,
         block_size=256,
         src_block_size=(128, 128),
         output_fp4=False,
         scale_max=FP4_MAX,
         fp4_block_size=512,
         workers=15):
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(input_path, "model.safetensors.index.json")) as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    shards = sorted(glob(os.path.join(input_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors in {input_path}")

    mode = "FP8[2D]→FP8[1D]"
    if output_fp4:
        mode += f" + MoE→FP4[bs={fp4_block_size}]"
    print(
        f"{len(shards)} shards | {mode} | block_size={block_size} | workers={workers}"
    )

    new_wmap = {}
    n_converted = n_pass = 0
    args = (input_path, output_path, weight_map, block_size, src_block_size,
            output_fp4, scale_max, fp4_block_size)

    if workers == 1:
        for sf in tqdm(shards, desc="Shards"):
            r = _process_shard(sf, *args)
            new_wmap.update(r["weight_map"])
            n_converted += r["converted"]
            n_pass += r["passthrough"]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_shard, sf, *args): sf for sf in shards}
            failed = []
            with tqdm(total=len(shards), desc="Shards") as pbar:
                for fut in as_completed(futs):
                    try:
                        r = fut.result()
                    except Exception as e:
                        failed.append(os.path.basename(futs[fut]))
                        log.error("%s FAILED: %s", failed[-1], e)
                        pbar.update(1)
                        continue
                    new_wmap.update(r["weight_map"])
                    n_converted += r["converted"]
                    n_pass += r["passthrough"]
                    pbar.update(1)
            if failed:
                print(f"\nWARNING: {len(failed)} shard(s) failed: {failed}")

    print(f"\nConverted: {n_converted}, Passthrough: {n_pass}")

    # write index
    idx = copy.deepcopy(model_index)
    idx["weight_map"] = new_wmap
    with open(os.path.join(output_path, "model.safetensors.index.json"),
              "w") as f:
        json.dump(idx, f, indent=2)

    # write config
    with open(os.path.join(input_path, "config.json")) as f:
        config = json.load(f)

    if "quantization_config" in config:
        config["quantization_config"]["weight_block_size"] = [1, block_size]
        if output_fp4:
            config["quantization_config"][
                "moe_weight_dtype"] = "fp4_packed_uint8"
            config["quantization_config"][
                "moe_fp4_block_size"] = fp4_block_size
    else:
        config["quantization_config"] = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [1, block_size],
        }

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # copy tokenizer etc
    for fname in [
            "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "configuration_deepseek.py", "modeling_deepseek.py"
    ]:
        src = os.path.join(input_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, fname))

    print(f"Output: {output_path}")


if __name__ == "__main__":
    p = ArgumentParser(
        description="DeepSeek-V3 FP8 2D→1D subchannel converter")
    p.add_argument("--input", required=True, help="Input model path")
    p.add_argument("--output", required=True, help="Output path")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--fp4",
                   action="store_true",
                   help="Also quantize MoE experts to FP4 packed uint8")
    p.add_argument("--fp4-block-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    main(args.input,
         args.output,
         args.block_size,
         output_fp4=args.fp4,
         fp4_block_size=args.fp4_block_size,
         workers=args.workers)
