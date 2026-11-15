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

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchax.ops.mappings import t2j

# NVFP4 block size (elements per scale group).
NVFP4_GROUP_SIZE = 16


def quantize_to_nvfp4(weight: torch.Tensor,
                      group_size: int = NVFP4_GROUP_SIZE):
    """Quantize a float32 weight to NVFP4 format (packed uint8 + scales).

    Returns:
        weight_packed: uint8 [out, in//2]
        weight_scale: float8_e4m3fn [out, in//group_size]
        weight_global_scale: float32 scalar
    """
    assert weight.ndim == 2
    out_size, in_size = weight.shape
    assert in_size % group_size == 0

    # Use JAX for FP4 quantization since torch doesn't have native FP4.
    w_jax = t2j(weight.float())

    # Compute per-block scales.
    num_blocks = in_size // group_size
    w_blocked = w_jax.reshape(out_size, num_blocks, group_size)
    block_abs_max = jnp.max(jnp.abs(w_blocked), axis=2, keepdims=True)

    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    block_scale = block_abs_max / fp4_max  # [out, num_blocks, 1]
    block_scale = jnp.where(block_scale == 0, 1.0, block_scale)

    # Compute global scale: max of all block scales.
    global_scale = jnp.max(block_scale)

    # Effective block scale = block_scale / global_scale (stored as FP8).
    effective_scale = (block_scale / global_scale).astype(jnp.float8_e4m3fn)

    # Quantize to FP4.
    scale_inv = jnp.where(block_scale == 0, 0.0, 1.0 / block_scale)
    w_q = jnp.clip(w_blocked * scale_inv, -fp4_max,
                   fp4_max).astype(jnp.float4_e2m1fn)
    w_q = w_q.reshape(out_size, in_size)

    # Pack FP4 into uint8 (2 values per byte).
    w_packed = w_q.reshape(out_size, in_size // 2, 2)
    w_packed = jax.lax.bitcast_convert_type(w_packed, jnp.uint8)

    effective_scale = effective_scale.reshape(out_size, num_blocks)

    # Convert via numpy to avoid j2t FP8 dtype issues.
    w_packed_t = torch.from_numpy(np.asarray(w_packed))
    scale_t = torch.from_numpy(np.asarray(effective_scale).view(
        np.uint8)).view(torch.float8_e4m3fn)
    global_t = torch.tensor(float(global_scale), dtype=torch.float32)
    return (w_packed_t, scale_t, global_t)


def ref_dequant_nvfp4(weight_packed, weight_scale, global_scale, group_size):
    """Reference dequantization: unpack → float32 using block_scale * global."""
    w_jax = jnp.array(weight_packed.numpy())
    # FP8 scale: go through uint8 view to avoid dtype conversion issues.
    s_np = weight_scale.view(torch.uint8).numpy()
    s_jax = jax.lax.bitcast_convert_type(jnp.array(s_np), jnp.float8_e4m3fn)
    g_jax = jnp.float32(global_scale.item())

    # Unpack uint8 → float4_e2m1fn.
    e2m1 = jax.lax.bitcast_convert_type(w_jax, jnp.float4_e2m1fn)
    fp4 = jnp.reshape(e2m1, e2m1.shape[:-2] + (-1, ))

    # Fold scales.
    eff_scale = s_jax.astype(jnp.float32) * g_jax
    out_size = fp4.shape[0]
    in_size = fp4.shape[1]
    num_blocks = in_size // group_size

    fp4_blocked = fp4.reshape(out_size, num_blocks, group_size)
    scale_expanded = eff_scale.reshape(out_size, num_blocks, 1)
    deq = (fp4_blocked.astype(jnp.float32) * scale_expanded).reshape(
        out_size, in_size)
    return torch.from_numpy(np.asarray(deq))
