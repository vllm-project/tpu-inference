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
"""Shared input validation for fused GDN kernels."""

from __future__ import annotations

import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental.pallas import tpu as pltpu


def validate_gdn_inputs(
    q,
    k,
    v,
    g,
    initial_state,
    state_indices,
    *,
    b=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
):
    """Validate shapes, dtypes, and TPU alignment for fused GDN kernels.

    Args:
        q: ``[T, H_qk, K]``.
        k: ``[T, H_qk, K]``.
        v: ``[T, H_v, V]``.
        g: ``[T, H_v, K]`` float32.
        initial_state: ``[num_states, H_v, K, V]`` float32.
        state_indices: ``[max_num_req]`` int32.
        b: ``[T, H_v, num_lanes]`` or ``None``.
        use_gate_in_kernel: Whether gate transformation is applied inside kernel.
        A_log: ``[H_v, num_lanes]`` float32 or ``None``.
        dt_bias: ``[H_v, num_lanes]`` float32 or ``None``.

    Returns:
        ``(T, H_qk, H_v, K, V, dtype, num_states, num_lanes, packing)``.
    """
    T, H_qk, K = q.shape
    H_v = v.shape[1]
    V = v.shape[2]
    dtype = q.dtype
    num_states = initial_state.shape[0]
    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 32 // dtypes.itemsize_bits(dtype)

    # Shape checks
    if k.shape != (T, H_qk, K):
        raise ValueError(f"k shape {k.shape} != q shape {q.shape}")
    if H_v % H_qk != 0:
        raise ValueError(f"H_v={H_v} must be a multiple of H_qk={H_qk}")
    if v.shape != (T, H_v, V):
        raise ValueError(f"v shape {v.shape} must be [{T}, {H_v}, {V}]")
    if g.shape != (T, H_v, K):
        raise ValueError(f"g shape {g.shape} must be [{T}, {H_v}, {K}]")
    if initial_state.shape[1:] != (H_v, K, V):
        raise ValueError(
            f"initial_state trailing dims {initial_state.shape[1:]} "
            f"must be ({H_v}, {K}, {V})")
    if b is not None and (b.ndim != 3 or b.shape[0] != T or b.shape[1] != H_v):
        raise ValueError(f"b shape {b.shape} must be [{T}, {H_v}, ...]")

    # TPU alignment
    if K % num_lanes != 0 or V % num_lanes != 0:
        raise ValueError(f"K={K}, V={V} must be multiples of {num_lanes}")
    if H_qk % packing != 0:
        raise ValueError(
            f"H_qk={H_qk} must be a multiple of packing={packing}")
    if H_v % packing != 0:
        raise ValueError(f"H_v={H_v} must be a multiple of packing={packing}")

    # Dtype checks
    if k.dtype != dtype or v.dtype != dtype:
        raise ValueError(f"q/k/v must share the same dtype, got q={dtype}, "
                         f"k={k.dtype}, v={v.dtype}")
    if g.dtype != jnp.float32:
        raise ValueError(f"g must be float32, got {g.dtype}")
    if initial_state.dtype not in (jnp.float32, jnp.bfloat16, jnp.float16):
        raise ValueError(
            f"initial_state must be float32, bfloat16, or float16, "
            f"got {initial_state.dtype}")
    if state_indices.dtype != jnp.int32:
        raise ValueError(
            f"state_indices must be int32, got {state_indices.dtype}")

    # Gate-in-kernel checks
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        if dt_bias is not None and (dt_bias.ndim != 2
                                    or dt_bias.shape[0] != H_v):
            raise ValueError(
                f"dt_bias shape {dt_bias.shape} must be [{H_v}, ...]")
        if dt_bias is not None and dt_bias.dtype != jnp.float32:
            raise ValueError(f"dt_bias must be float32, got {dt_bias.dtype}")

    return T, H_qk, H_v, K, V, dtype, num_states, num_lanes, packing
