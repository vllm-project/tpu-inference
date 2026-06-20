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
"""Curated KernelBench task subset, TPU-ported.

Each ``KernelTask`` has:
* A reference implementation in pure JAX (the *correctness oracle*).
* An evolve-targetable Pallas implementation file (the *baseline kernel*).
* Input generator and shape parameters.
* Dtype tolerance.

This module ships 10 representative tasks: 5 Level-1 (single primitives)
and 5 Level-2 (composed ops). The full Stanford set is 250.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class KernelTask:
    """One task in the benchmark suite."""
    name: str
    level: int  # 1 = primitive, 2 = composed, 3 = full model layer
    description: str
    # Pure-JAX reference that defines correctness.
    reference: Callable[..., jax.Array]
    # Input generator: takes a JAX PRNG key, returns a tuple of inputs.
    make_inputs: Callable[[jax.Array], tuple]
    # Dtype tolerance for the verifier (atol, rtol).
    atol: float
    rtol: float
    # Cosine floor for the verifier.
    cosine_floor: float = 0.999
    # Shape descriptor (used for keying telemetry).
    shape_key: str = ""


# ---- Level 1: single primitives ----------------------------------------------


def _mm_4096_4096(key):
    a = jax.random.normal(key, (4096, 4096), dtype=jnp.bfloat16)
    b = jax.random.normal(key, (4096, 4096), dtype=jnp.bfloat16)
    return (a, b)


def _mm_ref(a, b):
    return jnp.matmul(a, b).astype(a.dtype)


def _gelu_2k(key):
    return (jax.random.normal(key, (2048, 2048), dtype=jnp.bfloat16), )


def _gelu_ref(x):
    return jax.nn.gelu(x)


def _softmax_8k_2k(key):
    return (jax.random.normal(key, (8192, 2048), dtype=jnp.bfloat16), )


def _softmax_ref(x):
    return jax.nn.softmax(x, axis=-1)


def _layernorm_2k(key):
    return (jax.random.normal(key, (2048, 1024), dtype=jnp.bfloat16), )


def _layernorm_ref(x):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean)**2).mean(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)


def _reduce_sum(key):
    return (jax.random.normal(key, (1024, 4096), dtype=jnp.bfloat16), )


def _reduce_sum_ref(x):
    return jnp.sum(x, axis=-1)


# ---- Level 2: composed ----------------------------------------------------


def _qkv_proj(key):
    keys = jax.random.split(key, 4)
    x = jax.random.normal(keys[0], (2048, 1024), dtype=jnp.bfloat16)
    wq = jax.random.normal(keys[1], (1024, 1024), dtype=jnp.bfloat16)
    wk = jax.random.normal(keys[2], (1024, 512), dtype=jnp.bfloat16)
    wv = jax.random.normal(keys[3], (1024, 512), dtype=jnp.bfloat16)
    return (x, wq, wk, wv)


def _qkv_proj_ref(x, wq, wk, wv):
    return (jnp.matmul(x, wq), jnp.matmul(x, wk), jnp.matmul(x, wv))


def _scaled_dot_product(key):
    keys = jax.random.split(key, 3)
    q = jax.random.normal(keys[0], (16, 256, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (16, 256, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (16, 256, 64), dtype=jnp.bfloat16)
    return (q, k, v)


def _scaled_dot_product_ref(q, k, v):
    scale = jnp.float32(1.0 / jnp.sqrt(jnp.float32(q.shape[-1])))
    attn = jnp.einsum("hqd,hkd->hqk", q, k,
                      preferred_element_type=jnp.float32) * scale
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("hqk,hkd->hqd", attn, v)


def _ffn_silu_gate(key):
    keys = jax.random.split(key, 3)
    x = jax.random.normal(keys[0], (2048, 1024), dtype=jnp.bfloat16)
    w_gate = jax.random.normal(keys[1], (1024, 3072), dtype=jnp.bfloat16)
    w_up = jax.random.normal(keys[2], (1024, 3072), dtype=jnp.bfloat16)
    return (x, w_gate, w_up)


def _ffn_silu_gate_ref(x, w_gate, w_up):
    gate = jax.nn.silu(jnp.matmul(x, w_gate))
    up = jnp.matmul(x, w_up)
    return gate * up


def _rmsnorm(key):
    keys = jax.random.split(key, 2)
    x = jax.random.normal(keys[0], (2048, 1024), dtype=jnp.bfloat16)
    g = jax.random.normal(keys[1], (1024, ), dtype=jnp.bfloat16)
    return (x, g)


def _rmsnorm_ref(x, g):
    var = (x.astype(jnp.float32)**2).mean(axis=-1, keepdims=True)
    return ((x.astype(jnp.float32) / jnp.sqrt(var + 1e-6)) *
            g.astype(jnp.float32)).astype(x.dtype)


def _rope(key):
    keys = jax.random.split(key, 2)
    x = jax.random.normal(keys[0], (16, 256, 64), dtype=jnp.bfloat16)
    pos = jnp.arange(256)[:, None] * jnp.arange(32)[None, :] * 0.01
    return (x, pos.astype(jnp.float32))


def _rope_ref(x, pos):
    # Apply rotary embedding to last-dim pairs.
    cos = jnp.cos(pos).astype(x.dtype)
    sin = jnp.sin(pos).astype(x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos[None, :, :] - x_odd * sin[None, :, :]
    rotated_odd = x_even * sin[None, :, :] + x_odd * cos[None, :, :]
    out = jnp.stack([rotated_even, rotated_odd], axis=-1)
    return out.reshape(x.shape)


TASKS: list[KernelTask] = [
    # Level 1.
    KernelTask("matmul_4096",
               1,
               "bf16 matmul (4096, 4096) @ (4096, 4096)",
               _mm_ref,
               _mm_4096_4096,
               atol=0.1,
               rtol=0.1,
               shape_key="bf16_mm_4096"),
    KernelTask("gelu_2k",
               1,
               "bf16 GELU (2048, 2048)",
               _gelu_ref,
               _gelu_2k,
               atol=0.01,
               rtol=0.01,
               shape_key="bf16_gelu_2k"),
    KernelTask("softmax_8k_2k",
               1,
               "bf16 softmax (8192, 2048) along last dim",
               _softmax_ref,
               _softmax_8k_2k,
               atol=0.01,
               rtol=0.01,
               shape_key="bf16_softmax_8k2k"),
    KernelTask("layernorm_2k",
               1,
               "bf16 LayerNorm (2048, 1024)",
               _layernorm_ref,
               _layernorm_2k,
               atol=0.01,
               rtol=0.01,
               shape_key="bf16_ln_2k"),
    KernelTask("reduce_sum",
               1,
               "bf16 sum reduction along axis -1",
               _reduce_sum_ref,
               _reduce_sum,
               atol=0.5,
               rtol=0.05,
               shape_key="bf16_sum"),
    # Level 2.
    KernelTask("qkv_proj",
               2,
               "Q/K/V projection (3 matmuls share input)",
               _qkv_proj_ref,
               _qkv_proj,
               atol=0.1,
               rtol=0.1,
               shape_key="bf16_qkv"),
    KernelTask("scaled_dot_product",
               2,
               "multi-head scaled dot-product (h=16, q=k=256, d=64)",
               _scaled_dot_product_ref,
               _scaled_dot_product,
               atol=0.05,
               rtol=0.05,
               shape_key="bf16_sdp"),
    KernelTask("ffn_silu_gate",
               2,
               "FFN gate*up with SiLU",
               _ffn_silu_gate_ref,
               _ffn_silu_gate,
               atol=0.1,
               rtol=0.1,
               shape_key="bf16_ffn"),
    KernelTask("rmsnorm",
               2,
               "RMSNorm with bf16 store / fp32 compute",
               _rmsnorm_ref,
               _rmsnorm,
               atol=0.01,
               rtol=0.01,
               shape_key="bf16_rms"),
    KernelTask("rope",
               2,
               "RoPE applied to 16-head queries (256 seq, 64 dim)",
               _rope_ref,
               _rope,
               atol=0.05,
               rtol=0.05,
               shape_key="bf16_rope"),
]
