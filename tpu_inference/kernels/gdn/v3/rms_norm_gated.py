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
"""Single-pass Pallas RMSNormGated kernel for TPU (GDN prefill/mixed path)."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _rms_norm_gated_kernel(x_ref, z_ref, w_ref, out_ref, *, eps: float):
    xf = x_ref[...].astype(jnp.float32)
    zf = z_ref[...].astype(jnp.float32)
    wf = w_ref[...].astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    out = (xf * jax.lax.rsqrt(var + eps) * wf) * jax.nn.silu(zf)
    out_ref[...] = out.astype(out_ref.dtype)


def _jax_rms_norm_gated(
    x: jax.Array,
    z: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
    head_dim: int,
) -> jax.Array:
    orig_shape = x.shape
    xf = x.reshape(-1, head_dim).astype(jnp.float32)
    zf = z.reshape(-1, head_dim).astype(jnp.float32)
    wf = weight.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    out = (xf * jax.lax.rsqrt(var + eps) * wf) * jax.nn.silu(zf)
    return out.astype(x.dtype).reshape(orig_shape)


def _pick_block_rows(rows: int) -> int | None:
    if rows < 384:
        return None
    for cand in (3072, 1536, 768, 512, 384, 256, 128):
        if rows % cand == 0:
            return cand
    return None


def pallas_rms_norm_gated(
    x: jax.Array,
    z: jax.Array,
    weight: jax.Array,
    *,
    eps: float = 1e-6,
    head_dim: int = 128,
) -> jax.Array:
    """Single-pass RMSNormGated (norm_before_gate=True, activation='silu').

    Supports arbitrary leading shape (e.g. [T, n_v * d_v] or [T, n_v, d_v])
    where the total element count is divisible by head_dim and head_dim % 128 == 0.
    """
    orig_shape = x.shape
    total_elems = x.size
    if head_dim % 128 != 0 or total_elems % head_dim != 0:
        return _jax_rms_norm_gated(x, z, weight, eps=eps, head_dim=head_dim)

    rows = total_elems // head_dim
    block_rows = _pick_block_rows(rows)
    if block_rows is None:
        return _jax_rms_norm_gated(x, z, weight, eps=eps, head_dim=head_dim)

    x2 = x.reshape(rows, head_dim)
    z2 = z.reshape(rows, head_dim)
    w2 = weight.reshape(1, head_dim)
    row_spec = pl.BlockSpec((block_rows, head_dim), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, head_dim), lambda i: (0, 0))
    out2 = pl.pallas_call(
        functools.partial(_rms_norm_gated_kernel, eps=eps),
        grid=(rows // block_rows, ),
        in_specs=[row_spec, row_spec, w_spec],
        out_specs=row_spec,
        out_shape=jax.ShapeDtypeStruct((rows, head_dim), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", ),
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name=f"pallas_rmsnorm_gated_br{block_rows}_hd{head_dim}",
    )(x2, z2, w2)
    return out2.reshape(orig_shape)
