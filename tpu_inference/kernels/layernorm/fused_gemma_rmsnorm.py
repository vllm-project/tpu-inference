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
"""One-pass Pallas (fused add +) GemmaRMSNorm ported from vllm-torchtpu.

At prefill shapes (T >= 512), XLA lowers vllm's fused_add_rms_norm as two
separate HBM passes (reduction + apply). Keeping each [block_rows, hidden] tile
resident in VMEM performs add, variance reduction, rsqrt, and (1+w) scale in a
single HBM pass (1.21x speedup at T=8192, H=5120 on TPU v6e).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK_ROWS = 128


def _gemma_rmsnorm_kernel(*refs, eps: float, has_residual: bool):
    if has_residual:
        x_ref, r_ref, w_ref, y_ref, ro_ref = refs
        s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        ro_ref[...] = s.astype(ro_ref.dtype)
    else:
        x_ref, w_ref, y_ref = refs
        s = x_ref[...].astype(jnp.float32)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    # Match GemmaRMSNorm.forward_native: weight = self.weight.float() + 1.0,
    # applied in f32 before casting to orig_dtype.
    normed = s * jax.lax.rsqrt(var + eps)
    w = w_ref[...].astype(jnp.float32) + 1.0
    y_ref[...] = (normed * w).astype(y_ref.dtype)


def supports(
    num_rows: int,
    hidden: int,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    min_rows: int = 512,
) -> bool:
    return (num_rows >= min_rows and num_rows % block_rows == 0
            and hidden % 128 == 0 and hidden <= 16384)


@functools.partial(jax.jit, static_argnames=("eps", "block_rows"))
def fused_add_gemma_rms_norm(
    x: jax.Array,
    residual: jax.Array | None,
    weight: jax.Array,
    *,
    eps: float,
    block_rows: int = DEFAULT_BLOCK_ROWS,
):
    rows, hidden = x.shape
    has_residual = residual is not None
    row_spec = pl.BlockSpec((block_rows, hidden), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    out_shape = [jax.ShapeDtypeStruct((rows, hidden), x.dtype)]
    out_specs = [row_spec]
    in_specs = [row_spec]
    args = [x]
    if has_residual:
        in_specs.append(row_spec)
        args.append(residual)
        out_shape.append(jax.ShapeDtypeStruct((rows, hidden), x.dtype))
        out_specs.append(row_spec)
    in_specs.append(w_spec)
    args.append(weight.reshape(1, hidden))
    tile = block_rows * hidden * 4
    vmem_limit = min(64 * 1024 * 1024, max(32 * 1024 * 1024, 12 * tile))
    outs = pl.pallas_call(
        functools.partial(_gemma_rmsnorm_kernel,
                          eps=eps,
                          has_residual=has_residual),
        grid=(rows // block_rows, ),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", ),
            vmem_limit_bytes=vmem_limit,
        ),
        name=
        f"fused_add_gemma_rmsnorm_r{int(has_residual)}_b{block_rows}_h{hidden}",
    )(*args)
    if has_residual:
        return outs[0], outs[1]
    return outs[0]
