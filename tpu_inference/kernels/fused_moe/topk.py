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

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl


def _topk_kernel(x_ref, vals_ref, idxs_ref, *, k):
    """Pallas kernel body: k iterations of argmax + mask-out, entirely
    within one VMEM-resident block so the whole top-k computation is a
    single kernel launch instead of k separate XLA ops. On an exact-value
    tie, Mosaic's argmax picks the highest index (jax.lax.top_k picks the
    lowest) - assumed rare/inconsequential for real router logits.
    """
    out_dtype = vals_ref.dtype
    x = x_ref[...].astype(jnp.float32)  # Mosaic argmax/max reduce needs f32
    neg_inf = jnp.finfo(jnp.float32).min
    lane_iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for i in range(k):
        idx = jnp.argmax(x, axis=-1)
        val = jnp.max(x, axis=-1)
        vals_ref[:, i] = val.astype(out_dtype)
        idxs_ref[:, i] = idx.astype(jnp.int32)
        mask = lane_iota == idx[:, None]
        x = jnp.where(mask, neg_inf, x)


def iterative_top_k_kernel(x: jax.Array,
                           k: int) -> tuple[jax.Array, jax.Array]:
    """Top-k via a single Pallas kernel: values match jax.lax.top_k exactly,
    values and indices always correspond to each other.

    Processes the whole [T, E] input in one VMEM-resident block (no grid) -
    comfortably fits for MoE routing shapes (e.g. bf16[16384,128] is 4MiB).

    Args:
        x: [T, E] input.
        k: number of top values/indices to select.
    """
    num_tokens, num_experts = x.shape
    block_spec_in = pl.BlockSpec((num_tokens, num_experts), lambda: (0, 0))
    block_spec_out = pl.BlockSpec((num_tokens, k), lambda: (0, 0))

    return pl.pallas_call(
        functools.partial(_topk_kernel, k=k),
        in_specs=[block_spec_in],
        out_specs=[block_spec_out, block_spec_out],
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, k), x.dtype),
            jax.ShapeDtypeStruct((num_tokens, k), jnp.int32),
        ],
    )(x)
