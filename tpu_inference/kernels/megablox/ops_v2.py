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

"""Grouped matrix multiplication operations with custom VJPs."""

import functools

import jax
import jax.numpy as jnp

from tpu_inference.kernels.megablox.gmm_v2 import (
    TileFn,
    TileSizes,
    gmm_v2 as gmm_v2_impl,
    calculate_tiling,
)
from tpu_inference.kernels.megablox.tgmm_v2 import (
    tgmm_v2,
    calculate_tgmm_tiling,
)


@functools.partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
        "maybe_quantize_lhs",
        "zero_initialize",
        "fuse_act",
    ),
)
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    group_offset: jax.Array | None = None,  # int32[1]
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
):
  """GMM kernel."""
  return gmm_v2_impl(
      lhs,
      rhs,
      group_sizes,
      rhs_scale,
      rhs_bias,
      group_offset,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      precision=precision,
      preferred_element_type=preferred_element_type,
      acc_dtype=acc_dtype,
      maybe_quantize_lhs=maybe_quantize_lhs,
      zero_initialize=zero_initialize,
      fuse_act=fuse_act,
  )


def _gmm_v2_fwd(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    group_offset: jax.Array | None = None,  # int32[1]
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
):
  """Forward pass for GMM kernel."""
  out = gmm_v2_impl(
      lhs,
      rhs,
      group_sizes,
      rhs_scale,
      rhs_bias,
      group_offset,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      precision=precision,
      preferred_element_type=preferred_element_type,
      acc_dtype=acc_dtype,
      maybe_quantize_lhs=maybe_quantize_lhs,
      zero_initialize=zero_initialize,
      fuse_act=fuse_act,
  )
  num_actual_groups = rhs.shape[0]
  return out, (
      lhs,
      rhs,
      group_sizes,
      rhs_scale,
      rhs_bias,
      group_offset,
      num_actual_groups,
  )


def _gmm_v2_bwd(
    # non-diff argnames (passed as positional args before residuals and grad)
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    precision: jax.lax.Precision,
    preferred_element_type: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
    fuse_act: str | None,
    # residual
    residuals: tuple[
        jnp.ndarray,  # lhs
        jnp.ndarray,  # rhs
        jnp.ndarray,  # group_sizes
        jnp.ndarray,  # rhs_scale
        jnp.ndarray,  # rhs_bias
        jnp.ndarray,  # group_offset
        int,  # num_actual_groups
    ],
    # cotangent
    grad: jnp.ndarray,
):
  """Backward pass for GMM kernel."""
  (
      lhs,
      rhs,
      group_sizes,
      rhs_scale,
      rhs_bias,
      group_offset,
      num_actual_groups,
  ) = residuals
  # TODO: Consider supporting rhs_bias if needed.
  assert rhs_bias is None, "rhs_bias is not yet supported in TGMM."
  # d(lhs) = dout @ rhs^T — no bias term so rhs_scale and rhs_bias should be
  # None. So should the fuse_act.
  # TODO: Consider fusing the transposition of rhs into the gmm kernel.
  grad_lhs = gmm_v2_impl(
      grad,  # [m, n]
      rhs.swapaxes(1, 2),  # [num_groups, n, k]
      group_sizes,
      None,  # rhs_scale
      None,  # rhs_bias
      group_offset,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      precision=precision,
      preferred_element_type=lhs.dtype,
      acc_dtype=acc_dtype,
      maybe_quantize_lhs=maybe_quantize_lhs,
      zero_initialize=zero_initialize,
      fuse_act=None,
  )
  grad_rhs = tgmm_v2(
      lhs,  # [m, k]
      grad,  # [m, n]
      group_sizes,
      num_actual_groups,
      group_offset,
      # TODO: consider letting users provide tiling for bwd.
      tile_info=calculate_tgmm_tiling,
      vmem_limit_bytes=vmem_limit_bytes,
      precision=precision,
      # TODO: we may want a user provided preferred_element_type (drhs's dtype).
      preferred_element_type=rhs.dtype,
      acc_dtype=acc_dtype,
  )
  # Return a gradient per each differentiable argument except for the
  # nondiff_argnames.
  return grad_lhs, grad_rhs, None, None, None, None

gmm_v2.defvjp(_gmm_v2_fwd, _gmm_v2_bwd)
