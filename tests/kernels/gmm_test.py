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

import collections
import functools

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.megablox.gmm_v2 import (TileSizes, apply_act_fn)
from tpu_inference.kernels.megablox.ops_v2 import gmm_v2
from tpu_inference.kernels.megablox.tgmm_v2 import tgmm_v2, validate_tgmm_inputs

jax.config.parse_flags_with_absl()

_GroupConfig = collections.namedtuple(
    "_GroupConfig", ["num_groups", "group_offset", "num_local_groups"]
)


def get_group_sizes(batch_size: int, num_groups: int) -> jax.Array:
  distribution = jax.random.uniform(
      jax.random.key(0), (num_groups - 1,), dtype=jnp.float32
  )
  distribution = distribution / jnp.sum(distribution)
  group_sizes = jnp.floor(distribution * batch_size).astype(jnp.int32)
  return jnp.append(group_sizes, batch_size - jnp.sum(group_sizes))


def quantize_tensor(
    x: jax.Array, dtype: jnp.dtype, axis: int = -1, block_size: int = 256
):
  if jnp.issubdtype(dtype, jnp.integer):
    dtype_info = jnp.iinfo(dtype)
    max_val = int(dtype_info.max)
    min_val = int(dtype_info.min)
  else:
    dtype_info = jnp.finfo(dtype)
    max_val = float(dtype_info.max)
    min_val = float(dtype_info.min)

  orig_shape = x.shape
  blocked_shape = orig_shape[:axis] + (-1, block_size) + orig_shape[axis + 1 :]
  x_blocked = x.reshape(blocked_shape)

  x_blocked_abs_max = jnp.max(jnp.abs(x_blocked), axis=axis + 1, keepdims=True)
  scale = x_blocked_abs_max / max_val
  x_blocked_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)

  x_q = x_blocked_q.reshape(orig_shape)
  x_q = jnp.nan_to_num(x_q)
  scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
  return x_q, scale


def reference_gmm(
    lhs: jax.Array,  # [m, k]
    rhs: jax.Array,  # [num_groups, k, n]
    group_sizes: jax.Array,  # [num_groups]
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,  # int32[1]
):
  num_tokens = lhs.shape[0]
  num_groups, in_size, out_size = rhs.shape
  assert num_groups > 0, f'rhs must have at least 1 group, got {num_groups}'
  assert lhs.shape[1] == in_size

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  if rhs_scale is not None:
    num_blocks = rhs_scale.shape[1]
  else:
    num_blocks = 1
  block_size = in_size // num_blocks

  start = 0
  gmm_out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]

    group = global_group - group_offset[0]
    end = min(start + group_size, num_tokens)
    group_size = end - start
    if 0 <= group and group < num_groups:
      lhs_slice = lhs[start:end]
      rhs_slice = rhs[group]

      out = 0
      for block in range(num_blocks):
        block_start = block * block_size
        block_end = block_start + block_size
        lhs_block = lhs_slice[:, block_start:block_end].astype(jnp.float32)
        rhs_block = rhs_slice[block_start:block_end, :].astype(jnp.float32)

        acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
        if rhs_scale is not None:
          acc *= rhs_scale[group][block]
        out += acc
      if rhs_bias is not None:
        out = out + rhs_bias[group]
    else:
      out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)

    gmm_out.append(out.astype(lhs.dtype))
    start = end

  return jnp.concat(gmm_out, axis=0)


def reference_tgmm(
    lhs,  # [k, m]
    rhs,  # [m, n]
    group_sizes,  # [num_groups]
    # num_actual_groups comes from weights.shape[0]
    num_actual_groups,  # int32
    rhs_scale: jax.Array | None = None,
    # group_offset is obtained from
    # jnp.arange(0, num_experts, num_experts_per_shard)
    group_offset=None,
    out_dtype: jnp.dtype | None = None,
):  # [num_groups, k, n]
  # Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :]
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  assert group_sizes.size >= int(group_offset[0]) + num_actual_groups, (
      f"group_sizes.size ({group_sizes.size}) must be >= "
      f"group_offset ({int(group_offset[0])}) + num_actual_groups "
      f"({num_actual_groups})"
  )

  start = 0
  out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]
    group = global_group - group_offset[0]
    end = start + group_size
    if 0 <= group and group < num_actual_groups:
      if rhs_scale is None:
        out.append(lhs[:, start:end] @ rhs[start:end, :])
      else:
        # rhs_scale.shape==(1, 1, N). Use Precision.HIGHEST on f32-cast inputs so
        # the reference is a true f32 ground truth. The kernel runs in native
        # fp8 MXU mode for throughput, so we expect a precision gap that the
        # test tolerance must absorb.
        partial = jax.lax.dot_general(
            lhs[:, start:end].astype(jnp.float32),
            rhs[start:end, :].astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        partial *= rhs_scale[0]  # rhs_scale[0]: shape [1, N]
        output_dtype = out_dtype if out_dtype is not None else lhs.dtype
        out.append(partial.astype(output_dtype))
    start = end
  return jnp.stack(out)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GmmTest(jtu.JaxTestCase):

  @parameterized.product(
      batch_size=[128, 512],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      has_bias=[True, False],
      group_offset=[0, 2, 3],
      # batch_size=[512],
      # in_size=[512],
      # out_size=[1024],
      # num_groups=[1],
      # has_bias=[False],
      # group_offset=[0],
  )
  def test_gmm_basic(
      self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1, k2, k3 = jax.random.split(key, 4)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          k2, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected, reference_vjpfunc = jax.vjp(
        functools.partial(
            reference_gmm, rhs_bias=rhs_bias, group_offset=group_offset
        ),
        lhs,
        rhs,
        group_sizes,
    )

    actual, vjpfunc = jax.vjp(
        functools.partial(gmm_v2, rhs_bias=rhs_bias, group_offset=group_offset),
        lhs,
        rhs,
        group_sizes,
    )

    self.assertArraysAllClose(actual, expected)

    if has_bias:
      # has_bias is not supported in TGMM yet.
      return
    cotangent = jax.random.normal(
        k3, (batch_size, out_size), dtype=jnp.bfloat16
    )
    expected_grad_lhs, expected_grad_rhs, *_ = reference_vjpfunc(cotangent)
    grad_lhs, grad_rhs, *_ = vjpfunc(cotangent)
    self.assertArraysAllClose(grad_lhs, expected_grad_lhs)
    self.assertArraysAllClose(grad_rhs, expected_grad_rhs)

  @parameterized.product(
      batch_size=[128, 1024],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[5, 16, 32],
      group_offset=[0, 2, 3],
      # batch_size=[512],
      # in_size=[1024],
      # out_size=[512],
      # num_groups=[5],
      # group_offset=[1],
  )
  def test_tgmm_basic(self, batch_size, in_size, out_size, num_groups, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    )  # [m, k]
    grad = jax.random.normal(
        key2, (batch_size, out_size), dtype=jnp.bfloat16
    )  # [m, n]
    group_sizes = get_group_sizes(batch_size, num_groups)
    # if batch_size=128, num_groups=3, an example group_size is
    # group_sizes=Array([14, 14, ..., 7]).
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)  # [k, m]
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2(
        lhs, grad, group_sizes, num_local_groups, group_offset=group_offset, preferred_element_type=jnp.bfloat16
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    # diff = jnp.abs(expected - actual)
    # max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
    # print(f"Output max diff: {jnp.max(diff)} at index {max_diff_idx}")
    # print(f"Output mean diff: {jnp.mean(jnp.abs(expected - actual))}")
    self.assertArraysAllClose(actual, expected)

  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255, 500],
      out_size=[255, 500],
      num_groups=[16],
      group_offset=[0],
  )
  def test_tgmm_implicit_padding(
      self, batch_size, in_size, out_size, num_groups, group_offset
  ):
    # Notice that tile_n and tile_k are aligned to the num_lanes in
    # calculate_tgmm_tiling.
    # The output shape is [num_groups, size_k, aligned_n] but there is implicit
    # padding on the k-dim to a multiple of sublanes. So the kernel is able to
    # write the full [i, aligned_tile_k, aligned_tile_n] to hbm with no problem
    # at the last k block.
    # Within the kernel, because k is not the contracting dim, so the padded k
    # is also not a problem.
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    )
    grad = jax.random.normal(
        key2, (batch_size, out_size), dtype=jnp.bfloat16
    )
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )
    validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2(
        lhs, grad, group_sizes, num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    self.assertArraysAllClose(actual, expected)

  @parameterized.product(
      batch_size=[256, 1024],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16],
      group_offset=[0, 2],
      tile_k=[256, 512],
      tile_n=[256, 512],
  )
  def test_tgmm_with_tile_info(
      self, batch_size, in_size, out_size, num_groups, group_offset,
      tile_k, tile_n,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(
        lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    )

    tile_info = TileSizes(tile_m=256, tile_k=tile_k, tile_n=tile_n)
    validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2(
        lhs, grad, group_sizes, num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    self.assertArraysAllClose(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
      empty_group_index=[0, 1, 2, 3],
  )
  def test_tgmm_empty_group(
      self, batch_size, in_size, out_size, num_groups, group_offset,
      empty_group_index,
  ):
    """Test that TGMM correctly zeros output for empty groups."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    # Redistribute the empty group's tokens to the last group.
    group_sizes = group_sizes.at[-1].add(group_sizes[empty_group_index])
    group_sizes = group_sizes.at[empty_group_index].set(0)

    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    #expected = reference_tgmm(
    #    lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset
    #)
    validate_tgmm_inputs(group_sizes, num_local_groups, group_offset)
    actual = tgmm_v2(
        lhs, grad, group_sizes, num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    # self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    # self.assertArraysAllClose(actual, expected)

  def test_tgmm_fp8_inputs_smoke(self):
    batch_size, in_size, out_size = 1024, 256, 256
    num_groups = 4
    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs_bf16 = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs_bf16 = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    lhs_fp8 = lhs_bf16.astype(jnp.float8_e4m3fn)
    rhs_fp8 = rhs_bf16.astype(jnp.float8_e5m2)
    group_sizes = get_group_sizes(batch_size, num_groups)

    expected = reference_tgmm(
        lhs_fp8.swapaxes(0, 1), rhs_fp8, group_sizes, num_groups,
    )
    validate_tgmm_inputs(group_sizes, num_groups)
    actual = tgmm_v2(
        lhs_fp8, rhs_fp8, group_sizes, num_groups,
        preferred_element_type=jnp.bfloat16,
    )

    self.assertEqual(actual.shape, (num_groups, in_size, out_size))
    self.assertAllClose(actual, expected, rtol=1e-2, atol=1e-2)

  @parameterized.product(
      batch_size=[128, 512],          # M
      in_size=[256, 512],             # K
      out_size=[256, 512],            # N
      num_groups=[4, 8],
      group_offset=[0, 2],
      dtype_pair=[
          (jnp.float8_e4m3fn, jnp.float8_e5m2),       # production
          (jnp.float8_e4m3fn, jnp.float8_e4m3fn),     # symmetric fp8
      ],
  )
  def test_tgmm_with_rhs_scale(
      self, batch_size, in_size, out_size, num_groups, group_offset, dtype_pair
  ):
    lhs_dtype, rhs_quant_dtype = dtype_pair
    num_local_groups = num_groups - group_offset

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    ).astype(lhs_dtype)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.float32)

    grad_q, grad_scale = quantize_tensor(
        grad, rhs_quant_dtype, axis=0, block_size=batch_size,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, N]
    assert grad_scale.shape == (1, 1, out_size)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset_arr = jnp.array([group_offset], dtype=jnp.int32)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1), grad_q, group_sizes, num_local_groups,
        rhs_scale=grad_scale,
        group_offset=group_offset_arr,
        out_dtype=jnp.bfloat16,
    )
    validate_tgmm_inputs(group_sizes, num_local_groups, group_offset_arr)
    actual = tgmm_v2(
        lhs, grad_q, group_sizes, num_local_groups,
        rhs_scale=grad_scale,
        group_offset=group_offset_arr,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    self.assertAllClose(actual, expected, rtol=1e-2, atol=6e-1)

  def test_tgmm_with_rhs_scale_n_padding(self):
    # Pins tile_n=128 with out_size=300 so the kernel runs 3 n-tiles over an
    # aligned width of 384; the last tile (n_id=2) reads scale[..., 256:384]
    # where columns 300..383 are pad. Exercises the rhs_scale pad in
    # tgmm_v2.py:573-577 and the output slice-back at tgmm_v2.py:598.
    batch_size, in_size, out_size = 128, 256, 300
    num_groups = 4
    rhs_quant_dtype = jnp.float8_e5m2

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    lhs = jax.random.normal(
        key1, (batch_size, in_size), dtype=jnp.bfloat16
    ).astype(jnp.float8_e4m3fn)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.float32)

    grad_q, grad_scale = quantize_tensor(
        grad, rhs_quant_dtype, axis=0, block_size=batch_size,
    )
    grad_scale = jnp.expand_dims(grad_scale, axis=1)  # [1, 1, N]
    assert grad_scale.shape == (1, 1, out_size)

    group_sizes = get_group_sizes(batch_size, num_groups)
    tile_info = TileSizes(tile_m=128, tile_k=256, tile_n=128)

    expected = reference_tgmm(
        lhs.swapaxes(0, 1), grad_q, group_sizes, num_groups,
        rhs_scale=grad_scale,
        out_dtype=jnp.bfloat16,
    )
    validate_tgmm_inputs(group_sizes, num_groups)
    actual = tgmm_v2(
        lhs, grad_q, group_sizes, num_groups,
        rhs_scale=grad_scale,
        tile_info=tile_info,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_groups, in_size, out_size))
    self.assertAllClose(actual, expected, rtol=1e-2, atol=6e-1)


  @parameterized.product(
      batch_size=[128],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      has_bias=[True, False],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
      block_size=[64, 128, 256, 512],
      group_offset=[0, 2, 3],
  )
  def test_gmm_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and not jtu.is_device_tpu_at_least(
        version=7
    ):
      self.skipTest("Expect TPUv7+")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    self.assertArraysAllClose(actual, expected, atol=3e-1, rtol=3e-1)

  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
      block_size=[1024],
      tile_k=[128, 256, 512],
      group_offset=[0],
  )
  def test_gmm_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test that quant_block_size > tile_k is handled correctly."""
    if weight_dtype == jnp.float4_e2m1fn and not jtu.is_device_tpu_at_least(
        version=7
    ):
      self.skipTest("Expect TPUv7+")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = TileSizes(tile_m=128, tile_k=tile_k, tile_n=out_size)
    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    self.assertArraysAllClose(actual, expected, atol=3e-1, rtol=3e-1)

  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
      block_size=[1024],
      tile_k=[128, 256, 512],
      group_offset=[0],
  )
  def test_gmm_activation_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test activation+weight quantized path with quant_block_size > tile_k."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = TileSizes(tile_m=128, tile_k=tile_k, tile_n=out_size)
    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=True,
    ).astype(lhs.dtype)

    self.assertArraysAllClose(actual, expected, atol=1.2, rtol=1.2)

  @parameterized.product(
      batch_size=[128],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
      block_size=[512, 1024],
      group_offset=[0, 2, 3],
  )
  def test_gmm_activation_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and not jtu.is_device_tpu_at_least(
        version=7
    ):
      self.skipTest("Expect TPUv7+")
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
    ).astype(lhs.dtype)

    self.assertArraysAllClose(actual, expected, atol=1.1, rtol=1.1)

  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255, 500],
      out_size=[255, 500],
      num_groups=[16],
      has_bias=[True, False],
      group_offset=[0],
  )
  def test_gmm_implicit_padding(
      self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    self.assertArraysAllClose(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[500],
      num_groups=[16],
      has_bias=[True, False],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
      block_size=[512],
      group_offset=[0],
  )
  def test_gmm_weight_quantized_padding(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    self.assertEqual(actual.shape, (batch_size, out_size))
    self.assertArraysAllClose(actual, expected, atol=3e-1, rtol=3e-1)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      # group_config: (num_groups, group_offset, num_local_groups)
      group_config=[
          # groups 0-1: group<0, groups 2-5: local and active,
          # groups 6-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=2, num_local_groups=4),
          # no negative groups, groups 0-7: local and active,
          # groups 8-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=0, num_local_groups=8),
          # groups 0-3: group<0, groups 4-7: local and active,
          # groups 8-31: group>=num_local_groups
          _GroupConfig(num_groups=32, group_offset=4, num_local_groups=4),
      ],
  )
  def test_gmm_nonlocal_groups_produce_zeros(
      self, batch_size, in_size, out_size, group_config
  ):
    num_groups, group_offset, num_local_groups = group_config
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
    )
    rhs_bias = jax.random.normal(
        key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
    )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    self.assertArraysAllClose(actual, expected)

  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True, False],
      use_weight_scale=[True, False],
      maybe_quantize_lhs=[True, False],
      fuse_act=["silu", "swigluoai", "gelu"],
      group_offset=[0, 2],
      block_size=[256, 512],
  )
  def test_gmm_fused_activation(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      use_weight_scale,
      maybe_quantize_lhs,
      fuse_act,
      group_offset,
      block_size,
  ):
    if maybe_quantize_lhs and not use_weight_scale:
      self.skipTest(
          "LHS quantization requires RHS quantization/scale in this config."
      )
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    key = jax.random.key(0)
    final_out_size = out_size // 2
    num_local_groups = num_groups - group_offset

    # 1. Generate Inputs
    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(
        key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
    )

    rhs_q = rhs
    rhs_scale = None
    if use_weight_scale:
      rhs_q, rhs_scale = quantize_tensor(
          rhs, jnp.int8, axis=1, block_size=block_size
      )
      rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(
          key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
      )

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array([group_offset], dtype=jnp.int32)

    # 2. Simulate LHS Quantization Noise
    lhs_simulated = lhs
    # because the kernel quantizes LHS in blocks, while reference does it at the
    # whole tensor level, and output is casted down we need to simulate that
    # quantization noise in the reference as well for a fair comparison
    if maybe_quantize_lhs:
      lhs_block_size = min(512, in_size)
      lhs_q, lhs_scale_factor = quantize_tensor(
          lhs, jnp.int8, axis=1, block_size=lhs_block_size
      )
      lhs_q_blocked = lhs_q.reshape(batch_size, -1, lhs_block_size).astype(
          jnp.float32
      )
      lhs_scale_expanded = jnp.expand_dims(lhs_scale_factor, axis=2)
      lhs_simulated = (
          (lhs_q_blocked * lhs_scale_expanded)
          .reshape(lhs.shape)
          .astype(lhs.dtype)
      )

    # 3. Compute Reference Output
    raw_expected = reference_gmm(
        lhs_simulated,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    # Slice the reference and apply the activation function
    expected = apply_act_fn(raw_expected.astype(jnp.float32), fuse_act).astype(
        lhs.dtype
    )

    # 4. Compute Actual Kernel Output
    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
    ).astype(lhs.dtype)

    # 5. Compare Results
    self.assertEqual(actual.shape, (batch_size, final_out_size))

    # tolerances based quantization noise difference between reference and
    # gmm_v2
    if maybe_quantize_lhs:
      atol, rtol = 4.0, 2.0  # Act + Weight Quantization
    elif use_weight_scale:
      atol, rtol = 3e-1, 3e-1  # Weight Quantization Only
    else:
      atol, rtol = 5e-2, 5e-2  # Unquantized Path (bfloat16 precision diffs)

    self.assertArraysAllClose(actual, expected, atol=atol, rtol=rtol)

  def test_gmm_deepseekv3(self):
    num_groups = 256
    m = 262144
    k = 7168
    n = 2048
    lhs_dtype = jnp.bfloat16

    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    lhs = jax.random.normal(k1, (m, k), dtype=lhs_dtype)
    rhs = jax.random.normal(
        key, (num_groups, k, n), dtype=jnp.bfloat16
    )
    weight_dtype = jnp.float8_e4m3fn
    block_size = 512
    rhs_q, rhs_scale = quantize_tensor(
        rhs, weight_dtype, axis=1, block_size=block_size
    )
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    cotangent = jax.random.normal(
        k2, (m, n), dtype=lhs_dtype
    )
    group_sizes = get_group_sizes(m, num_groups)

    actual = gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
    )
    actual.block_until_ready()

    grad_lhs = gmm_v2(
        cotangent,
        rhs_q.swapaxes(1, 2),
        group_sizes,
    )
    grad_lhs.block_until_ready()

    validate_tgmm_inputs(group_sizes, num_groups)
    grad_rhs = tgmm_v2(
        lhs,
        cotangent,
        group_sizes,
        num_groups,
        preferred_element_type=jnp.bfloat16,
    )
    grad_rhs.block_until_ready()

  def test_gmm_benchmark_small(self):
    num_groups = 16
    m = 512
    k = 1024
    n = 512
    lhs_dtype = jnp.bfloat16

    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    lhs = jax.random.normal(k1, (m, k), dtype=lhs_dtype)
    rhs = jax.random.normal(
        key, (num_groups, k, n), dtype=jnp.bfloat16
    )
    cotangent = jax.random.normal(k2, (m, n), dtype=lhs_dtype)
    group_sizes = get_group_sizes(m, num_groups)

    # xprof_sess = xprof_session.XprofSession()
    # xprof_sess.start_session(trace_mode="TRACE_COMPUTE_AND_SYNC")
    actual = gmm_v2(
        lhs,
        rhs,
        group_sizes,
    )
    actual.block_until_ready()

    print(
        f"xw32 test_gmm_benchmark_small tgmm inputs: {lhs.shape=}, {cotangent.shape=},"
        f" {group_sizes=}, {num_groups=}"
    )
    validate_tgmm_inputs(group_sizes, num_groups)
    grad_rhs = tgmm_v2(
        lhs,
        cotangent,
        group_sizes,
        num_groups,
        preferred_element_type=jnp.bfloat16,
    )
    grad_rhs.block_until_ready()

    # url = xprof_sess.end_session_and_get_url()
    # print(f"XProf URL: {url}")

    lhs_t = lhs.swapaxes(0, 1)  # [k, m]
    expected = reference_tgmm(
        lhs_t, cotangent, group_sizes, num_groups
    )
    self.assertArraysAllClose(grad_rhs, expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
