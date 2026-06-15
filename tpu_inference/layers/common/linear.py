# Copyright 2025 Google LLC
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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2 import \
    PallasMosaicTpuV2RaggedDot
from tpu_inference.kernels.quantized_matmul.util import (
    quantize_tensor, xla_quantized_batched_matmul)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def xla_quantized_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation=True,
) -> jax.Array:
    """
    Reference (pure JAX) implementation of the quantized matmul kernel below.

    Args:
        x:  Activation.
        w_q: Weight quantized array. [n_input_features, n_output_features]
        w_s: Weight quantization scale. [n_output_features]
        mesh: Mesh to shard on.
        weight_sharding: PartitionSpec for the weight tensor.

    Returns:
        Output of the quantized matmul.
    """
    skip_scale = False
    if w_scale is not None and w_scale.ndim == 2:
        skip_scale = True
        in_features, out_features = w_q.shape
        in_blocks, out_blocks = w_scale.shape
        block_size_in = in_features // in_blocks
        block_size_out = out_features // out_blocks

        w_q_reshaped = w_q.reshape(in_blocks, block_size_in, out_blocks,
                                   block_size_out)
        w_q = (w_q_reshaped.astype(jnp.float32) *
               w_scale[:, jnp.newaxis, :, jnp.newaxis]).reshape(
                   in_features, out_features).astype(x.dtype)

        # in this case, we don't want to quantize the activations
        quantize_activation = False
        logger.info_once(
            "Skipping activation quantization due to weight requantization being disabled."
        )

    if quantize_activation:
        acc_dtype = jnp.float32
        if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
            acc_dtype = jnp.int32

        x_q, x_scale = quantize_tensor(x, w_q.dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(((1, ), (0, )), ((), ())),
            preferred_element_type=acc_dtype,
        ).astype(jnp.float32)
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            dimension_numbers=(((1, ), (0, )), ((), ())),
            preferred_element_type=jnp.float32,
        )
    if not skip_scale:
        out *= jnp.expand_dims(w_scale, 0)
    return out.astype(x.dtype)


def _get_x_q_dtype(w_q_dtype: jnp.dtype) -> jnp.dtype:
    """Return 8-bit float or integer dtype depending on w_q_dtype."""
    if jnp.issubdtype(w_q_dtype, jnp.integer):
        return jnp.int8
    elif jnp.issubdtype(w_q_dtype, jnp.floating):
        return jnp.float8_e4m3fn
    # TODO: we need a new flag for 4bit activation later such as w4a4.
    else:
        raise ValueError(
            f"Unsupported quantized dtype: {w_q_dtype}, it should be integer or float"
        )


def sharded_quantized_matmul(x: jax.Array,
                             w_q: jax.Array,
                             w_s: jax.Array,
                             weight_sharding: P | NamedSharding,
                             *,
                             mesh: Mesh | None = None,
                             x_q_dtype: jnp.dtype | None = None) -> jax.Array:
    """
    Wrapper around the quantized matmul kernel.

    Args:
        x:  Activation.
        w_q: Weight quantized array. [n_input_features, n_output_features]
        w_s: Weight quantization scale. [n_output_features] for xla quantized matmul, [n_blocks, 1, n_output_features] for quantized matmul kernel
        weight_sharding: PartitionSpec or NamedSharding for the weight tensor.
        mesh: (Optional) Mesh to shard on. If None, mesh from current context is used, similar to jax.shard_map().
        x_q_dtype: (Optional) Quantized dtype for the activation. If None, inferred from w_q dtype (int -> int8, float -> float8).

    Returns:
        Output of the quantized matmul.
    """

    if isinstance(weight_sharding, NamedSharding):
        if mesh is None:
            mesh = weight_sharding.mesh
        weight_spec = weight_sharding.spec
    else:
        weight_spec = weight_sharding

    # NOTE (jacobplatin/kyuyeunk) there have been numeric issues (concerning) NaNs
    # with the kernel and thus we disable it for now.
    in_axis, out_axis = weight_spec
    x_sharding = P(ShardingAxisName.ATTN_DATA, in_axis)
    enable_quantized_matmul_kernel = w_s is not None and (len(
        w_s.shape) == 3 or len(w_s.shape) == 4)
    if enable_quantized_matmul_kernel:
        if w_s.ndim == 4:
            _, num_blocks, __, ___ = w_s.shape
            scale_sharding = P(None, in_axis if num_blocks > 1 else None, None,
                               out_axis)
        else:
            num_blocks, _, __ = w_s.shape
            scale_sharding = P(in_axis if num_blocks > 1 else None, None,
                               out_axis)
    else:
        # 2D-Blockwise case (e.g. from skipped re-quantization)
        if w_s is not None and len(w_s.shape) == 2:
            scale_sharding = weight_spec
        else:
            # 1D (channelwise) case
            scale_sharding = P(out_axis, )
    out_sharding = P(ShardingAxisName.ATTN_DATA, out_axis)

    if x_q_dtype is None:
        x_q_dtype = _get_x_q_dtype(w_q.dtype)
    x = jax.lax.with_sharding_constraint(
        x,
        NamedSharding(mesh, x_sharding) if mesh else x_sharding)

    def wrapper(x, w_q, w_s):
        if enable_quantized_matmul_kernel:
            output = PallasMosaicTpuV2RaggedDot()(
                x,
                jnp.expand_dims(w_q, 0),
                group_sizes=jnp.array([x.shape[0]], dtype=jnp.int32),
                rhs_scale=w_s if w_s.ndim == 4 else jnp.expand_dims(w_s, 0),
                rhs_bias=None,
                group_offset=jnp.array([0], dtype=jnp.int32),
                zero_initialize=False,
                preferred_element_type=x.dtype,
                maybe_quantize_lhs=True,
            )
        else:
            output = xla_quantized_matmul(x, w_q, w_s)
        if in_axis:
            output = jax.lax.psum(output, axis_name=in_axis)
        return output

    return jax.shard_map(
        wrapper,
        mesh=mesh,
        in_specs=(x_sharding, weight_spec, scale_sharding),
        out_specs=(out_sharding),
        check_vma=False,
    )(x, w_q, w_s)


def _parse_einsum_dims(einsum_str: str):
    """Parse an einsum string to extract dimension classifications.

    Returns:
        Tuple of (contract_dims_x, contract_dims_w, batch_dims_x,
        batch_dims_w, output_perm) where output_perm is the permutation
        needed to go from dot_general output order to the einsum output order.
    """
    lhs, output_axis = einsum_str.replace(" ", "").split("->")
    x_axis, w_axis = lhs.split(",")

    shared = set(x_axis) & set(w_axis)
    batch_axes = shared & set(output_axis)
    contracting_axes = shared - batch_axes

    contract_dims_x = tuple(i for i, c in enumerate(x_axis)
                            if c in contracting_axes)
    contract_dims_w = tuple(i for i, c in enumerate(w_axis)
                            if c in contracting_axes)
    batch_dims_x = tuple(i for i, c in enumerate(x_axis) if c in batch_axes)
    batch_dims_w = tuple(i for i, c in enumerate(w_axis) if c in batch_axes)

    # dot_general output order: batch dims, lhs free dims, rhs free dims.
    dg_output_labels = []
    for i, c in enumerate(x_axis):
        if c in batch_axes:
            dg_output_labels.append(c)
    for i, c in enumerate(x_axis):
        if c not in shared:
            dg_output_labels.append(c)
    for i, c in enumerate(w_axis):
        if c not in shared:
            dg_output_labels.append(c)

    # Permutation to go from dot_general output to desired einsum output.
    output_perm = tuple(dg_output_labels.index(c) for c in output_axis)

    return (contract_dims_x, contract_dims_w, batch_dims_x, batch_dims_w,
            output_perm)


def sharded_quantized_batched_matmul(x: jax.Array,
                                     w_q: jax.Array,
                                     w_s: jax.Array,
                                     einsum_str: str,
                                     weight_sharding: P | NamedSharding,
                                     *,
                                     mesh: Mesh | None = None) -> jax.Array:
    """Sharded quantized matmul with batch dimensions.

    Like ``sharded_quantized_matmul`` but supports einsum patterns where some
    axes are shared between both operands **and** appear in the output (batch
    dims).  Uses ``jax.lax.dot_general`` with native batch dimensions inside
    ``shard_map`` — the Pallas kernel is not used because it is 2D-only.

    Args:
        x: Activation tensor (e.g. shape ``[T, N, H]``).
        w_q: Quantized weight (e.g. shape ``[A, N, H]``).
        w_s: Weight scale. Shape ``(out,)`` for tensorwise.
        einsum_str: Full einsum equation (e.g. ``"TNH,ANH->NTA"``).
        weight_sharding: PartitionSpec or NamedSharding for ``w_q``.
        mesh: Optional mesh.

    Returns:
        Output shaped according to ``einsum_str``.
    """
    if isinstance(weight_sharding, NamedSharding):
        if mesh is None:
            mesh = weight_sharding.mesh
        weight_spec = weight_sharding.spec
    else:
        weight_spec = weight_sharding

    (contract_dims_x, contract_dims_w, batch_dims_x, batch_dims_w,
     output_perm) = _parse_einsum_dims(einsum_str)

    dimension_numbers = (
        (contract_dims_x, contract_dims_w),
        (batch_dims_x, batch_dims_w),
    )

    # Build PartitionSpecs for shard_map from the weight spec and einsum
    # structure. The weight_spec maps to the weight's axes directly.
    lhs, _ = einsum_str.replace(" ", "").split("->")
    x_axis, w_axis = lhs.split(",")

    # Build a per-axis sharding map from the weight spec.
    w_spec_padded = weight_spec + tuple(
        None for _ in range(len(w_axis) - len(weight_spec)))
    axis_shard = {c: w_spec_padded[i] for i, c in enumerate(w_axis)}

    # Determine the token axis as the lhs-free axis (in x but not w) for
    # ATTN_DATA sharding. Falls back to x_axis[0] if all axes are shared.
    _shared = set(x_axis) & set(w_axis)
    _lhs_free = [c for c in x_axis if c not in _shared]
    _dp_axis = _lhs_free[0] if _lhs_free else x_axis[0]
    act_shard = {_dp_axis: ShardingAxisName.ATTN_DATA}

    # Input sharding: activation takes precedence; fall back to weight sharding
    # for shared (batch) axes where activation info is absent.
    x_spec = tuple(act_shard.get(c, axis_shard.get(c, None)) for c in x_axis)
    x_sharding = P(*x_spec)

    # Scale sharding: scale is 1D (out_features) for tensorwise.
    # Find the output axis from the weight (rhs free dims).
    shared = set(x_axis) & set(w_axis)
    rhs_free = [c for c in w_axis if c not in shared]
    scale_spec = tuple(axis_shard.get(c, None) for c in rhs_free)
    scale_sharding = P(*scale_spec) if scale_spec else P()

    # We first compute dg_out_spec using dot_general's physical output order
    # (batch, lhs_free, rhs_free) and then permute via output_perm to reach the
    # einsum logical output order that shard_map actually sees.
    batch_labels = [
        c for c in x_axis
        if c in (set(x_axis) & set(w_axis) & set(einsum_str.split("->")[1]))
    ]
    lhs_free = [c for c in x_axis if c not in shared]
    dg_out_labels = batch_labels + lhs_free + rhs_free
    # Output sharding: weight sharding takes precedence; fall back to activation sharding.
    dg_out_spec = tuple(
        axis_shard.get(c, act_shard.get(c, None)) for c in dg_out_labels)
    out_spec = tuple(dg_out_spec[i] for i in output_perm)
    out_sharding = P(*out_spec)

    # Determine the contracting axis name for psum (if sharded).
    contract_axis_names = set()
    for i in contract_dims_w:
        s = w_spec_padded[i]
        if s is not None:
            contract_axis_names.add(s)

    x = jax.lax.with_sharding_constraint(
        x,
        NamedSharding(mesh, x_sharding) if mesh else x_sharding)

    def wrapper(x, w_q, w_s):
        _should_quantize_act = x.dtype.itemsize > 1
        output = xla_quantized_batched_matmul(
            x,
            w_q,
            w_s,
            dimension_numbers,
            quantize_activation=_should_quantize_act)
        for axis_name in contract_axis_names:
            output = jax.lax.psum(output, axis_name=axis_name)
        # Transpose from dot_general output order to einsum output order.
        if output_perm != tuple(range(len(output_perm))):
            output = jnp.transpose(output, output_perm)
        return output

    return jax.shard_map(
        wrapper,
        mesh=mesh,
        in_specs=(x_sharding, weight_spec, scale_sharding),
        out_specs=out_sharding,
        check_vma=False,
    )(x, w_q, w_s)
