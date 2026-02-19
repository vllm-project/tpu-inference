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

from tpu_inference.kernels.quantized_matmul.blockwise_kernel import \
    quantized_matmul_kernel as blockwise_quantized_matmul_kernel
from tpu_inference.kernels.quantized_matmul.util import (
    xla_quantized_batched_matmul, xla_quantized_matmul)


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
                             mesh: Mesh | None = None) -> jax.Array:
    """
    Wrapper around the quantized matmul kernel.

    Args:
        x:  Activation.
        w_q: Weight quantized array. [n_output_features, n_input_features]
        w_s: Weight quantization scale. [n_output_features] for xla quantized matmul, [n_blocks, 1, n_output_features] for quantized matmul kernel
        weight_sharding: PartitionSpec or NamedSharding for the weight tensor.
        mesh: (Optional) Mesh to shard on. If None, mesh from current context is used, similar to jax.shard_map().

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
    out_axis, in_axis = weight_spec
    x_sharding = P(None, in_axis)
    enable_quantized_matmul_kernel = len(w_s.shape) == 3
    if enable_quantized_matmul_kernel:
        num_blocks, _, __ = w_s.shape
        scale_sharding = P(
            in_axis if num_blocks > 1 else None,
            None,
            out_axis,
        )
    else:
        scale_sharding = P(out_axis, )
    out_sharding = P(None, out_axis)

    x_q_dtype = _get_x_q_dtype(w_q.dtype)
    x = jax.lax.with_sharding_constraint(
        x,
        NamedSharding(mesh, x_sharding) if mesh else x_sharding)

    def wrapper(x, w_q, w_s):
        if enable_quantized_matmul_kernel:
            k_dim = x.shape[1]
            sharded_num_blocks, _, __ = w_s.shape
            block_size = k_dim // sharded_num_blocks
            output = blockwise_quantized_matmul_kernel(x,
                                                       w_q,
                                                       w_s,
                                                       x_q_dtype=x_q_dtype,
                                                       block_size=block_size)
        else:
            print(
                f"clkbp {x.shape=}, {w_q.shape=}, {w_s.shape=}, using xla_quantized_matmul"
            )  # TODO: remove this debug print
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
        einsum_str: Full einsum equation (e.g. ``"TNH,ANH->TNA"``).
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

    # Input sharding: match weight sharding for shared axes, None for free.
    x_spec = tuple(axis_shard.get(c, None) for c in x_axis)
    x_sharding = P(*x_spec)

    # Scale sharding: scale is 1D (out_features) for tensorwise.
    # Find the output axis from the weight (rhs free dims).
    shared = set(x_axis) & set(w_axis)
    rhs_free = [c for c in w_axis if c not in shared]
    scale_spec = tuple(axis_shard.get(c, None) for c in rhs_free)
    scale_sharding = P(*scale_spec) if scale_spec else P()

    # Output sharding: dot_general output order is batch, lhs_free, rhs_free.
    batch_labels = [
        c for c in x_axis
        if c in (set(x_axis) & set(w_axis) & set(einsum_str.split("->")[1]))
    ]
    lhs_free = [c for c in x_axis if c not in shared]
    dg_out_labels = batch_labels + lhs_free + rhs_free
    dg_out_spec = tuple(axis_shard.get(c, None) for c in dg_out_labels)
    # Apply the output permutation to match the einsum output.
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
        output = xla_quantized_batched_matmul(x,
                                              w_q,
                                              w_s,
                                              dimension_numbers,
                                              quantize_activation=True)
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
