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
import functools
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from jax.experimental.layout import Layout, with_layout_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.tensor import Tensor

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    general_device_put, reorder_concatenated_tensor_for_sharding)
from tpu_inference.utils import align_to, get_mesh_shape_product

P = PartitionSpec


@jax.tree_util.register_dataclass
@dataclass
class FusedMoEWeights:
    """Fused moe weights. weights can be either jax or torchax array."""
    w13_weight: jax.Array | Tensor
    w13_weight_scale: jax.Array | Tensor | None
    w13_weight_zero_point: jax.Array | Tensor | None
    w13_bias: jax.Array | Tensor | None
    w2_weight: jax.Array | Tensor
    w2_weight_scale: jax.Array | Tensor | None
    w2_weight_zero_point: jax.Array | Tensor | None
    w2_bias: jax.Array | Tensor | None


@jax.tree_util.register_dataclass
@dataclass
class UnfusedMoEWeights:
    """Unfused moe weights. weights can be either jax or torchax array."""
    w1_weight: jax.Array | Tensor
    w1_weight_scale: jax.Array | Tensor | None
    w1_bias: jax.Array | Tensor | None
    w2_weight: jax.Array | Tensor
    w2_weight_scale: jax.Array | Tensor | None
    w2_bias: jax.Array | Tensor | None
    w3_weight: jax.Array | Tensor
    w3_weight_scale: jax.Array | Tensor | None
    w3_bias: jax.Array | Tensor | None


def quantize_moe_weights(
    weights: FusedMoEWeights,
    dtype: jnp.dtype,
    block_size: int | None,
) -> FusedMoEWeights:
    """Quantize fused moe weights into a given dtype and block size.

    Args:
        weights: fused moe weights.
        dtype: dtype to perform quantization.
        block_size: Specify block quantization size. If non, use per-channel
            quantization. If contracting dim is not divisible by block size,
            the dim will be automatically padded and corresponding dim on bias
            and the other weight (w13_weight <-> w2_weight) is also padded.

    Returns:
        Quantized fused moe weights that may have also been padded.
    """

    # If scale is present, it means the weights are already quantized.
    # Ensure that weights are not quantized by checking if scales are None.
    assert weights.w13_weight_scale is None
    assert weights.w2_weight_scale is None
    assert weights.w13_weight_zero_point is None
    assert weights.w2_weight_zero_point is None

    w13_weight = weights.w13_weight
    w2_weight = weights.w2_weight

    if block_size is None:
        # Use per-channel quantizaiton.
        w13_block_size = w13_weight.shape[-1]
        w2_block_size = w2_weight.shape[-1]
    else:
        w13_block_size = w2_block_size = block_size

    _, orig_hidden_size, orig_intermediate_size = w2_weight.shape

    hidden_size = align_to(orig_hidden_size, w13_block_size)
    intermediate_size = align_to(orig_intermediate_size, w2_block_size)

    w13_pad_widths = [[0, 0] for _ in range(3)]
    w13_pad_widths[1][1] = 2 * (intermediate_size - orig_intermediate_size)
    w13_pad_widths[2][1] = hidden_size - orig_hidden_size
    w2_pad_widths = [[0, 0] for _ in range(3)]
    w2_pad_widths[1][1] = hidden_size - orig_hidden_size
    w2_pad_widths[2][1] = intermediate_size - orig_intermediate_size

    w13_weight = jnp.pad(w13_weight, w13_pad_widths)
    w2_weight = jnp.pad(w2_weight, w2_pad_widths)

    if (w13_bias := weights.w13_bias) is not None:
        weights.w13_bias = jnp.pad(w13_bias, w13_pad_widths[:2])
    if (w2_bias := weights.w2_bias) is not None:
        weights.w2_bias = jnp.pad(w2_bias, w2_pad_widths[:2])

    w13_weight, w13_weight_scale = quantize_tensor(dtype, w13_weight, 2,
                                                   w13_block_size)
    w2_weight, w2_weight_scale = quantize_tensor(dtype, w2_weight, 2,
                                                 w2_block_size)

    weights.w13_weight = w13_weight
    weights.w13_weight_scale = w13_weight_scale
    weights.w13_weight_zero_point = None
    weights.w2_weight = w2_weight
    weights.w2_weight_scale = w2_weight_scale
    weights.w2_weight_zero_point = None

    return weights


def process_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    w13_reorder_size: int | None = None,
    w13_interleave: bool = False,
) -> FusedMoEWeights:
    """Process fused moe weights to a layout that moe backend expects.

    Args:
        weights: fused moe weights.
        moe_backend: backend type the weights should be processed for.
        w13_reorder_size: only used when backend type is GMM_TP. in order to
            eliminate collective operations when using tensor parallelism,
            group w13_weight into w13_reorder_size number of chunks where each
            chunk stores both w1 and w3 weights.
        w13_interleave: used when loaded w13_weight is stored in interleaved
            pattern where even index element is w1 and odd index element is w3.
            we uninterleave so that first half is w1 and second half is w3.

    Returns:
        MoE weights that are processed for specified backend.
    """

    w13_weight = weights.w13_weight
    w13_weight_scale = weights.w13_weight_scale
    w13_weight_zero_point = weights.w13_weight_zero_point
    w13_bias = weights.w13_bias
    w2_weight = weights.w2_weight
    w2_weight_scale = weights.w2_weight_scale
    w2_weight_zero_point = weights.w2_weight_zero_point
    w2_bias = weights.w2_bias

    num_experts, hidden_size, intermediate_size = w2_weight.shape

    if w13_interleave:
        w1_weight = w13_weight[:, ::2, :]
        w3_weight = w13_weight[:, 1::2, :]
        w13_weight = jnp.concat([w1_weight, w3_weight], axis=1)

        if w13_weight_scale is not None:
            w1_weight_scale = w13_weight_scale[:, ::2, :]
            w3_weight_scale = w13_weight_scale[:, 1::2, :]
            w13_weight_scale = jnp.concat([w1_weight_scale, w3_weight_scale],
                                          axis=1)

        if w13_weight_zero_point is not None:
            w1_weight_zp = w13_weight_zero_point[:, ::2, :]
            w3_weight_zp = w13_weight_zero_point[:, 1::2, :]
            w13_weight_zero_point = jnp.concat([w1_weight_zp, w3_weight_zp],
                                               axis=1)

        if w13_bias is not None:
            w1_bias = w13_bias[:, ::2]
            w3_bias = w13_bias[:, 1::2]
            w13_bias = jnp.concat([w1_bias, w3_bias], axis=1)

    # Transpose non-constracting dim to right most dim
    w13_weight = jnp.swapaxes(w13_weight, 1, 2)
    w2_weight = jnp.swapaxes(w2_weight, 1, 2)

    # Workaround for JAX error "must have valid byte strides"
    w13_weight = with_layout_constraint(w13_weight, Layout((0, 1, 2)))
    w2_weight = with_layout_constraint(w2_weight, Layout((0, 1, 2)))

    if w13_weight_scale is not None:
        w13_weight_scale = w13_weight_scale.astype(jnp.float32)
        w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
        w13_weight_scale = jnp.expand_dims(w13_weight_scale, 2)
    if w2_weight_scale is not None:
        w2_weight_scale = w2_weight_scale.astype(jnp.float32)
        w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 2)
        w2_weight_scale = jnp.expand_dims(w2_weight_scale, 2)
    if w13_weight_zero_point is not None:
        w13_weight_zero_point = jnp.swapaxes(w13_weight_zero_point, 1, 2)
        w13_weight_zero_point = jnp.expand_dims(w13_weight_zero_point, 2)
    if w2_weight_zero_point is not None:
        w2_weight_zero_point = jnp.swapaxes(w2_weight_zero_point, 1, 2)
        w2_weight_zero_point = jnp.expand_dims(w2_weight_zero_point, 2)
    if w13_bias is not None:
        w13_bias = w13_bias.astype(jnp.float32)
        w13_bias = jnp.expand_dims(w13_bias, 1)
    if w2_bias is not None:
        w2_bias = w2_bias.astype(jnp.float32)
        w2_bias = jnp.expand_dims(w2_bias, 1)

    match moe_backend:
        case MoEBackend.FUSED_MOE:
            # Kernel expects:
            # w13: (num_experts, 2, hidden_size, intermediate_size)
            # w2: (num_experts, intermediate_size, hidden_size)
            # Current format:
            # w13_weight: (num_experts, 2*intermediate_size, hidden_size)
            # w2_weight: (num_experts, hidden_size, intermediate_size)

            w13_weight = w13_weight.reshape(
                num_experts,
                hidden_size,
                2,
                intermediate_size,
            )
            w13_weight = jnp.swapaxes(w13_weight, 1, 2)
            w13_weight = with_layout_constraint(w13_weight, Layout(
                (0, 1, 2, 3)))

            # Fused moe kernel expects dims to be multiple of 256.
            pad_width_intermediate_size = align_to(intermediate_size,
                                                   256) - intermediate_size
            pad_width_hidden_size = align_to(hidden_size, 256) - hidden_size

            w13_weight = jnp.pad(
                w13_weight,
                ((0, 0), (0, 0), (0, pad_width_hidden_size),
                 (0, pad_width_intermediate_size)),
            )

            w2_weight = jnp.pad(
                w2_weight,
                ((0, 0), (0, pad_width_intermediate_size),
                 (0, pad_width_hidden_size)),
            )

            if w13_weight_scale is not None:
                w13_weight_scale = w13_weight_scale.reshape(
                    num_experts, -1, 2, 1, intermediate_size)
                w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
                w13_weight_scale = jnp.pad(
                    w13_weight_scale,
                    ((0, 0), (0, 0), (0, pad_width_hidden_size), (0, 0),
                     (0, pad_width_intermediate_size)),
                )
            if w2_weight_scale is not None:
                w2_weight_scale = jnp.pad(
                    w2_weight_scale,
                    ((0, 0), (0, pad_width_intermediate_size), (0, 0),
                     (0, pad_width_hidden_size)),
                )

            if w13_weight_zero_point is not None:
                w13_weight_zero_point = w13_weight_zero_point.reshape(
                    num_experts, -1, 2, 1, intermediate_size)
                w13_weight_zero_point = jnp.swapaxes(w13_weight_zero_point, 1,
                                                     2)
                w13_weight_zero_point = jnp.pad(
                    w13_weight_zero_point,
                    ((0, 0), (0, 0), (0, pad_width_hidden_size), (0, 0),
                     (0, pad_width_intermediate_size)),
                )
            if w2_weight_zero_point is not None:
                w2_weight_zero_point = jnp.pad(
                    w2_weight_zero_point,
                    ((0, 0), (0, pad_width_intermediate_size), (0, 0),
                     (0, pad_width_hidden_size)),
                )

            if w13_bias is not None:
                w13_bias = w13_bias.reshape(num_experts, 2, 1,
                                            intermediate_size)
                w13_bias = jnp.pad(
                    w13_bias,
                    ((0, 0), (0, 0), (0, 0), (0, pad_width_intermediate_size)),
                )
            if w2_bias is not None:
                w2_bias = jnp.pad(
                    w2_bias,
                    ((0, 0), (0, 0), (0, pad_width_hidden_size)),
                )

        case MoEBackend.GMM_TP:
            assert w13_reorder_size is not None
            assert intermediate_size % w13_reorder_size == 0
            output_sizes = [intermediate_size, intermediate_size]
            w13_weight = reorder_concatenated_tensor_for_sharding(
                w13_weight,
                output_sizes,
                w13_reorder_size,
                dim=2,
            )
            if w13_weight_scale is not None:
                w13_weight_scale = reorder_concatenated_tensor_for_sharding(
                    w13_weight_scale,
                    output_sizes,
                    w13_reorder_size,
                    dim=3,
                )
            if w13_weight_zero_point is not None:
                w13_weight_zero_point = (
                    reorder_concatenated_tensor_for_sharding(
                        w13_weight_zero_point,
                        output_sizes,
                        w13_reorder_size,
                        dim=3,
                    ))
            if w13_bias is not None:
                w13_bias = reorder_concatenated_tensor_for_sharding(
                    w13_bias,
                    output_sizes,
                    w13_reorder_size,
                    dim=2,
                )
        case MoEBackend.GMM_EP:
            # No additional processing is needed for GMM_EP.
            pass

        case MoEBackend.DENSE_MAT:
            # TODO (jacobplatin)
            raise NotImplementedError(
                "process_moe_weights is not yet implemented for dense matmul backend."
            )
        case MoEBackend.MEGABLX_GMM:
            # TODO (jacobplatin)
            raise NotImplementedError(
                "process_moe_weights is not yet implemented for megablox gmm backend"
            )

    return FusedMoEWeights(
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w13_weight_zero_point=w13_weight_zero_point,
        w13_bias=w13_bias,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        w2_weight_zero_point=w2_weight_zero_point,
        w2_bias=w2_bias,
    )


def shard_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh: Mesh,
) -> FusedMoEWeights:

    match moe_backend:
        case MoEBackend.FUSED_MOE | MoEBackend.GMM_EP:
            ep_sharding = NamedSharding(mesh, P(ShardingAxisName.EXPERT))
            weight_shardings = FusedMoEWeights(
                w13_weight=ep_sharding,
                w13_weight_scale=ep_sharding,
                w13_weight_zero_point=ep_sharding,
                w13_bias=ep_sharding,
                w2_weight=ep_sharding,
                w2_weight_scale=ep_sharding,
                w2_weight_zero_point=ep_sharding,
                w2_bias=ep_sharding,
            )
        case MoEBackend.GMM_TP:
            # When using per-channel, in_dim // block_size == 1. This means we
            # are unable to shard w2_weight_scale along 1st dim. Therefore, we
            # fully replicate it instead.
            if (weights.w2_weight_scale is not None
                    and weights.w2_weight_scale.shape[1] == 1):
                w2_weight_scale_p_spec = P()
            else:
                w2_weight_scale_p_spec = P(None, ShardingAxisName.MLP_TENSOR)

            # Same logic as w2_weight_scale.
            if (weights.w2_weight_zero_point is not None
                    and weights.w2_weight_zero_point.shape[1] == 1):
                w2_weight_zp_p_spec = P()
            else:
                w2_weight_zp_p_spec = P(None, ShardingAxisName.MLP_TENSOR)

            weight_shardings = FusedMoEWeights(
                w13_weight=NamedSharding(
                    mesh,
                    P(None, None, ShardingAxisName.MLP_TENSOR),
                ),  # (num_experts, out_dim, in_dim)
                w13_weight_scale=NamedSharding(
                    mesh,
                    P(None, None, None, ShardingAxisName.MLP_TENSOR),
                ),  # (num_experts, in_dim // block_size, 1, out_dim)
                w13_weight_zero_point=NamedSharding(
                    mesh,
                    P(None, None, None, ShardingAxisName.MLP_TENSOR),
                ),  # (num_experts, in_dim // block_size, 1, out_dim)
                w13_bias=NamedSharding(
                    mesh,
                    P(None, None, ShardingAxisName.MLP_TENSOR),
                ),  # (num_experts, 1, out_dim)
                w2_weight=NamedSharding(
                    mesh,
                    P(None, ShardingAxisName.MLP_TENSOR, None),
                ),  # (num_experts, out_dim, in_dim)
                w2_weight_scale=NamedSharding(
                    mesh, w2_weight_scale_p_spec
                ),  # (num_experts, in_dim // block_size, 1, out_dim)
                w2_weight_zero_point=NamedSharding(
                    mesh, w2_weight_zp_p_spec
                ),  # (num_experts, in_dim // block_size, 1, out_dim)
                w2_bias=NamedSharding(
                    mesh,
                    P(None, None, None),
                ),  # (num_experts, 1, out_dim)
            )

    match moe_backend:
        case MoEBackend.FUSED_MOE:
            weight_layouts = FusedMoEWeights(
                w13_weight=Layout((0, 1, 2, 3)),
                w13_weight_scale=Layout((0, 1, 2, 3, 4)),
                w13_weight_zero_point=Layout((0, 1, 2, 3, 4)),
                w13_bias=Layout((0, 1, 2, 3)),
                w2_weight=Layout((0, 1, 2)),
                w2_weight_scale=Layout((0, 1, 2, 3)),
                w2_weight_zero_point=Layout((0, 1, 2, 3)),
                w2_bias=Layout((0, 1, 2)),
            )
        case MoEBackend.GMM_TP | MoEBackend.GMM_EP:
            weight_layouts = FusedMoEWeights(
                w13_weight=Layout((0, 1, 2)),
                w13_weight_scale=Layout((0, 1, 2, 3)),
                w13_weight_zero_point=Layout((0, 1, 2, 3)),
                w13_bias=Layout((0, 1, 2)),
                w2_weight=Layout((0, 1, 2)),
                w2_weight_scale=Layout((0, 1, 2, 3)),
                w2_weight_zero_point=Layout((0, 1, 2, 3)),
                w2_bias=Layout((0, 1, 2)),
            )

    for field in fields(FusedMoEWeights):
        key = field.name
        if (weight := getattr(weights, key, None)) is not None:
            layout = getattr(weight_layouts, key)
            sharding = getattr(weight_shardings, key)
            weight = general_device_put(weight, sharding, layout=layout)
            setattr(weights, key, weight)
    return weights


@functools.partial(jax.jit,
                   static_argnames=(
                       "moe_backend",
                       "mesh",
                       "activation",
                       "weight_block_size",
                   ))
def process_fp8_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh: Mesh,
    activation: str,
    weight_block_size: tuple[int, ...] | None = None,
) -> FusedMoEWeights:
    w13_weight = weights.w13_weight
    w13_weight_scale = weights.w13_weight_scale
    w2_weight = weights.w2_weight
    w2_weight_scale = weights.w2_weight_scale

    # Dequantize fp8 2d block quantized weights into fp32.
    w13_weight = dequantize_tensor(w13_weight,
                                   w13_weight_scale, (1, 2),
                                   jnp.float32,
                                   block_size=weight_block_size)
    w2_weight = dequantize_tensor(w2_weight,
                                  w2_weight_scale, (1, 2),
                                  jnp.float32,
                                  block_size=weight_block_size)

    w13_interleave = activation == "swigluoai"
    w13_reorder_size = get_mesh_shape_product(mesh,
                                              ShardingAxisName.MLP_TENSOR)

    weights = quantize_moe_weights(
        FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=None,
            w13_weight_zero_point=None,
            w13_bias=None,
            w2_weight=w2_weight,
            w2_weight_scale=None,
            w2_weight_zero_point=None,
            w2_bias=None,
        ),
        jnp.float8_e4m3fn,
        None,
    )
    return process_moe_weights(
        weights,
        moe_backend=moe_backend,
        w13_reorder_size=w13_reorder_size,
        w13_interleave=w13_interleave,
    )
