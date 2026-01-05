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

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from jax.experimental.layout import Format, Layout, with_layout_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.tensor import Tensor

from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.vllm.fused_moe import FusedMoEBackend
from tpu_inference.utils import align_to

P = PartitionSpec


@jax.tree_util.register_dataclass
@dataclass
class FusedMoEWeights:
    """Fused moe weights. weights can be either jax or torchax array."""
    w13_weight: jax.Array | Tensor
    w13_weight_scale: jax.Array | Tensor | None
    w13_bias: jax.Array | Tensor | None
    w2_weight: jax.Array | Tensor
    w2_weight_scale: jax.Array | Tensor | None
    w2_bias: jax.Array | Tensor | None


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

    w13_weight = weights.w13_weight
    w2_weight = weights.w2_weight

    if block_size is None:
        # Use per-channel quantizaiton.
        w13_block_size = w13_weight.shape[-1]
        w2_block_size = w2_weight.shape[-1]
    else:
        w13_block_size = w2_block_size = block_size

    _, orig_hidden_size, orig_intermediate_size = w2_weight.shape

    w13_weight, w13_weight_scale = quantize_tensor(dtype, w13_weight, 2,
                                                   w13_block_size, True)
    w2_weight, w2_weight_scale = quantize_tensor(dtype, w2_weight, 2,
                                                 w2_block_size, True)

    intermediate_size = w2_weight.shape[-1]
    hidden_size = w13_weight.shape[-1]

    # Dims may have been padded to align with subchannel size during
    # quantization. We pad the corresponding dim on other weight.
    # NOTE: We perform padding after quantization as padding value can
    # affect quantization numerics.
    w13_pad_widths = [[0, 0] for _ in range(3)]
    w13_pad_widths[1][1] = 2 * (intermediate_size - orig_intermediate_size)
    w2_pad_widths = [[0, 0] for _ in range(3)]
    w2_pad_widths[1][1] = hidden_size - orig_hidden_size

    weights.w13_weight = jnp.pad(w13_weight, w13_pad_widths)
    weights.w13_weight_scale = jnp.pad(w13_weight_scale, w13_pad_widths)
    weights.w2_weight = jnp.pad(w2_weight, w2_pad_widths)
    weights.w2_weight_scale = jnp.pad(w2_weight_scale, w2_pad_widths)

    if (w13_bias := weights.w13_bias) is not None:
        weights.w13_bias = jnp.pad(w13_bias, w13_pad_widths[:2])
    if (w2_bias := weights.w2_bias) is not None:
        weights.w2_bias = jnp.pad(w2_bias, w2_pad_widths[:2])

    return weights


def process_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: FusedMoEBackend,
    w13_reorder_size: int | None = None,
    w13_interleave: bool = False,
) -> FusedMoEWeights:
    """Process fused moe weights to a layout that moe backend expects.

    Args:
        weights: fused moe weights.
        moe_backend: backend type the weights should be processed for.
        w13_reorder_size: only used when backend type is GMM_TP. in order to
            eliminate collective operations when using tensor parallelism, 
            group w13_weight into w13_reorder_size number of chuncks where each
            chunk stores both w1 and w3 weights.
        w13_interleave: used when loaded w13_weight is stored in interleaved
            pattern where even index element is w1 and odd index element is w3.
            we uninterleave so that first half is w1 and second half is w3.

    Returns:
        MoE weights that are processed for specified backend.
    """

    w13_weight = weights.w13_weight
    w13_weight_scale = weights.w13_weight_scale
    w13_bias = weights.w13_bias
    w2_weight = weights.w2_weight
    w2_weight_scale = weights.w2_weight_scale
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

        if w13_bias is not None:
            w1_bias = w13_bias[:, ::2]
            w3_bias = w13_bias[:, 1::2]
            w13_bias = jnp.concat([w1_bias, w3_bias], axis=1)

    if w13_weight_scale is not None:
        w13_weight_scale = w13_weight_scale.astype(jnp.float32)
    if w2_weight_scale is not None:
        w2_weight_scale = w2_weight_scale.astype(jnp.float32)
    if w13_bias is not None:
        w13_bias = w13_bias.astype(jnp.float32)
    if w2_bias is not None:
        w2_bias = w2_bias.astype(jnp.float32)

    match moe_backend:
        case FusedMoEBackend.FUSED_MOE:
            # Kernel expects:
            # w13: (num_experts, 2, hidden_size, intermediate_size)
            # w2: (num_experts, intermediate_size, hidden_size)
            # Current format:
            # w13_weight: (num_experts, 2*intermediate_size, hidden_size)
            # w2_weight: (num_experts, hidden_size, intermediate_size)

            # Fused moe kernel expects dims to be multiple of 256.
            pad_width_intermediate_size = align_to(intermediate_size,
                                                   256) - intermediate_size
            pad_width_hidden_size = align_to(hidden_size, 256) - hidden_size

            w13_weight = w13_weight.reshape(
                num_experts,
                2,
                intermediate_size,
                hidden_size,
            )

            # Transpose non-constracting dim to right most dim
            w13_weight = jnp.swapaxes(w13_weight, 2, 3)
            w2_weight = jnp.swapaxes(w2_weight, 1, 2)

            # Workaround for JAX error "must have valid byte strides"
            w13_weight = with_layout_constraint(w13_weight, Layout(
                (0, 1, 2, 3)))
            w2_weight = with_layout_constraint(w2_weight, Layout((0, 1, 2)))

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
                    num_experts, 2, intermediate_size, 1, -1)
                w13_weight_scale = jnp.swapaxes(w13_weight_scale, 2, 4)
                w13_weight_scale = jnp.pad(
                    w13_weight_scale,
                    ((0, 0), (0, 0), (0, pad_width_hidden_size), (0, 0),
                     (0, pad_width_intermediate_size)),
                )
            if w2_weight_scale is not None:
                w2_weight_scale = w2_weight_scale.reshape(
                    num_experts, hidden_size, 1, -1)
                w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 3)
                w2_weight_scale = jnp.pad(
                    w2_weight_scale,
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
                w2_bias = w2_bias.reshape(num_experts, 1, hidden_size)
                w2_bias = jnp.pad(
                    w2_bias,
                    ((0, 0), (0, 0), (0, pad_width_hidden_size)),
                )

        case FusedMoEBackend.GMM_EP | FusedMoEBackend.GMM_TP:
            if w13_weight_scale is not None:
                w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
                w13_weight_scale = jnp.expand_dims(w13_weight_scale, 2)
            if w2_weight_scale is not None:
                w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 2)
                w2_weight_scale = jnp.expand_dims(w2_weight_scale, 2)
            if w13_bias is not None:
                w13_bias = jnp.expand_dims(w13_bias, 1)
            if w2_bias is not None:
                w2_bias = jnp.expand_dims(w2_bias, 1)

            if moe_backend == FusedMoEBackend.GMM_TP:
                assert w13_reorder_size is not None
                assert intermediate_size % w13_reorder_size == 0
                output_sizes = [intermediate_size, intermediate_size]
                w13_weight = reorder_concatenated_tensor_for_sharding(
                    w13_weight,
                    output_sizes,
                    w13_reorder_size,
                    dim=1,
                )
                if w13_weight_scale is not None:
                    w13_weight_scale = reorder_concatenated_tensor_for_sharding(
                        w13_weight_scale,
                        output_sizes,
                        w13_reorder_size,
                        dim=3,
                    )
                if w13_bias is not None:
                    w13_bias = reorder_concatenated_tensor_for_sharding(
                        w13_bias,
                        output_sizes,
                        w13_reorder_size,
                        dim=2,
                    )

    return FusedMoEWeights(
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w13_bias=w13_bias,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        w2_bias=w2_bias,
    )


def shard_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: FusedMoEBackend,
    mesh: Mesh,
) -> FusedMoEWeights:

    match moe_backend:
        case FusedMoEBackend.FUSED_MOE | FusedMoEBackend.GMM_EP:
            ep_sharding = NamedSharding(mesh, P(ShardingAxisName.EXPERT))
            weight_shardings = FusedMoEWeights(
                w13_weight=ep_sharding,
                w13_weight_scale=ep_sharding,
                w13_bias=ep_sharding,
                w2_weight=ep_sharding,
                w2_weight_scale=ep_sharding,
                w2_bias=ep_sharding,
            )
        case FusedMoEBackend.GMM_TP:
            # When using per-channel, in_dim // block_size == 1. This means we
            # are unable to shard w2_weight_scale along 1st dim. Therefore, we
            # fully replicate it instead.
            if (weights.w2_weight_scale is not None
                    and weights.w2_weight_scale.shape[1] == 1):
                w2_weight_scale_p_spec = P()
            else:
                w2_weight_scale_p_spec = P(None, ShardingAxisName.MLP_TENSOR)
            weight_shardings = FusedMoEWeights(
                w13_weight=NamedSharding(
                    mesh,
                    P(None, ShardingAxisName.MLP_TENSOR, None),
                ),  # (num_experts, out_dim, in_dim)
                w13_weight_scale=NamedSharding(
                    mesh,
                    P(None, None, None, ShardingAxisName.MLP_TENSOR),
                ),  # (num_experts, in_dim // block_size, 1, out_dim)
                w13_bias=NamedSharding(
                    mesh,
                    P(None, None, ShardingAxisName.MLP_TENSOR),
                ),  # (num_experts, 1, out_dim)
                w2_weight=NamedSharding(
                    mesh,
                    P(None, None, ShardingAxisName.MLP_TENSOR),
                ),  # (num_experts, out_dim, in_dim)
                w2_weight_scale=NamedSharding(
                    mesh, w2_weight_scale_p_spec
                ),  # (num_experts, in_dim // block_size, 1, out_dim)
                w2_bias=NamedSharding(
                    mesh,
                    P(None, None, None),
                ),  # (num_experts, 1, out_dim)
            )

    match moe_backend:
        case FusedMoEBackend.FUSED_MOE:
            weight_layouts = FusedMoEWeights(
                w13_weight=Layout((0, 1, 2, 3)),
                w13_weight_scale=Layout((0, 1, 2, 3, 4)),
                w13_bias=Layout((0, 1, 2, 3)),
                w2_weight=Layout((0, 1, 2)),
                w2_weight_scale=Layout((0, 1, 2, 3)),
                w2_bias=Layout((0, 1, 2)),
            )
        case FusedMoEBackend.GMM_TP | FusedMoEBackend.GMM_EP:
            weight_layouts = FusedMoEWeights(
                w13_weight=Layout((0, 1, 2)),
                w13_weight_scale=Layout((0, 1, 2, 3)),
                w13_bias=Layout((0, 1, 2)),
                w2_weight=Layout((0, 1, 2)),
                w2_weight_scale=Layout((0, 1, 2, 3)),
                w2_bias=Layout((0, 1, 2)),
            )

    for field in fields(FusedMoEWeights):
        key = field.name
        if (weight := getattr(weights, key, None)) is not None:
            layout = getattr(weight_layouts, key)
            sharding = getattr(weight_shardings, key)
            weight = jax.device_put(weight, Format(layout, sharding))
            setattr(weights, key, weight)
    return weights
