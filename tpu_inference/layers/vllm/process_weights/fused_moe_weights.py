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
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.vllm.fused_moe import FusedMoEBackend
from tpu_inference.utils import align_to

P = PartitionSpec
N = NamedSharding


def quantize_moe_weights(w13_weight: jax.Array, w13_bias: jax.Array | None,
                         w2_weight: jax.Array, w2_bias: jax.Array | None,
                         dtype: jnp.dtype, block_size: int | None):
    if block_size is None:
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

    w13_weight = jnp.pad(w13_weight, w13_pad_widths)
    w13_weight_scale = jnp.pad(w13_weight_scale, w13_pad_widths)
    if w13_bias is not None:
        w13_bias = jnp.pad(w13_bias, w13_pad_widths[:2])

    w2_pad_widths = [[0, 0] for _ in range(3)]
    w2_pad_widths[1][1] = hidden_size - orig_hidden_size

    w2_weight = jnp.pad(w2_weight, w2_pad_widths)
    w2_weight_scale = jnp.pad(w2_weight_scale, w2_pad_widths)
    if w2_bias is not None:
        w2_bias = jnp.pad(w2_bias, w2_pad_widths[:2])

    return w13_weight, w13_weight_scale, w13_bias, w2_weight, w2_weight_scale, w2_bias


def process_moe_weights(
    w13_weight: jax.Array,
    w13_weight_scale: jax.Array | None,
    w13_bias: jax.Array | None,
    w2_weight: jax.Array,
    w2_weight_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    *,
    moe_backend: FusedMoEBackend,
    w13_reorder_size: int | None = None,
    w13_interleave: bool = False,
):
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

    match moe_backend:
        case FusedMoEBackend.FUSED_MOE:
            # Kernel expects:
            # w13: (num_experts, 2, hidden_size, intermediate_size)
            # w2: (num_experts, intermediate_size, hidden_size)
            # Current format:
            # w13_weight: (num_experts, 2*intermediate_size, hidden_size)
            # w2_weight: (num_experts, hidden_size, intermediate_size)

            pad_width_intermediate_size = 2 * (
                align_to(intermediate_size, 256) - intermediate_size)
            pad_width_hidden_size = align_to(hidden_size, 256) - hidden_size

            w13_pad_width = [[0, 0] for _ in range(3)]
            w13_pad_width[1][1] = pad_width_hidden_size
            w13_pad_width[2][1] = pad_width_intermediate_size

            w2_pad_width = [[0, 0] for _ in range(3)]
            w2_pad_width[1][1] = pad_width_intermediate_size
            w2_pad_width[2][1] = pad_width_hidden_size

            w13_weight = jnp.pad(w13_weight, w13_pad_width)
            w2_weight = jnp.pad(w2_weight, w2_pad_width)

            w13_weight = w13_weight.reshape(num_experts, 2, intermediate_size,
                                            hidden_size)

            # Transpose non-constracting dim to right most dim
            w13_weight = jnp.swapaxes(w13_weight, 2, 3)
            w2_weight = jnp.swapaxes(w2_weight, 1, 2)

            if w13_weight_scale is not None:
                w13_weight_scale = jnp.pad(w13_weight_scale,
                                           w13_pad_width[:2] + [[0, 0]])
                w13_weight_scale = w13_weight_scale.reshape(
                    num_experts, 2, intermediate_size, 1, -1)
                w13_weight_scale = jnp.swapaxes(w13_weight_scale, 2, 4)
            if w2_weight_scale is not None:
                w2_weight_scale = jnp.pad(w2_weight_scale,
                                          w2_pad_width[:2] + [[0, 0]])
                w2_weight_scale = w2_weight_scale.reshape(
                    num_experts, hidden_size, 1, -1)
                w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 3)

            if w13_bias is not None:
                w13_bias = jnp.pad(w13_bias, w13_pad_width[:2])
                w13_bias = w13_bias.astype(jnp.float32).reshape(
                    num_experts, 2, 1, intermediate_size)
            if w2_bias is not None:
                w2_bias = jnp.pad(w2_bias, w2_pad_width[:2])
                w2_bias = w2_bias.astype(jnp.float32).reshape(
                    num_experts, 1, hidden_size)

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

    return w13_weight, w13_weight_scale, w13_bias, w2_weight, w2_weight_scale, w2_bias


def shard_moe_weights(weights: list[jax.Array | None],
                      *,
                      moe_backend: FusedMoEBackend,
                      mesh: Mesh,
                      per_channel=False):
    # TODO: Use proper class structure.
    assert len(weights) == 6

    match moe_backend:
        case FusedMoEBackend.FUSED_MOE | FusedMoEBackend.GMM_EP:
            ep_sharding = N(mesh, P(ShardingAxisName.EXPERT))
            weight_shardings = [ep_sharding] * 6
        case FusedMoEBackend.GMM_TP:
            w2_weight_scale_p_spec = P() if per_channel else P(
                None,
                ShardingAxisName.MLP_TENSOR,
            )
            weight_shardings = [
                Format(
                    Layout((0, 1, 2)),
                    N(mesh, P(None, ShardingAxisName.MLP_TENSOR, None)),
                ),
                Format(
                    Layout((0, 1, 2, 3)),
                    N(mesh, P(None, None, None, ShardingAxisName.MLP_TENSOR)),
                ),
                Format(Layout((0, 1, 2)),
                       N(mesh, P(None, None, ShardingAxisName.MLP_TENSOR))),
                Format(Layout((0, 1, 2)),
                       N(mesh, P(None, None, ShardingAxisName.MLP_TENSOR))),
                Format(Layout((0, 1, 2, 3)), N(mesh, w2_weight_scale_p_spec)),
                Format(Layout((0, 1, 2)), N(mesh, P(None, None, None))),
            ]

    outs = []
    for weight, sharding in zip(weights, weight_shardings):
        if weight is not None:
            weight = jax.device_put(weight, sharding)
        outs.append(weight)
    return outs
