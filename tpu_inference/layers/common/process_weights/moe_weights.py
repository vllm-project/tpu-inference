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
from jax.experimental.layout import Layout, with_layout_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.tensor import Tensor
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

import tpu_inference.envs as envs
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    general_device_put, reorder_concatenated_tensor_for_sharding)
from tpu_inference.logger import init_logger
from tpu_inference.utils import align_to, get_mesh_shape_product, to_jax_dtype

P = PartitionSpec

logger = init_logger(__name__)


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
    w13_interleave: bool = False,
) -> FusedMoEWeights:
    """Quantize fused moe weights into a given dtype and block size.

    Args:
        weights: fused moe weights.
        dtype: dtype to perform quantization.
        block_size: Specify block quantization size. If non, use per-channel
            quantization. If contracting dim is not divisible by block size,
            the dim will be automatically padded and corresponding dim on bias
            and the other weight (w13_weight <-> w2_weight) is also padded.
        w13_interleave: used when loaded w13_weight is stored in interleaved
            pattern where even index element is w1 and odd index element is w3
            we uninterleave so that first half is w1 and second half is w3.

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

    hidden_size = align_to(orig_hidden_size, w13_block_size)
    intermediate_size = align_to(orig_intermediate_size, w2_block_size)

    inter_pad = intermediate_size - orig_intermediate_size
    hidden_pad = hidden_size - orig_hidden_size

    if w13_interleave:
        w13_pad_widths = [[0, 0] for _ in range(3)]
        w13_pad_widths[1][1] = 2 * inter_pad
        w13_pad_widths[2][1] = hidden_pad
        w13_weight = jnp.pad(w13_weight, w13_pad_widths)
        if (w13_bias := weights.w13_bias) is not None:
            weights.w13_bias = jnp.pad(w13_bias, w13_pad_widths[:2])
    else:
        w1 = w13_weight[:, :orig_intermediate_size, :]
        w3 = w13_weight[:, orig_intermediate_size:, :]
        w13_pad_widths = [[0, 0], [0, inter_pad], [0, hidden_pad]]
        w1 = jnp.pad(w1, w13_pad_widths)
        w3 = jnp.pad(w3, w13_pad_widths)
        w13_weight = jnp.concatenate([w1, w3], axis=1)
        if (w13_bias := weights.w13_bias) is not None:
            b1 = w13_bias[:, :orig_intermediate_size]
            b3 = w13_bias[:, orig_intermediate_size:]
            b1 = jnp.pad(b1, w13_pad_widths[:2])
            b3 = jnp.pad(b3, w13_pad_widths[:2])
            weights.w13_bias = jnp.concatenate([b1, b3], axis=1)

    w2_pad_widths = [[0, 0], [0, hidden_pad], [0, inter_pad]]
    w2_weight = jnp.pad(w2_weight, w2_pad_widths)
    if (w2_bias := weights.w2_bias) is not None:
        weights.w2_bias = jnp.pad(w2_bias, w2_pad_widths[:2])

    w13_weight, w13_weight_scale = quantize_tensor(dtype, w13_weight, 2,
                                                   w13_block_size)
    w2_weight, w2_weight_scale = quantize_tensor(dtype, w2_weight, 2,
                                                 w2_block_size)

    weights.w13_weight = w13_weight
    weights.w13_weight_scale = w13_weight_scale
    weights.w2_weight = w2_weight
    weights.w2_weight_scale = w2_weight_scale

    return weights


@dataclass
class W13PaddingConfig:
    intermediate_size: int
    w13_reorder_size: int
    local_intermediate_size: int
    pad_amount: int
    padded_intermediate_size: int


def get_w13_padding_config(intermediate_size: int,
                           reorder_size: int,
                           align: int = 128,
                           outer_block_size: int = 1) -> W13PaddingConfig:
    """Calculates padded dimensions and pad amounts for w13 tensors.

    Args:
        intermediate_size: size of the intermediate dimension.
        reorder_size: size of the reorder dimension.
        align: alignment of the padded dimensions.
        outer_block_size: outer block size of the quantized weights.
            It is 1 for 1D block quantization and > 1 for 2D block quantization.

    Returns:
        W13PaddingConfig
    """
    local_intermediate_size = intermediate_size // reorder_size

    padded_local_intermediate_size = align_to(local_intermediate_size, align)
    padded_intermediate_size = padded_local_intermediate_size * reorder_size
    pad_amount = padded_local_intermediate_size - local_intermediate_size

    assert padded_intermediate_size % outer_block_size == 0
    assert pad_amount % outer_block_size == 0
    assert local_intermediate_size % outer_block_size == 0
    assert intermediate_size % outer_block_size == 0

    return W13PaddingConfig(
        intermediate_size=intermediate_size // outer_block_size,
        w13_reorder_size=reorder_size,
        local_intermediate_size=local_intermediate_size // outer_block_size,
        pad_amount=pad_amount // outer_block_size,
        padded_intermediate_size=padded_intermediate_size // outer_block_size)


def process_w13_for_gmm(tensor,
                        concat_dim: int,
                        config: W13PaddingConfig,
                        padded_output_sizes: list[int] | None = None,
                        name: str = "w13"):
    """Splits, pads, concatenates, and optionally reorders W13 tensors for GMM backends.

    This function takes a fused W13 tensor (which contains both W1 and W3 weights
    or their corresponding scales), splits them apart, applies specific padding
    for alignment, and then recombines them.

    Args:
        tensor: The input JAX array. Can be the actual weight tensor or its
            corresponding block-quantized scale tensor.
        concat_dim: The axis dimension along which W1 and W3 are concatenated.
        config: A `W13PaddingConfig` object containing the unscaled sizes
            and padding amounts calculated based on the full *weight*
            tensor dimensions.
        padded_output_sizes: Optional list of sizes for the padded W1 and W3
            blocks. If provided, triggers a reordering of the concatenated
            tensor for optimal TP sharding.
        name: String identifier used for logging tensor shapes.

    Returns:
        The processed JAX array, appropriately padded and dimensionally aligned
        for the target MoE hardware backend.
    """

    # 1. Split into W1 and W3
    w1 = tensor[..., :config.intermediate_size]
    w3 = tensor[..., config.intermediate_size:]

    # 2. Pad the intermediate dimension
    def _pad_tensor(t):
        dims = t.shape[:-1]
        # Reshape to expose local_intermediate_size
        t = t.reshape(*dims, config.w13_reorder_size,
                      config.local_intermediate_size)

        # Dynamically create pad widths based on the reshaped tensor's rank
        pad_widths = [(0, 0)] * t.ndim
        # Padding for the last dimension
        pad_widths[-1] = (0, config.pad_amount)
        t = jnp.pad(t, pad_widths)

        # Reshape back
        return t.reshape(*dims, config.padded_intermediate_size)

    # Apply padding
    padded_w1 = _pad_tensor(w1)
    padded_w3 = _pad_tensor(w3)

    logger.info(f"{name}_w1 shape after padding: {padded_w1.shape}")
    logger.info(f"{name}_w3 shape after padding: {padded_w3.shape}")

    # 3. Concatenate and Reorder for avoiding TP sharding comms
    w13_concat = jnp.concatenate([padded_w1, padded_w3], axis=concat_dim)
    if padded_output_sizes is not None:
        w13_concat = reorder_concatenated_tensor_for_sharding(
            w13_concat,
            padded_output_sizes,
            config.w13_reorder_size,
            dim=concat_dim,
        )
    return w13_concat


def process_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    w13_reorder_size: int | None = None,
    w13_interleave: bool = False,
    disable_weight_requantization: bool = False,
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
        disable_weight_requantization: whether to keep scales broad for GMM
            setups.

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
            # If scale is block-quantized along the inner dimension, adjust stride
            if w13_weight_scale.shape[1] == w13_weight.shape[1]:
                w1_weight_scale = w13_weight_scale[:, ::2, :]
                w3_weight_scale = w13_weight_scale[:, 1::2, :]
                w13_weight_scale = jnp.concat(
                    [w1_weight_scale, w3_weight_scale], axis=1)
            else:
                block_size = w13_weight.shape[1] // w13_weight_scale.shape[1]
                assert block_size % 2 == 0, (
                    f"Block size {block_size} must be even for "
                    "interleaved weights")

        if w13_bias is not None:
            w1_bias = w13_bias[:, ::2]
            w3_bias = w13_bias[:, 1::2]
            w13_bias = jnp.concat([w1_bias, w3_bias], axis=1)

    # Transpose non-contracting dim to right most dim
    w13_weight = jnp.swapaxes(w13_weight, 1, 2)
    w2_weight = jnp.swapaxes(w2_weight, 1, 2)

    # Workaround for JAX error "must have valid byte strides"
    w13_weight = with_layout_constraint(w13_weight, Layout((0, 1, 2)))
    w2_weight = with_layout_constraint(w2_weight, Layout((0, 1, 2)))

    if w13_weight_scale is not None:
        # For block scales (experts, out_blocks, in_blocks), we need to maintain
        # the block dims
        w13_weight_scale = w13_weight_scale.astype(jnp.float32)
        w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
        w13_weight_scale = jnp.expand_dims(w13_weight_scale, 2)

    if w2_weight_scale is not None:
        w2_weight_scale = w2_weight_scale.astype(jnp.float32)
        w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 2)
        w2_weight_scale = jnp.expand_dims(w2_weight_scale, 2)

    w13_outer_block_size = 1
    w2_outer_block_size = 1
    if disable_weight_requantization:
        if w13_weight_scale is not None:
            assert w13_weight.shape[2] % w13_weight_scale.shape[3] == 0
            w13_outer_block_size = (w13_weight.shape[2] //
                                    w13_weight_scale.shape[3])
        if w2_weight_scale is not None:
            assert w2_weight.shape[2] % w2_weight_scale.shape[3] == 0
            w2_outer_block_size = (w2_weight.shape[2] //
                                   w2_weight_scale.shape[3])

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
            pad_width_intermediate_size = (align_to(intermediate_size, 256) -
                                           intermediate_size)
            pad_width_hidden_size = align_to(hidden_size, 256) - hidden_size

            w13_weight = jnp.pad(w13_weight,
                                 ((0, 0), (0, 0), (0, pad_width_hidden_size),
                                  (0, pad_width_intermediate_size)))

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

            pad_config_weight = get_w13_padding_config(intermediate_size,
                                                       w13_reorder_size,
                                                       align=128)

            padded_output_sizes = [
                pad_config_weight.padded_intermediate_size,
                pad_config_weight.padded_intermediate_size
            ]

            w13_weight = process_w13_for_gmm(
                tensor=w13_weight,
                concat_dim=2,
                config=pad_config_weight,
                padded_output_sizes=padded_output_sizes,
                name="w13_weight")

            if w13_weight_scale is not None:
                pad_config_scale = get_w13_padding_config(
                    intermediate_size,
                    w13_reorder_size,
                    align=128,
                    outer_block_size=w13_outer_block_size)
                padded_output_sizes_scales = [
                    pad_config_scale.padded_intermediate_size,
                    pad_config_scale.padded_intermediate_size
                ]
                w13_weight_scale = process_w13_for_gmm(
                    tensor=w13_weight_scale,
                    concat_dim=3,
                    config=pad_config_scale,
                    padded_output_sizes=padded_output_sizes_scales,
                    name="w13_weight_scale")
                if w13_outer_block_size > 1:
                    # GMM currently expects scales to be broadcasted to
                    # full shape along the contracting dimension when
                    # skipping requantization.
                    w13_weight_scale = jnp.repeat(w13_weight_scale,
                                                  w13_outer_block_size,
                                                  axis=3)
            if w13_bias is not None:
                w13_bias = process_w13_for_gmm(
                    tensor=w13_bias,
                    concat_dim=2,
                    config=pad_config_weight,
                    padded_output_sizes=padded_output_sizes,
                    name="w13_bias")
            if w2_weight_scale is not None:
                if w2_outer_block_size > 1:
                    # GMM currently expects scales to be broadcasted to
                    # full shape along the contracting dimension when
                    # skipping requantization.
                    w2_weight_scale = jnp.repeat(w2_weight_scale,
                                                 w2_outer_block_size,
                                                 axis=3)

        case MoEBackend.GMM_EP:
            pad_config_weight = get_w13_padding_config(intermediate_size,
                                                       reorder_size=1,
                                                       align=128)

            w13_weight = process_w13_for_gmm(tensor=w13_weight,
                                             concat_dim=2,
                                             config=pad_config_weight,
                                             name="w13_weight")

            if w13_weight_scale is not None:
                pad_config_scale = get_w13_padding_config(
                    intermediate_size,
                    reorder_size=1,
                    align=128,
                    outer_block_size=w13_outer_block_size)
                w13_weight_scale = process_w13_for_gmm(tensor=w13_weight_scale,
                                                       concat_dim=3,
                                                       config=pad_config_scale,
                                                       name="w13_weight_scale")
                if w13_outer_block_size > 1:
                    # GMM currently expects scales to be broadcasted to
                    # full shape along the contracting dimension when
                    # skipping requantization.
                    w13_weight_scale = jnp.repeat(w13_weight_scale,
                                                  w13_outer_block_size,
                                                  axis=3)

            if w13_bias is not None:
                w13_bias = process_w13_for_gmm(tensor=w13_bias,
                                               concat_dim=2,
                                               config=pad_config_weight,
                                               name="w13_bias")

            if w2_weight_scale is not None:
                if w2_outer_block_size > 1:
                    # GMM currently expects scales to be broadcasted to
                    # full shape along the contracting dimension when
                    # skipping requantization.
                    w2_weight_scale = jnp.repeat(w2_weight_scale,
                                                 w2_outer_block_size,
                                                 axis=3)

        case MoEBackend.DENSE_MAT:
            # TODO (jacobplatin)
            raise NotImplementedError(
                "process_moe_weights is not yet implemented for dense matmul "
                "backend.")
        case MoEBackend.MEGABLX_GMM:
            # TODO (jacobplatin)
            raise NotImplementedError(
                "process_moe_weights is not yet implemented for megablox gmm "
                "backend")

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
    moe_backend: MoEBackend,
    mesh: Mesh,
) -> FusedMoEWeights:
    match moe_backend:
        case MoEBackend.FUSED_MOE | MoEBackend.GMM_EP:
            ep_sharding = NamedSharding(mesh, P(ShardingAxisName.EXPERT))
            weight_shardings = FusedMoEWeights(
                w13_weight=ep_sharding,
                w13_weight_scale=ep_sharding,
                w13_bias=ep_sharding,
                w2_weight=ep_sharding,
                w2_weight_scale=ep_sharding,
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
            weight_shardings = FusedMoEWeights(
                w13_weight=NamedSharding(
                    mesh,
                    P(None, None, ShardingAxisName.MLP_TENSOR),
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
                    P(None, ShardingAxisName.MLP_TENSOR, None),
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
        case MoEBackend.FUSED_MOE:
            weight_layouts = FusedMoEWeights(
                w13_weight=Layout((0, 1, 2, 3)),
                w13_weight_scale=Layout((0, 1, 2, 3, 4)),
                w13_bias=Layout((0, 1, 2, 3)),
                w2_weight=Layout((0, 1, 2)),
                w2_weight_scale=Layout((0, 1, 2, 3)),
                w2_bias=Layout((0, 1, 2)),
            )
        case MoEBackend.GMM_TP | MoEBackend.GMM_EP:
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
            weight = general_device_put(weight, sharding, layout=layout)
            setattr(weights, key, weight)
    return weights


@jax.jit(static_argnames=(
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

    w13_interleave = activation == "swigluoai"
    w13_reorder_size = get_mesh_shape_product(mesh,
                                              ShardingAxisName.MLP_TENSOR)

    if envs.DISABLE_WEIGHT_REQUANTIZATION:
        logger.info_once("Disabled weight requantization")
        assert not envs.MOE_REQUANTIZE_WEIGHT_DTYPE, (
            "MOE_REQUANTIZE_WEIGHT_DTYPE should not be set when weight "
            "requantization is disabled.")

        assert weight_block_size is not None
        in_block_size = weight_block_size[1]
        if w13_weight_scale is not None and w13_weight_scale.ndim == 3:
            # out_dim = 2 * inter, in_dim = hidden
            # we want (experts, out_blocks, in_blocks)
            # check if it is (experts, in_blocks, out_blocks)
            in_blocks_13 = w13_weight.shape[2] // in_block_size
            if (w13_weight_scale.shape[1] == in_blocks_13
                    and w13_weight_scale.shape[2] != in_blocks_13):
                w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
        if w2_weight_scale is not None and w2_weight_scale.ndim == 3:
            # out_dim = hidden, in_dim = inter
            # we want (experts, out_blocks, in_blocks)
            in_blocks_2 = w2_weight.shape[2] // in_block_size
            if (w2_weight_scale.shape[1] == in_blocks_2
                    and w2_weight_scale.shape[2] != in_blocks_2):
                w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 2)

        # TODO (jacobplatin): add support for bias
        weights = FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w13_bias=None,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            w2_bias=None,
        )

    else:
        if desired_quant_dtype_from_env := envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
            desired_quant_dtype = to_jax_dtype(desired_quant_dtype_from_env)
        else:
            desired_quant_dtype = w13_weight.dtype
            if w13_weight.dtype != w2_weight.dtype:
                raise ValueError(
                    "Expected w13_weight and w2_weight to have the same dtype, "
                    f"but got {w13_weight.dtype} and {w2_weight.dtype}")
        requant_block_size = None
        if requant_block_size_from_env := envs.MOE_REQUANTIZE_BLOCK_SIZE:
            requant_block_size = (int(requant_block_size_from_env)
                                  if requant_block_size_from_env else None)

        moe_logging_str = (
            "[MoE requantization]: re-quantizing MoE weights to "
            f"{desired_quant_dtype}")
        if requant_block_size is not None:
            moe_logging_str += f" with block size {requant_block_size}"
        logger.info_once(moe_logging_str)

        # Dequantize fp8 2d block quantized weights into fp32.
        w13_weight = dequantize_tensor(w13_weight,
                                       w13_weight_scale, (1, 2),
                                       jnp.float32,
                                       block_size=weight_block_size)
        w2_weight = dequantize_tensor(w2_weight,
                                      w2_weight_scale, (1, 2),
                                      jnp.float32,
                                      block_size=weight_block_size)

        weights = quantize_moe_weights(
            FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=None,
                w2_bias=None,
            ),
            desired_quant_dtype,
            requant_block_size,
            w13_interleave=w13_interleave,
        )
    return process_moe_weights(
        weights,
        moe_backend=moe_backend,
        w13_reorder_size=w13_reorder_size,
        w13_interleave=w13_interleave,
        disable_weight_requantization=envs.DISABLE_WEIGHT_REQUANTIZATION,
    )


@jax.jit(static_argnames=('mesh', 'activation', 'moe_backend'))
def process_unquantized_moe_weights(
    *,
    mesh: Mesh,
    moe_backend: MoEBackend,
    activation: MoEActivation,
    w13_weight: jax.Array,
    w13_bias: jax.Array | None,
    w2_weight: jax.Array,
    w2_bias: jax.Array | None,
) -> FusedMoEWeights:
    """Jit'ed version to process unquantized moe weights. See `process_moe_weights` for details.
    """
    if envs.DISABLE_WEIGHT_REQUANTIZATION:
        logger.info_once("Disabled weight requantization")
        assert not envs.MOE_REQUANTIZE_WEIGHT_DTYPE, (
            "MOE_REQUANTIZE_WEIGHT_DTYPE should not be set when weight "
            "requantization is disabled.")
    if desired_quant_dtype_from_env := envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
        desired_quant_dtype = to_jax_dtype(desired_quant_dtype_from_env)
        requant_block_size = None
        if requant_block_size_from_env := envs.MOE_REQUANTIZE_BLOCK_SIZE:
            requant_block_size = (int(requant_block_size_from_env)
                                  if requant_block_size_from_env else None)
        moe_logging_str = (
            "[MoE requantization]: re-quantizing MoE weights to "
            f"{desired_quant_dtype}")
        if requant_block_size is not None:
            moe_logging_str += f" with block size {requant_block_size}"
        logger.info_once(moe_logging_str)

        weights = quantize_moe_weights(
            FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=None,
                w2_bias=None,
            ),
            desired_quant_dtype,
            requant_block_size,
        )
    else:
        weights = FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=None,
            w13_bias=w13_bias,
            w2_weight=w2_weight,
            w2_weight_scale=None,
            w2_bias=w2_bias,
        )

    w13_interleave = activation == MoEActivation.SWIGLUOAI
    w13_reorder_size = get_mesh_shape_product(mesh,
                                              ShardingAxisName.MLP_TENSOR)

    return process_moe_weights(
        weights,
        moe_backend=moe_backend,
        w13_reorder_size=w13_reorder_size,
        w13_interleave=w13_interleave,
        disable_weight_requantization=envs.DISABLE_WEIGHT_REQUANTIZATION,
    )
