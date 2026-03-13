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
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.tensor import Tensor

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
    weights.w2_weight = w2_weight
    weights.w2_weight_scale = w2_weight_scale

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
        w13_bias=w13_bias,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        w2_bias=w2_bias,
    )


def _get_moe_weight_shardings(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh: Mesh,
) -> FusedMoEWeights:
    """Build sharding specs for MoE weights based on the backend type.

    Returns a FusedMoEWeights where each field is a NamedSharding.
    Used by both shard_moe_weights (for device_put) and
    process_fp8_moe_weights (for sharding constraints inside JIT).
    """
    match moe_backend:
        case MoEBackend.FUSED_MOE | MoEBackend.GMM_EP:
            ep_sharding = NamedSharding(mesh, P(ShardingAxisName.EXPERT))
            return FusedMoEWeights(
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
            return FusedMoEWeights(
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


def shard_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh: Mesh,
) -> FusedMoEWeights:

    weight_shardings = _get_moe_weight_shardings(weights, moe_backend, mesh)

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


def shard_fp8_moe_weights_to_tpu(
    weights: FusedMoEWeights,
    mesh: Mesh,
    source_mesh: Mesh | None = None,
) -> FusedMoEWeights:
    """Shard FP8 MoE weights onto TPU before requantization.

    Transfers FP8 weights from CPU to TPU with expert-dimension sharding
    so that the subsequent dequant/requant in process_fp8_moe_weights runs
    on TPU in parallel across experts. This avoids OOM (no single TPU holds
    the full unsharded weight) and is much faster than CPU requantization.

    For meshes without an EXPERT axis (e.g. GMM_TP), falls back to the
    first mesh axis to distribute experts across devices.

    Args:
        weights: FP8 MoE weights (typically on CPU).
        mesh: The TPU device mesh for inference.
        source_mesh: The mesh the weights currently reside on (e.g.
            cpu_mesh()). None when weights are plain CPU arrays.

    Returns:
        FusedMoEWeights sharded across TPU devices.
    """
    expert_axis = ShardingAxisName.EXPERT
    if isinstance(expert_axis, str):
        if expert_axis in mesh.axis_names:
            shard_axis = expert_axis
        else:
            shard_axis = mesh.axis_names[0]
    else:
        if all(a in mesh.axis_names for a in expert_axis):
            shard_axis = expert_axis
        else:
            shard_axis = mesh.axis_names[0]
    ep_sharding = NamedSharding(mesh, P(shard_axis))

    result_fields = {}
    for field in fields(FusedMoEWeights):
        key = field.name
        weight = getattr(weights, key)
        if weight is not None:
            result_fields[key] = general_device_put(weight,
                                                    ep_sharding,
                                                    source_mesh=source_mesh)
        else:
            result_fields[key] = None
    return FusedMoEWeights(**result_fields)


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

    if desired_quant_dtype_from_env := envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
        desired_quant_dtype = to_jax_dtype(desired_quant_dtype_from_env)
    else:
        desired_quant_dtype = w13_weight.dtype
        if w13_weight.dtype != w2_weight.dtype:
            raise ValueError(
                f"Expected w13_weight and w2_weight to have the same dtype, but got {w13_weight.dtype} and {w2_weight.dtype}"
            )
    requant_block_size = None
    if requant_block_size_from_env := envs.MOE_REQUANTIZE_BLOCK_SIZE:
        requant_block_size = (int(requant_block_size_from_env)
                              if requant_block_size_from_env else None)

    moe_logging_str = f"[MoE requantization]: re-quantizing MoE weights to {desired_quant_dtype}"
    if requant_block_size is not None:
        moe_logging_str += f" with block size {requant_block_size}"
    logger.info(moe_logging_str)

    w13_interleave = activation == "swigluoai"
    w13_reorder_size = get_mesh_shape_product(mesh,
                                              ShardingAxisName.MLP_TENSOR)

    if not envs.MOE_REQUANTIZE_ON_TPU:
        # Default path: direct dequant → quantize → process (matches main branch).
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
        )
        return process_moe_weights(
            weights,
            moe_backend=moe_backend,
            w13_reorder_size=w13_reorder_size,
            w13_interleave=w13_interleave,
        )

    # TPU path: shard_map + lax.scan for lower XLA reservation.

    # Pre-compute pad widths and block sizes for requantization.
    _, orig_hidden_size, orig_intermediate_size = w2_weight.shape
    if requant_block_size is None:
        w13_block_size = w13_weight.shape[-1]
        w2_block_size = w2_weight.shape[-1]
    else:
        w13_block_size = w2_block_size = requant_block_size
    hidden_size = align_to(orig_hidden_size, w13_block_size)
    intermediate_size = align_to(orig_intermediate_size, w2_block_size)
    w13_pad = ((0, 2 * (intermediate_size - orig_intermediate_size)),
               (0, hidden_size - orig_hidden_size))
    w2_pad = ((0, hidden_size - orig_hidden_size),
              (0, intermediate_size - orig_intermediate_size))

    # Determine which mesh axis the expert dim is sharded across.
    expert_axis = ShardingAxisName.EXPERT
    if isinstance(expert_axis, str):
        if expert_axis in mesh.axis_names:
            shard_axis = expert_axis
        else:
            shard_axis = mesh.axis_names[0]
    else:
        if all(a in mesh.axis_names for a in expert_axis):
            shard_axis = expert_axis
        else:
            shard_axis = mesh.axis_names[0]

    scan_batch_size = 1
    w13_pad_3d = ((0, 0), ) + w13_pad
    w2_pad_3d = ((0, 0), ) + w2_pad

    expert_p = P(shard_axis)

    def _requant_and_process_local(w13, w13_scale, w2, w2_scale):
        """Per-device requant + process. Shapes are local [local_experts, ...]."""
        n_local = w13.shape[0]
        n_batches = n_local // scan_batch_size

        def _requant_expert_batch(carry, batch_inputs):
            w13_b, w13_s_b, w2_b, w2_s_b = batch_inputs
            w13_fp32 = dequantize_tensor(w13_b,
                                         w13_s_b, (1, 2),
                                         jnp.float32,
                                         block_size=weight_block_size)
            w2_fp32 = dequantize_tensor(w2_b,
                                        w2_s_b, (1, 2),
                                        jnp.float32,
                                        block_size=weight_block_size)
            w13_fp32 = jnp.pad(w13_fp32, w13_pad_3d)
            w2_fp32 = jnp.pad(w2_fp32, w2_pad_3d)
            w13_q, w13_s_new = quantize_tensor(desired_quant_dtype, w13_fp32,
                                               2, w13_block_size)
            w2_q, w2_s_new = quantize_tensor(desired_quant_dtype, w2_fp32, 2,
                                             w2_block_size)
            return carry, (w13_q, w13_s_new, w2_q, w2_s_new)

        def _reshape_to_batches(x):
            return x.reshape(n_batches, scan_batch_size, *x.shape[1:])

        def _reshape_from_batches(x):
            return x.reshape(n_local, *x.shape[2:])

        xs = jax.tree.map(_reshape_to_batches, (w13, w13_scale, w2, w2_scale))
        _, (w13_q, w13_s, w2_q, w2_s) = jax.lax.scan(_requant_expert_batch,
                                                     init=None,
                                                     xs=xs)
        w13_q, w13_s, w2_q, w2_s = jax.tree.map(_reshape_from_batches,
                                                (w13_q, w13_s, w2_q, w2_s))

        out = process_moe_weights(
            FusedMoEWeights(
                w13_weight=w13_q,
                w13_weight_scale=w13_s,
                w13_bias=None,
                w2_weight=w2_q,
                w2_weight_scale=w2_s,
                w2_bias=None,
            ),
            moe_backend=moe_backend,
            w13_reorder_size=w13_reorder_size,
            w13_interleave=w13_interleave,
        )
        return (out.w13_weight, out.w13_weight_scale, out.w2_weight,
                out.w2_weight_scale)

    w13_q, w13_s, w2_q, w2_s = shard_map(
        _requant_and_process_local,
        mesh=mesh,
        in_specs=(expert_p, expert_p, expert_p, expert_p),
        out_specs=(expert_p, expert_p, expert_p, expert_p),
        check_rep=False,
    )(w13_weight, w13_weight_scale, w2_weight, w2_weight_scale)
    jax.block_until_ready((w13_q, w13_s, w2_q, w2_s))

    out = FusedMoEWeights(
        w13_weight=w13_q,
        w13_weight_scale=w13_s,
        w13_bias=None,
        w2_weight=w2_q,
        w2_weight_scale=w2_s,
        w2_bias=None,
    )

    # Apply sharding constraints so the JIT output matches what
    # shard_moe_weights expects.
    target_shardings = _get_moe_weight_shardings(out, moe_backend, mesh)
    for field in fields(FusedMoEWeights):
        key = field.name
        weight = getattr(out, key)
        if weight is not None:
            sharding = getattr(target_shardings, key)
            setattr(out, key,
                    jax.lax.with_sharding_constraint(weight, sharding))
    return out
