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
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.layout import Layout, with_layout_constraint
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


@dataclass
class W13PaddingConfig:
    intermediate_size: int
    w13_reorder_size: int
    local_intermediate_size: int
    pad_amount: int
    padded_intermediate_size: int


def get_w13_padding_config(intermediate_size: int,
                           reorder_size: int,
                           align: int = 128) -> W13PaddingConfig:
    """Calculates padded dimensions and pad amounts for w13 tensors."""
    local_intermediate_size = intermediate_size // reorder_size

    padded_local_intermediate_size = align_to(local_intermediate_size, align)
    padded_intermediate_size = padded_local_intermediate_size * reorder_size
    pad_amount = padded_local_intermediate_size - local_intermediate_size

    return W13PaddingConfig(
        intermediate_size=intermediate_size,
        w13_reorder_size=reorder_size,
        local_intermediate_size=local_intermediate_size,
        pad_amount=pad_amount,
        padded_intermediate_size=padded_intermediate_size,
    )


def process_w13_for_gmm(tensor,
                        concat_dim: int,
                        config: W13PaddingConfig,
                        padded_output_sizes: list[int] | None = None,
                        name: str = "w13"):
    """helper to split, pad, concatenate, and reorder w13 tensors."""

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

    logger.debug(f"{name}_w1 shape after padding: {padded_w1.shape}")
    logger.debug(f"{name}_w3 shape after padding: {padded_w3.shape}")

    # 3. Concatenate and Reorder for avoiding TP sharding comms
    w13_concat = jnp.concatenate([padded_w1, padded_w3], axis=concat_dim)
    if padded_output_sizes is not None:
        return reorder_concatenated_tensor_for_sharding(
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
    compact_scale_fields: tuple[str, ...] = (),
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
        compact_scale_fields: field names (e.g. "w2_weight_scale") whose
            scale should stay 3D (E, K_blocks, N_blocks) rather than being
            expand_dims'd to 4D. Used by the direct FP8 path to feed the
            kernel's per-block scalar lookup (scale_n_block_size mode),
            saving the 128× host-side memory blow-up.

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

    # Workaround for JAX error "must have valid byte strides". Keep it for the
    # fused-MoE path, but skip it for GMM_EP because that path already ends up
    # with EP-only sharding and this extra layout constraint can trigger a huge
    # TP all-gather during lowering.
    if moe_backend != MoEBackend.GMM_EP:
        w13_weight = with_layout_constraint(w13_weight, Layout((0, 1, 2)))
        w2_weight = with_layout_constraint(w2_weight, Layout((0, 1, 2)))

    # Scale layout: swap (N_blocks↔K_blocks) so kernel's b_id iterates over
    # K_blocks (contracting dim). By default expand_dims(axis=2) to get 4D
    # (E, K_blocks, 1, N_full) expected by legacy per-element-N kernel path.
    # Fields in `compact_scale_fields` keep the 3D (E, K_blocks, N_blocks)
    # layout — kernel's gmm_v2 auto-enables scale_n_block_size mode and does
    # per-block scalar lookup. See process_fp8_moe_weights_direct.
    if w13_weight_scale is not None:
        w13_weight_scale = w13_weight_scale.astype(jnp.float32)
        w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
        if "w13_weight_scale" not in compact_scale_fields:
            w13_weight_scale = jnp.expand_dims(w13_weight_scale, 2)
    if w2_weight_scale is not None:
        w2_weight_scale = w2_weight_scale.astype(jnp.float32)
        w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 2)
        if "w2_weight_scale" not in compact_scale_fields:
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

            pad_config = get_w13_padding_config(intermediate_size,
                                                w13_reorder_size,
                                                align=128)

            padded_output_sizes = [
                pad_config.padded_intermediate_size,
                pad_config.padded_intermediate_size
            ]

            process_w13_tp = partial(process_w13_for_gmm,
                                     config=pad_config,
                                     padded_output_sizes=padded_output_sizes)

            w13_weight = process_w13_tp(tensor=w13_weight,
                                        concat_dim=2,
                                        name="w13_weight")

            if w13_weight_scale is not None:
                w13_weight_scale = process_w13_tp(tensor=w13_weight_scale,
                                                  concat_dim=3,
                                                  name="w13_weight_scale")

            if w13_bias is not None:
                w13_bias = process_w13_tp(tensor=w13_bias,
                                          concat_dim=2,
                                          name="w13_bias")

        case MoEBackend.GMM_EP:
            pad_config = get_w13_padding_config(intermediate_size,
                                                reorder_size=1,
                                                align=128)

            process_w13_ep = partial(process_w13_for_gmm, config=pad_config)

            w13_weight = process_w13_ep(tensor=w13_weight,
                                        concat_dim=2,
                                        name="w13_weight")

            if w13_weight_scale is not None:
                w13_weight_scale = process_w13_ep(tensor=w13_weight_scale,
                                                  concat_dim=3,
                                                  name="w13_weight_scale")

            if w13_bias is not None:
                w13_bias = process_w13_ep(tensor=w13_bias,
                                          concat_dim=2,
                                          name="w13_bias")

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


def shard_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh: Mesh,
    num_experts_global: int | None = None,
) -> FusedMoEWeights:
    # Only set by GMM_TP when we expand w2_weight_scale via jnp.repeat, so
    # the main loop can override axis-1 of its global_shape.
    _w2_ws_global_axis1 = None
    match moe_backend:
        case MoEBackend.FUSED_MOE:
            ep_sharding = NamedSharding(mesh, P(ShardingAxisName.EXPERT))
            weight_shardings = FusedMoEWeights(
                w13_weight=ep_sharding,
                w13_weight_scale=ep_sharding,
                w13_bias=ep_sharding,
                w2_weight=ep_sharding,
                w2_weight_scale=ep_sharding,
                w2_bias=ep_sharding,
            )
        case MoEBackend.GMM_EP:
            # For the 8EP4TP NEW_MODEL_DESIGN bring-up, shard MoE weights on
            # the full EXPERT axis tuple so the 32-way mesh placement matches
            # the intended 8-way EP x 4-way TP layout.
            gmm_ep_sharding = NamedSharding(mesh, P(ShardingAxisName.EXPERT))
            weight_shardings = FusedMoEWeights(
                w13_weight=gmm_ep_sharding,
                w13_weight_scale=gmm_ep_sharding,
                w13_bias=gmm_ep_sharding,
                w2_weight=gmm_ep_sharding,
                w2_weight_scale=gmm_ep_sharding,
                w2_bias=gmm_ep_sharding,
            )
        case MoEBackend.GMM_TP:
            # w2_weight_scale must align with w2_weight's TP sharding along
            # the intermediate (contracting) axis. For GLM-5.1 the raw scale
            # has only 16 K-blocks while TP=32, so we repeat axis-1 up to 32
            # *after* the compact-scale optimization, then shard it. Each
            # pair of adjacent chips thus owns the same (small) block — the
            # kernel reads scale[b_id=0, n_block_idx] as a scalar and applies
            # it to that chip's 64 local intermediate elements (half of a
            # 128-element global block; correct per block-quantization
            # semantics, since all elements in a block share one scale).
            _mlp_tensor_size = get_mesh_shape_product(
                mesh, list(ShardingAxisName.MLP_TENSOR))
            _w2_ws = weights.w2_weight_scale
            if _w2_ws is None:
                w2_weight_scale_p_spec = P(None, ShardingAxisName.MLP_TENSOR)
            else:
                _process_count = jax.process_count()
                _local_axis1 = _w2_ws.shape[1]
                _global_axis1 = _local_axis1 * _process_count
                if _global_axis1 % _mlp_tensor_size == 0:
                    w2_weight_scale_p_spec = P(
                        None, ShardingAxisName.MLP_TENSOR)
                elif _mlp_tensor_size % _global_axis1 == 0:
                    _rf = _mlp_tensor_size // _global_axis1
                    _w2_ws_global_axis1 = _global_axis1 * _rf
                    weights.w2_weight_scale = jnp.repeat(_w2_ws, _rf, axis=1)
                    _w2_ws = None  # free pre-repeat buffer
                    w2_weight_scale_p_spec = P(
                        None, ShardingAxisName.MLP_TENSOR)
                else:
                    w2_weight_scale_p_spec = P()
                logger.debug(
                    "[shard_moe_weights GMM_TP] w2_weight_scale ndim=%d "
                    "shape=%s global_axis1=%d mlp=%d → %s",
                    weights.w2_weight_scale.ndim,
                    weights.w2_weight_scale.shape,
                    _w2_ws_global_axis1 or _global_axis1,
                    _mlp_tensor_size,
                    "shard axis-1" if w2_weight_scale_p_spec != P()
                    else "replicate")
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
        case MoEBackend.GMM_TP:
            weight_layouts = FusedMoEWeights(
                w13_weight=Layout((0, 1, 2)),
                w13_weight_scale=Layout((0, 1, 2, 3)),
                w13_bias=Layout((0, 1, 2)),
                w2_weight=Layout((0, 1, 2)),
                w2_weight_scale=Layout((0, 1, 2, 3)),
                w2_bias=Layout((0, 1, 2)),
            )
        case MoEBackend.GMM_EP:
            # Skip Layout constraints for GMM_EP to avoid the ~1.5GB XLA
            # compilation/transposition temp buffer that jax.device_put(Format(...))
            # requires. With large models (e.g., GLM-5.1-FP8 744B) that fill nearly
            # all HBM with weights, this temp buffer causes RESOURCE_EXHAUSTED.
            # Weights are stored in natural C-order; the GMM kernel still produces
            # correct results, just without the Fortran-order cache optimization.
            weight_layouts = FusedMoEWeights(
                w13_weight=None,
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=None,
                w2_weight_scale=None,
                w2_bias=None,
            )

    for field in fields(FusedMoEWeights):
        key = field.name
        if (weight := getattr(weights, key, None)) is not None:
            layout = getattr(weight_layouts, key)
            # Compact 3D scale (kernel 2D block mode) — static layout tuples
            # above are 4D; drop the constraint so JAX picks a compatible
            # layout for ndim=3. Covers the direct-FP8 path for w2 (and in
            # the future w13) where scale is kept as (E, K_blocks, N_blocks).
            if key in ("w13_weight_scale", "w2_weight_scale") and \
                    weight.ndim == 3:
                layout = None
            sharding = getattr(weight_shardings, key)
            # make_array_from_process_local_data wants global shape. axis-0
            # is always experts (provided explicitly via num_experts_global);
            # axis-1 needs override only when shard_moe_weights already did
            # a host-side repeat that blew up the local axis to
            # `_w2_ws_global_axis1` per host × process_count globally.
            g_shape = None
            if num_experts_global is not None:
                g_shape = (num_experts_global,) + weight.shape[1:]
            if (moe_backend == MoEBackend.GMM_TP
                    and key == "w2_weight_scale"
                    and _w2_ws_global_axis1 is not None
                    and g_shape is not None):
                g_shape = (g_shape[0], _w2_ws_global_axis1) + g_shape[2:]
            logger.debug(
                "[shard_moe_weights] field=%s weight.shape=%s g_shape=%s",
                key, getattr(weight, "shape", "?"), g_shape)
            weight = general_device_put(
                weight, sharding, layout=layout, global_shape=g_shape)
            setattr(weights, key, weight)
    # Force host→device DMAs to complete before returning so the CPU source
    # buffers (t2j'd FP8 tensors, repeated scales, etc) are no longer pinned
    # by async transfers. Without this, JAX holds the CPU buffers for each
    # leaf until the DMA completes across all processes — host RAM accumulates
    # across the 75 MoE layers and trips the Ray 95% OOM kill threshold.
    jax.block_until_ready(weights)
    return weights


def _slice_fused_moe_weights(
    weights: FusedMoEWeights,
    expert_slice: slice,
) -> FusedMoEWeights:
    sliced_weights = {}
    for field in fields(FusedMoEWeights):
        value = getattr(weights, field.name)
        sliced_weights[field.name] = None if value is None else value[
            expert_slice]
    return FusedMoEWeights(**sliced_weights)


def _concat_fused_moe_weight_chunks(
    chunks: list[FusedMoEWeights],
) -> FusedMoEWeights:
    if not chunks:
        raise ValueError("Expected at least one MoE weight chunk")

    concatenated_weights = {}
    for field in fields(FusedMoEWeights):
        value = getattr(chunks[0], field.name)
        if value is None:
            concatenated_weights[field.name] = None
            continue

        concatenated_weights[field.name] = jnp.concatenate(
            [getattr(chunk, field.name) for chunk in chunks], axis=0)

    return FusedMoEWeights(**concatenated_weights)


@partial(
    jax.jit,
    static_argnames=(
        "moe_backend",
        "mesh",
        "activation",
        "desired_quant_dtype",
        "requant_block_size",
        "weight_block_size",
    ),
)
def _process_fp8_moe_weight_chunk(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh: Mesh,
    activation: str,
    desired_quant_dtype: jnp.dtype,
    requant_block_size: int | None,
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


def process_fp8_moe_weights(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh: Mesh,
    activation: str,
    weight_block_size: tuple[int, ...] | None = None,
) -> FusedMoEWeights:
    if desired_quant_dtype_from_env := envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
        desired_quant_dtype = to_jax_dtype(desired_quant_dtype_from_env)
    else:
        desired_quant_dtype = weights.w13_weight.dtype
        if weights.w13_weight.dtype != weights.w2_weight.dtype:
            raise ValueError(
                "Expected w13_weight and w2_weight to have the same dtype, "
                f"but got {weights.w13_weight.dtype} and {weights.w2_weight.dtype}"
            )

    requant_block_size = None
    if requant_block_size_from_env := envs.MOE_REQUANTIZE_BLOCK_SIZE:
        requant_block_size = (int(requant_block_size_from_env)
                              if requant_block_size_from_env else None)

    expert_chunk_size = envs.MOE_REQUANTIZE_EXPERT_CHUNK_SIZE
    if (expert_chunk_size is not None and expert_chunk_size <= 0):
        expert_chunk_size = None

    moe_logging_str = (
        f"[MoE requantization]: re-quantizing MoE weights to {desired_quant_dtype}"
    )
    if requant_block_size is not None:
        moe_logging_str += f" with block size {requant_block_size}"
    if expert_chunk_size is not None:
        moe_logging_str += f" using expert chunk size {expert_chunk_size}"
    logger.info(moe_logging_str)

    num_experts = weights.w13_weight.shape[0]
    if expert_chunk_size is None or expert_chunk_size >= num_experts:
        return _process_fp8_moe_weight_chunk(
            weights,
            moe_backend=moe_backend,
            mesh=mesh,
            activation=activation,
            desired_quant_dtype=desired_quant_dtype,
            requant_block_size=requant_block_size,
            weight_block_size=weight_block_size,
        )

    # Chunk the experts in *equal* slabs so every call to the jitted
    # `_process_fp8_moe_weight_chunk` sees the same input shapes and reuses
    # the same compiled artifact. A tail chunk of a different size would
    # trigger a second JIT compilation for only one chunk and undo the
    # cost amortization this helper exists for.
    if num_experts % expert_chunk_size != 0:
        # Round up to the nearest chunk size that divides num_experts evenly.
        for candidate in range(expert_chunk_size, num_experts + 1):
            if num_experts % candidate == 0:
                expert_chunk_size = candidate
                break
        else:
            expert_chunk_size = num_experts

    processed_chunks = []
    for expert_start in range(0, num_experts, expert_chunk_size):
        expert_end = expert_start + expert_chunk_size
        chunk = _slice_fused_moe_weights(weights, slice(expert_start,
                                                        expert_end))
        processed_chunks.append(
            _process_fp8_moe_weight_chunk(
                chunk,
                moe_backend=moe_backend,
                mesh=mesh,
                activation=activation,
                desired_quant_dtype=desired_quant_dtype,
                requant_block_size=requant_block_size,
                weight_block_size=weight_block_size,
            ))

    return _concat_fused_moe_weight_chunks(processed_chunks)


def expand_2d_block_scale(
    scale: jax.Array,
    block_size_n: int,
) -> jax.Array:
    """Expand 2D block scale's N-block dimension to full N resolution.

    Checkpoint 2D block scale: (E, N_blocks, K_blocks)
    After expansion: (E, N_full, K_blocks) where N_full = N_blocks * block_size_n

    process_moe_weights will then do swapaxes(1,2) + expand_dims(2)
    to get the kernel-expected shape (E, K_blocks, 1, N_full).

    Args:
        scale: 2D block scale with shape (E, N_blocks, K_blocks).
        block_size_n: The block size along the N dimension (typically 128).

    Returns:
        Expanded scale with shape (E, N_full, K_blocks).
    """
    return jnp.repeat(scale, block_size_n, axis=1)


def process_fp8_moe_weights_direct(
    weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh: Mesh,
    activation: str,
    weight_block_size: tuple[int, ...],
) -> FusedMoEWeights:
    """Skip dequant/requant — directly do shape transforms on FP8 weights.

    This is mathematically equivalent to process_fp8_moe_weights but
    avoids the FP32 intermediate representation and JIT compilation,
    resulting in ~30x faster startup for large MoE models.

    The key insight: checkpoint uses 2D block-quantized scale
    (E, N_blocks, K_blocks), while the GMM kernel expects
    (E, K_blocks, 1, N_full). We can expand the scale using jnp.repeat
    instead of full dequant→requant cycle.

    Args:
        weights: FP8 MoE weights with 2D block-quantized scales.
        moe_backend: The MoE backend to use.
        mesh: The JAX mesh for sharding.
        activation: Activation function name.
        weight_block_size: The 2D block size (block_n, block_k).

    Returns:
        Processed FP8 MoE weights with expanded scales.
    """
    block_size_n = weight_block_size[0]  # 128

    # w13: keep the legacy expansion. GMM_TP's `process_w13_for_gmm` does
    # split+reorder on axis-3 which requires the expanded (E, N_full, K_blocks)
    # layout.
    w13_scale = expand_2d_block_scale(weights.w13_weight_scale, block_size_n)
    # Keep legacy 4D expanded w2 scale for all backends.  Compact 3D was tried
    # for GMM_EP but Mosaic's DMA slicer requires the last dim of the scale
    # memref be divisible by the tile size (128), which fails on GLM-5.1 where
    # N_blocks=hidden/128=48.  The CPU-RAM motivation for the compact path is
    # already handled upstream by `tpu_streaming_loader` + the EP weight
    # filter, which gives each host only 1/ep_size of the experts so the 128×
    # host-side expand cost is bounded (~120 MB per host for 75 layers).
    w2_scale = expand_2d_block_scale(weights.w2_weight_scale, block_size_n)

    w13_interleave = activation == "swigluoai"
    w13_reorder_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)

    logger.debug(
        "[MoE direct FP8]: w13 expanded scale shape=%s, w2 expanded scale "
        "shape=%s (block_size_n=%d)",
        None if w13_scale is None else tuple(w13_scale.shape),
        None if w2_scale is None else tuple(w2_scale.shape),
        block_size_n,
    )

    return process_moe_weights(
        FusedMoEWeights(
            w13_weight=weights.w13_weight,     # FP8 unchanged
            w13_weight_scale=w13_scale,         # expanded 4D
            w13_bias=None,
            w2_weight=weights.w2_weight,        # FP8 unchanged
            w2_weight_scale=w2_scale,           # expanded 4D
            w2_bias=None,
        ),
        moe_backend=moe_backend,
        w13_reorder_size=w13_reorder_size,
        w13_interleave=w13_interleave,
    )
