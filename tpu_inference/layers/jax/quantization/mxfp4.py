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

from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, quantize_moe_weights,
    shard_moe_weights)
from tpu_inference.layers.common.quantization import (
    MXFP4_REQUANTIZED_BLOCK_SIZE, dequantize_tensor_from_mxfp4_packed)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import cpu_mesh, cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.moe.moe import JaxMoE, JaxRoutedExperts
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import \
    jax_array_from_reshaped_torch
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

# TODO(#2952): remove once MXFP4 supports all fused MoE backends.
MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS = [
    MoEBackend.GMM_EP, MoEBackend.GMM_TP
]

_GPT_OSS_MXFP4_EXPERT_TENSOR_ATTRS = {
    "gate_up_proj_blocks": "w13_blocks",
    "gate_up_proj_scales": "w13_scales",
    "gate_up_proj_bias": "w13_bias",
    "down_proj_blocks": "w2_blocks",
    "down_proj_scales": "w2_scales",
    "down_proj_bias": "w2_bias",
}


class Mxfp4FusedMoEMethod(QuantizeMethodBase):
    """MXFP4 method for GPT-OSS routed experts."""

    def __init__(self):
        self.extra_backend_kwargs = {}

    def load_weights(self, *, layer: JaxRoutedExperts,
                     original_load_weights_fn,
                     weights: Iterable[tuple[str, torch.Tensor]]) -> set:
        """Stage the six GPT-OSS MXFP4 expert tensors on the layer."""
        # FP8 delegates to `original_load_weights_fn`, but that loader expects
        # per-expert `.<expert_id>.<param>.weight` tensors; GPT-OSS checkpoints
        # ship whole packed (E, ...) tensors, so we load all six ourselves.
        loaded_names = set()
        unexpected = []
        for torch_name, torch_weight in weights:
            suffix = torch_name.rsplit(".", maxsplit=1)[-1]
            attr_name = _GPT_OSS_MXFP4_EXPERT_TENSOR_ATTRS.get(suffix)
            if attr_name is None:
                unexpected.append(torch_name)
                continue
            jax_param = getattr(layer, attr_name, None)
            assert isinstance(jax_param, nnx.Param)
            staging_shape = tuple(jax_param.shape)
            # Checkpoint blocks come in 4-D (E, rows, groups, 16); flatten the
            # group axes back down to the 3-D staging shape. The identity
            # permute is just to stop the helper's default 2-D transpose.
            reshape_dims = (staging_shape
                            if suffix.endswith("_blocks") else None)
            jax_weight = jax_array_from_reshaped_torch(
                torch_weight,
                reshape_dims=reshape_dims,
                permute_dims=tuple(range(len(staging_shape))))
            if tuple(jax_weight.shape) != staging_shape:
                raise ValueError(
                    f"Converted MXFP4 tensor {torch_name} has shape "
                    f"{tuple(jax_weight.shape)}, expected staging shape "
                    f"{staging_shape} for {attr_name}.")
            jax_param.set_raw_value(jax_weight)
            jax_param.set_metadata("_is_loaded", True)
            loaded_names.add(attr_name)

        if unexpected:
            raise ValueError(
                "Mxfp4FusedMoEMethod only handles GPT-OSS MXFP4 expert "
                f"tensors, got unexpected checkpoint tensors: {unexpected}")

        logger.debug(
            f"Loaded {len(loaded_names)} MXFP4 tensors for {layer.prefix} MoE layer."
        )

        return loaded_names

    def create_weights_jax(self, layer: JaxRoutedExperts, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        """
        Create the quant method-specific weights.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to create weights for.
        """
        if layer.moe_backend in MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            E = layer.num_local_experts
            D = layer.hidden_size
            F = layer.intermediate_size_moe

            for param_name in [
                    "kernel_gating_EDF", "kernel_up_proj_EDF",
                    "kernel_down_proj_EFD"
            ]:
                param = getattr(layer, param_name, None)
                assert isinstance(
                    param, nnx.Param
                ), f"Expected nnx.Param for {param_name}, got {type(param)}"
                delattr(layer, param_name)

            for param_name, shape, dtype in [
                ("w13_blocks", (E, 2 * F, D // 2), jnp.uint8),
                ("w13_scales", (E, 2 * F, D // 32), jnp.uint8),
                ("w13_bias", (E, 2 * F), layer.dtype),
                ("w2_blocks", (E, D, F // 2), jnp.uint8),
                ("w2_scales", (E, D, F // 32), jnp.uint8),
                ("w2_bias", (E, D), layer.dtype),
            ]:
                param = nnx.Param(jnp.zeros(shape, dtype=dtype),
                                  eager_sharding=False)
                param.set_metadata('mesh', cpu_mesh())
                setattr(layer, param_name, param)
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

    def process_weights_after_loading(self, layer: JaxRoutedExperts) -> bool:
        """
        Process the staged GPT-OSS MXFP4 expert tensors.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to process.
        """
        if layer.moe_backend in MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            if not all(
                    getattr(layer, name).get_metadata("_is_loaded", False)
                    for name in _GPT_OSS_MXFP4_EXPERT_TENSOR_ATTRS.values()):
                # A module's weights can be split across multiple files, so
                # this can get called more than once. Wait until all of them
                # are loaded before processing.
                return False

            w13_interleave = layer.activation == "swigluoai"
            w13_reorder_size = get_mesh_shape_product(
                layer.mesh, ShardingAxisName.MLP_TENSOR)

            # Keep the large dequantized expert tensors off device until
            # sharding.
            with cpu_mesh_context():
                weights = FusedMoEWeights(
                    w13_weight=dequantize_tensor_from_mxfp4_packed(
                        layer.w13_blocks[...], layer.w13_scales[...], 2,
                        jnp.float32),
                    w13_weight_scale=None,
                    w13_bias=layer.w13_bias[...],
                    w2_weight=dequantize_tensor_from_mxfp4_packed(
                        layer.w2_blocks[...], layer.w2_scales[...], 2,
                        jnp.float32),
                    w2_weight_scale=None,
                    w2_bias=layer.w2_bias[...],
                )
                weights = quantize_moe_weights(weights,
                                               jnp.float4_e2m1fn,
                                               MXFP4_REQUANTIZED_BLOCK_SIZE,
                                               w13_interleave=w13_interleave)
                weights = process_moe_weights(
                    weights,
                    moe_backend=layer.moe_backend,
                    w13_reorder_size=w13_reorder_size,
                    w13_interleave=w13_interleave)

                for name in _GPT_OSS_MXFP4_EXPERT_TENSOR_ATTRS.values():
                    delattr(layer, name)

            weights = shard_moe_weights(weights,
                                        moe_backend=layer.moe_backend,
                                        mesh=layer.mesh)

            layer.kernel_gating_upproj_EDF = nnx.Param(weights.w13_weight)
            layer.kernel_gating_upproj_EDF_weight_scale = nnx.Param(
                weights.w13_weight_scale)
            layer.kernel_gating_upproj_EDF_bias = nnx.Param(weights.w13_bias)
            layer.kernel_down_proj_EFD = nnx.Param(weights.w2_weight)
            layer.kernel_down_proj_EFD_weight_scale = nnx.Param(
                weights.w2_weight_scale)
            layer.kernel_down_proj_EFD_bias = nnx.Param(weights.w2_bias)
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array, *,
                  router_logits: jax.Array) -> jax.Array:
        """
        Run the forward pass of the MoE layer.

        Args:
            layer: The layer to apply the quantization method to.
            x: The input to the layer.

        Returns:
            The MoE output.
        """
        assert isinstance(layer, (JaxMoE, JaxRoutedExperts))

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            jax.sharding.NamedSharding(layer.mesh,
                                       P(*layer.activation_ffw_td)))

        if layer.moe_backend in MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            weights = FusedMoEWeights(
                w13_weight=layer.kernel_gating_upproj_EDF[...],
                w13_weight_scale=layer.kernel_gating_upproj_EDF_weight_scale[
                    ...],
                w13_bias=layer.kernel_gating_upproj_EDF_bias[...],
                w2_weight=layer.kernel_down_proj_EFD[...],
                w2_weight_scale=layer.kernel_down_proj_EFD_weight_scale[...],
                w2_bias=layer.kernel_down_proj_EFD_bias[...],
            )
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


class Mxfp4Config(QuantizationConfig):
    """Quantization config for GPT-OSS MXFP4 routed experts.

    GPT-OSS MXFP4 has no configurable JAX-side options yet.
    """

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxRoutedExperts):
            return Mxfp4FusedMoEMethod()
        return None
