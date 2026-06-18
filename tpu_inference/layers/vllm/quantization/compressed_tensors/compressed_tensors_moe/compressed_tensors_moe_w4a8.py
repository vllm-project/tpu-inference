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

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import QuantizationArgs
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import (FusedMoEConfig,
                                                  FusedMoeWeightScaleSupported,
                                                  RoutedExperts)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import \
    CompressedTensorsMoEMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types

import tpu_inference.envs as envs
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_quantized_moe_weights, shard_moe_weights_to_tpu)
from tpu_inference.layers.common.quantization import u32_unpack_i4
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j, to_jax_dtype

logger = init_logger(__name__)


class VllmCompressedTensorsW4A8MoEMethod(CompressedTensorsMoEMethod,
                                         VllmQuantConfig):
    """
    MoE method for int4 weights and 8 bit activations.

    Uses fp8 activations for TPU generations that support fp8 compute and int8
    activations for generations that support int8 compute.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        mesh: jax.sharding.Mesh,
        ep_axis_name: str = "model",
    ):
        super().__init__(moe)

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name, )
        self.wtype = scalar_types.uint4

        self.weight_quant = weight_quant
        self.input_quant = input_quant

        self.group_size = self.weight_quant.group_size
        self.num_bits = self.weight_quant.num_bits
        self.packed_factor = 32 // self.num_bits

        assert self.weight_quant.symmetric, (
            "Only symmetric quantization is supported for W4A8 MoE")
        assert self.weight_quant.actorder != "group"

    @property
    def is_monolithic(self) -> bool:
        """Indicates if the MoE operation is monolithic."""
        return True

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        """
        Initializes the weights and scales for the FusedMoE layer.
        Handles packed int4 weights and grouped/channelwise scales.

        This method differs from the VLLM CompressedTensorsW4A8Fp8MoEMethod's
        create_weights in that it does not require that the hidden_size and
        intermediate_size be divisible by 256 and instead only requires them to
        be divisible by the packed factor.
        https://github.com/vllm-project/vllm/blob/9db4650e5e4c726eb5ae29330cd55e796567469c/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8_fp8.py#L68

        :param layer: The FusedMoE layer to initialize.
        :param num_experts: Total number of experts.
        :param hidden_size: Hidden dimension size.
        :param intermediate_size: Intermediate dimension size.
        :param params_dtype: Data type for parameters like scale and bias.
        :param kwargs: Additional arguments like weight_loader.
        """
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        assert hidden_size % self.packed_factor == 0, (
            f"Hidden size ({hidden_size}) must be divisible by packed factor "
            f"({self.packed_factor}).")
        assert intermediate_size_per_partition % self.packed_factor == 0, (
            f"Intermediate size ({intermediate_size_per_partition}) must be divisible by "
            f"packed factor ({self.packed_factor}).")

        # storage type, pack 8xint4 into int32
        params_dtype = torch.int32

        # WEIGHTS
        w13_weight_packed = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.packed_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight_packed)
        set_weight_attrs(w13_weight_packed, extra_weight_attrs)

        w2_weight_packed = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.packed_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight_packed)
        set_weight_attrs(w2_weight_packed, extra_weight_attrs)

        # SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=layer.orig_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=layer.orig_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-GROUP quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # weight shapes
        w2_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        # don't use input scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Processes and shards MoE weights after loading.

        :param self: The method for the layer responsible for processing the weights.
        :param layer: The source PyTorch layer containing the raw, un-sharded weights from the loaded checkpoint.
        :type layer: torch.nn.Module
        """
        assert isinstance(layer, RoutedExperts)

        # N.B
        # layer.w13_weight: [num_experts, 2*moe_intermediate_size, hidden_size]
        # layer.w13_weight_scale: [num_experts, 2*moe_intermediate_size, 1]
        # layer.w2_weight: [num_experts, hidden_size, moe_intermediate_size]
        # layer.w2_weight_scale: [num_experts, hidden_size, 1]
        # Unpack uint4 weights to int32 (values 0-15)
        # Transfer packed weights directly to TPU
        w13_weight_packed = t2j(layer.w13_weight_packed.view(torch.int32))
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)

        w2_weight_packed = t2j(layer.w2_weight_packed.view(torch.int32))
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
        else:
            w13_bias = w2_bias = None

        # Create FusedMoEWeights with packed weights
        weights = FusedMoEWeights(
            w13_weight=w13_weight_packed,
            w13_weight_scale=w13_weight_scale,
            w13_bias=w13_bias,
            w2_weight=w2_weight_packed,
            w2_weight_scale=w2_weight_scale,
            w2_bias=w2_bias,
        )

        # Shard packed weights to TPU before unpacking to avoid OOM
        weights = shard_moe_weights_to_tpu(weights, self.mesh)

        desired_quant_dtype = to_jax_dtype(
            envs.MOE_REQUANTIZE_WEIGHT_DTYPE
        ) if envs.MOE_REQUANTIZE_WEIGHT_DTYPE else None
        requant_block_size = int(envs.MOE_REQUANTIZE_BLOCK_SIZE
                                 ) if envs.MOE_REQUANTIZE_BLOCK_SIZE else None

        activation_str = "swigluoai" if layer.activation == MoEActivation.SWIGLUOAI else ""

        @jax.jit(static_argnames=("desired_quant_dtype", "requant_block_size"))
        def unpack_and_process(
            weights: FusedMoEWeights,
            desired_quant_dtype: jnp.dtype | None,
            requant_block_size: int | None,
        ) -> FusedMoEWeights:

            w13_unpacked = u32_unpack_i4(weights.w13_weight)
            w2_unpacked = u32_unpack_i4(weights.w2_weight)

            weights_unpacked = FusedMoEWeights(
                w13_weight=w13_unpacked,
                w13_weight_scale=weights.w13_weight_scale,
                w13_bias=weights.w13_bias,
                w2_weight=w2_unpacked,
                w2_weight_scale=weights.w2_weight_scale,
                w2_bias=weights.w2_bias,
            )

            return process_quantized_moe_weights(
                weights=weights_unpacked,
                moe_backend=self.moe_backend,
                mesh=self.mesh,
                activation=activation_str,
                weight_block_size=(1, self.group_size),
                desired_quant_dtype=desired_quant_dtype,
                requant_block_size=requant_block_size,
            )

        # Use the new fast requantization logic with TPU-side unpacking
        weights = unpack_and_process(
            weights,
            desired_quant_dtype,
            requant_block_size,
        )

        weights = torch_view(weights)

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

        # Clean up packed parameters and shape metadata
        if hasattr(layer, "w13_weight_packed"):
            delattr(layer, "w13_weight_packed")
        if hasattr(layer, "w2_weight_packed"):
            delattr(layer, "w2_weight_packed")
        if hasattr(layer, "w13_weight_shape"):
            delattr(layer, "w13_weight_shape")
        if hasattr(layer, "w2_weight_shape"):
            delattr(layer, "w2_weight_shape")

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        # Quantization is handled in the kernel.
        return None

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits,
                              input_ids=input_ids)
