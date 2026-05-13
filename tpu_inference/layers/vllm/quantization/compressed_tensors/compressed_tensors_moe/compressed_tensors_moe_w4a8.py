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

from dataclasses import fields

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import QuantizationArgs
from jax.sharding import PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import \
    CompressedTensorsMoEMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types

P = PartitionSpec

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, _get_moe_weight_shardings, process_moe_weights,
    shard_fp8_moe_weights_to_tpu)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, t2j

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
        assert isinstance(layer, FusedMoE)

        # N.B
        # layer.w13_weight: [num_experts, 2*moe_intermediate_size, hidden_size]
        # layer.w13_weight_scale: [num_experts, 2*moe_intermediate_size, 1]
        # layer.w2_weight: [num_experts, hidden_size, moe_intermediate_size]
        # layer.w2_weight_scale: [num_experts, hidden_size, 1]
        w13_weight_packed = t2j(layer.w13_weight_packed, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w2_weight_packed = t2j(layer.w2_weight_packed, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
        else:
            w13_bias = w2_bias = None

        # Pack packed weights into FusedMoEWeights for sharding
        input_weights = FusedMoEWeights(
            w13_weight=w13_weight_packed,
            w13_weight_scale=w13_weight_scale,
            w13_bias=w13_bias,
            w2_weight=w2_weight_packed,
            w2_weight_scale=w2_weight_scale,
            w2_bias=w2_bias,
        )

        # Shard packed weights to TPU before processing to avoid OOM
        input_weights = shard_fp8_moe_weights_to_tpu(input_weights, self.mesh)

        # Determine shard axis for shard_map
        expert_axis = ShardingAxisName.EXPERT
        if isinstance(expert_axis, str):
            if expert_axis in self.mesh.axis_names:
                shard_axis = expert_axis
            else:
                shard_axis = self.mesh.axis_names[0]
        else:
            if all(a in self.mesh.axis_names for a in expert_axis):
                shard_axis = expert_axis
            else:
                shard_axis = self.mesh.axis_names[0]

        expert_p = P(shard_axis)
        bias_p = expert_p if self.moe.has_bias else None

        def _process_local(w13_p, w13_s, w13_b, w2_p, w2_s, w2_b):
            n_local = w13_p.shape[0]
            scan_batch_size = 1  # Process 1 expert at a time to save memory
            n_batches = n_local // scan_batch_size

            def _process_expert_batch(carry, batch_inputs):
                w13_p_b, w13_s_b, w13_b_b, w2_p_b, w2_s_b, w2_b_b = batch_inputs

                # Unpack on TPU
                w13_uint4 = jax.lax.bitcast_convert_type(w13_p_b, jnp.uint4)
                w13_uint4 = w13_uint4.reshape(w13_uint4.shape[:-2] + (-1, ))

                w2_uint4 = jax.lax.bitcast_convert_type(w2_p_b, jnp.uint4)
                w2_uint4 = w2_uint4.reshape(w2_uint4.shape[:-2] + (-1, ))

                w13_int4 = (w13_uint4 - 8).astype(jnp.int4)
                w2_int4 = (w2_uint4 - 8).astype(jnp.int4)

                out = process_moe_weights(
                    weights=FusedMoEWeights(
                        w13_weight=w13_int4,
                        w13_weight_scale=w13_s_b,
                        w13_bias=w13_b_b,
                        w2_weight=w2_int4,
                        w2_weight_scale=w2_s_b,
                        w2_bias=w2_b_b,
                    ),
                    moe_backend=self.moe_backend,
                    w13_reorder_size=w13_reorder_size,
                    w13_interleave=w13_interleave,
                )
                return carry, (out.w13_weight, out.w13_weight_scale,
                               out.w13_bias, out.w2_weight,
                               out.w2_weight_scale, out.w2_bias)

            def _reshape_to_batches(x):
                if x is None:
                    return None
                return x.reshape(n_batches, scan_batch_size, *x.shape[1:])

            def _reshape_from_batches(x):
                if x is None:
                    return None
                return x.reshape(n_local, *x.shape[2:])

            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            xs = jax.tree.map(_reshape_to_batches,
                              (w13_p, w13_s, w13_b, w2_p, w2_s, w2_b))
            _, (w13_q, w13_s, w13_b, w2_q, w2_s,
                w2_b) = jax.lax.scan(_process_expert_batch, init=None, xs=xs)

            w13_q, w13_s, w13_b, w2_q, w2_s, w2_b = jax.tree.map(
                _reshape_from_batches, (w13_q, w13_s, w13_b, w2_q, w2_s, w2_b))

            return (w13_q, w13_s, w13_b, w2_q, w2_s, w2_b)

        # Apply shard_map
        w13_q, w13_s, w13_b, w2_q, w2_s, w2_b = jax.shard_map(
            _process_local,
            mesh=self.mesh,
            in_specs=(expert_p, expert_p, bias_p, expert_p, expert_p, bias_p),
            out_specs=(expert_p, expert_p, bias_p, expert_p, expert_p, bias_p),
            check_vma=False,
        )(input_weights.w13_weight, input_weights.w13_weight_scale,
          input_weights.w13_bias, input_weights.w2_weight,
          input_weights.w2_weight_scale, input_weights.w2_bias)

        out = FusedMoEWeights(
            w13_weight=w13_q,
            w13_weight_scale=w13_s,
            w13_bias=w13_b,
            w2_weight=w2_q,
            w2_weight_scale=w2_s,
            w2_bias=w2_b,
        )

        # Apply sharding constraints
        target_shardings = _get_moe_weight_shardings(out, self.moe_backend,
                                                     self.mesh)
        for field in fields(FusedMoEWeights):
            key = field.name
            weight = getattr(out, key)
            if weight is not None:
                sharding = getattr(target_shardings, key)
                setattr(out, key,
                        jax.lax.with_sharding_constraint(weight, sharding))

        weights = torch_view(out)

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
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight).astype(jnp.int4),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight).astype(jnp.int4),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
