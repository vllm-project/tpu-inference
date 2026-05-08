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
"""NVFP4 (ModelOpt FP4) quantization support for TPU.

This module provides the TPU-specific implementation for loading and executing
NVIDIA's ModelOpt FP4 (NVFP4) quantized checkpoints in vLLM.

Important notes:
 - Weight Processing: During `process_weights_after_loading`, we:
   - Unpack the 8-bit packed into unpacked FP4 weights
   - Pre-fuse the FP8 block scales and the FP32 global scale into a single,
     unified FP32 1D-Blockwise scale.

 - Execution:
   - Linear (via sharded_quantized_matmul) and MoE (via GMM_v2) will dequantize weights
     on-the-fly.
"""

from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config, ModelOptNvFp4FusedMoE, ModelOptNvFp4LinearMethod)
from vllm.model_executor.utils import set_weight_attrs

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import NVFP4
from tpu_inference.layers.common.quantization import u8_unpack_e2m1
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, t2j

logger = init_logger(__name__)


@register_quantization_config(NVFP4)
class VllmNvfp4Config(ModelOptNvFp4Config, VllmQuantConfig):
    """NVFP4 config for TPU. Inherits config parsing from upstream."""

    @classmethod
    def get_name(cls):
        return NVFP4

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[Union[QuantizeMethodBase]]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if self.is_layer_excluded(prefix):
                return VllmUnquantizedLinearMethod(linear_config)
            return VllmNvfp4LinearMethod(self, linear_config)
        elif isinstance(layer, FusedMoE):
            if self.is_layer_excluded(prefix):
                return VllmUnquantizedFusedMoEMethod(layer.moe_config)
            layer.moe_config = self.get_moe_config(layer)
            return VllmNvfp4MoEMethod(self, layer, self.mesh)
        elif isinstance(layer, Attention):
            logger.warning_once(
                "NVFP4 attention quantization is not implemented. "
                "Skipping quantization for this layer.")
        return None


class VllmNvfp4LinearMethod(VllmUnquantizedLinearMethod):
    """NVFP4 linear for TPU.

    Reuses upstream ModelOptNvFp4LinearMethod.create_weights for parameter
    registration. process_weights_after_loading unpacks FP4 and computes
    block scales. apply routes to vllm_linear_apply for OTF dequantization.
    """

    def __init__(self, quant_config: 'VllmNvfp4Config',
                 linear_config: VllmQuantLinearConfig):
        VllmUnquantizedLinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        ModelOptNvFp4LinearMethod.create_weights(
            self, layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, params_dtype, **extra_weight_attrs)

        def scalar_weight_loader(param, loaded_weight, *args, **kwargs):
            assert loaded_weight.numel() == 1
            value = loaded_weight.item()
            param.data.fill_(value)

        for name in ("input_scale", "weight_scale_2"):
            if hasattr(layer, name):
                getattr(layer, name).weight_loader = scalar_weight_loader

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, LinearBase)
        weight_packed = t2j(layer.weight, use_dlpack=False)
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        # vLLM allocates this based on the output sizes, so it's the same scalar broadcasted (potentially) multiple times,
        # so we can just grab one, see here:
        # https://github.com/vllm-project/vllm/blob/10ebb40d62e024116c1a4473f8e357a3e72761ed/vllm/model_executor/layers/quantization/modelopt.py#L1149
        weight_global_scale = t2j(layer.weight_scale_2, use_dlpack=False)
        # assert that all elements are the same
        assert jnp.all(weight_global_scale == weight_global_scale[0])
        weight_global_scale = weight_global_scale[0]

        for attr in ('weight', 'weight_scale', 'weight_scale_2',
                     'input_scale'):
            if hasattr(layer, attr):
                delattr(layer, attr)

        if hasattr(
                layer,
                'bias') and layer.bias is not None and not layer.skip_bias_add:
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, 'bias')
        else:
            bias = None

        @jax.jit
        def _unpack_and_scale(weight_packed, weight_scale,
                              weight_global_scale):
            fp4 = u8_unpack_e2m1(weight_packed)  # (O, I) float4_e2m1fn

            # Combine FP8 block scale & FP32 global scale
            block_scale = weight_scale.astype(
                jnp.float32) * weight_global_scale  # (O, I/group)

            return fp4, block_scale

        weight_fp4, block_scale = _unpack_and_scale(weight_packed,
                                                    weight_scale,
                                                    weight_global_scale)

        weights = process_linear_weights(
            LinearWeights(weight=weight_fp4,
                          weight_scale=block_scale,
                          zero_point=None,
                          bias=bias),
            fused=self.linear_config.fuse_matmuls,
            output_sizes=self.linear_config.output_sizes,
            reorder_size=self.linear_config.n_shards,
        )

        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            layer.weight_scale = Parameter(weights.weight_scale,
                                           requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            layer.weight_scale = to_parameter_list(weights.weight_scale)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            x_jax = jax_view(x)
            bias_jax = jax_view(
                bias) if bias is not None and not layer.skip_bias_add else None
            if self.linear_config.fuse_matmuls:
                weight_jax = jax_view(layer.weight)
                weight_scale_jax = jax_view(layer.weight_scale)
                out = self._apply_fused(x_jax, weight_jax, weight_scale_jax,
                                        bias_jax)
            else:
                assert isinstance(layer.weight, torch.nn.ParameterList)
                assert isinstance(layer.weight_scale, torch.nn.ParameterList)
                # jax_view cannot handle ParameterList directly, so we explicitly
                # convert them to list of jax.Array.
                weight_and_scale = [
                    (jax_view(w), jax_view(s))
                    for w, s in zip(layer.weight, layer.weight_scale)
                ]
                if bias is not None and not layer.skip_bias_add:
                    assert isinstance(bias, torch.nn.ParameterList)
                    bias_jax = [jax_view(b) for b in bias]
                out = self._apply_split(x_jax, weight_and_scale, bias_jax)
            return torch_view(out)

    def _apply_fused(self,
                     x: jax.Array,
                     weight_jax: jax.Array,
                     weight_scale_jax: jax.Array,
                     bias: Optional[jax.Array] = None) -> jax.Array:

        outs = sharded_quantized_matmul(x,
                                        weight_jax,
                                        weight_scale_jax,
                                        self.linear_config.weight_sharding,
                                        mesh=self.linear_config.mesh)

        if bias is not None:
            outs += bias
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        return jnp.concatenate(outs, axis=-1)

    def _apply_split(self,
                     x: jax.Array,
                     weight_and_scale: Sequence[tuple[jax.Array, jax.Array]],
                     bias: Optional[Sequence[jax.Array]] = None,
                     mesh: Optional[Mesh] = None) -> jax.Array:

        outs = []
        for i, (weight, weight_scale) in enumerate(weight_and_scale):

            out = sharded_quantized_matmul(x,
                                           weight,
                                           weight_scale,
                                           self.linear_config.weight_sharding,
                                           mesh=mesh)

            if bias is not None:
                out += bias[i]
            outs.append(out)
        return jnp.concatenate(outs, axis=-1)


class VllmNvfp4MoEMethod(FusedMoEMethodBase):
    """NVFP4 MoE for TPU.

    Reuses upstream ModelOptNvFp4FusedMoE.create_weights for parameter
    registration; adds bias parameters upstream does not register.
    Weights stay packed E2M1 in HBM; the FP32 global × FP8 block scales are
    combined into a single FP32 blockwise scale of size 16 and the matmul
    runs through gmm_v2's dequant-in-VMEM path.
    """

    def __init__(self, quant_config, layer, mesh, ep_axis_name="model"):
        # Skip ModelOptNvFp4FusedMoE.__init__ (selects GPU experts backend).
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        # Upstream's create_weights references self.use_global_sf to size the
        # input_scale params; we don't use them so any value works — keep
        # False to match a CPU/TPU "no global SF" path.
        self.use_global_sf = False
        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name)

    @property
    def is_monolithic(self):
        return True

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        layer.params_dtype = params_dtype
        ModelOptNvFp4FusedMoE.create_weights(self, layer, num_experts,
                                             hidden_size,
                                             intermediate_size_per_partition,
                                             params_dtype,
                                             **extra_weight_attrs)
        # Upstream's create_weights only updates `extra_weight_attrs` after
        # the scale params are registered, so vLLM's MoE weight loader does
        # not see `quant_method` on them and rejects the load. Set them
        # explicitly here.
        from vllm.model_executor.layers.fused_moe.layer import \
            FusedMoeWeightScaleSupported
        for name in ("w13_weight_scale", "w2_weight_scale"):
            set_weight_attrs(
                getattr(layer, name),
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            set_weight_attrs(
                getattr(layer, name),
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        # Upstream NVFP4 MoE does not register bias; some models (e.g.
        # gpt-oss) need it. Keep our own bias registration.
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype),
                                          requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(torch.zeros(num_experts,
                                                     hidden_size,
                                                     dtype=params_dtype),
                                         requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, FusedMoE)
        # Drop unused activation-quant scales registered by upstream.
        for attr in ('w13_input_scale', 'w2_input_scale'):
            if hasattr(layer, attr):
                delattr(layer, attr)

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)
        w13_global_scale = t2j(layer.w13_weight_scale_2, use_dlpack=False)
        w2_global_scale = t2j(layer.w2_weight_scale_2, use_dlpack=False)
        w13_bias = t2j(layer.w13_bias,
                       use_dlpack=False) if self.moe.has_bias else None
        w2_bias = t2j(layer.w2_bias,
                      use_dlpack=False) if self.moe.has_bias else None

        @jax.jit
        def process_nvfp4_moe_weights(w13_weight, w13_weight_scale,
                                      w13_global_scale, w2_weight,
                                      w2_weight_scale, w2_global_scale,
                                      w13_bias, w2_bias):
            # Unpack packed uint8 -> native float4_e2m1fn (no value change).
            w13_fp4 = u8_unpack_e2m1(w13_weight)
            w2_fp4 = u8_unpack_e2m1(w2_weight)

            # Combine FP32 global × FP8 block -> single FP32 block scale.
            if w13_global_scale.ndim == 2:  # (E, 2) — separate w1/w3 globals.
                half = w13_weight_scale.shape[1] // 2
                w1_eff = w13_weight_scale[:, :half].astype(
                    jnp.float32) * w13_global_scale[:, 0:1].reshape(-1, 1, 1)
                w3_eff = w13_weight_scale[:, half:].astype(
                    jnp.float32) * w13_global_scale[:, 1:2].reshape(-1, 1, 1)
                w13_scale_eff = jnp.concatenate([w1_eff, w3_eff], axis=1)
            else:
                w13_scale_eff = w13_weight_scale.astype(
                    jnp.float32) * w13_global_scale.reshape(-1, 1, 1)
            w2_scale_eff = w2_weight_scale.astype(
                jnp.float32) * w2_global_scale.reshape(-1, 1, 1)

            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            weights = FusedMoEWeights(
                w13_weight=w13_fp4,
                w13_weight_scale=w13_scale_eff,
                w13_bias=w13_bias,
                w2_weight=w2_fp4,
                w2_weight_scale=w2_scale_eff,
                w2_bias=w2_bias,
            )
            return process_moe_weights(weights,
                                       moe_backend=self.moe_backend,
                                       w13_reorder_size=w13_reorder_size,
                                       w13_interleave=w13_interleave)

        weights = process_nvfp4_moe_weights(w13_weight, w13_weight_scale,
                                            w13_global_scale, w2_weight,
                                            w2_weight_scale, w2_global_scale,
                                            w13_bias, w2_bias)
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

    def apply_monolithic(self, layer, x, router_logits, **kwargs):
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
                              router_logits=router_logits)
