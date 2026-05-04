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

Subclasses upstream vLLM's NVFP4 config + linear/MoE methods for parameter
registration; overrides only the TPU-specific paths:

    - Config: subclass ModelOptNvFp4Config (config parsing).
    - Linear: subclass VllmUnquantizedLinearMethod for the bf16 matmul apply
      path; delegate create_weights to ModelOptNvFp4LinearMethod; override
      process_weights_after_loading to dequant FP4 -> bf16 once.
    - MoE: subclass FusedMoEMethodBase; delegate create_weights to
      ModelOptNvFp4FusedMoE (and add bias params upstream omits); override
      process_weights_after_loading to combine scales and reshape for
      gmm_v2's dequant-in-VMEM path; override apply_monolithic to route
      through vllm_moe_apply.

Weights stay as packed E2M1 in HBM with a single FP32 blockwise scale
(combined from FP8 block × FP32 global). Dequantization happens in VMEM
inside gmm_v2 (block_size=16 < mxu_column_size=128 auto-trips the
should_dequantize_before_matmul branch).
"""

from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
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

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import NVFP4
from tpu_inference.layers.common.quantization import u8_unpack_e2m1
from tpu_inference.layers.common.sharding import ShardingAxisName
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


# ---------------------------------------------------------------------------
# Linear: delegate create_weights to upstream; dequant once to bf16; inherit
# unquantized bf16 matmul apply.
# ---------------------------------------------------------------------------
class VllmNvfp4LinearMethod(VllmUnquantizedLinearMethod):
    """NVFP4 linear for TPU.

    Reuses upstream ModelOptNvFp4LinearMethod.create_weights for parameter
    registration. process_weights_after_loading dequants FP4 -> bf16 using
    the original two-level scales. apply is inherited from
    VllmUnquantizedLinearMethod (plain bf16 matmul).
    """

    def __init__(self, quant_config: 'VllmNvfp4Config',
                 linear_config: VllmQuantLinearConfig):
        # Skip ModelOptNvFp4LinearMethod.__init__ (calls GPU cutlass
        # init_nvfp4_linear_kernel). Initialize the unquantized base instead.
        VllmUnquantizedLinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        ModelOptNvFp4LinearMethod.create_weights(
            self, layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, params_dtype, **extra_weight_attrs)

        # For fused-QKV layers, upstream registers `input_scale` and
        # `weight_scale_2` as `PerTensorScaleParameter` with shape
        # `(len(output_partition_sizes),)`. vLLM's QKVParallelLinear
        # weight_loader expects the loaded scalar to match this shape and
        # asserts (linear.py: `assert param_data.shape == loaded_weight.shape`),
        # which fails because the checkpoint provides a single scalar per
        # shard. Override the loader to fill the slot regardless of
        # the loaded weight's shape (we only use the max in
        # process_weights_after_loading anyway).
        def scalar_weight_loader(param, loaded_weight, *args, **kwargs):
            value = (loaded_weight.item() if loaded_weight.numel() == 1 else
                     loaded_weight.max().item())
            param.data.fill_(value)

        for name in ("input_scale", "weight_scale_2"):
            if hasattr(layer, name):
                set_weight_attrs(getattr(layer, name),
                                 {"weight_loader": scalar_weight_loader})

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, LinearBase)
        weight_packed = t2j(layer.weight, use_dlpack=False)
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        # Upstream registers weight_scale_2 / input_scale as
        # PerTensorScaleParameter of shape (num_partitions,). Take max so a
        # fused-QKV layer with differing per-projection globals collapses to
        # one scalar (matches upstream's process_weights_after_loading).
        weight_global_scale = jnp.max(
            t2j(layer.weight_scale_2, use_dlpack=False))

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
        def _dequant_to_bf16(weight_packed, weight_scale, weight_global_scale):
            fp4 = u8_unpack_e2m1(weight_packed)  # (O, I) float4_e2m1fn
            fp4_fp32 = fp4.astype(jnp.float32)
            block_scale = weight_scale.astype(
                jnp.float32) * weight_global_scale  # (O, I/group)
            group_size = fp4.shape[-1] // block_scale.shape[-1]
            scaled = fp4_fp32.reshape(fp4.shape[0], block_scale.shape[-1],
                                      group_size) * block_scale[:, :, None]
            return scaled.reshape(fp4.shape).astype(jnp.bfloat16)

        weight_bf16 = _dequant_to_bf16(weight_packed, weight_scale,
                                       weight_global_scale)

        weights = process_linear_weights(
            LinearWeights(weight=weight_bf16,
                          weight_scale=None,
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
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    # apply() inherited from VllmUnquantizedLinearMethod.


# ---------------------------------------------------------------------------
# MoE: delegate create_weights to upstream (and add bias params upstream
# omits); combine scales + reshape for gmm_v2; route apply through
# vllm_moe_apply.
# ---------------------------------------------------------------------------
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
