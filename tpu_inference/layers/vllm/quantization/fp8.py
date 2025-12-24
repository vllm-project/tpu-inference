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

from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.attention.layer import Attention
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoERouter
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
                                                         Fp8LinearMethod,
                                                         Fp8MoEMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.layers.common.quant_methods import FP8, get_tpu_quant_method
from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.fused_moe import (FusedMoEBackend,
                                                 fused_moe_apply,
                                                 select_moe_backend)
from tpu_inference.layers.vllm.linear import sharded_quantized_matmul
from tpu_inference.layers.vllm.process_weights.fused_moe_weights import (
    FusedMoEWeights, process_moe_weights, quantize_moe_weights,
    shard_moe_weights)
from tpu_inference.layers.vllm.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(FP8))
class VllmFp8Config(Fp8Config, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return FP8

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedLinearMethod(linear_config)
            return VllmFp8LinearMethod(self, linear_config)
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedFusedMoEMethod(layer.moe_config)
            if self.is_checkpoint_fp8_serialized:
                layer.moe_config = self.get_moe_config(layer)
                return VllmFp8MoEMethod(self, layer, self.mesh)
            else:
                raise NotImplementedError(
                    "FP8OnelineMoEMethod is not supported.")
        elif isinstance(layer, Attention):
            logger.warning_once("FP8KVCacheMethod is not implemented. "
                                "Skipping quantization for this layer.")
        return None


class VllmFp8LinearMethod(Fp8LinearMethod):

    def __init__(self, quant_config: VllmFp8Config,
                 linear_config: VllmQuantLinearConfig):
        super().__init__(quant_config)
        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, LinearBase)

        assert self.block_quant
        weight = t2j(layer.weight, use_dlpack=False)
        delattr(layer, "weight")

        weight_scale = t2j(layer.weight_scale_inv, use_dlpack=False)
        delattr(layer, "weight_scale_inv")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def process_fp8_linear_weights(
            weight: jax.Array,
            weight_scale: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            weights = []
            weight_scales = []
            block_size = self.weight_block_size[0]
            start = 0
            for output_size in self.linear_config.output_sizes:
                end = start + output_size

                weight_slice = weight[start:end]
                weight_scale_slice = weight_scale[start // block_size:end //
                                                  block_size]
                dequantzed_weight = dequantize_tensor(
                    weight_slice,
                    weight_scale_slice,
                    (0, 1),
                )
                weight_slice, weight_scale_slice = quantize_tensor(
                    jnp.float8_e4m3fn, dequantzed_weight, block_size=512)

                weights.append(weight_slice)
                weight_scales.append(weight_scale_slice)

                start = end

            weight = jnp.concat(weights, axis=0)
            weight_scale = jnp.concat(weight_scales, axis=0)

            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )

        weights = process_fp8_linear_weights(weight, weight_scale, bias)
        weights.weight_scale = jnp.expand_dims(jnp.transpose(weights.weight_scale), axis=1)
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

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                return self._apply_fused(layer, x, bias)
            else:
                return self._apply_split(layer, x, bias)

    def _apply_fused(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_jax = jax_view(x)
        weight_jax = jax_view(layer.weight)
        weight_scale_jax = jax_view(layer.weight_scale)

        outs = sharded_quantized_matmul(x_jax, weight_jax, weight_scale_jax,
                                        self.linear_config.mesh,
                                        self.linear_config.weight_sharding, sc_size=512)

        if bias is not None and not layer.skip_bias_add:
            outs += jax_view(bias)
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        return torch_view(jnp.concatenate(outs, axis=-1))

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, (weight, weight_scale) in enumerate(
                zip(layer.weight, layer.weight_scale)):
            weight_jax = jax_view(weight)
            weight_scale_jax = jax_view(weight_scale)

            #print("Printing quantized matmul 2")
            out = sharded_quantized_matmul(x_jax, weight_jax, weight_scale_jax,
                                           self.linear_config.mesh,
                                           self.linear_config.weight_sharding)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])
            outs.append(out)
        return torch_view(jnp.concatenate(outs, axis=-1))


class VllmFp8MoEMethod(Fp8MoEMethod):

    def __init__(self,
                 quant_config: Fp8Config,
                 layer: torch.nn.Module,
                 mesh: Mesh,
                 ep_axis_name: str = "model"):
        super().__init__(quant_config, layer)

        self.mesh = mesh
        self.moe_backend = select_moe_backend(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == FusedMoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(
                ep_axis_name=ep_axis_name,
                # TODO: Use autotune table once we have it.
                bt=256,
                bf=1024,
                bd1=1024,
                bd2=1024,
                btc=256,
                bfc=1024,
                bd1c=1024,
                bd2c=1024,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        assert self.block_quant
        assert not self.moe.has_bias

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale_inv, use_dlpack=False)

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale_inv, use_dlpack=False)

        @jax.jit
        def process_fp8_moe_weights(
            w13_weight: jax.Array,
            w13_weight_scale: jax.Array,
            w2_weight: jax.Array,
            w2_weight_scale: jax.Array,
        ) -> FusedMoEWeights:
            # Dequantize fp8 2d block quantized weights into fp32.
            w13_weight = dequantize_tensor(w13_weight, w13_weight_scale,
                                           (1, 2))
            w2_weight = dequantize_tensor(w2_weight, w2_weight_scale, (1, 2))

            w13_interleave = layer.activation == "swigluoai"
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            weights = quantize_moe_weights(
                FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=None,
                    w13_bias=None,
                    w2_weight=w2_weight,
                    w2_weight_scale=None,
                    w2_bias=None,
                ),
                jnp.float8_e4m3fn,
                None,
            )

            return process_moe_weights(
                weights,
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_fp8_moe_weights(
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
        )
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale_inv = Parameter(weights.w13_weight_scale,
                                               requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(weights.w2_weight_scale,
                                              requires_grad=False)

    def apply(
        self,
        layer: FusedMoE,
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale_inv),
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale_inv),
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )

        return fused_moe_apply(
            layer,
            x,
            router_logits,
            weights,
            self.moe_backend,
            self.mesh,
            self.extra_backend_kwargs,
        weight.shape, scale.shape)
