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
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.quantization import fp8 as vllm_fp8
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.layers.common.moe import \
    FusedMoEMethodBase as TpuFusedMoEMethodBase
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_quantized_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import FP8
from tpu_inference.layers.common.quantization import fp8 as common_fp8
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(FP8)
class VllmFp8Config(vllm_fp8.Fp8Config, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return FP8

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union[vllm_linear.LinearMethodBase, QuantizeMethodBase]]:
        match layer:
            case vllm_linear.LinearBase():
                linear_config = self.get_linear_config(layer)
                if is_layer_skipped(
                        prefix=prefix,
                        ignored_layers=self.ignored_layers,
                        fused_mapping=self.packed_modules_mapping,
                ):
                    return VllmUnquantizedLinearMethod(linear_config)
                return VllmFp8LinearMethod(self, linear_config)
            case RoutedExperts():
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
            case Attention():
                logger.warning_once("FP8KVCacheMethod is not implemented. "
                                    "Skipping quantization for this layer.")
                return None
            case _:
                return None


class VllmFp8LinearMethod(vllm_fp8.Fp8LinearMethod,
                          common_fp8.Fp8LinearMethod):

    # Dynamically register this method to support weight_loader_v2 in vLLM.
    # This avoids hardcoding TPU-specific methods in the upstream vLLM repository.
    if "VllmFp8LinearMethod" not in vllm_linear.WEIGHT_LOADER_V2_SUPPORTED:
        vllm_linear.WEIGHT_LOADER_V2_SUPPORTED.append("VllmFp8LinearMethod")

    def __init__(
        self,
        quant_config: VllmFp8Config,
        linear_config: VllmQuantLinearConfig,
    ):

        # Per https://github.com/vllm-project/vllm/pull/32929,
        # init_fp8_linear_kernel is now called by super().__init__
        # but does not support TPU backends as expected.
        # use_marlin was also changed to be determined via isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel).
        # We need to monkeypatch init_fp8_linear_kernel and explicitly set use_marlin = True
        # in order to bypass using native vLLM's vllm/vllm/model_executor/layers/quantization/utils/quant_utils.py:scaled_quantize.
        vllm_fp8.init_fp8_linear_kernel = lambda *args, **kwargs: None
        super().__init__(quant_config)
        self.use_marlin = True

        self.linear_config = linear_config
        if self.linear_config.enable_quantized_matmul_kernel and not self.linear_config.requant_block_size:
            raise ValueError(
                "You should set REQUANTIZE_BLOCK_SIZE to enable quantized matmul kernel. Please set the value or disable the quantized matmul kernel."
            )
        if not self.linear_config.enable_quantized_matmul_kernel and self.linear_config.requant_block_size:
            raise ValueError(
                "Blockwise quantization is supported by quantized matmul kernel. Please enable quantized_matmul_kernel or unset the quantize block size to trigger XLA per-channel quantization."
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Per https://github.com/vllm-project/vllm/pull/33892, use_marlin is set again
        # in vllm/model_executor/layers/quantization/fp8.py `create_weights`.
        # The flag is set on whether a specific type of GPU kernel is being used which
        # means it is set to False for TPU.
        # We need to return it back to True here.
        super().create_weights(layer, input_size_per_partition,
                               output_partition_sizes, input_size, output_size,
                               params_dtype, **extra_weight_attrs)
        self.use_marlin = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, vllm_linear.LinearBase)

        weight = t2j(layer.weight, use_dlpack=False)
        weight = jnp.transpose(weight)
        delattr(layer, "weight")

        if self.block_quant:
            weight_scale_inv = layer.weight_scale_inv
            # Float8_e8m0fnu (ue8m0) scales cannot be converted via numpy in t2j.
            # TODO: consider get rid of the f32 conversion, to optimize the HBM usage.
            if weight_scale_inv.dtype == torch.float8_e8m0fnu:
                weight_scale_inv = weight_scale_inv.to(torch.float32)
            weight_scale = t2j(weight_scale_inv, use_dlpack=False)
            weight_scale = jnp.transpose(weight_scale)
            delattr(layer, "weight_scale_inv")
        else:
            weight_scale = layer.weight_scale
            if weight_scale.dtype == torch.float8_e8m0fnu:
                weight_scale = weight_scale.to(torch.float32)
            weight_scale = t2j(weight_scale, use_dlpack=False)
            delattr(layer, "weight_scale")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        if self.block_quant:
            weights = common_fp8.process_blockwise_fp8_linear_weights(
                weight,
                weight_scale,
                bias=bias,
                weight_block_size=tuple(self.weight_block_size),
                requant_block_size=self.linear_config.requant_block_size,
                output_sizes=tuple(self.linear_config.output_sizes),
                requant_weight_dtype=self.linear_config.requant_weight_dtype,
                fuse_matmuls=self.linear_config.fuse_matmuls,
                n_shards=self.linear_config.n_shards,
                enable_kernel=self.linear_config.enable_quantized_matmul_kernel
            )
        else:
            if self.linear_config.fuse_matmuls and weight_scale.ndim == 1:
                # Handle fused scales for non-block quantized weights (e.g. Mistral Small 4).
                # Models might provide a single 1D scale vector for a fused layer
                # (like qkv_proj where there are multiple output partitions).
                # We need to replicate these scales so that `process_linear_weights`
                # can slice them correctly alongside the concatenated weights.
                weight_scale = jnp.concatenate([
                    jnp.full((size, ), s) for size, s in zip(
                        self.linear_config.output_sizes, weight_scale)
                ])

            weights = process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=tuple(self.linear_config.output_sizes),
                reorder_size=self.linear_config.n_shards,
                per_tensor=weight_scale.ndim == 0,
            )

        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
                per_tensor=not self.block_quant,
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
                out = self._apply_split(x_jax,
                                        weight_and_scale,
                                        bias_jax,
                                        mesh=self.linear_config.mesh)
            return torch_view(out)


class VllmFp8MoEMethod(vllm_fp8.Fp8MoEMethod, TpuFusedMoEMethodBase):

    def __init__(self,
                 quant_config: vllm_fp8.Fp8Config,
                 layer: torch.nn.Module,
                 mesh: Mesh,
                 ep_axis_name: str = "model"):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")
        self.fp8_backend = None

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        TpuFusedMoEMethodBase.__init__(self, self.moe_backend, ep_axis_name)

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, RoutedExperts)

        assert not self.moe.has_bias

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)

        if self.block_quant:
            w13_weight_scale = t2j(layer.w13_weight_scale_inv,
                                   use_dlpack=False)
            w2_weight_scale = t2j(layer.w2_weight_scale_inv, use_dlpack=False)
        else:
            w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
            w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        # TODO: do we need to support bias?
        input_weights = FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w13_bias=None,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            w2_bias=None,
        )

        weight_block_size = None
        if self.weight_block_size is not None:
            weight_block_size = tuple(self.weight_block_size)

        weights = process_quantized_moe_weights(
            input_weights,
            moe_backend=self.moe_backend,
            mesh=self.mesh,
            activation=layer.activation.value,
            # Convert to tuple so jax jit can hash it
            weight_block_size=weight_block_size,
        )
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        # Use setattr to dynamically assign the correct scale parameter name
        # based on the quantization type. vLLM uses 'weight_scale_inv' for
        # block-quantized scales and 'weight_scale' for per-tensor/per-channel scales.
        setattr(layer, f"w13_{self.weight_scale_name}",
                Parameter(weights.w13_weight_scale, requires_grad=False))
        setattr(layer, f"w2_{self.weight_scale_name}",
                Parameter(weights.w2_weight_scale, requires_grad=False))

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Use getattr to retrieve the scales dynamically since the attribute
        # name varies depending on if block quantization is active.
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(
                getattr(layer, f"w13_{self.weight_scale_name}")),
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(
                getattr(layer, f"w2_{self.weight_scale_name}")),
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits,
                              input_ids=input_ids)
