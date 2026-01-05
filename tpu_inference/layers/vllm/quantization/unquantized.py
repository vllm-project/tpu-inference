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

from typing import Any, Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.attention.layer import Attention
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEConfig, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from tpu_inference.layers.common.quant_methods import (UNQUANTIZED,
                                                       get_tpu_quant_method)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.fused_moe import (FusedMoEBackend,
                                                 fused_moe_apply,
                                                 select_moe_backend)
from tpu_inference.layers.vllm.process_weights.fused_moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.vllm.process_weights.linear_weights import (
    LinearWeights, process_lienar_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(get_tpu_quant_method(UNQUANTIZED))
class VllmUnquantizedConfig(QuantizationConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls) -> str:
        return UNQUANTIZED

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # Always supported

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # No extra configs required.

    @classmethod
    def from_config(cls, _: dict[str, Any]) -> "VllmUnquantizedConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            return VllmUnquantizedLinearMethod(linear_config)
        if isinstance(layer, FusedMoE):
            moe_config = self.get_moe_config(layer)
            return VllmUnquantizedFusedMoEMethod(moe_config, self.mesh)
        if isinstance(layer, Attention):
            return None
        return None


class VllmUnquantizedLinearMethod(UnquantizedLinearMethod):

    def __init__(self, linear_config: VllmQuantLinearConfig):
        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = t2j(layer.weight, use_dlpack=False)
        delattr(layer, "weight")
        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def process_unquantized_linear_weights(
            weight: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            return process_lienar_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=None,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )

        weights = process_unquantized_linear_weights(weight, bias)
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

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer, LinearBase)

        with jax.named_scope(layer._get_name()):
            if in_sharding := self.linear_config.get_input_sharding(x):
                x.shard_(NamedSharding(self.linear_config.mesh, in_sharding))

            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

            if out_sharding := self.linear_config.get_output_sharding(out):
                out.shard_(NamedSharding(self.linear_config.mesh,
                                         out_sharding))

        return out

    def _apply_fused(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_jax = jax_view(x)
        weight_jax = jax_view(layer.weight)

        outs = jnp.einsum("mn,pn->mp", x_jax, weight_jax)
        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = x.jax()
        outs = []
        for i, weight in enumerate(layer.weight):
            weight_jax = jax_view(weight)

            out = jnp.einsum("mn,pn->mp", x_jax, weight_jax)
            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)


class VllmUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(
        self,
        moe: FusedMoEConfig,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        super().__init__(moe)
        self.mesh = mesh
        self.moe_backend = select_moe_backend(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == FusedMoEBackend.FUSED_MOE:
            # When fused moe kernle is used, we pass extra arguments like
            # tuned block sizes to the kernel.
            self.extra_backend_kwargs = dict(
                ep_axis_name=ep_axis_name,
                # TODO: Use autotune table once we have it.
                bt=64,
                bf=1024,
                bd1=1536,
                bd2=1536,
                btc=64,
                bfc=1024,
                bd1c=1536,
                bd2c=1536,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
        else:
            w13_bias = w2_bias = None

        @jax.jit
        def process_unquantized_moe_weights(
            w13_weight: jax.Array,
            w13_bias: jax.Array | None,
            w2_weight: jax.Array,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:

            w13_interleave = layer.activation == "swigluoai"
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            return process_moe_weights(
                FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=None,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=None,
                    w2_bias=w2_bias,
                ),
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_unquantized_moe_weights(
            w13_weight,
            w13_bias,
            w2_weight,
            w2_bias,
        )
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        return fused_moe_apply(
            layer,
            x,
            router_logits,
            self.moe_backend,
            self.mesh,
            self.extra_backend_kwargs,
        )
