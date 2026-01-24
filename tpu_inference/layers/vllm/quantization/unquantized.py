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

from typing import Any, Callable, Optional

import jax
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import (UNQUANTIZED,
                                                       get_tpu_quant_method)
from tpu_inference.layers.common.quantization import \
    unquantized as common_unquantized
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
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
        if isinstance(layer, vllm_linear.LinearBase):
            linear_config = self.get_linear_config(layer)
            return VllmUnquantizedLinearMethod(linear_config)
        if isinstance(layer, FusedMoE):
            moe_config = self.get_moe_config(layer)
            return VllmUnquantizedFusedMoEMethod(moe_config, self.mesh)
        if isinstance(layer, Attention):
            return None
        return None


class VllmUnquantizedLinearMethod(vllm_linear.UnquantizedLinearMethod,
                                  common_unquantized.UnquantizedLinearMethod):

    def __init__(self, linear_config: VllmQuantLinearConfig):
        super().__init__(linear_config)

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
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype,
                                       device="meta"),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        setattr(weight, "weight_loader", self._get_weight_loader(layer))

        if self.linear_config.bias:
            bias = Parameter(torch.empty(sum(output_partition_sizes),
                                         dtype=params_dtype,
                                         device="meta"),
                             requires_grad=False)
            layer.register_parameter("bias", bias)
            setattr(bias, "weight_loader", self._get_weight_loader(layer))

    def _get_weight_loader(self, layer: torch.nn.Module):

        def weight_loader(param: torch.nn.Parameter,
                          loaded_weight: torch.Tensor):
            # If the weight is already on TPU, we don't need to do anything.
            if isinstance(loaded_weight, torchax.tensor.Tensor):
                param.data = loaded_weight
                return

            weight_name = None
            for name, p in layer.named_parameters():
                if p is param:
                    weight_name = name
                    break
            assert weight_name is not None, "param not found in layer"

            # Helper to check if we are loading bias
            is_bias = weight_name == "bias"
            if is_bias:
                bias = t2j(loaded_weight, use_dlpack=False)
                # When loading bias, we need to create a dummy weight for the process function
                weight = jnp.zeros(layer.weight.shape, dtype=bias.dtype)
            else:
                weight = t2j(loaded_weight, use_dlpack=False)
                # When loading weight, we might not have bias yet, or bias is already loaded
                if layer.bias is not None and isinstance(
                        layer.bias, torchax.tensor.Tensor):
                    bias = jax_view(layer.bias)
                else:
                    bias = None

            # Definition of the processing function
            @jax.jit
            def process_unquantized_linear_weights(
                weight: jax.Array,
                bias: jax.Array | None,
            ) -> LinearWeights:
                return process_linear_weights(
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

            # Process and shard
            # NOTE: We only update the tensor that is currently being loaded.
            # If we are loading weight, we update weight.
            # If we are loading bias, we update bias.
            weights = process_unquantized_linear_weights(weight, bias)
            weights = torch_view(
                shard_linear_weights(
                    weights,
                    mesh=self.linear_config.mesh,
                    weight_p_spec=self.linear_config.weight_sharding,
                    bias_p_spec=self.linear_config.bias_sharding,
                ))

            if is_bias:
                param.data = weights.bias
            else:
                param.data = weights.weight

        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer, vllm_linear.LinearBase)

        with jax.named_scope(layer._get_name()):
            if in_sharding := self.linear_config.get_input_sharding(x):
                x.shard_(NamedSharding(self.linear_config.mesh, in_sharding))

            x_jax = jax_view(x)
            bias_jax = jax_view(
                bias) if bias is not None and not layer.skip_bias_add else None
            if self.linear_config.fuse_matmuls:
                weight_jax = jax_view(layer.weight)
                out_jax = self._apply_fused(x_jax, weight_jax, bias_jax)
                out: torch.Tensor = torch_view(out_jax)
            else:
                assert isinstance(layer.weight, torch.nn.ParameterList)
                # jax_view cannot handle ParameterList directly, so explicitly
                # convert to list.
                weight_jax = [jax_view(w) for w in layer.weight]
                if bias_jax is not None:
                    assert isinstance(layer.bias, torch.nn.ParameterList)
                    bias_jax = [jax_view(b) for b in layer.bias]
                out_jax = self._apply_split(x_jax, weight_jax, bias_jax)
                out: torch.Tensor = torch_view(out_jax)

            if out_sharding := self.linear_config.get_output_sharding(out):
                out.shard_(NamedSharding(self.linear_config.mesh,
                                         out_sharding))

        return out


class VllmUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(
        self,
        moe: FusedMoEConfig,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        super().__init__(moe)
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            # When fused moe kernle is used, we pass extra arguments like
            # tuned block sizes to the kernel.
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name, )

    @property
    def is_monolithic(self) -> bool:
        return True

    def _select_monolithic(self) -> Callable:
        return self.apply_monolithic

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)
        # Check if weights are already sharded (if create_weights was fully implemented in future)
        if isinstance(layer.w13_weight, torchax.tensor.Tensor):
            return

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

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=None,
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=None,
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )

        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
