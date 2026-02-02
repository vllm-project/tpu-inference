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
from torch.nn import Parameter
from torchax.interop import jax_view, torch_view
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
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    _tensor_is_in_cpu
from tpu_inference.layers.vllm.quantization.base import VllmQuantizationMethod
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.logger import init_logger

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
                                  common_unquantized.UnquantizedLinearMethod,
                                  VllmQuantizationMethod):

    def __init__(self, linear_config: VllmQuantLinearConfig):
        super().__init__(linear_config)

    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        """Check if all weights are loaded for the layer. If so, process and shard the weights."""
        if isinstance(layer, vllm_linear.QKVParallelLinear):
            assert len(args) == 1, "Expecting shard_id as the only argument"
            shard_id = args[0]
            # Keep track of loaded weights for QKVLinear, e.g. (('weight', 'q'), ('bias', 'q'), ('weight', 'k'), ('bias', 'k'), ...)
            layer._loaded_weights.add((param_name, shard_id))
        else:
            # Keep track of loaded weights for other linear layers, e.g. ('weight', 'bias')
            layer._loaded_weights.add(param_name)

        if len(layer._loaded_weights) == self.linear_config.num_proj * len(
                dict(layer.named_parameters(recurse=False))):
            self.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _tensor_is_in_cpu(layer.weight):
            # Already processed and sharded.
            return
        weight = t2j(layer.weight, use_dlpack=False)
        # Free CPU memory immediately
        layer.weight.untyped_storage().resize_(0)
        delattr(layer, 'weight')
        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            layer.bias.untyped_storage().resize_(0)
            delattr(layer, 'bias')
        else:
            bias = None

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


class VllmUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod,
                                    VllmQuantizationMethod):

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

    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        """Check if all weights are loaded for the layer. If so, process and shard the weights."""
        expert_id = kwargs.get('expert_id')
        shard_id = kwargs.get('shard_id')
        assert expert_id is not None, "Expecting expert_id argument"
        assert shard_id is not None, "Expecting shard_id argument"
        # Keep track of loaded weights for MoE layers, e.g. (('0', 'w1'), ('0', 'w2'), ('0', 'w3'), ('1', 'w1'), ...)
        layer._loaded_weights.add((expert_id, shard_id))
        if len(layer._loaded_weights) == layer.global_num_experts * len(
            ('w1', 'w2', 'w3')):
            self.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _tensor_is_in_cpu(layer.w13_weight):
            # Already processed and sharded.
            return
        assert isinstance(layer, FusedMoE)

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        # Free CPU memory immediately
        layer.w13_weight.untyped_storage().resize_(0)
        layer.w2_weight.untyped_storage().resize_(0)
        delattr(layer, 'w13_weight')
        delattr(layer, 'w2_weight')

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
            layer.w13_bias.untyped_storage().resize_(0)
            layer.w2_bias.untyped_storage().resize_(0)
            delattr(layer, 'w13_bias')
            delattr(layer, 'w2_bias')
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
