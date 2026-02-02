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
from tpu_inference.layers.vllm.fused_moe import (FusedMoEBackend,
                                                 fused_moe_apply,
                                                 select_moe_backend)
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import (
    _tensor_is_in_cpu, process_unquantized_linear_weight,
    process_unquantized_moe_weight)
from tpu_inference.layers.vllm.process_weights.fused_moe_weights import \
    FusedMoEWeights
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
                                  common_unquantized.UnquantizedLinearMethod):

    def __init__(self, linear_config: VllmQuantLinearConfig):
        super().__init__(linear_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _tensor_is_in_cpu(layer.weight):
            # Already processed and sharded.
            return
        process_unquantized_linear_weight(layer)

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
        if not _tensor_is_in_cpu(layer.w13_weight):
            # Already processed and sharded.
            return
        assert isinstance(layer, FusedMoE)
        process_unquantized_moe_weight(layer)

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
