from typing import Any, Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from tpu_commons.models.vllm.jax_linear_common import (
    slice_sharded_tensor_for_concatenation, torch_to_jax_param)
from tpu_commons.models.vllm.quantization.common import (JaxCommonConfig,
                                                         JaxCommonLinearConfig)

P = PartitionSpec
logger = init_logger(__name__)


@register_quantization_config("jax-unquantized")
class JaxUnquantizedConfig(QuantizationConfig, JaxCommonConfig):

    @classmethod
    def get_name(cls) -> str:
        return "jax-unquantized"

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
    def from_config(cls, _: dict[str, Any]) -> "JaxUnquantizedConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            return JaxUnquantizedLinearMethod(linear_config)
        if isinstance(layer, FusedMoE):
            return UnquantizedFusedMoEMethod(layer.moe_config)
        if isinstance(layer, Attention):
            return None
        return None


class JaxUnquantizedLinearMethod(UnquantizedLinearMethod):

    def __init__(self, jax_config: JaxCommonLinearConfig):
        self.jax_config = jax_config

    def move_weights_to_torchax_tensor(self, layer: torch.nn.Module) -> None:
        weight = torch_to_jax_param(
            layer.weight,
            NamedSharding(self.jax_config.mesh,
                          self.jax_config.weight_sharding),
            self.jax_config.output_sizes,
            self.jax_config.n_shards,
            self.jax_config.fuse_matmuls,
        )
        delattr(layer, 'weight')
        layer.weight = weight

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")

            bias = torch_to_jax_param(
                layer.bias,
                NamedSharding(self.jax_config.mesh,
                              self.jax_config.bias_sharding),
                self.jax_config.output_sizes,
                self.jax_config.n_shards,
                self.jax_config.fuse_matmuls,
            )
            delattr(layer, 'bias')
            layer.bias = bias

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if in_sharding := self.jax_config.get_input_sharding(x):
                x.shard_(NamedSharding(self.jax_config.mesh, in_sharding))

            if self.jax_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

            if out_sharding := self.jax_config.get_output_sharding(out):
                out.shard_(NamedSharding(self.jax_config.mesh, out_sharding))

        return out

    def _apply_fused(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_jax = x.jax()
        weight_jax = layer.weight.jax()

        outs = jnp.einsum('mn,pn->mp', x_jax, weight_jax)
        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.jax_config.output_sizes, self.jax_config.n_shards)
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
            weight_jax = weight.jax()

            out = jnp.einsum('mn,pn->mp', x_jax, weight_jax)
            if bias is not None and not layer.skip_bias_add:
                out += bias[i].jax()

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)
