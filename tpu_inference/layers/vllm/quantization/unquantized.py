import functools
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.attention.layer import Attention
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEConfig, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from tpu_inference.layers.vllm.jax_fused_moe import jax_fused_moe_func_padded
from tpu_inference.layers.vllm.jax_linear_common import (
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation, torch_to_jax_param)
from tpu_inference.layers.vllm.quantization.common import (
    JaxCommonConfig, JaxCommonLinearConfig)

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
            moe_config = self.get_moe_config(layer)
            return JaxUnquantizedFusedMoEMethod(moe_config, self.mesh)
        if isinstance(layer, Attention):
            return None
        return None


class JaxUnquantizedLinearMethod(UnquantizedLinearMethod):

    def __init__(self, jax_config: JaxCommonLinearConfig):
        self.jax_config = jax_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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


class JaxUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig, mesh: Mesh):
        super().__init__(moe)
        self.mesh = mesh

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        moe: FusedMoEConfig,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        raise NotImplementedError(
            "Selecting gemm implementation is currently not supported.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w13_weight = t2j(layer.w13_weight, use_dlpack=False)

        if layer.use_ep:
            w13_weight = jax.device_put(
                w13_weight,
                Format(Layout((0, 1, 2)),
                       NamedSharding(self.mesh, P('model', None, None))))
            w2_weight = jax.device_put(
                w2_weight,
                Format(Layout((0, 1, 2)),
                       NamedSharding(self.mesh, P('model', None, None))))
        else:
            intermediate_size = w13_weight.shape[1] // 2
            assert intermediate_size == w2_weight.shape[-1]
            output_sizes = [intermediate_size, intermediate_size]
            n_shards = self.mesh.shape['model']
            assert intermediate_size % n_shards == 0
            w13_weight = reorder_concatenated_tensor_for_sharding(w13_weight,
                                                                  output_sizes,
                                                                  n_shards,
                                                                  dim=1)
            w13_weight = jax.device_put(
                w13_weight,
                Format(Layout((0, 1, 2)),
                       NamedSharding(self.mesh, P(None, 'model', None))))
            w2_weight = jax.device_put(
                w2_weight,
                Format(Layout((0, 1, 2)),
                       NamedSharding(self.mesh, P(None, None, 'model'))))
        w13_weight = Parameter(torch_view(w13_weight), requires_grad=False)
        w2_weight = Parameter(torch_view(w2_weight), requires_grad=False)

        layer.w13_weight = w13_weight
        layer.w2_weight = w2_weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(layer, FusedMoE)
        if activation != "silu":
            raise NotImplementedError(
                "Only silu is supported for activation function.")
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax is supported for scoring_func")

        _fused_moe_func = functools.partial(
            jax.jit(jax_fused_moe_func_padded,
                    static_argnames=[
                        "topk", "global_num_experts", "renormalize",
                        "reduce_results", "mesh", "use_ep"
                    ]),
            topk=top_k,
            global_num_experts=global_num_experts,
            renormalize=renormalize,
            reduce_results=layer.reduce_results,
            mesh=self.mesh,
            use_ep=layer.use_ep)

        output = _fused_moe_func(
            jax_view(x),
            jax_view(layer.w13_weight),
            jax_view(layer.w2_weight),
            jax_view(router_logits),
        )

        return torch_view(output)
