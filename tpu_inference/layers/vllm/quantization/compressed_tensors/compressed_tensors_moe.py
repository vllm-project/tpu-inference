from typing import Union

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.nn.parameter import Parameter
from torchax.interop import call_jax, torch_view
from torchax.ops.mappings import t2j
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import \
    CompressedTensorsW8A8Fp8MoEMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS, WNA16_SUPPORTED_TYPES_MAP)

from tpu_inference.layers.vllm.quantization.common import JaxCommonConfig

logger = init_logger(__name__)


class VllmCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod,
                                            JaxCommonConfig):

    def __init__(self, quant_config: "CompressedTensorsConfig",
                 moe: FusedMoEConfig, mesh: Mesh):
        super().__init__(quant_config, moe)
        self.mesh = mesh
        self.quant_config = quant_config

        # disable GPU paths
        self.use_marlin = False
        self.rocm_aiter_moe_enabled = False  # is_rocm_aiter_moe_enabled()
        self.is_fp8_w8a8_sm100 = False
        self.use_cutlass = False
        self.disable_expert_map = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        intermediate_size = layer.w13_weight.shape[1] // 2
        w1_weight = layer.w13_weight[:, :intermediate_size]
        w3_weight = layer.w13_weight[:, intermediate_size:]
        w1_weight_scale = layer.w13_weight_scale[:, :intermediate_size]
        w3_weight_scale = layer.w13_weight_scale[:, intermediate_size:]

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale.to(torch.bfloat16),
                              use_dlpack=False)
        w1_weight = t2j(w1_weight, use_dlpack=False)
        w1_weight_scale = t2j(w1_weight_scale.to(torch.bfloat16),
                              use_dlpack=False)
        w3_weight = t2j(w3_weight, use_dlpack=False)
        w3_weight_scale = t2j(w3_weight_scale.to(torch.bfloat16),
                              use_dlpack=False)

        if layer.use_ep:
            format = Format(Layout((0, 1, 2)),
                            NamedSharding(self.mesh, P("model", None, None)))
            w1_weight = jax.device_put(w1_weight, format)
            w1_weight_scale = jax.device_put(w1_weight_scale, format)
            w3_weight = jax.device_put(w3_weight, format)
            w3_weight_scale = jax.device_put(w3_weight_scale, format)
            w2_weight = jax.device_put(w2_weight, format)
            w2_weight_scale = jax.device_put(w2_weight_scale, format)
        else:
            assert intermediate_size == w2_weight.shape[-1]
            n_shards = self.mesh.shape["model"]
            assert intermediate_size % n_shards == 0

            # TODO: enable this if using fused weights
            # output_sizes = [intermediate_size, intermediate_size]
            # w13_weight = reorder_concatenated_tensor_for_sharding(
            #    w13_weight, output_sizes, n_shards, dim=1
            # )

            w13_format = Format(
                Layout((0, 1, 2)),
                NamedSharding(self.mesh, P(None, "model", None)))
            w1_weight = jax.device_put(w1_weight, w13_format)
            w1_weight_scale = jax.device_put(w1_weight_scale, w13_format)
            w3_weight = jax.device_put(w3_weight, w13_format)
            w3_weight_scale = jax.device_put(w3_weight_scale, w13_format)
            w2_weight = jax.device_put(
                w2_weight,
                Format(Layout((0, 1, 2)),
                       NamedSharding(self.mesh, P(None, None, "model"))),
            )
            w2_weight_scale = jax.device_put(
                w2_weight_scale,
                Format(Layout((0, 1, 2)), NamedSharding(self.mesh, P())),
            )  # replicate

        w1_weight = Parameter(torch_view(w1_weight), requires_grad=False)
        w1_weight_scale = Parameter(torch_view(w1_weight_scale),
                                    requires_grad=False)
        w2_weight = Parameter(torch_view(w2_weight), requires_grad=False)
        w2_weight_scale = Parameter(torch_view(w2_weight_scale),
                                    requires_grad=False)
        w3_weight = Parameter(torch_view(w3_weight), requires_grad=False)
        w3_weight_scale = Parameter(torch_view(w3_weight_scale),
                                    requires_grad=False)

        # TODO dont reuse variable
        layer.w13_weight = w1_weight
        layer.w13_weight_scale = w1_weight_scale
        layer.w2_weight = w2_weight
        layer.w2_weight_scale = w2_weight_scale
        layer.w3_weight = w3_weight
        layer.w3_weight_scale = w3_weight_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(layer, FusedMoE)
        if layer.activation != "silu":
            raise NotImplementedError(
                "Only silu is supported for activation function.")
        if layer.scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax is supported for scoring_func")

        # import sys
        # sys.stdin = open(0)
        # breakpoint()

        # TODO: Use MoE kernel when it supports fp8

        seqlen = x.shape[0]

        expert_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(expert_weights,
                                                    layer.top_k,
                                                    dim=-1)
        if layer.renormalize:
            expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

        # cond ffn
        # e = total num of exp = 160
        # t = seqlen
        # o = config.imtermediate size
        # i = config.dim
        #torch.einsum("ti, eoi -> teo", x, layer.w13_weight) * self.w13_weight_scale)
        ux1 = call_jax(jax.lax.dot,
                       x,
                       layer.w13_weight,
                       dimension_numbers=(((1, ), (2, )), ((), ())),
                       preferred_element_type=jnp.bfloat16.dtype)
        x1 = F.silu(ux1 * layer.w13_weight_scale.squeeze(2))

        #x3 = torch.einsum("ti, eoi -> teo", x, layer.w3_weight) * self.w3_weight_scale
        x3 = call_jax(jax.lax.dot,
                      x,
                      layer.w3_weight,
                      dimension_numbers=(((1, ), (2, )), ((), ())),
                      preferred_element_type=jnp.bfloat16.dtype
                      ) * layer.w3_weight_scale.squeeze(2)

        #expert_outs = torch.einsum("teo, eio -> tei", (x1 * x3), self.w2_weight) * self.w2_weight_scale
        expert_outs = call_jax(
            jax.lax.dot,
            x1 * x3,
            layer.w2_weight,
            dimension_numbers=(((2, ), (2, )), ((1, ), (0, ))),
            preferred_element_type=jnp.bfloat16.dtype).transpose(
                0, 1) * layer.w2_weight_scale.squeeze(2)

        seq_indexes = torch.arange(seqlen, device='jax').unsqueeze(1)
        expert_outs = expert_outs[seq_indexes, expert_indices]

        # out = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        out = call_jax(jax.lax.dot,
                       expert_outs,
                       expert_weights,
                       dimension_numbers=(((1, ), (1, )), ((0, ), (0, ))),
                       preferred_element_type=jnp.bfloat16.dtype)

        return out
