from typing import Union

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from compressed_tensors.quantization import QuantizationArgs
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.nn.parameter import Parameter
from torchax.interop import call_jax, torch_view, jax_view
from tpu_inference.layers.vllm.quantization.mxfp4 import (
    dequantize_block_weight, 
    quantize_block_weight,
    u8_unpack_e2m1,
    e8m0_to_fp32)
    
from tpu_inference.layers.vllm.linear_common import \
    reorder_concatenated_tensor_for_sharding
from torchax.ops.mappings import t2j
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import \
    CompressedTensorsW8A8Fp8MoEMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS, WNA16_SUPPORTED_TYPES_MAP)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig,  mxfp4_mxfp8_moe_quant_config)

from tpu_inference.layers.vllm.quantization.common import JaxCommonConfig
from tpu_inference.layers.vllm.fused_moe import fused_moe_func

logger = init_logger(__name__)
MXFP8_BLOCK_SIZE = 16


class VllmCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod,
                                            JaxCommonConfig):

    def __init__(self, quant_config: "CompressedTensorsConfig",
                 moe: FusedMoEConfig, mesh: Mesh):
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        super().__init__(weight_quant, input_quant,moe)
        self.mesh = mesh
        self.quant_config = quant_config

        # disable GPU paths
        self.use_marlin = False
        self.rocm_aiter_moe_enabled = False  # is_rocm_aiter_moe_enabled()
        self.is_fp8_w8a8_sm100 = False
        self.use_cutlass = False
        self.disable_expert_map = False

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return mxfp4_mxfp8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Docstring for process_weights_after_loading
        
        :param self: Description
        :param layer: Description
        :type layer: torch.nn.Module
        
        Steps:
        1. Read weights from layer object and convert to jax arrays
        2. Interleave concat w13 weights
        3. Shard weights for tp (rowwise w13, colwise w2)
        4. Initialize Params as torch.nn.Parameter
            a. w13_weight - float8_e4m3fn shape: (num_experts, 2 x intermediate_size, input_size)
            b. w2_weight - float8_e4m3fn shape: (num_experts, output_size, intermediate_size)
            c. w13_weight_scale - FP32 shape: (num_experts, 2 x intermediate_size, 1)
            d. w2_weight_scale - FP32shape: (num_experts, output_size, 1)
        """
        assert isinstance(layer, FusedMoE)
        
        # Read weights from layer object
        w13_weight = layer.w13_weight # float8_e4m3fn shape: (num_experts, 2 x intermediate_size, input_size)
        w2_weight = layer.w2_weight # float8_e4m3fn shape: (num_experts, output_size, intermediate_size)
        w13_weight_scale = layer.w13_weight_scale # FP32 shape: (num_experts, 2 x intermediate_size, 1)
        w2_weight_scale = layer.w2_weight_scale # FP32shape: (num_experts, output_size, 1)
        
        w13_weight = t2j(w13_weight)
        w2_weight = t2j(w2_weight)
        w13_weight_scale = t2j(w13_weight_scale.to(torch.bfloat16),
                              use_dlpack=False)
        w2_weight_scale = t2j(w2_weight_scale.to(torch.bfloat16),
                              use_dlpack=False)
        
        logger.info(f"Shapes and dtypes of params "
                    f"w13_weight: {w13_weight.shape}, {w13_weight.dtype}, "
                    f"w2_weight: {w2_weight.shape}, {w2_weight.dtype}, "
                    f"w13_weight_scale: {w13_weight_scale.shape}, {w13_weight_scale.dtype},"
                    f"w2_weight_scale: {w2_weight_scale.shape}, {w2_weight_scale.dtype}")
        
        intermediate_size = layer.w13_weight.shape[1] // 2
        assert intermediate_size == w2_weight.shape[-1]
        n_shards = self.mesh.shape["model"]
        assert intermediate_size % n_shards == 0
        
        # Interleave concat w13 weights
        w13_weight = reorder_concatenated_tensor_for_sharding(
            w13_weight,
            split_sizes=(intermediate_size, intermediate_size),
            dim=1,
            n_shards=n_shards)
        # Interleave concat w13 weight scales
        w13_weight_scale = reorder_concatenated_tensor_for_sharding(
            w13_weight_scale,
            split_sizes=(intermediate_size, intermediate_size),
            dim=1,
            n_shards=n_shards)
        
        # 160,5120,1 -> 160,1,5120 
        w13_weight_scale = jnp.swapaxes(w13_weight_scale, 1, 2)
        # 160,1,5120 -> 160, 1, 1, 5120   (num_experts, num_blocks, 1, outer_dim)
        w13_weight_scale = jnp.expand_dims(w13_weight_scale, 2)
        
        # Shard weights for tp (rowwise w13, colwise w2)
        w13_format = Format(
                Layout((0, 1, 2)), # expert, 2xintermed, input
                NamedSharding(
                    self.mesh, 
                    P(None, "model", None))) # rowwise sharding on intermed dim
        
        w13_scale_format = Format(
                Layout((0, 1, 2, 3)), #  (num_experts, num_blocks, 1, outer_dim)
                NamedSharding(
                    self.mesh, 
                    P(None, None, None, "model"))) # col wise GMM sharding on intermed dim
        
        # Local shard shape: (num_experts, 2 x (intermediate_size // n_shards), input_size)
        w13_weight = jax.device_put(w13_weight, w13_format) 
        # Local shard shape: (num_experts, (intermediate_size // n_shards), 1)
        w13_weight_scale = jax.device_put(w13_weight_scale, w13_scale_format) 
        
        
        # Shard weights for tp (colwise w2)
        w2_format = Format(
                       Layout((0, 1, 2)), # expert, intermed, hidden
                       NamedSharding(self.mesh, P(None, None, "model")))
        # Local shard shape: (num_experts, hidden, (intermediate_size // n_shards))
        # #  (num_experts, num_blocks, 1, outer_dim)
        w2_weight = jax.device_put(w2_weight, w2_format)
        w2_weight_scale = jnp.swapaxes(w2_weight_scale, 1, 2)
        w2_weight_scale = jnp.expand_dims(w2_weight_scale, 2)
        w2_scale_format = Format(
                       Layout((0, 1, 2, 3)), # expert, intermed, 1
                       NamedSharding(self.mesh, P(None, None, None, None)))
        # Local shard shape: (num_experts, intermediate_size // n_shards, 1)
        w2_weight_scale = jax.device_put(w2_weight_scale, w2_scale_format)

        w13_weight = Parameter(torch_view(w13_weight), requires_grad=False)
        w13_weight_scale = Parameter(torch_view(w13_weight_scale),
                                    requires_grad=False)
        w2_weight = Parameter(torch_view(w2_weight), requires_grad=False)
        w2_weight_scale = Parameter(torch_view(w2_weight_scale),
                                    requires_grad=False)
        
        layer.w13_weight = w13_weight
        layer.w13_weight_scale = w13_weight_scale
        layer.w2_weight = w2_weight
        layer.w2_weight_scale = w2_weight_scale

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

        # TODO: Use MoE kernel when it supports fp8
        x = jax_view(x)
        w13_weight = jax_view(layer.w13_weight)
        w2_weight = jax_view(layer.w2_weight)
        w13_weight_scale = jax_view(layer.w13_weight_scale)
        w2_weight_scale = jax_view(layer.w2_weight_scale)
        # w13_bias = jax_view(layer.w13_bias)
        # w2_bias = jax_view(layer.w2_bias)
        gating_output = jax_view(router_logits)
        seqlen = x.shape[0]
        use_jax = False
        if use_jax:
            expert_weights = F.softmax(router_logits, dim=-1)
            expert_weights, expert_indices = torch.topk(expert_weights,
                                                        top_k,
                                                        dim=-1)
            if renormalize:
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
        else:
            out = torch_view(
                    fused_moe_func(
                hidden_states=x,
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                w13_weight_scale=w13_weight_scale,
                w2_weight_scale=w2_weight_scale,
                w13_bias=None,
                w2_bias=None,
                gating_output=gating_output,
                topk=top_k,
                renormalize=renormalize,
                mesh=self.mesh,
                use_ep=layer.use_ep,
                activation=activation,
            ))

        return out
