from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig, biased_moe_quant_config)
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEMethodBase)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.mxfp4 import (Mxfp4Backend,
                                                           Mxfp4Config,
                                                           Mxfp4MoEMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference import envs
from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.common.quant_methods import (MXFP4,
                                                       get_tpu_quant_method)
from tpu_inference.layers.vllm.fused_moe import fused_moe_func
from tpu_inference.layers.vllm.linear_common import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.vllm.quantization.common import JaxCommonConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod

MXFP4_BLOCK_SIZE = 32

P = PartitionSpec
logger = init_logger(__name__)


# TODO(kyuyeunk): Move these functions into a common utility file.
def u8_unpack_e2m1(u8_packed_e2m1: jax.Array) -> jax.Array:
    assert u8_packed_e2m1.dtype == jnp.uint8
    e2m1 = jax.lax.bitcast_convert_type(u8_packed_e2m1, jnp.float4_e2m1fn)
    # bitcast creates one more dimension that splits 8 bits into two e2m1.
    # we flatten them with the last dim.
    return jnp.reshape(e2m1, e2m1.shape[:-2] + (-1, ))


def e8m0_to_fp32(u8: jax.Array) -> jax.Array:
    e8_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    exponents = u8.astype(jnp.int32) + e8_finfo.minexp
    ones = jnp.ones_like(u8, dtype=jnp.float32)
    return jnp.ldexp(ones, exponents)


def dequantize_block_weight(weight: jax.Array,
                            scale: jax.Array,
                            block_size: int,
                            out_dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
    orig_shape = weight.shape
    weight_block = weight.reshape(orig_shape[:-1] + (-1, block_size))
    weight_dequantized = weight_block.astype(jnp.float32) * jnp.expand_dims(
        scale, -1)
    return weight_dequantized.reshape(orig_shape).astype(out_dtype)


@register_quantization_config(get_tpu_quant_method(MXFP4))
class VllmMxfp4Config(Mxfp4Config, JaxCommonConfig):

    @classmethod
    def get_name(cls):
        return MXFP4

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedLinearMethod(linear_config)
            logger.warning_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.")
            return VllmUnquantizedLinearMethod(linear_config)
        elif isinstance(layer, FusedMoE):
            moe_config = self.get_moe_config(layer)
            return VllmMxfp4MoEMethod(moe_config, self.mesh)
        elif isinstance(layer, Attention):
            logger.warning_once("MXFP4 attention layer is not implemented. "
                                "Skipping quantization for this layer.")
        return None


class VllmMxfp4MoEMethod(Mxfp4MoEMethod):

    def __init__(self,
                 moe: FusedMoEConfig,
                 mesh: Mesh,
                 ep_axis_name: str = 'model'):
        FusedMoEMethodBase.__init__(self, moe)

        # We piggyback on triton implementation as it applies minimal hardware
        # specific post processing to the weights.
        self.mxfp4_backend = Mxfp4Backend.TRITON

        self.mesh = mesh
        self.use_kernel = envs.USE_MOE_EP_KERNEL and moe.use_ep
        self.ep_axis_name = ep_axis_name
        # TODO: Use autotune table once we have it.
        self.block_size = {
            "bt": 64,
            "bf": 1024,
            "bd1": 1536,
            "bd2": 1536,
            "btc": 64,
            "bfc": 1024,
            "bd1c": 1536,
            "bd2c": 1536,
        }

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        # Because we have dequantized weights, we only need biased moe config.
        # TODO(kyuyeunk): Add native support for MXFP4.
        return biased_moe_quant_config(
            layer.w13_bias,
            layer.w2_bias,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, FusedMoE)
        assert layer.moe_config.has_bias, "mxfp4 quantization alwyas use bias."

        w13_weight = u8_unpack_e2m1(t2j(layer.w13_weight, use_dlpack=False))
        w13_weight_scale = e8m0_to_fp32(
            t2j(layer.w13_weight_scale, use_dlpack=False))
        w13_bias = t2j(layer.w13_bias, use_dlpack=False)

        w2_weight = u8_unpack_e2m1(t2j(layer.w2_weight, use_dlpack=False))
        w2_weight_scale = e8m0_to_fp32(
            t2j(layer.w2_weight_scale, use_dlpack=False))
        w2_bias = t2j(layer.w2_bias, use_dlpack=False)

        # We dequantize fp4 weights into bf16.
        # TODO(kyuyeunk): Add native support for MXFP4.
        w13_weight = dequantize_block_weight(w13_weight, w13_weight_scale,
                                             MXFP4_BLOCK_SIZE, jnp.bfloat16)
        w2_weight = dequantize_block_weight(w2_weight, w2_weight_scale,
                                            MXFP4_BLOCK_SIZE, jnp.bfloat16)

        num_experts, hidden_size, intermediate_size = w2_weight.shape

        # Because we have dequantized weights, scales are not used anymore.
        delattr(layer, "w13_weight_scale")
        delattr(layer, "w2_weight_scale")

        if layer.activation == "swigluoai":
            # When using swigluoai, vLLM splits gmm output in a interleaved way.
            # However, interleaved split is not performant on TPU. Therefore,
            # we preprocess the weight so that splitting gmm output by middle
            # can still get the same result.
            w1_weight = w13_weight[:, ::2, :]
            w3_weight = w13_weight[:, 1::2, :]
            w13_weight = jnp.concat([w1_weight, w3_weight], axis=1)

            w1_bias = w13_bias[:, ::2]
            w3_bias = w13_bias[:, 1::2]
            w13_bias = jnp.concat([w1_bias, w3_bias], axis=1)

        if self.use_kernel:
            # Kernel expects:
            # w13: (num_experts, 2, hidden_size, intermediate_size)
            # w2: (num_experts, intermediate_size, hidden_size)
            # Current format:
            # w13_weight: (num_experts, 2*intermediate_size, hidden_size)
            # w2_weight: (num_experts, hidden_size, intermediate_size)

            w13_reshaped = w13_weight.reshape(num_experts, 2,
                                              intermediate_size, hidden_size)

            # Transpose non-constracting dim to right most dim
            w13_weight_transposed = jnp.swapaxes(w13_reshaped, 2, 3)
            w2_weight_transposed = jnp.swapaxes(w2_weight, 1, 2)

            # Apply EP sharding
            ep_sharding = NamedSharding(self.mesh, P("model"))

            w13_weight = jax.device_put(
                w13_weight_transposed, Format(Layout((0, 1, 2, 3)),
                                              ep_sharding))
            w2_weight = jax.device_put(w2_weight_transposed,
                                       Format(Layout((0, 1, 2)), ep_sharding))

            w13_bias = w13_bias.reshape(num_experts, 2, intermediate_size)
            w13_bias = jax.device_put(w13_bias,
                                      Format(Layout((0, 1, 2)), ep_sharding))
            w2_bias = jax.device_put(w2_bias,
                                     Format(Layout((0, 1)), ep_sharding))

        else:
            if layer.use_ep:
                ep_sharding = NamedSharding(self.mesh, P("model"))
                w13_weight = jax.device_put(
                    w13_weight, Format(Layout((0, 1, 2)), ep_sharding))
                w2_weight = jax.device_put(
                    w2_weight, Format(Layout((0, 1, 2)), ep_sharding))

                w13_bias = jax.device_put(w13_bias,
                                          Format(Layout((0, 1)), ep_sharding))
                w2_bias = jax.device_put(w2_bias,
                                         Format(Layout((0, 1)), ep_sharding))

            else:
                output_sizes = [intermediate_size, intermediate_size]
                n_shards = self.mesh.shape["model"]
                assert intermediate_size % n_shards == 0

                w13_weight = reorder_concatenated_tensor_for_sharding(
                    w13_weight,
                    output_sizes,
                    n_shards,
                    dim=1,
                )
                w13_weight = jax.device_put(
                    w13_weight,
                    Format(Layout((0, 1, 2)),
                           NamedSharding(self.mesh, P(None, "model", None))))
                w2_weight = jax.device_put(
                    w2_weight,
                    Format(Layout((0, 1, 2)),
                           NamedSharding(self.mesh, P(None, None, "model"))))

                w13_bias = reorder_concatenated_tensor_for_sharding(
                    w13_bias,
                    output_sizes,
                    n_shards,
                    dim=1,
                )
                w13_bias = jax.device_put(
                    w13_bias,
                    Format(Layout((0, 1)),
                           NamedSharding(self.mesh, P(None, "model"))))
                w2_bias = jax.device_put(
                    w2_bias,
                    Format(Layout((0, 1)),
                           NamedSharding(self.mesh, P(None, None))))

        layer.w13_weight = Parameter(torch_view(w13_weight),
                                     requires_grad=False)
        layer.w2_weight = Parameter(torch_view(w2_weight), requires_grad=False)

        layer.w13_bias = Parameter(torch_view(w13_bias), requires_grad=False)
        layer.w2_bias = Parameter(torch_view(w2_bias), requires_grad=False)

        pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(layer, FusedMoE)
        if layer.scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax is supported for scoring_func")

        x = jax_view(x)
        w13_weight = jax_view(layer.w13_weight)
        w2_weight = jax_view(layer.w2_weight)
        w13_bias = jax_view(layer.w13_bias)
        w2_bias = jax_view(layer.w2_bias)
        gating_output = jax_view(router_logits)

        if self.use_kernel:
            output = fused_ep_moe(
                mesh=self.mesh,
                tokens=x,
                w1=w13_weight,
                w2=w2_weight,
                b1=w13_bias,
                b2=w2_bias,
                gating_output=gating_output,
                top_k=layer.top_k,
                ep_axis_name=self.ep_axis_name,
                renormalize_topk_logits=layer.renormalize,
                act_fn=layer.activation,
                **self.block_size,
            )
        else:
            output = fused_moe_func(
                hidden_states=x,
                w1=w13_weight,
                w2=w2_weight,
                w1_bias=w13_bias,
                w2_bias=w2_bias,
                gating_output=gating_output,
                topk=layer.top_k,
                renormalize=layer.renormalize,
                mesh=self.mesh,
                use_ep=layer.use_ep,
                activation=layer.activation,
            )

        return torch_view(output)
