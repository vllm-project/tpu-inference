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
import jax.numpy as jnp
import torch
import vllm.envs as vllm_envs
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoEConfig,
                                                  RoutedExperts,
                                                  UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod, VocabParallelEmbedding)

from tpu_inference.layers.common.moe import \
    FusedMoEMethodBase as TpuFusedMoEMethodBase
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_unquantized_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import UNQUANTIZED
from tpu_inference.layers.common.quantization import \
    unquantized as common_unquantized
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    _tensor_is_in_cpu
from tpu_inference.layers.vllm.quantization.base import VllmQuantizationMethod
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.logger import init_logger
from tpu_inference.models.common.pathways_dummy_loader import (
    create_dummy_weights_on_tpu, is_pathways_dummy_load)
from tpu_inference.utils import to_jax_dtype

P = PartitionSpec

logger = init_logger(__name__)


def _load_weight_for_layer(
    layer: torch.nn.Module,
    param_name: str,
    sharding: NamedSharding,
) -> jax.Array:
    """Load a layer's weight parameter onto the TPU mesh.
    """
    tensor = getattr(layer, param_name)

    if tensor.device == torch.device("meta"):
        vllm_config = get_current_vllm_config()

        # Hardening for Multimodal Embedding models (e.g., Qwen3-VL-Embedding-8B):
        # vLLM V1 uses lazy loading which may leave some tensors on the 'meta' device.
        # Since the TPU/JAX backend requires concrete data to perform t2j (Torch-to-JAX)
        # sharding, we must force materialization to CPU memory for pooling tasks.
        # Note: we cannot use `is_pooling_model` here because a model instance is not available
        # in this layer-level loading function.
        if vllm_config.model_config.runner_type == "pooling":
            logger.warning(
                f"Materializing meta tensor '{param_name}' for layer "
                f"{layer.__class__.__name__} to CPU RAM for StepPooler compatibility."
            )
            # Allocate real memory on CPU
            real_data = torch.empty_like(tensor, device='cpu')
            new_param = torch.nn.Parameter(real_data, requires_grad=False)

            # Bypass setters to satisfy PyTorch registration rules.
            # pop from __dict__ avoids shadowing KeyError;
            # inject into _parameters establishes Parameter identity.
            layer.__dict__.pop(param_name, None)
            layer._parameters[param_name] = new_param

            # Synchronize local handle for the subsequent t2j call
            tensor = new_param

    if not vllm_envs.VLLM_TPU_USING_PATHWAYS:
        return t2j(tensor, use_dlpack=False)

    if is_pathways_dummy_load():
        # Dummy weights are created directly on the TPU mesh, no CPU→TPU transfer needed
        tensor_shape = tuple(tensor.shape)
        tensor_dtype = tensor.dtype
        tensor.untyped_storage().resize_(0)
        dtype = to_jax_dtype(tensor_dtype)
        return create_dummy_weights_on_tpu(
            sharding=sharding,
            weight_shape=tensor_shape,
            weight_dtype=dtype,
        )

    # Pathways real-weight path
    dtype = to_jax_dtype(tensor.dtype)
    np_tensor = tensor.detach().cpu().to(torch.float32).numpy()
    return jax.device_put(np_tensor, sharding).astype(dtype)


@register_quantization_config(UNQUANTIZED)
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
        match layer:
            case vllm_linear.LinearBase():
                linear_config = self.get_linear_config(layer)
                return VllmUnquantizedLinearMethod(linear_config)
            case RoutedExperts():
                moe_config = self.get_moe_config(layer)
                return VllmUnquantizedFusedMoEMethod(moe_config, self.mesh)
            case Attention():
                return None
            case VocabParallelEmbedding():
                return VllmUnquantizedEmbeddingMethod(self.mesh)
            case _:
                return None


class VllmUnquantizedEmbeddingMethod(UnquantizedEmbeddingMethod):

    def __init__(self, mesh):
        self.mesh = mesh

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_sharding = NamedSharding(self.mesh,
                                        P(ShardingAxisName.MLP_TENSOR, None))
        weight = _load_weight_for_layer(layer, "weight", weight_sharding)
        delattr(layer, 'weight')
        weight = general_device_put(weight, weight_sharding)
        is_pooling = get_current_vllm_config(
        ).model_config.runner_type == "pooling"

        if is_pooling:
            layer.__dict__.pop("weight", None)
            layer._parameters["weight"] = Parameter(torch_view(weight),
                                                    requires_grad=False)
        else:
            layer.weight = Parameter(torch_view(weight), requires_grad=False)

        if isinstance(layer, ParallelLMHead) and layer.bias is not None:
            bias_sharding = NamedSharding(self.mesh,
                                          P(ShardingAxisName.MLP_TENSOR))
            bias = _load_weight_for_layer(layer, "bias", bias_sharding)
            delattr(layer, 'bias')
            bias = general_device_put(bias, bias_sharding)

            if is_pooling:
                layer.__dict__.pop("bias", None)
                layer._parameters["bias"] = Parameter(torch_view(bias),
                                                      requires_grad=False)
            else:
                layer.bias = Parameter(torch_view(bias), requires_grad=False)


class VllmUnquantizedLinearMethod(vllm_linear.UnquantizedLinearMethod,
                                  common_unquantized.UnquantizedLinearMethod,
                                  VllmQuantizationMethod):

    # Dynamically register this method to support weight_loader_v2 in vLLM.
    if "VllmUnquantizedLinearMethod" not in vllm_linear.WEIGHT_LOADER_V2_SUPPORTED:
        vllm_linear.WEIGHT_LOADER_V2_SUPPORTED.append(
            "VllmUnquantizedLinearMethod")

    def __init__(self, linear_config: VllmQuantLinearConfig):
        super().__init__(linear_config)

    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        """Check if all weights are loaded for the layer. If so, process and shard the weights.

        Note on Fused Weight Loading (e.g., Qwen3-VL / Qwen3-VL-MoE / Qwen2.5-VL Vision Encoder):
        Historically, LLMs stored attention Q/K/V weights separately in HuggingFace checkpoints,
        and vLLM loaded them per shard (passing shard_id in `args`).
        However, newer multimodal models like Qwen3-VL / Qwen3-VL-MoE store vision attention weights
        already fused on disk (e.g., `attn.qkv.weight`). In such cases, vLLM's weight loader is invoked
        without a shard_id (`args` is empty), and vLLM internally slices and copies the fused weight.
        To support TPU incremental sharding for these fused weights, we detect when `args` is empty
        and immediately register all underlying shards ('q', 'k', 'v') as loaded to satisfy the sharding trigger.
        """
        if isinstance(layer, vllm_linear.QKVParallelLinear):
            if len(args) == 1:
                shard_id = args[0]
                layer._loaded_weights.add((param_name, shard_id))
            else:
                # Fused weight loaded in one go (e.g., Qwen3-VL / Qwen3-VL-MoE vision encoder `attn.qkv.weight`).
                # vLLM's QKVParallelLinear.weight_loader internally slices the weight into q, k, v.
                # We register all 3 shards to immediately trigger process_weights_after_loading.
                layer._loaded_weights.add((param_name, 'q'))
                layer._loaded_weights.add((param_name, 'k'))
                layer._loaded_weights.add((param_name, 'v'))
        elif isinstance(layer, vllm_linear.MergedColumnParallelLinear):
            if len(args) == 1:
                shard_id = args[0]
                layer._loaded_weights.add((param_name, shard_id))
            else:
                # Fused weight loaded in one go (e.g., MLP gate_up_proj fused on disk).
                # Register all output partitions to immediately trigger process_weights_after_loading.
                for i in range(len(layer.output_sizes)):
                    layer._loaded_weights.add((param_name, i))
        else:
            # Keep track of loaded weights for other linear layers, e.g. ('weight', 'bias')
            layer._loaded_weights.add(param_name)

        if len(layer._loaded_weights) == self.linear_config.num_proj * len(
                dict(layer.named_parameters(recurse=False))):
            logger.debug(f"Start sharding weights for layer {type(layer)}")
            self.process_weights_after_loading(layer)
            logger.debug(f"Complete sharding weights for layer {type(layer)}")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _tensor_is_in_cpu(layer.weight):
            # Already processed and sharded.
            return
        # Under Pathways, shard weights directly onto the TPU mesh to avoid
        # placing a full unsharded copy on a single device (OOM).
        loading_sharding = NamedSharding(
            self.linear_config.mesh,
            PartitionSpec(*self.linear_config.weight_sharding[::-1]))
        weight = _load_weight_for_layer(layer, "weight", loading_sharding)
        weight = jnp.transpose(weight)

        # Free CPU memory immediately
        layer.weight.untyped_storage().resize_(0)
        delattr(layer, 'weight')
        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias_sharding = NamedSharding(self.linear_config.mesh,
                                          self.linear_config.bias_sharding)
            bias = _load_weight_for_layer(layer, "bias", bias_sharding)
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
        UnquantizedFusedMoEMethod.__init__(self, moe)
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        TpuFusedMoEMethodBase.__init__(self, self.moe_backend, ep_axis_name)

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
            logger.debug(f"Start sharding weights for layer {type(layer)}")
            self.process_weights_after_loading(layer)
            logger.debug(f"Complete sharding weights for layer {type(layer)}")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _tensor_is_in_cpu(layer.w13_weight):
            # Already processed and sharded.
            return
        assert isinstance(layer, RoutedExperts)

        # Under Pathways, shard weights directly onto the TPU mesh to avoid
        # placing a full unsharded copy on a single device (OOM for large MoE).
        ep_sharding = NamedSharding(self.mesh, P(ShardingAxisName.EXPERT))
        w13_weight = _load_weight_for_layer(layer, "w13_weight", ep_sharding)
        w2_weight = _load_weight_for_layer(layer, "w2_weight", ep_sharding)
        # Free CPU memory immediately
        layer.w13_weight.untyped_storage().resize_(0)
        layer.w2_weight.untyped_storage().resize_(0)
        delattr(layer, 'w13_weight')
        delattr(layer, 'w2_weight')

        if self.moe.has_bias:
            w13_bias = _load_weight_for_layer(layer, "w13_bias", ep_sharding)
            w2_bias = _load_weight_for_layer(layer, "w2_bias", ep_sharding)
            layer.w13_bias.untyped_storage().resize_(0)
            layer.w2_bias.untyped_storage().resize_(0)
            delattr(layer, 'w13_bias')
            delattr(layer, 'w2_bias')
        else:
            w13_bias = w2_bias = None

        weights = process_unquantized_moe_weights(mesh=self.mesh,
                                                  moe_backend=self.moe_backend,
                                                  activation=layer.activation,
                                                  w13_weight=w13_weight,
                                                  w13_bias=w13_bias,
                                                  w2_weight=w2_weight,
                                                  w2_bias=w2_bias)

        del w13_weight, w2_weight, w13_bias, w2_bias

        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))
        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

        # Force JAX to release intermediate buffers before processing the next
        # layer.  Without this barrier, async dispatch can keep old weight
        # buffers alive across layers, accumulating until OOM.
        jax.effects_barrier()

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
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
                              router_logits=router_logits,
                              input_ids=input_ids)
