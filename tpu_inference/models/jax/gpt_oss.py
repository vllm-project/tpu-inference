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

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.layers.common.quant_methods import MXFP4
from tpu_inference.layers.jax.attention.gpt_oss_attention import (
    AttentionMetadata, GptOssAttention)
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.moe.gpt_oss_moe import GptOssMoE, GptOssRouter
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.transformer_block import TransformerBlock
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    get_param, model_weights_generator, print_param_info)

logger = init_logger(__name__)

# A map from JAX dtype to the corresponding PyTorch integer dtype for raw memory viewing.
DTYPE_VIEW_MAP = {
    jnp.dtype(jnp.float8_e4m3fn): torch.uint8,
    jnp.dtype(jnp.bfloat16): torch.uint16,
    jnp.dtype(jnp.float32): torch.uint32,
}

# Everything under `mlp.experts.` belongs to the experts' quant method, which
# owns the tensor-name contract and rejects names it doesn't recognize.
_EXPERT_TENSOR_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\..+")


@dataclass(kw_only=True)
class GptOssMoEBlock(nnx.Module):
    """Calls the GPT-OSS router, then the routed experts on its logits."""
    router: GptOssRouter
    experts: GptOssMoE

    def __call__(self, x_TD: jax.Array):
        router_logits = self.router(x_TD)
        return self.experts(x_TD, router_logits)


@dataclass
class GptOss(nnx.Module):
    """
    JAX implementation of the GPT-OSS model architecture.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: jax.Array,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        self.hf_config = vllm_config.model_config.hf_config
        self.rng = nnx.Rngs(rng)

        num_layers: int = self.hf_config.num_hidden_layers
        num_experts: int = self.hf_config.num_local_experts
        vocab_size: int = self.hf_config.vocab_size
        num_attention_heads: int = self.hf_config.num_attention_heads
        num_key_value_heads: int = self.hf_config.num_key_value_heads
        head_dim: int = self.hf_config.head_dim
        hidden_size: int = self.hf_config.hidden_size
        ffw_intermediate_size: int = self.hf_config.intermediate_size
        num_experts_per_token: int = self.hf_config.num_experts_per_tok
        rms_norm_eps: float = self.hf_config.rms_norm_eps
        swiglu_limit: float = self.hf_config.swiglu_limit
        # The shared swigluoai GMM kernel hardcodes limit=7.0.
        if swiglu_limit != 7.0:
            raise NotImplementedError(
                "GPT-OSS native MXFP4 MoE only supports swiglu_limit=7.0, "
                f"got {swiglu_limit}.")

        rope_theta: float = getattr(
            self.hf_config, "rope_theta",
            None) or self.hf_config.rope_scaling["rope_theta"]
        rope_scaling_factor: float = self.hf_config.rope_scaling["factor"]
        rope_ntk_alpha: float = self.hf_config.rope_scaling["beta_slow"]
        rope_ntk_beta: float = self.hf_config.rope_scaling["beta_fast"]
        initial_context_length: int = self.hf_config.rope_scaling[
            "original_max_position_embeddings"]

        dtype: jnp.dtype = jnp.bfloat16

        self.sliding_window = self.hf_config.sliding_window

        self.random_init = force_random_weights or self.vllm_config.additional_config.get(
            "random_weights", False)
        self.enable_return_routed_experts = self.vllm_config.model_config.enable_return_routed_experts
        self.mesh = mesh
        quant_config = get_tpu_quantization_config(vllm_config)

        self.embedder = Embedder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dtype=dtype,
            rngs=self.rng,
            vd_sharding=P('model', None),
            random_init=self.random_init,
        )

        self.layers = nnx.List([])
        for i in range(num_layers):
            attn = GptOssAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dtype=dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                rope_theta=rope_theta,
                initial_context_length=initial_context_length,
                rope_scaling_factor=rope_scaling_factor,
                rope_ntk_alpha=rope_ntk_alpha,
                rope_ntk_beta=rope_ntk_beta,
                rngs=self.rng,
                random_init=self.random_init,
                query_tnh=P("data", 'model', None),
                keyvalue_skh=P("data", 'model', None),
                attn_o_tnh=P("data", 'model', None),
                dnh_sharding=P(None, 'model', None),
                dkh_sharding=P(None, 'model', None),
                nhd_sharding=P('model', None, None),
                mesh=self.mesh,
            )

            experts = GptOssMoE(
                rngs=self.rng,
                dtype=dtype,
                mesh=self.mesh,
                quant_config=quant_config,
                num_local_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size_moe=ffw_intermediate_size,
                num_experts_per_tok=num_experts_per_token,
                random_init=self.random_init,
                enable_return_routed_experts=self.enable_return_routed_experts,
                prefix=f"model.layers.{i}.mlp.experts",
            )
            router = GptOssRouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_token,
                rngs=self.rng,
                dtype=dtype,
                router_act='softmax',
                random_init=self.random_init,
                activation_ffw_td=P('data', None),
                ed_sharding=P('model', None),
                e_sharding=P('model'),
                moe_backend=experts.moe_backend,
                mesh=self.mesh,
            )
            moe_mlp = GptOssMoEBlock(router=router, experts=experts)

            block = TransformerBlock(
                pre_attention_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    dtype=dtype,
                    rngs=self.rng,
                    activation_ffw_td=P('data', None),
                ),
                pre_mlp_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    dtype=dtype,
                    rngs=self.rng,
                    activation_ffw_td=P('data', None),
                ),
                attn=attn,
                custom_module=moe_mlp,
                enable_return_routed_experts=self.enable_return_routed_experts,
            )
            self.layers.append(block)
        # Note: ALL RMSNorm does not upcast input to float32, while the pytorch does
        self.final_norm = RMSNorm(
            dims=hidden_size,
            rngs=self.rng,
            random_init=self.random_init,
            epsilon=rms_norm_eps,
            dtype=dtype,
            activation_ffw_td=P('data', None),
        )

        self.lm_head = LMhead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dtype=dtype,
            rngs=self.rng,
            vd_sharding=P('model', None),
            dv_sharding=P(None, 'model'),
            prelogit_td=P('data', None),
            random_init=self.random_init,
        )

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        """Loads and transforms all weights from a checkpoint"""
        self.rng = nnx.Rngs(rng)

        # Determine quantization method from HF config (config.json)
        quant_method = (self.hf_config.quantization_config["quant_method"]
                        if hasattr(self.hf_config, "quantization_config") else
                        None)

        # Format: 'hf_key': ('jax_model_path', transform_function, target_shape)
        transforms = {
            "transpose_reshape": lambda w, shape: w.T.reshape(shape),
            "reshape": lambda b, shape: b.reshape(shape),
            "transpose": lambda w, _: w.T,
        }

        mappings = {
            # Embeddings, Norms, and LM Head
            "model.embed_tokens.weight": ("embedder.input_embedding_table_VD",
                                          None, None),
            "lm_head.weight": ("lm_head.input_embedding_table_DV",
                               transforms["transpose"], None),
            "model.norm.weight": ("final_norm.scale", None, None),
            "model.layers.*.input_layernorm.weight":
            ("layers.*.pre_attention_norm.scale", None, None),
            "model.layers.*.post_attention_layernorm.weight":
            ("layers.*.pre_mlp_norm.scale", None, None),

            # Attention Weights
            "model.layers.*.self_attn.q_proj.weight":
            ("layers.*.attn.kernel_q_DNH", transforms["transpose_reshape"],
             (self.hf_config.hidden_size, self.hf_config.num_attention_heads,
              self.hf_config.head_dim)),
            "model.layers.*.self_attn.k_proj.weight":
            ("layers.*.attn.kernel_k_DKH", transforms["transpose_reshape"],
             (self.hf_config.hidden_size, self.hf_config.num_key_value_heads,
              self.hf_config.head_dim)),
            "model.layers.*.self_attn.v_proj.weight":
            ("layers.*.attn.kernel_v_DKH", transforms["transpose_reshape"],
             (self.hf_config.hidden_size, self.hf_config.num_key_value_heads,
              self.hf_config.head_dim)),
            "model.layers.*.self_attn.o_proj.weight":
            ("layers.*.attn.kernel_o_proj_NHD",
             transforms["transpose_reshape"],
             (self.hf_config.num_attention_heads, self.hf_config.head_dim,
              self.hf_config.hidden_size)),

            # Attention Biases
            "model.layers.*.self_attn.q_proj.bias":
            ("layers.*.attn.bias_q_NH", transforms["reshape"],
             (self.hf_config.num_attention_heads, self.hf_config.head_dim)),
            "model.layers.*.self_attn.k_proj.bias":
            ("layers.*.attn.bias_k_KH", transforms["reshape"],
             (self.hf_config.num_key_value_heads, self.hf_config.head_dim)),
            "model.layers.*.self_attn.v_proj.bias":
            ("layers.*.attn.bias_v_KH", transforms["reshape"],
             (self.hf_config.num_key_value_heads, self.hf_config.head_dim)),
            "model.layers.*.self_attn.o_proj.bias": ("layers.*.attn.bias_o_D",
                                                     None, None),

            # Sinks
            "model.layers.*.self_attn.sinks": ("layers.*.attn.sinks_N", None,
                                               None),

            # MoE Weights
            "model.layers.*.mlp.router.weight":
            ("layers.*.custom_module.router.kernel_DE",
             transforms["transpose"], None),
            "model.layers.*.mlp.router.bias":
            ("layers.*.custom_module.router.bias_E", None, None),
        }

        model_params = nnx.state(self)
        is_verbose = self.vllm_config.additional_config.get(
            "is_verbose", False)

        names_and_weights_generator = model_weights_generator(
            model_name_or_path=self.vllm_config.model_config.model,
            framework="pt",
            download_dir=self.vllm_config.load_config.download_dir)

        pool: dict[str, torch.Tensor] = {
            loaded_name: loaded_weight
            for loaded_name, loaded_weight in names_and_weights_generator
        }
        expert_weights_by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for loaded_name, loaded_weight in pool.items():
            layer_num = self._match_expert_layer(loaded_name)
            if layer_num is None:
                continue
            if quant_method != MXFP4:
                raise ValueError(
                    "Only MXFP4 GPT-OSS expert tensors are supported on the "
                    f"flax_nnx path, got quant_method={quant_method!r} for "
                    f"{loaded_name}.")
            expert_weights_by_layer.setdefault(layer_num, []).append(
                (loaded_name, loaded_weight))

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in pool.items():
                if self._match_expert_layer(loaded_name) is not None:
                    continue

                hf_pattern = re.sub(r"layers\.(\d+)", "layers.*", loaded_name)
                if hf_pattern not in mappings:
                    continue

                jax_path_template, transform_fn, target_shape = mappings[
                    hf_pattern]
                layer_num_match = re.search(r"layers\.(\d+)", loaded_name)
                layer_num = layer_num_match.group(
                    1) if layer_num_match else None

                jax_path = jax_path_template.replace(
                    "*", layer_num) if layer_num else jax_path_template
                model_weight = get_param(model_params, jax_path)
                self._load_regular_param(model_weight, loaded_weight,
                                         model_weight.value.dtype,
                                         transform_fn, target_shape,
                                         jax_path_template)

                if is_verbose:
                    print_param_info(model_weight, loaded_name)

        nnx.update(self, model_params)

        for layer_num, expert_weights in expert_weights_by_layer.items():
            experts = self.layers[layer_num].custom_module.experts
            experts.load_weights(expert_weights)
            # process_weights_after_loading returns False until all six staged
            # expert tensors are present.
            processed = experts.quant_method.process_weights_after_loading(
                experts)
            if not processed:
                raise RuntimeError(
                    "Incomplete GPT-OSS MXFP4 expert tensors for layer "
                    f"{layer_num}")

    def _match_expert_layer(self, loaded_name: str) -> Optional[int]:
        match = _EXPERT_TENSOR_RE.fullmatch(loaded_name)
        return int(match.group(1)) if match else None

    def _load_regular_param(self, model_weight, loaded_weight: torch.Tensor,
                            cast_type, transform_fn, target_shape,
                            jax_path_template: str):
        """Assign a regular tensor (non-MXFP4) into the model param with transform applied."""
        if jax_path_template == "layers.*.attn.sinks_N":
            # Checkpoint is bf16, but we have to upcast sinks to f32, as required by RPA_v3 kernel
            weight_np = jnp.array(loaded_weight.to(torch.float32).numpy())
        else:
            torch_view_type = DTYPE_VIEW_MAP.get(jnp.dtype(cast_type))
            if torch_view_type:
                weight_np = jnp.array(
                    loaded_weight.view(torch_view_type).numpy()).view(
                        cast_type)
            else:
                raise ValueError(
                    f"Unsupported dtype for tensor conversion: {cast_type}")

        transformed_weight = transform_fn(
            weight_np, target_shape) if transform_fn else weight_np

        if model_weight.value.shape != transformed_weight.shape:
            raise ValueError(
                f"Shape mismatch: model expects {model_weight.value.shape}, but got {transformed_weight.shape} after transform."
            )

        def get_slice(index):
            return transformed_weight[index]

        sharded_array = jax.make_array_from_callback(
            transformed_weight.shape,
            NamedSharding(self.mesh, P(*model_weight.sharding)), get_slice)
        model_weight.value = sharded_array

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array, List[jax.Array],
               Optional[jax.Array]]:
        is_prefill = False
        x = self.embedder.encode(input_ids)

        all_expert_ids = []
        for i, block in enumerate(self.layers):
            kv_cache = kv_caches[i]
            current_sliding_window = self.sliding_window if i % 2 == 0 else None
            attention_metadata.sliding_window = current_sliding_window

            new_kv_cache, x, expert_ids = block(x, is_prefill, kv_cache,
                                                attention_metadata)
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)

        stacked_expert_ids = jnp.stack(all_expert_ids,
                                       axis=0) if all_expert_ids else None
        return kv_caches, final_activation, [], stacked_expert_ids

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head.decode(hidden_states)
