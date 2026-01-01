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
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.attention.llama4_attention import Llama4Attention
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import DenseFFW, Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.misc import shard_put
from tpu_inference.layers.jax.transformer_block import TransformerBlock
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    get_param, model_weights_generator, print_param_info, reshape_params,
    transpose_params)

logger = init_logger(__name__)


class LlamaGuard4ForCausalLM(nnx.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        logger.warning(
            "🚨🚨🚨WARNING🚨🚨🚨 🚨🚨🚨WARNING🚨🚨🚨 🚨🚨🚨WARNING🚨🚨🚨\n"
            "Llama Guard 4 (JAX) is WIP: Only the text modality is currently implemented.  "
            "Multimodal inputs will fail.\n"
            "🚨🚨🚨WARNING🚨🚨🚨 🚨🚨🚨WARNING🚨🚨🚨 🚨🚨🚨WARNING🚨🚨🚨")
        assert mesh is not None

        self.vllm_config = vllm_config
        self.vllm_config.model_config.dtype = torch.bfloat16
        model_config = vllm_config.model_config
        text_config = model_config.hf_config.text_config

        self.mesh = mesh
        self.is_verbose = getattr(self.vllm_config.additional_config,
                                  "is_verbose", False)

        self.use_qk_norm = getattr(text_config, "use_qk_norm", True)

        vocab_size = model_config.get_vocab_size()
        self.hidden_size = model_config.get_hidden_size()

        self.dtype: jnp.dtype = jnp.bfloat16

        self.num_layers: int = getattr(text_config, "num_layers", 48)
        hidden_act: str = getattr(text_config, "hidden_act", "silu")

        rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-5)
        self.num_attention_heads = getattr(text_config, "num_attention_heads",
                                           40)
        self.num_key_value_heads = getattr(text_config, "num_key_value_heads",
                                           8)
        self.head_dim = getattr(text_config, "head_dim", 128)

        intermediate_size = getattr(text_config, "intermediate_size", 8192)

        self.rope_theta_text = getattr(text_config, "rope_theta", 500000.0)
        self.rope_scaling = getattr(text_config, "rope_scaling")

        self.rng = nnx.Rngs(rng)

        self.embedder = Embedder(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            vd_sharding=(('data', 'model'), None),
            rngs=self.rng,
            random_init=force_random_weights,
        )

        self.layers = []

        for i in range(self.num_layers):
            use_attention_rope = True

            custom_module = DenseFFW(dtype=self.dtype,
                                     hidden_act=hidden_act,
                                     hidden_size=self.hidden_size,
                                     intermediate_size=intermediate_size,
                                     random_init=force_random_weights,
                                     rngs=self.rng,
                                     df_sharding=P(None, 'model'),
                                     fd_sharding=P('model', None),
                                     activation_ffw_td=P('data', None))

            attn = Llama4Attention(
                hidden_size=self.hidden_size,
                dtype=self.dtype,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                rope_theta=self.rope_theta_text,
                rope_scaling={
                    "scale_factor":
                    self.rope_scaling["factor"],
                    "low_freq_factor":
                    self.rope_scaling["low_freq_factor"],
                    "high_freq_factor":
                    self.rope_scaling["high_freq_factor"],
                    "original_max_position_embeddings":
                    self.rope_scaling["original_max_position_embeddings"]
                },
                rngs=self.rng,
                rope_input_ordering="interleaved",
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                temperature_tuning=True,
                temperature_tuning_scale=0.1,
                temperature_tuning_floor_scale=8192,
                use_qk_norm=self.use_qk_norm,
                attention_chunk_size=None if use_attention_rope else 8192,
                mesh=self.mesh,
                random_init=force_random_weights,
                activation_attention_td=('data', 'model'),
                activation_q_td=('data', 'model'),
                query_tnh=P('data', 'model', None),
                keyvalue_skh=P('data', 'model', None),
                activation_attention_out_td=('data', 'model'),
                attn_o_tnh=P('data', 'model', None),
                dnh_sharding=(None, 'model', None),
                dkh_sharding=(None, 'model', None),
                nhd_sharding=('model', None, None),
            )

            pre_attention_norm = RMSNorm(
                dims=self.hidden_size,
                random_init=force_random_weights,
                epsilon=rms_norm_eps,
                rngs=self.rng,
                activation_ffw_td=('data', None),
                with_scale=True,
                dtype=self.dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=self.hidden_size,
                activation_ffw_td=('data', None),
                epsilon=rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=self.dtype,
                random_init=force_random_weights,
            )

            block = TransformerBlock(custom_module=custom_module,
                                     attn=attn,
                                     pre_attention_norm=pre_attention_norm,
                                     pre_mlp_norm=pre_mlp_norm,
                                     use_attention_rope=use_attention_rope)
            self.layers.append(block)

        self.final_norm = RMSNorm(
            dims=self.hidden_size,
            activation_ffw_td=P(),
            epsilon=rms_norm_eps,
            rngs=self.rng,
            with_scale=True,
            dtype=self.dtype,
            random_init=force_random_weights,
        )

        self.lm_head = LMhead(vocab_size=vocab_size,
                              hidden_size=self.hidden_size,
                              dtype=self.dtype,
                              rngs=self.rng,
                              vd_sharding=(('data', 'model'), None),
                              dv_sharding=(None, ('data', 'model')),
                              random_init=force_random_weights)
        if self.is_verbose:
            self._print_model_architecture()

    def _print_model_architecture(self):

        logger.info("### Embedding ###")
        nnx.display(self.embedder)

        logger.info("\n### Layers ###")
        for i, layer in enumerate(self.layers):
            logger.info(f"\n--- Layer {i} ---")
            nnx.display(layer)

        logger.info("\n### LM Head ###")
        nnx.display(self.lm_head)

    def load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None):
        self.rng = nnx.Rngs(rng)

        weight_loader = LlamaGuard4WeightLoader(
            vllm_config=self.vllm_config,
            hidden_size=self.hidden_size,
            attn_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            attn_head_dim=self.head_dim)
        weight_loader.load_weights(self)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        layer_metadata_tuple: Optional[Tuple] = None,
        lora_metadata: Optional[Any] = None,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False

        if inputs_embeds is not None:
            x_TD = inputs_embeds
        elif input_ids is not None:
            x_TD = self.embedder.encode(input_ids)
        else:
            raise ValueError(
                "Cannot run forward pass: Both input_ids and inputs_embeds are None."
            )

        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x_TD = block(x_TD, is_prefill, kv_cache,
                                       attention_metadata)
            jax.block_until_ready(x_TD)
            kv_caches[i] = new_kv_cache

        final_activation_TD = self.final_norm(x_TD)

        return kv_caches, final_activation_TD, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits_TV = jnp.dot(hidden_states,
                            self.lm_head.input_embedding_table_DV.value)
        return logits_TV

    def embed_input_ids(
            self,
            input_ids: jax.Array,
            multimodal_embeddings: Optional[List[jax.Array]] = None
    ) -> jax.Array:
        """
        Computes the embeddings for text input (used for input to fusion).
        """
        return self.embedder.encode(input_ids)


class LlamaGuard4WeightLoader:

    def __init__(self, vllm_config: VllmConfig, hidden_size, attn_heads,
                 num_key_value_heads, attn_head_dim):
        self.names_and_weights_generator = model_weights_generator(
            model_name_or_path=vllm_config.model_config.model,
            framework="flax",
            filter_regex="language_model",
            download_dir=vllm_config.load_config.download_dir)
        self.is_verbose = getattr(vllm_config.additional_config, "is_verbose",
                                  False)
        self._transpose_map = {
            "q_proj": (2, 0, 1),
            "k_proj": (2, 0, 1),
            "v_proj": (2, 0, 1),
            "o_proj": (1, 2, 0),
            "lm_head": (1, 0),
            "feed_forward.down_proj": (1, 0),
            "feed_forward.gate_proj": (1, 0),
            "feed_forward.up_proj": (1, 0),
            "mlp.down_proj": (1, 0),
            "mlp.gate_proj": (1, 0),
            "mlp.up_proj": (1, 0),
        }
        self._weight_shape_map = {
            "q_proj": (attn_heads, attn_head_dim, hidden_size),
            "k_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "v_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "o_proj": (hidden_size, attn_heads, attn_head_dim),
        }

        self._loaded_to_standardized_keys = {
            "language_model.model.embed_tokens.weight":
            "embedder.input_embedding_table_VD",
            "language_model.lm_head.weight":
            "lm_head.input_embedding_table_DV",
            "language_model.model.norm.weight":
            "final_norm.scale",
            "language_model.model.layers.*.input_layernorm.weight":
            "layers.*.pre_attention_norm.scale",
            "language_model.model.layers.*.post_attention_layernorm.weight":
            "layers.*.pre_mlp_norm.scale",
            "language_model.model.layers.*.self_attn.q_proj.weight":
            "layers.*.attn.kernel_q_proj_DNH",
            "language_model.model.layers.*.self_attn.k_proj.weight":
            "layers.*.attn.kernel_k_proj_DKH",
            "language_model.model.layers.*.self_attn.v_proj.weight":
            "layers.*.attn.kernel_v_proj_DKH",
            "language_model.model.layers.*.self_attn.o_proj.weight":
            "layers.*.attn.kernel_o_proj_NHD",
            "language_model.model.layers.*.feed_forward.gate_proj.weight":
            "layers.*.custom_module.kernel_gating_DF",
            "language_model.model.layers.*.feed_forward.up_proj.weight":
            "layers.*.custom_module.kernel_up_proj_DF",
            "language_model.model.layers.*.feed_forward.down_proj.weight":
            "layers.*.custom_module.kernel_down_proj_FD",
        }

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        if "layer" in loaded_key:
            layer_num = re.search(r"layers\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
            mapped_key = self._loaded_to_standardized_keys.get(
                layer_key, loaded_key)
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)
        else:
            mapped_key = self._loaded_to_standardized_keys.get(
                loaded_key, loaded_key)
        return mapped_key

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                if loaded_name.endswith(".bias"):
                    continue
                if "vision_model" in loaded_name or "multi_modal_projector" in loaded_name:
                    continue

                mapped_name = self.map_loaded_to_standardized_name(loaded_name)
                model_weight = get_param(model_params, mapped_name)

                if not loaded_name.endswith(".bias"):
                    # For other layers, continue to use the transpose_params helper.
                    loaded_weight = reshape_params(loaded_name, loaded_weight,
                                                   self._weight_shape_map)
                    loaded_weight = transpose_params(loaded_name,
                                                     loaded_weight,
                                                     self._transpose_map)
                if model_weight.value.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                        f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                    )
                logger.debug(
                    f"Transformed parameter {loaded_name} to {mapped_name}: {loaded_weight.shape} --> {model_weight.value.shape}"
                )

                model_weight.value = shard_put(loaded_weight,
                                               model_weight.sharding,
                                               mesh=model_for_loading.mesh)
                if self.is_verbose:
                    print_param_info(model_weight, loaded_name)

        nnx.update(model_for_loading, model_params)
