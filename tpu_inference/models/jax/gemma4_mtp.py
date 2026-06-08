# Copyright 2026 Google LLC
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

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma4TextConfig
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear, JaxLmHead
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.gemma4 import Gemma4MLP
from tpu_inference.models.jax.utils.weight_utils import (LoadableWithIterator,
                                                         StandardWeightLoader)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class Gemma4MTPMaskedEmbedder(JaxModule):
    """Sparse logit computation via centroid-based vocabulary masking in JAX.

    Projects hidden states to centroid scores, selects the top centroids, and
    computes logits only for the subset of tokens belonging to those centroids.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        centroid_intermediate_top_k: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = centroid_intermediate_top_k * self.vocab_size_per_centroid

        self.centroids = JaxLinear(
            hidden_size,
            num_centroids,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
            prefix=prefix + ".centroids",
        )
        # token_ordering is loaded as a buffer/weight
        self.token_ordering = nnx.Param(
            jnp.zeros((vocab_size, ), dtype=jnp.int32),
            eager_sharding=False,
        )

    def _select_and_score(
        self,
        hidden_states: jax.Array,
        lm_head_weight: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Centroid selection + sparse dot product."""
        num_tokens = hidden_states.shape[0]

        # Handle transposed lm_head_weight if necessary.
        # lm_head.weight is (hidden_size, vocab_size) due to weight loader permutation.
        # embed_tokens.weight is (vocab_size, hidden_size).
        if lm_head_weight.shape == (self.hidden_size, self.vocab_size):
            lm_head_weight = lm_head_weight.T
        elif lm_head_weight.shape != (self.vocab_size, self.hidden_size):
            raise ValueError(
                f"Unexpected lm_head_weight shape: {lm_head_weight.shape}, "
                f"expected ({self.hidden_size}, {self.vocab_size}) or ({self.vocab_size}, {self.hidden_size})"
            )

        centroid_logits = self.centroids(
            hidden_states)  # (num_tokens, num_centroids)
        _, top_k_indices = jax.lax.top_k(
            centroid_logits,
            k=self.centroid_intermediate_top_k)  # (num_tokens, top_k)

        clusters = self.token_ordering.value.reshape(
            self.num_centroids, self.vocab_size_per_centroid)
        selected = clusters[
            top_k_indices]  # (num_tokens, top_k, vocab_size_per_centroid)
        selected_flat = selected.reshape(num_tokens, self.num_selected)

        embeddings = jnp.take(
            lm_head_weight, selected_flat,
            axis=0)  # (num_tokens, num_selected, hidden_size)
        logits = jnp.einsum("td,tsd->ts", hidden_states, embeddings)
        return logits, selected_flat

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head_weight: jax.Array,
    ) -> jax.Array:
        """Full-vocab logits with non-selected positions masked to -inf."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        num_tokens = hidden_states.shape[0]
        output = jnp.full(
            (num_tokens, self.vocab_size),
            jnp.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
        )
        row_indices = jnp.arange(num_tokens)[:, None]
        output = output.at[row_indices, indices].set(logits)
        return output

    def get_top_tokens(
        self,
        hidden_states: jax.Array,
        lm_head_weight: jax.Array,
    ) -> jax.Array:
        """Sparse argmax — returns vocab token IDs directly."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        best_idx = jnp.argmax(logits, axis=-1, keepdims=True)
        return jnp.take_along_axis(indices, best_idx, axis=-1).squeeze(-1)


class Gemma4MTPAttention(JaxModule):
    """Q-only attention for Gemma4 MTP layers.

    K/V come from the target model's KV cache, queried at runtime by passing
    update_kv_cache=False.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
        quant_config: VllmQuantConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.rms_norm_eps = config.rms_norm_eps

        self.scaling = 1.0

        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_idx < len(
                config.layer_types):
            self.layer_type = config.layer_types[layer_idx]

        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        rope_parameters = getattr(config, "rope_parameters", {})
        if self.layer_type in rope_parameters:
            rope_parameters = rope_parameters[self.layer_type]
            self.rope_theta = rope_parameters.get(
                "rope_theta", getattr(config, "rope_theta", 10000.0))
            self.rope_scaling = rope_parameters.get(
                "rope_scaling", getattr(config, "rope_scaling", None))
            self.rope_proportion = rope_parameters.get("partial_rotary_factor",
                                                       1.0)
        else:
            self.rope_theta = (getattr(config, "rope_local_base_freq", 10000.0)
                               if self.is_sliding else config.rope_theta)
            self.rope_scaling = getattr(config, "rope_scaling", None)
            self.rope_proportion = 0.25 if not self.is_sliding else 1.0

        if not self.is_sliding:
            self.head_dim_original = config.global_head_dim
        else:
            self.head_dim_original = config.head_dim

        use_k_eq_v = (not self.is_sliding) and getattr(
            config, "attention_k_eq_v", False)
        if use_k_eq_v:
            self.num_kv_heads = (config.num_global_key_value_heads
                                 or config.num_key_value_heads)
        else:
            self.num_kv_heads = config.num_key_value_heads

        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)
        self.mesh = mesh

        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            bias_shape=(self.num_heads,
                        self.head_dim) if config.attention_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, ("model", None))
            if config.attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.q_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_norm",
        )

        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            bias_shape=(self.hidden_size, ) if config.attention_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, ))
            if config.attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self.is_kv_shared_layer = True
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        q = self.q_proj(x)
        q = self.q_norm(q)

        q = apply_rope(
            q,
            md.input_positions,
            self.head_dim_original,
            self.rope_theta,
            self.rope_scaling,
            rope_proportion=self.rope_proportion,
        )

        num_tokens = q.shape[0]
        dummy_dtype = self.kv_cache_quantized_dtype or q.dtype
        dummy_k = jnp.zeros((num_tokens, self.num_kv_heads, self.head_dim),
                            dtype=dummy_dtype)
        dummy_v = jnp.zeros((num_tokens, self.num_kv_heads, self.head_dim),
                            dtype=dummy_dtype)

        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            dummy_k,
            dummy_v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            sm_scale=self.scaling,
            attention_chunk_size=self.sliding_window,
            update_kv_cache=False,  # Read-only shared cache query
        )
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Gemma4MTPDecoderLayer(JaxModule):

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
        quant_config: VllmQuantConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_idx < len(
                config.layer_types):
            self.layer_type = config.layer_types[layer_idx]

        self.is_sliding = self.layer_type == "sliding_attention"
        self.layer_scalar = nnx.Param(jnp.ones((1, ), dtype=dtype))

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Gemma4MTPAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
            quant_config=quant_config,
            prefix=prefix + ".self_attn",
        )
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.pre_feedforward_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".pre_feedforward_layernorm",
        )
        self.mlp = Gemma4MLP(
            config=config,
            dtype=dtype,
            rng=rng,
            quant_config=quant_config,
            intermediate_size=config.intermediate_size,
            prefix=prefix + ".mlp",
        )
        self.post_feedforward_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_feedforward_layernorm",
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        residual = x
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(residual)
        hidden_states = self.mlp(hidden_states)

        mlp_output = self.post_feedforward_layernorm(hidden_states)
        outputs = residual + mlp_output

        outputs = outputs * self.layer_scalar.value

        return kv_cache, outputs, None


class Gemma4MultiTokenPredictor(JaxModule):

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
        prefix: str = "",
    ) -> None:
        super().__init__()

        draft_config = vllm_config.speculative_config.draft_model_config.hf_config
        text_config = draft_config.text_config
        self.config = text_config
        dtype = vllm_config.model_config.dtype

        self.hidden_size = text_config.hidden_size
        self.backbone_hidden_size = getattr(draft_config,
                                            "backbone_hidden_size",
                                            self.hidden_size)
        self.vocab_size = text_config.vocab_size
        self.num_mtp_layers = text_config.num_hidden_layers

        self.embed_tokens = JaxEmbed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=vllm_config.quant_config,
            prefix=prefix + ".embed_tokens",
        )

        self.pre_projection = JaxLinear(
            2 * self.backbone_hidden_size,
            self.hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rng,
            quant_config=vllm_config.quant_config,
            prefix=prefix + ".pre_projection",
        )

        self.post_projection = JaxLinear(
            self.hidden_size,
            self.backbone_hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rng,
            quant_config=vllm_config.quant_config,
            prefix=prefix + ".post_projection",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.num_mtp_layers,
            lambda layer_index: Gemma4MTPDecoderLayer(
                config=self.config,
                layer_idx=layer_index,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ),
        )

        self.norm = JaxRmsNorm(
            self.hidden_size,
            epsilon=text_config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=vllm_config.quant_config,
            prefix=prefix + ".norm",
        )

        self.normalizer = self.backbone_hidden_size**0.5

    def embed_input_ids(self, input_ids: jax.Array) -> jax.Array:
        return self.embed_tokens(input_ids) * self.normalizer

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
        layer_name_to_kv_cache: Optional[dict] = None,
    ) -> Tuple[List[jax.Array], jax.Array, jax.Array]:
        inputs_embeds = self.embed_input_ids(input_ids)

        combined = jnp.concatenate([inputs_embeds, hidden_states], axis=-1)
        hidden_states = self.pre_projection(combined)

        for i, layer in enumerate(self.layers):
            layer_name = f"draft_layer.{i}"
            if layer_name_to_kv_cache and layer_name in layer_name_to_kv_cache:
                cache_idx = layer_name_to_kv_cache[layer_name]
            else:
                cache_idx = i

            kv_cache = kv_caches[cache_idx]
            kv_cache, hidden_states, _ = layer(
                kv_cache,
                hidden_states,
                attention_metadata,
            )
            kv_caches[cache_idx] = kv_cache

        draft_hidden_states = self.norm(hidden_states)
        backbone_hidden_states = self.post_projection(draft_hidden_states)

        return kv_caches, draft_hidden_states, backbone_hidden_states


class Gemma4MTPForCausalLM(JaxModule, LoadableWithIterator):
    WeightLoader = StandardWeightLoader

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh,
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Gemma4MultiTokenPredictor(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )

        draft_config = vllm_config.speculative_config.draft_model_config.hf_config
        text_config = draft_config.text_config
        dtype = vllm_config.model_config.dtype

        self.final_logit_softcapping = getattr(text_config,
                                               "final_logit_softcapping", None)

        self.lm_head = JaxLmHead(
            text_config.hidden_size,
            text_config.vocab_size,
            rngs=rng,
            dtype=dtype,
            prefix="lm_head",
        )

        if getattr(draft_config, "use_ordered_embeddings", False):
            num_centroids = getattr(draft_config, "num_centroids", 2048)
            top_k = getattr(draft_config, "centroid_intermediate_top_k", 32)
            self.masked_embedding = Gemma4MTPMaskedEmbedder(
                hidden_size=text_config.hidden_size,
                vocab_size=text_config.vocab_size,
                num_centroids=num_centroids,
                centroid_intermediate_top_k=top_k,
                dtype=dtype,
                rngs=rng,
                prefix="masked_embedding",
            )
        else:
            self.masked_embedding = None

    def load_weights(self, weights: Iterable[Tuple[str, Any]]):
        allowed_layers = set(f"layers.{i}."
                             for i in range(len(self.model.layers)))

        def clean_and_map(w_iter):
            loaded_keys = set()
            for name, tensor in w_iter:
                clean_name = name.replace("language_model.", "")
                if clean_name.startswith("mtp."):
                    clean_name = clean_name.replace("mtp.", "model.")
                if clean_name.startswith("pre_projection."):
                    clean_name = "model." + clean_name
                elif clean_name.startswith("post_projection."):
                    clean_name = "model." + clean_name

                if clean_name.startswith(
                    ("model.", "lm_head", "masked_embedding")):
                    loaded_keys.add(clean_name)
                    yield clean_name, tensor
                    if clean_name == "model.embed_tokens.weight" and getattr(
                            self.vllm_config.speculative_config.
                            draft_model_config.hf_config,
                            "tie_word_embeddings", True):
                        loaded_keys.add("lm_head.weight")
                        yield "lm_head.weight", tensor

            if getattr(self, "masked_embedding", None) is not None:
                for key in [
                        "masked_embedding.token_ordering",
                        "masked_embedding.centroids.weight"
                ]:
                    if key not in loaded_keys:
                        raise ValueError(
                            f"Ordered embeddings masking is enabled at runtime, but the "
                            f"required parameter '{key}' is missing from the loaded checkpoint. "
                            f"Please use a draft checkpoint that was trained with centroids masking."
                        )

        mapped_weights = clean_and_map(weights)

        filtered_weights = ((name, tensor) for name, tensor in mapped_weights
                            if not ("layers." in name and not any(
                                layer_prefix in name
                                for layer_prefix in allowed_layers)))

        from tpu_inference.models.jax.utils.weight_utils import \
            JaxAutoWeightsLoader

        pytorch_pooler = getattr(getattr(self, "vllm_config", None),
                                 "pytorch_pooler", None)

        loader = JaxAutoWeightsLoader(
            self,
            pytorch_pooler=pytorch_pooler,
            skip_prefixes=(["lm_head"]
                           if not hasattr(self, 'lm_head') else None),
            ignore_unexpected_prefixes=[
                "model.embed_vision", "model.vision_tower"
            ],
        )
        return loader.load_weights(filtered_weights)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
        layer_name_to_kvcache_index: Optional[Sequence[Tuple[str,
                                                             int]]] = None,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array],
               Optional[jax.Array]]:
        layer_name_to_kv_cache = (dict(layer_name_to_kvcache_index)
                                  if layer_name_to_kvcache_index else None)

        kv_caches, draft_hidden_states, backbone_hidden_states = self.model(
            kv_caches,
            input_ids,
            hidden_states,
            attention_metadata,
            layer_name_to_kv_cache=layer_name_to_kv_cache,
        )

        return kv_caches, draft_hidden_states, [backbone_hidden_states], None

    def _get_full_lm_head_weight(self) -> jax.Array:
        if hasattr(self, "lm_head"):
            return self.lm_head.weight.value
        else:
            return self.model.embed_tokens.embedding.value

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.masked_embedding is not None:
            return self.masked_embedding(
                hidden_states,
                self._get_full_lm_head_weight(),
            )

        if hasattr(self, "lm_head"):
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.decode(hidden_states)

        if self.final_logit_softcapping is not None:
            logits = (jnp.tanh(logits / self.final_logit_softcapping) *
                      self.final_logit_softcapping)
        return logits

    def get_top_tokens(self, hidden_states: jax.Array) -> jax.Array:
        if self.masked_embedding is not None:
            return self.masked_embedding.get_top_tokens(
                hidden_states,
                self._get_full_lm_head_weight(),
            )
        return jnp.argmax(self.compute_logits(hidden_states), axis=-1)
