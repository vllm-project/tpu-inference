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

from functools import partial
from itertools import islice
from typing import Any, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma4TextConfig
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (LoadableWithIterator,
                                                         StandardWeightLoader)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


# MLP arch is the same as Gemma3
class Gemma4MLP(JaxModule):

    def __init__(self,
                 config: Gemma4TextConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )
        self.act_fn = partial(nnx.gelu, approximate=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


class Gemma4Attention(JaxModule):

    def __init__(self,
                 config: Gemma4TextConfig,
                 layer_idx: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.rms_norm_eps = config.rms_norm_eps

        # Assuming Gemma 4 also uses a custom scalar, not 1/sqrt(head_dim)
        self.scaling = config.query_pre_attn_scalar**-0.5  # may not be true, need to check with config

        # Same as Gemma3: use layer_idx to handle GLOBAL/LOCAL layer
        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_idx < len(
                config.layer_types):
            self.layer_type = config.layer_types[layer_idx]

        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # Gemma4 use partial rope (0.25) in GLOBAL layer
        rope_parameters = getattr(config, "rope_parameters", {})
        if self.layer_type in rope_parameters:
            # Transformers v5 rope config.
            rope_parameters = rope_parameters[self.layer_type]
            self.rope_theta = rope_parameters.get(
                "rope_theta", getattr(config, "rope_theta", 10000.0))
            self.rope_scaling = rope_parameters.get(
                "rope_scaling", getattr(config, "rope_scaling", None))
            if not self.is_sliding:  # GLOBAL layer
                self.rope_proportion = rope_parameters.get(
                    "partial_rotary_factor",
                    getattr(config, "global_partial_rotary_factor", 0.25))
            else:  # LOCAL layer
                self.rope_proportion = rope_parameters.get(
                    "partial_rotary_factor", 1.0)
        else:
            # Transformers v4 rope config.
            # Fallback for config backward compatibility
            self.rope_theta = config.rope_local_base_freq if self.is_sliding else config.rope_theta
            self.rope_scaling = getattr(config, "rope_scaling", None)
            self.rope_proportion = getattr(
                config, "global_partial_rotary_factor",
                0.25) if not self.is_sliding else 1.0

        # Gemma4: use different num_kv_heads and head_dim in GLOBAL/LOCAL layers
        if not self.is_sliding:
            # GLOBAL layers
            self.num_kv_heads = config.num_global_key_value_heads
            self.head_dim_original = config.global_head_dim
        else:
            # LOCAL layers
            self.num_kv_heads = config.num_key_value_heads
            self.head_dim_original = config.head_dim

        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

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

        # --- Shared KV Projection Logic ---
        if not self.is_sliding:  # Shared KV (GLOBAL layers)
            self.k_proj = JaxEinsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                bias_shape=(self.num_kv_heads,
                            self.head_dim) if config.attention_bias else None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                bias_init=nnx.with_partitioning(init_fn, ("model", None))
                if config.attention_bias else None,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".k_proj",
            )
            self.v_proj = None
        else:
            # Standard separate K/V (LOCAL layers)
            self.k_proj = JaxEinsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                bias_shape=(self.num_kv_heads,
                            self.head_dim) if config.attention_bias else None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                bias_init=nnx.with_partitioning(init_fn, ("model", None))
                if config.attention_bias else None,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".k_proj",
            )
            self.v_proj = JaxEinsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                bias_shape=(self.num_kv_heads,
                            self.head_dim) if config.attention_bias else None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                bias_init=nnx.with_partitioning(init_fn, ("model", None))
                if config.attention_bias else None,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".v_proj",
            )

        self.k_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_norm",
        )
        # Gemma4: Use value norm (need to disable scaling)
        self.v_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(
                init_fn, (None, )),  # need to disable scaling for value norm
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_norm",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            bias_shape=(self.hidden_size, ) if config.attention_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            bias_init=nnx.with_partitioning(init_fn, (None, ))
            if config.attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        # q: (T, N, H)
        q = self.q_proj(x)
        q = self.q_norm(q)

        q = apply_rope(q,
                       md.input_positions,
                       self.head_dim_original,
                       self.rope_theta,
                       self.rope_scaling,
                       rope_proportion=self.rope_proportion)

        # k: (T, K, H)
        # v: (T, K, H)
        if not self.is_sliding:
            shared_kv = self.k_proj(x)
            k = self.k_norm(shared_kv)
            v = self.v_norm(shared_kv)
        else:
            k = self.k_proj(x)
            k = self.k_norm(k)
            v = self.v_proj(x)
            v = self.v_norm(v)

        k = apply_rope(k,
                       md.input_positions,
                       self.head_dim_original,
                       self.rope_theta,
                       self.rope_scaling,
                       rope_proportion=self.rope_proportion)

        # o: (T, N, H)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            # q_scale = self._q_scale
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)
        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            sm_scale=self.scaling,
            attention_chunk_size=self.sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Gemma4DecoderLayer(JaxModule):

    def __init__(self,
                 config: Gemma4TextConfig,
                 layer_idx: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        # Same as Gemma3: use layer_idx to handle GLOBAL/LOCAL layer
        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_idx < len(
                config.layer_types):
            self.layer_type = config.layer_types[layer_idx]

        self.is_sliding = self.layer_type == "sliding_attention"

        if not self.is_sliding:
            self.skip_scale = nnx.Param(jnp.ones(
                (1, ),
                dtype=dtype))  # placeholder for learnable skip_scale value
        else:
            self.skip_scale = None

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Gemma4Attention(config=config,
                                         layer_idx=layer_idx,
                                         dtype=dtype,
                                         rng=rng,
                                         mesh=mesh,
                                         kv_cache_dtype=kv_cache_dtype,
                                         quant_config=quant_config,
                                         prefix=prefix + ".self_attn")
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.pre_feedforward_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
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
            prefix=prefix + ".mlp",
        )
        self.post_feedforward_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
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
    ) -> Tuple[jax.Array, jax.Array]:
        residual = x
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        residual = residual + attn_output

        hidden_states = self.pre_feedforward_layernorm(residual)
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        outputs = residual + mlp_output

        if self.skip_scale is not None:
            outputs = outputs * self.skip_scale.value

        return kv_cache, outputs


class Gemma4Model(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        text_config = hf_config.text_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = text_config.rms_norm_eps
        hidden_size = text_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        # Gemma 4: Embeddings are scaled by sqrt(hidden_size)
        self.embedding_scale = hidden_size**0.5

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda layer_index: Gemma4DecoderLayer(
                config=text_config,
                layer_idx=layer_index,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        layer_name_to_kv_cache: Optional[dict] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:

        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
            # Gemma4: Apply embedding scaling
            x = x * self.embedding_scale

        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            layer_name = f"layer.{i + self.start_layer}"
            if isinstance(attention_metadata, dict):
                layer_attn_metadata = attention_metadata[layer_name]
            else:
                layer_attn_metadata = attention_metadata

            if layer_name_to_kv_cache and layer_name in layer_name_to_kv_cache:
                cache_idx = layer_name_to_kv_cache[layer_name]
            else:
                cache_idx = i + self.start_layer

            kv_cache = kv_caches[cache_idx]
            kv_cache, x = layer(
                kv_cache,
                x,
                layer_attn_metadata,
            )
            kv_caches[cache_idx] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class Gemma4ForCausalLM(JaxModule, LoadableWithIterator):
    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Gemma4Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config

        # Gemma 4: soft-capping in the final logits.
        self.final_logit_softcapping = getattr(
            model_config.hf_config.text_config, "final_logit_softcapping",
            None)

        if not model_config.hf_config.tie_word_embeddings:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                hidden_size = model_config.hf_config.text_config.hidden_size
                self.lm_head = JaxEinsum(
                    einsum_str="TD,DV->TV",
                    kernel_shape=(hidden_size, vocab_size),
                    dtype=model_config.dtype,
                    rngs=rng,
                    quant_config=vllm_config.quant_config,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[Tuple[str, Any]]):
        stripped_weights = (
            (clean_name, 1.0 + tensor if "norm" in clean_name else tensor)
            for name, tensor in weights
            if (clean_name := name.replace("language_model.", "")).startswith((
                "model.", "lm_head")))

        return super().load_weights(stripped_weights)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array]]:

        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        layer_name_to_kv_cache = dict(
            _layer_name_to_kv_cache) if _layer_name_to_kv_cache else None
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            layer_name_to_kv_cache=layer_name_to_kv_cache,
        )

        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )

        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.decode(hidden_states)

        # Gemma4: Use Logit Soft-capping
        if self.final_logit_softcapping is not None:
            logits = jnp.tanh(
                logits /
                self.final_logit_softcapping) * self.final_logit_softcapping
        return logits
