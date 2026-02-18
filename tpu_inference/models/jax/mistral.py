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

from itertools import islice
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import MistralConfig, modeling_flax_utils
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


class MistralMLP(JaxModule):
    """MLP module for the Mistral architecture.

    Uses SwiGLU activation (gate_proj + up_proj with SiLU activation).
    """

    def __init__(self,
                 config: MistralConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.gate_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )
        self.act_fn = modeling_flax_utils.ACT2FN[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


class MistralAttention(JaxModule):
    """Multi-head attention module for the Mistral architecture.

    Supports Grouped Query Attention (GQA) and optional YaRN RoPE scaling.
    """

    def __init__(self,
                 config: MistralConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.rope_theta = getattr(config, "rope_theta", 1000000.0)

        # YaRN RoPE scaling configuration (if present)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and rope_scaling.get("type") == "yarn":
            self.rope_scaling = rope_scaling
        else:
            self.rope_scaling = None

        self.head_dim_original = getattr(config, "head_dim", None)
        if self.head_dim_original is None:
            self.head_dim_original = self.hidden_size // self.num_heads
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        # Mistral has no bias in attention projections
        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_proj",
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_proj",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
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
        q = apply_rope(q, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # k: (T, K, H)
        k = self.k_proj(x)
        k = apply_rope(k, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # v: (T, K, H)
        v = self.v_proj(x)
        # o: (T, N, H)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
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
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class MistralDecoderLayer(JaxModule):
    """Transformer decoder layer for the Mistral architecture.

    Consists of self-attention and MLP with pre-normalization and residual
    connections.
    """

    def __init__(self,
                 config: MistralConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = MistralAttention(config=config,
                                          dtype=dtype,
                                          rng=rng,
                                          mesh=mesh,
                                          kv_cache_dtype=kv_cache_dtype,
                                          quant_config=quant_config,
                                          prefix=prefix + ".self_attn")
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.mlp = MistralMLP(
            config=config,
            dtype=dtype,
            rng=rng,
            quant_config=quant_config,
            prefix=prefix + ".mlp",
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output += x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs


class MistralModel(JaxModule):
    """The bare Mistral model outputting raw hidden states without a head."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: MistralDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ),
        )
        self.layers = nnx.List(layers)
        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                dtype=dtype,
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
    ) -> Tuple[List[jax.Array], jax.Array]:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class MistralForCausalLM(JaxModule, LoadableWithIterator):
    """Mistral model with a language modeling head for causal generation."""

    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = MistralModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )

        model_config = vllm_config.model_config
        # Create a separate lm_head when embeddings are not tied
        if not model_config.hf_config.tie_word_embeddings:
            vocab_size = model_config.get_vocab_size()
            hidden_size = model_config.hf_config.hidden_size
            self.lm_head = JaxEinsum(
                einsum_str="TD,DV->TV",
                kernel_shape=(hidden_size, vocab_size),
                dtype=model_config.dtype,
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix="lm_head",
            )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            return self.lm_head(hidden_states)
        return self.model.embed_tokens.decode(hidden_states)
