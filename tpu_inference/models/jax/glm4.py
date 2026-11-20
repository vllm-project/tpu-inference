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
"""GLM-4 dense (``Glm4ForCausalLM``, e.g. ``zai-org/GLM-4-9B-0414``).

Differences from Qwen2:
- two extra "sandwich" RMSNorms per layer (``post_self_attn_layernorm`` after
  attention and ``post_mlp_layernorm`` after the MLP),
- fused ``gate_up_proj`` checkpoint weight,
- partial (``partial_rotary_factor``) RoPE with interleaved (GPT-J style)
  pairs; the remaining channels of each head are not rotated.
"""

from itertools import islice
from typing import Any, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.linear import (JaxEinsum, JaxLinear, JaxLmHead,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import (apply_rope,
                                                     get_rope_scaling,
                                                     get_rope_theta)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (LoadableWithIterator,
                                                         StandardWeightLoader)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()
modeling_flax_utils = FlaxUtils()


def _get_partial_rotary_factor(config: PretrainedConfig) -> float:
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if "partial_rotary_factor" in rope_parameters:
        return float(rope_parameters["partial_rotary_factor"])
    return float(getattr(config, "partial_rotary_factor", 0.5))


class Glm4MLP(JaxModule):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_up_proj = JaxMergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_up_proj",
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
        self.act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate_up = self.gate_up_proj(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Glm4Attention(JaxModule):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = get_rope_theta(config, default=10000.0)
        self.rope_scaling = get_rope_scaling(config)

        self.head_dim_original = getattr(
            config, "head_dim", None) or (self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)
        # GLM-4 only rotates the first `rotary_dim` channels of each head.
        self.rotary_dim = int(self.head_dim_original *
                              _get_partial_rotary_factor(config))
        attention_bias = getattr(config, "attention_bias", True)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        def _proj(name, num_heads):
            return JaxEinsum(
                "TD,DNH->TNH",
                (self.hidden_size, num_heads, self.head_dim),
                bias_shape=(num_heads,
                            self.head_dim) if attention_bias else None,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                bias_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + "." + name,
            )

        self.q_proj = _proj("q_proj", self.num_heads)
        self.k_proj = _proj("k_proj", self.num_kv_heads)
        self.v_proj = _proj("v_proj", self.num_kv_heads)
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

    def _apply_partial_rope(self, x: jax.Array,
                            positions: jax.Array) -> jax.Array:
        x_rot = apply_rope(x[..., :self.rotary_dim],
                           positions,
                           self.rotary_dim,
                           self.rope_theta,
                           self.rope_scaling,
                           rope_input_ordering="interleaved")
        return jnp.concatenate([x_rot, x[..., self.rotary_dim:]], axis=-1)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        # q: (T, N, H)
        q = self._apply_partial_rope(self.q_proj(x), md.input_positions)
        # k: (T, K, H)
        k = self._apply_partial_rope(self.k_proj(x), md.input_positions)
        # v: (T, K, H)
        v = self.v_proj(x)
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


class Glm4DecoderLayer(JaxModule):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):

        def _norm(name):
            return JaxRmsNorm(
                config.hidden_size,
                epsilon=config.rms_norm_eps,
                dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + "." + name,
            )

        self.input_layernorm = _norm("input_layernorm")
        self.self_attn = Glm4Attention(config=config,
                                       dtype=dtype,
                                       rng=rng,
                                       mesh=mesh,
                                       kv_cache_dtype=kv_cache_dtype,
                                       quant_config=quant_config,
                                       prefix=prefix + ".self_attn")
        self.post_self_attn_layernorm = _norm("post_self_attn_layernorm")
        self.post_attention_layernorm = _norm("post_attention_layernorm")
        self.mlp = Glm4MLP(
            config=config,
            dtype=dtype,
            rng=rng,
            quant_config=quant_config,
            prefix=prefix + ".mlp",
        )
        self.post_mlp_layernorm = _norm("post_mlp_layernorm")

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = x
        hidden_states = self.input_layernorm(x)
        kv_cache, hidden_states = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return kv_cache, hidden_states


class Glm4Model(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
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

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Glm4DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ),
        )
        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=hf_config.rms_norm_eps,
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


class Glm4ForCausalLM(JaxModule, LoadableWithIterator):
    packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Glm4Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config
        if not model_config.hf_config.tie_word_embeddings:
            if self.model.is_last_rank:
                self.lm_head = JaxLmHead(
                    hidden_size=model_config.hf_config.hidden_size,
                    vocab_size=model_config.get_vocab_size(),
                    dtype=model_config.dtype,
                    rngs=rng,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[Tuple[str, Any]]) -> set[str]:
        # GLM-4 checkpoints store the MLP as a single fused `gate_up_proj`
        # tensor. Split it into the `gate_proj` / `up_proj` shards so the
        # merged linear's quant method can fuse (and shard) them per device.
        def _split_gate_up(weights):
            for name, weight in weights:
                if "mlp.gate_up_proj." in name:
                    gate, up = weight.chunk(2, dim=0)
                    yield name.replace("gate_up_proj", "gate_proj"), gate
                    yield name.replace("gate_up_proj", "up_proj"), up
                else:
                    yield name, weight

        return super().load_weights(_split_gate_up(weights))

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
               List[jax.Array], Optional[jax.Array]]:
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
        return kv_caches, x, [], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            return self.lm_head(hidden_states)
        return self.model.embed_tokens.decode(hidden_states)
