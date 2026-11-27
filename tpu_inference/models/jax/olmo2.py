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

from collections.abc import Iterable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from torch import Tensor
from transformers import Olmo2Config
from vllm.config import VllmConfig

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxLmHead
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.models.jax.qwen2 import (Qwen2ForCausalLM, Qwen2MLP,
                                            Qwen2Model)
from tpu_inference.models.jax.qwen3 import Qwen3Attention
from tpu_inference.utils import to_torch_dtype

init_fn = nnx.initializers.uniform()


class Olmo2QKNorm(JaxRmsNorm):
    """OLMo 2 normalizes a token's complete projection, not each head."""

    def __call__(self, x: jax.Array) -> jax.Array:
        shape = x.shape
        return super().__call__(x.reshape(shape[0], -1)).reshape(shape)


class Olmo2Attention(Qwen3Attention):

    def __init__(self,
                 config: Olmo2Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        if config.attention_bias:
            raise ValueError("OLMo 2 attention biases are not supported")
        super().__init__(config, dtype, rng, mesh, kv_cache_dtype,
                         quant_config, prefix)
        if (self.head_dim != self.head_dim_original
                or self.num_heads != config.num_attention_heads
                or self.num_kv_heads != config.num_key_value_heads):
            raise ValueError("OLMo 2 does not support padded attention heads")
        self.q_norm = Olmo2QKNorm(
            self.num_heads * self.head_dim,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_norm",
        )
        self.k_norm = Olmo2QKNorm(
            self.num_kv_heads * self.head_dim,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_norm",
        )


class Olmo2DecoderLayer(JaxModule):

    def __init__(self,
                 config: Olmo2Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.self_attn = Olmo2Attention(config, dtype, rng, mesh,
                                        kv_cache_dtype, quant_config,
                                        prefix + ".self_attn")
        self.mlp = Qwen2MLP(config, dtype, rng, quant_config, prefix + ".mlp")
        self.post_attention_layernorm = JaxRmsNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.post_feedforward_layernorm = JaxRmsNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_feedforward_layernorm",
        )

    def __call__(
            self, kv_cache: jax.Array, x: jax.Array,
            attention_metadata: AttentionMetadata
    ) -> tuple[jax.Array, jax.Array]:
        kv_cache, attn_output = self.self_attn(kv_cache, x, attention_metadata)
        x = x + self.post_attention_layernorm(attn_output)
        return kv_cache, x + self.post_feedforward_layernorm(self.mlp(x))


class Olmo2Model(Qwen2Model):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model"):
        model_config = vllm_config.model_config
        config = model_config.hf_config
        dtype = model_config.dtype
        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank
        if self.is_first_rank or (config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=model_config.get_vocab_size(),
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, lambda index: Olmo2DecoderLayer(
                config, dtype, rng, mesh, vllm_config.cache_config.cache_dtype,
                vllm_config.quant_config, f"{prefix}.layers.{index}"))
        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                config.hidden_size,
                epsilon=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()


class Olmo2ForCausalLM(Qwen2ForCausalLM):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh):
        self.vllm_config = vllm_config
        self.mesh = mesh
        rng = nnx.Rngs(rng_key)
        self.model = Olmo2Model(vllm_config, rng, mesh)
        model_config = vllm_config.model_config
        if (self.model.is_last_rank
                and not model_config.hf_config.tie_word_embeddings):
            self.lm_head = JaxLmHead(
                hidden_size=model_config.hf_config.hidden_size,
                vocab_size=model_config.get_vocab_size(),
                dtype=model_config.dtype,
                param_dtype=model_config.dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
                rngs=rng,
                prefix="lm_head",
            )

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        # Official checkpoints store FP32 weights; keep the requested model
        # dtype when the iterator loader replaces the initialized parameters.
        dtypes = {
            name: to_torch_dtype(param.get_value().dtype)
            for name, param in self.named_parameters()
        }
        return super().load_weights(
            (name, weight.to(dtype=dtypes.get(name, weight.dtype)))
            for name, weight in weights)
