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

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import GemmaConfig
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import \
    sharded_ragged_paged_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import StandardWeightLoader

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class RMSNorm(nnx.Module):

    def __init__(
        self,
        dim: int,
        config: GemmaConfig,
        dtype: jnp.dtype,
    ):
        self.rms_norm_eps = config.rms_norm_eps
        self.weight = nnx.Param(jnp.zeros(dim, dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(
            jnp.mean(x**2, axis=-1, keepdims=True) + self.rms_norm_eps)
        return (self.weight + 1) * (x / rms)


class Gemma3MLP(nnx.Module):

    def __init__(self, config: GemmaConfig, dtype: jnp.dtype, rng: nnx.Rngs):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_activation
        self.gate_proj = nnx.Linear(in_features=self.hidden_size,
                                    out_features=self.intermediate_size,
                                    use_bias=False,
                                    param_dtype=dtype,
                                    kernel_init=nnx.with_partitioning(
                                        init_fn, (None, 'model')),
                                    rngs=rng)
        self.up_proj = nnx.Linear(in_features=self.hidden_size,
                                  out_features=self.intermediate_size,
                                  use_bias=False,
                                  param_dtype=dtype,
                                  kernel_init=nnx.with_partitioning(
                                      init_fn, (None, 'model')),
                                  rngs=rng)
        self.down_proj = nnx.Linear(in_features=self.intermediate_size,
                                    out_features=self.hidden_size,
                                    use_bias=False,
                                    param_dtype=dtype,
                                    kernel_init=nnx.with_partitioning(
                                        init_fn, ('model', None)),
                                    rngs=rng)
        self.act_fn = {
            'gelu_pytorch_tanh': lambda x: jax.nn.gelu(x, approximate=True)
        }[self.hidden_act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        return self.down_proj(fuse)


class Gemma3Attention(nnx.Module):
    """
    T - seq. length; 
    D - hidden size; 
    H - head dim.;
    N - num. query heads;
    K - num. kv heads;
    """

    def __init__(self,
                 config: GemmaConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 is_local: bool = False):
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta if not is_local else config.rope_local_base_freq
        self.sliding_window = None if not is_local else config.sliding_window
        self.query_pre_attn_scalar = config.query_pre_attn_scalar
        self.mesh = mesh
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    self.mesh.shape["model"])
        self.num_kv_heads = utils.get_padded_num_heads(
            self.num_kv_heads, self.mesh.shape["model"])

        self.q_norm = RMSNorm(dim=self.head_dim, config=config, dtype=dtype)
        self.k_norm = RMSNorm(dim=self.head_dim, config=config, dtype=dtype)

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH", (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, 'model', None)),
            rngs=rng)
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, 'model', None)),
            rngs=rng)
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, 'model', None)),
            rngs=rng)
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD", (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ('model', None, None)),
            rngs=rng)

    def __call__(
            self, kv_cache: Optional[jax.Array], x: jax.Array,
            attention_metadata: AttentionMetadata
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata

        # q: (T, N, H)
        q = self.q_proj(x)
        q = self.q_norm(q)
        q = apply_rope(q,
                       positions=md.input_positions,
                       head_dim=self.head_dim,
                       rope_theta=self.rope_theta)
        q = q * (self.query_pre_attn_scalar**-0.5)

        # k: (T, K, H)
        k = self.k_proj(x)
        k = self.k_norm(k)
        k = apply_rope(k,
                       positions=md.input_positions,
                       head_dim=self.head_dim,
                       rope_theta=self.rope_theta)

        # v: (T, K, H)
        v = self.v_proj(x)

        # o: (T, N, H)
        outputs, new_kv_cache = sharded_ragged_paged_attention(
            self.mesh,
            q,
            k,
            v,
            kv_cache,
            md.seq_lens,
            md.block_tables,
            md.query_start_loc,
            md.request_distribution,
            attention_sink=None,
            sm_scale=1.0,
            attention_chunk_size=self.sliding_window,
            q_scale=None,
            k_scale=None,
            v_scale=None)

        # o: (T, D)
        o = self.o_proj(outputs)

        return new_kv_cache, o


class Gemma3DecoderLayer(nnx.Module):

    def __init__(self,
                 config: GemmaConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 is_local: bool = False):
        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(dim=self.hidden_size,
                                       config=config,
                                       dtype=dtype)
        self.self_attn = Gemma3Attention(config, dtype, rng, mesh,
                                         kv_cache_dtype, is_local)
        self.post_attention_layernorm = RMSNorm(dim=self.hidden_size,
                                                config=config,
                                                dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(dim=self.hidden_size,
                                                 config=config,
                                                 dtype=dtype)
        self.mlp = Gemma3MLP(config, dtype, rng)
        self.post_feedforward_layernorm = RMSNorm(dim=self.hidden_size,
                                                  config=config,
                                                  dtype=dtype)

    def __call__(
            self, kv_cache: Optional[jax.Array], x: jax.Array,
            attention_metadata: AttentionMetadata
    ) -> Tuple[jax.Array, jax.Array]:
        x_norm = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(kv_cache, x_norm,
                                               attention_metadata)
        attn_output = self.post_attention_layernorm(attn_output)
        attn_output += x

        outputs = self.pre_feedforward_layernorm(attn_output)
        outputs = self.mlp(outputs)
        outputs = self.post_feedforward_layernorm(outputs)
        outputs += attn_output

        return kv_cache, outputs


class Gemma3Model(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                 mesh: jax.sharding.Mesh):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config

        self.vocab_size = model_config.get_vocab_size()
        self.dtype = model_config.dtype
        self.hidden_size = hf_config.hidden_size
        self.sliding_window_pattern = hf_config.sliding_window_pattern

        self.embed = nnx.Embed(num_embeddings=self.vocab_size,
                               features=self.hidden_size,
                               param_dtype=self.dtype,
                               embedding_init=nnx.with_partitioning(
                                   init_fn, ("model", None)),
                               rngs=rng)
        self.layers = nnx.List([
            Gemma3DecoderLayer(
                config=hf_config,
                dtype=self.dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                is_local=(i + 1) % self.sliding_window_pattern != 0,
            ) for i in range(hf_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(self.hidden_size, hf_config, self.dtype)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        x = self.embed(input_ids)
        x *= jnp.sqrt(self.hidden_size).astype(x.dtype)

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            kv_caches[i] = kv_cache

        x = self.norm(x)

        return kv_caches, x


class Gemma3ForCausalLM(nnx.Module):
    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                 mesh: jax.sharding.Mesh):
        self.vllm_config = vllm_config
        self.model_config = self.vllm_config.model_config
        self.hf_config = self.model_config.hf_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh

        self.vocab_size = self.model_config.get_vocab_size()
        self.dtype = self.model_config.dtype
        self.hidden_size = self.hf_config.hidden_size

        self.model = Gemma3Model(self.vllm_config, self.rng, self.mesh)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        kv_caches, x = self.model(kv_caches, input_ids, attention_metadata)
        return kv_caches, x, []

    def load_weights(self, rng_key: jax.Array):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng_key)

        # Key: path to a HF layer weight
        # Value: path to a nnx layer weight
        mappings = {
            "model.embed_tokens.weight": "model.embed.embedding",
            "model.layers.*.mlp.down_proj.weight":
            "model.layers.*.mlp.down_proj.kernel",
            "model.layers.*.mlp.gate_proj.weight":
            "model.layers.*.mlp.gate_proj.kernel",
            "model.layers.*.mlp.up_proj.weight":
            "model.layers.*.mlp.up_proj.kernel",
            "model.layers.*.self_attn.k_proj.weight":
            "model.layers.*.self_attn.k_proj.kernel",
            "model.layers.*.self_attn.o_proj.weight":
            "model.layers.*.self_attn.o_proj.kernel",
            "model.layers.*.self_attn.q_proj.weight":
            "model.layers.*.self_attn.q_proj.kernel",
            "model.layers.*.self_attn.v_proj.weight":
            "model.layers.*.self_attn.v_proj.kernel",
            "model.layers.*.input_layernorm.weight":
            "model.layers.*.input_layernorm.weight",
            "model.layers.*.post_attention_layernorm.weight":
            "model.layers.*.post_attention_layernorm.weight",
            "model.layers.*.pre_feedforward_layernorm.weight":
            "model.layers.*.pre_feedforward_layernorm.weight",
            "model.layers.*.post_feedforward_layernorm.weight":
            "model.layers.*.post_feedforward_layernorm.weight",
            "model.layers.*.self_attn.q_norm.weight":
            "model.layers.*.self_attn.q_norm.weight",
            "model.layers.*.self_attn.k_norm.weight":
            "model.layers.*.self_attn.k_norm.weight",
            "model.norm.weight": "model.norm.weight",
        }

        loader = self.WeightLoader(self.vllm_config, self.mesh)
        keep_hf_weight_suffix_when_match = ['model']
        loader.load_weights(
            self,
            mappings,
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match)

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return hidden_states @ self.model.embed.embedding.T
