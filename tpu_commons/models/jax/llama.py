from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
from jax.sharding import Mesh
from transformers import LlamaConfig, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.attention import (sharded_flash_attention,
                                                     sharded_paged_attention,
                                                     update_cache)
from tpu_commons.models.jax.layers.chunked_prefill_attention import (
    sharded_chunked_prefill_attention, sharded_chunked_prefill_update_cache)
from tpu_commons.models.jax.layers.misc import (Einsum, Embedder, RMSNorm,
                                                shard_put)
from tpu_commons.models.jax.layers.params import sharding_init
from tpu_commons.models.jax.layers.rope import apply_rope
from tpu_commons.models.jax.layers.sampling import sample
from tpu_commons.models.jax.utils.weight_utils import hf_model_weights_iterator

logger = init_logger(__name__)

KVCache = Tuple[jax.Array, jax.Array]


class LlamaMLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    act: str
    dtype: jnp.dtype
    mesh: Mesh

    @nn.compact
    def __call__(self, x) -> jax.Array:
        gate_proj = self.param(
            "gate_proj",
            sharding_init((None, "model"), self.mesh),
            (self.hidden_size, self.intermediate_size),
            self.dtype,
        )

        up_proj = self.param(
            "up_proj",
            sharding_init((None, "model"), self.mesh),
            (self.hidden_size, self.intermediate_size),
            self.dtype,
        )
        down_proj = self.param(
            "down_proj",
            sharding_init(("model", None), self.mesh),
            (self.intermediate_size, self.hidden_size),
            self.dtype,
        )

        up = jnp.dot(x, up_proj)
        fuse = up
        gate = jnp.dot(x, gate_proj)
        gate = modeling_flax_utils.ACT2FN[self.act](gate)
        fuse = gate * up
        return jnp.dot(fuse, down_proj)


class LlamaAttention(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype
    mesh: Mesh

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.rope_theta = self.config.rope_theta
        self.rope_scaling = getattr(self.config, "rope_scaling", None)
        self.head_dim = self.config.head_dim

        self.q_proj = Einsum(
            shape=(self.num_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
        )
        self.k_proj = Einsum(
            shape=(self.num_kv_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
        )
        self.v_proj = Einsum(
            shape=(self.num_kv_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
        )
        self.o_proj = Einsum(
            shape=(self.num_heads, self.head_dim, self.hidden_size),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
        )
        self.flash_attention = sharded_flash_attention(self.mesh)
        self.paged_attention = sharded_paged_attention(self.mesh)

    def __call__(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[KVCache, jax.Array, Optional[jax.Array]]:
        # B: batch_size
        # T: seq_len
        # N: num_heads
        # K: num_kv_heads
        # D: hidden_size
        # H: head_dim
        # L: num_blocks
        # S: block_size

        md = attention_metadata

        # q: (B, N, T, H)
        q = self.q_proj("BTD,NDH->BNTH", x)
        q = apply_rope(q, md.input_positions, self.head_dim, self.rope_theta,
                       self.rope_scaling)
        q = q * self.head_dim**-0.5

        # k: (B, K, T, H)
        k = self.k_proj("BTD,KDH->BKTH", x)
        k = apply_rope(k, md.input_positions, self.head_dim, self.rope_theta,
                       self.rope_scaling)

        # v: (B, K, T, H)
        v = self.v_proj("BTD,KDH->BKTH", x)

        # (K, L, S, H)
        k_cache, v_cache = kv_cache
        if md.chunked_prefill_enabled:
            k_cache = sharded_chunked_prefill_update_cache(self.mesh)(
                k_cache, md.kv_cache_write_indices, k, md.num_decode_seqs)
            v_cache = sharded_chunked_prefill_update_cache(self.mesh)(
                v_cache, md.kv_cache_write_indices, v, md.num_decode_seqs)
            outputs = sharded_chunked_prefill_attention(self.mesh)(
                q,
                k_cache,
                v_cache,
                attention_metadata.decode_lengths,
                attention_metadata.decode_page_indices,
                attention_metadata.num_decode_seqs,
                attention_metadata.prefill_lengths,
                attention_metadata.prefill_page_indices,
                attention_metadata.prefill_query_start_offsets,
                attention_metadata.num_prefill_seqs,
            )
        else:
            k_cache = update_cache(is_prefill, k_cache,
                                   md.kv_cache_write_indices, k)
            v_cache = update_cache(is_prefill, v_cache,
                                   md.kv_cache_write_indices, v)
            if is_prefill:
                # (B, N, T, H)
                # TODO(xiang): support MQA and GQA
                if self.num_kv_heads != self.num_heads:
                    k = jnp.repeat(k,
                                   self.num_heads // self.num_kv_heads,
                                   axis=1)
                    v = jnp.repeat(v,
                                   self.num_heads // self.num_kv_heads,
                                   axis=1)
                outputs = sharded_flash_attention(self.mesh)(q, k, v)
            else:
                # (B, N, H)
                q = jnp.squeeze(q, 2)
                outputs = sharded_paged_attention(self.mesh)(q, k_cache,
                                                             v_cache,
                                                             md.seq_lens,
                                                             md.block_indices)
                # (B, N, 1, H)
                outputs = jnp.expand_dims(outputs, 2)

        # (B, T, D)
        o = self.o_proj("BNTH,NHD->BTD", outputs)
        return (k_cache, v_cache), o


class LlamaDecoderLayer(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype
    mesh: Mesh

    def setup(self) -> None:
        rms_norm_eps = self.config.rms_norm_eps
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        act = self.config.hidden_act

        self.input_layernorm = RMSNorm(
            rms_norm_eps=rms_norm_eps,
            dtype=self.dtype,
            mesh=self.mesh,
        )

        self.self_attn = LlamaAttention(
            config=self.config,
            dtype=self.dtype,
            mesh=self.mesh,
        )

        self.post_attention_layernorm = RMSNorm(
            rms_norm_eps=rms_norm_eps,
            dtype=self.dtype,
            mesh=self.mesh,
        )

        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            act=act,
            dtype=self.dtype,
            mesh=self.mesh,
        )

    def __call__(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[KVCache, jax.Array]:
        # Self attention.
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            is_prefill,
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output += x

        # MLP.
        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs


class LlamaModel(nn.Module):
    vllm_config: VllmConfig
    mesh: Mesh

    def setup(self) -> None:
        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config

        self.layers = [
            LlamaDecoderLayer(
                config=hf_config,
                dtype=model_config.dtype,
                mesh=self.mesh,
            ) for i in range(hf_config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            rms_norm_eps=hf_config.rms_norm_eps,
            dtype=model_config.dtype,
            mesh=self.mesh,
        )

    def __call__(
        self,
        is_prefill: bool,
        kv_caches: List[KVCache],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[List[KVCache], jax.Array]:
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                is_prefill,
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class LlamaForCausalLM(nn.Module):
    vllm_config: VllmConfig
    rng: jax.Array
    mesh: Mesh

    def setup(self) -> None:
        model_config = self.vllm_config.model_config
        self.embed_tokens = Embedder(
            vocab_size=model_config.get_vocab_size(),
            hidden_size=model_config.get_hidden_size(),
            dtype=model_config.dtype,
            mesh=self.mesh,
        )
        self.model = LlamaModel(
            vllm_config=self.vllm_config,
            mesh=self.mesh,
        )
        try:
            self.lm_head = self.param(
                "lm_head",
                sharding_init(
                    (None, "model"),
                    self.mesh,
                ),
                (model_config.get_hidden_size(),
                 model_config.get_vocab_size()),
                model_config.dtype,
            )
        except Exception:
            self.lm_head = None

    def __call__(
        self,
        is_prefill: bool,
        do_sampling: bool,
        kv_caches: List[KVCache],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        temperatures: jax.Array = None,
        top_ps: jax.Array = None,
        top_ks: jax.Array = None,
        *args,
    ) -> Tuple[List[KVCache], jax.Array, jax.Array]:
        x = self.embed_tokens.encode(input_ids)

        kv_caches, x = self.model(
            is_prefill,
            kv_caches,
            x,
            attention_metadata,
        )

        if self.lm_head is not None:
            logits = jnp.dot(x, self.lm_head)
        else:
            logits = self.embed_tokens.decode(x)

        next_tokens = sample(
            is_prefill,
            do_sampling,
            self.rng,
            self.mesh,
            logits,
            attention_metadata.seq_lens,
            temperatures,
            top_ps,
            top_ks,
            attention_metadata.chunked_prefill_enabled,
        )
        return kv_caches, next_tokens, logits

    # TODO(xiangxu): extract this to a common weights loading utility
    def load_weights(
        self,
        model_name_or_path: str,
    ) -> Dict[str, Any]:
        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config

        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = hf_config.head_dim

        params_dict = {}
        shard = partial(shard_put, mesh=self.mesh)
        for name, checkpoint_weight in hf_model_weights_iterator(
                model_name_or_path, framework="flax"):
            if "inv_freq" in name:
                pass

            key = name.replace("model", "params")
            key = key.replace("layers.", "model.layers_")
            key = key.replace("params.norm", "params.model.norm")
            if "gate_proj" in key or "up_proj" in key or "down_proj" in key:
                key = key.strip(".weight")
            if "lm_head" in key:
                key = "params.lm_head"

            weight = checkpoint_weight.astype(model_config.dtype)

            if "embed_tokens" in key:
                weight = shard(weight, ("model", None))
            elif "lm_head" in key:
                weight = jnp.transpose(weight)
                weight = shard(weight, (None, "model"))
            if "gate_proj" in key or "up_proj" in key:
                weight = jnp.transpose(weight)
                weight = shard(weight, (None, "model"))
            elif "down_proj" in key:
                weight = jnp.transpose(weight)
                weight = shard(weight, ("model", None))
            elif "q_proj" in key:
                weight = jnp.reshape(
                    weight,
                    (
                        num_heads,
                        head_dim,
                        hidden_size,
                    ),
                )
                weight = jnp.transpose(weight, (0, 2, 1))
                weight = shard(weight, ("model", None, None))
            elif "k_proj" in key or "v_proj" in key:
                weight = jnp.reshape(
                    weight,
                    (
                        num_kv_heads,
                        head_dim,
                        hidden_size,
                    ),
                )
                weight = jnp.transpose(weight, (0, 2, 1))
                weight = shard(weight, ("model", None, None))
            elif "o_proj" in key:
                weight = jnp.reshape(
                    weight,
                    (
                        hidden_size,
                        num_heads,
                        head_dim,
                    ),
                )
                weight = jnp.transpose(weight, (1, 2, 0))
                weight = shard(weight, ("model", None, None))
            elif "norm" in key:
                weight = shard(weight, (None, ))

            params_dict[key] = weight

        return unflatten_dict(params_dict, ".")
