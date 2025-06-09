from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_interface import KVCache, attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.rope import apply_rope
from tpu_commons.models.jax.layers.sampling import sample

logger = init_logger(__name__)

# TODO(xiang): is this faster than nnx.initializers.zeros_init()?
init_fn = nnx.initializers.uniform()


class LlamaMLP(nnx.Module):

    def __init__(self, config: LlamaConfig, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.act_fn = modeling_flax_utils.ACT2FN[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


class LlamaAttention(nnx.Module):

    def __init__(self, config: LlamaConfig, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.head_dim = config.head_dim

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "BTD,NDH->BNTH",
            (self.num_heads, self.hidden_size, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "BTD,KDH->BKTH",
            (self.num_kv_heads, self.hidden_size, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "BTD,KDH->BKTH",
            (self.num_kv_heads, self.hidden_size, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "BNTH,NHD->BTD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
        )

    def __call__(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[KVCache, jax.Array]:
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
        q = self.q_proj(x)
        q = apply_rope(q, md.input_positions, self.head_dim, self.rope_theta,
                       self.rope_scaling)
        q = q * self.head_dim**-0.5

        # k: (B, K, T, H)
        k = self.k_proj(x)
        k = apply_rope(k, md.input_positions, self.head_dim, self.rope_theta,
                       self.rope_scaling)

        # v: (B, K, T, H)
        v = self.v_proj(x)

        new_kv_cache, outputs = attention(
            is_prefill,
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.num_heads,
            self.num_kv_heads,
        )

        # (B, T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class LlamaDecoderLayer(nnx.Module):

    def __init__(self, config: LlamaConfig, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.attention = LlamaAttention(config=config,
                                        dtype=dtype,
                                        rng=rng,
                                        mesh=mesh)
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.mlp = LlamaMLP(
            config=config,
            dtype=dtype,
            rng=rng,
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
        kv_cache, attn_output = self.attention(
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


class LlamaModel(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.layers = [
            LlamaDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
            ) for i in range(hf_config.num_hidden_layers)
        ]
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
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


class LlamaForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: jax.Array,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        vocab_size = model_config.get_vocab_size()
        hidden_size = model_config.get_hidden_size()
        dtype = model_config.dtype

        self.rng = nnx.Rngs(rng)
        self.mesh = mesh

        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=self.rng,
        )
        self.model = LlamaModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

        # TODO(xiang): Llama3.2 does not use lm_head
        self.lm_head = nnx.Param(
            init_fn(self.rng.params(), (hidden_size, vocab_size), dtype),
            sharding=(None, "model"),
        )

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
        x = self.embed(input_ids)

        kv_caches, x = self.model(
            is_prefill,
            kv_caches,
            x,
            attention_metadata,
        )

        logits = jnp.dot(x, self.lm_head.value)

        next_tokens = sample(
            is_prefill,
            do_sampling,
            self.rng.params(),
            self.mesh,
            logits,
            attention_metadata.seq_lens,
            temperatures,
            top_ps,
            top_ks,
            attention_metadata.chunked_prefill_enabled,
        )
        return kv_caches, next_tokens, logits
        # return explict_sharding(kv_caches, next_tokens, logits)


# def explict_sharding(kv_caches, next_tokens, logits):
#     kv_caches = jax.lax.with_sharding_constraint(kv_caches,
#                                                  PartitionSpec("model"))
#     next_tokens = jax.lax.with_sharding_constraint(next_tokens,
#                                                    PartitionSpec(None))
#     logits = jax.lax.with_sharding_constraint(logits, PartitionSpec(None))
#     return kv_caches, next_tokens, logits
