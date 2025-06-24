from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Qwen2Config, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_interface import KVCache, attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.rope import apply_rope
from tpu_commons.models.jax.layers.sampling import sample
from tpu_commons.models.jax.utils.weight_utils import load_hf_weights

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class Qwen2MLP(nnx.Module):

    def __init__(self, config: Qwen2Config, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rng,
        )
        self.act_fn = modeling_flax_utils.ACT2FN[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


class Qwen2Attention(nnx.Module):

    def __init__(self, config: Qwen2Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "BTD,NDH->BNTH",
            (self.num_heads, self.hidden_size, self.head_dim),
            (self.num_heads, self.head_dim),  # bias shape [N, H]
            param_dtype=dtype,
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "BTD,KDH->BKTH",
            (self.num_kv_heads, self.hidden_size, self.head_dim),
            (self.num_kv_heads, self.head_dim),  # bias shape [K, H])
            param_dtype=dtype,
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "BTD,KDH->BKTH",
            (self.num_kv_heads, self.hidden_size, self.head_dim),
            (self.num_kv_heads, self.head_dim),  # bias shape [K, H])
            param_dtype=dtype,
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "BNTH,NHD->BTD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            rngs=rng,
        )

    def __call__(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[KVCache, jax.Array]:
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

        # o: (B, N, T, H)
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


class Qwen2DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen2Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            rngs=rng,
        )
        self.self_attn = Qwen2Attention(config=config,
                                        dtype=dtype,
                                        rng=rng,
                                        mesh=mesh)
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            rngs=rng,
        )
        self.mlp = Qwen2MLP(
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


class Qwen2Model(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.layers = [
            Qwen2DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
            ) for _ in range(hf_config.num_hidden_layers)
        ]
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
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
            jax.debug.print(
                "[Qwen2Model] hidden_states after layer {layer_idx}: shape={shape}, dtype={dtype}, first 10: {first_10}, sum: {sum}",
                layer_idx=i,
                shape=x.shape,
                first_10=x.flatten()[:10],
                sum=jnp.sum(x[:, :5, :]),
                dtype=x.dtype,
            )
        x = self.norm(x)
        return kv_caches, x


class Qwen2ForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        vocab_size = model_config.get_vocab_size()
        hidden_size = model_config.get_hidden_size()
        dtype = model_config.dtype

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            rngs=self.rng,
        )
        self.model = Qwen2Model(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

        hf_config = model_config.hf_config
        if hf_config.tie_word_embeddings:
            self.lm_head = self.embed.embedding
        else:
            self.lm_head = nnx.Param(
                init_fn(self.rng.params(), (hidden_size, vocab_size), dtype), )

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

        jax.debug.print(
            "[Qwen2ForCausalLM] input_ids: shape={shape}, dtype={dtype}, first 10: {first_10}, sum: {sum}",
            shape=input_ids.shape,
            first_10=input_ids.flatten()[:10],
            sum=jnp.sum(input_ids),
            dtype=input_ids.dtype,
        )
        x = self.embed(input_ids)
        jax.debug.print(
            "[Qwen2ForCausalLM] input after embedding: shape={shape}, dtype={dtype}, first 10: {first_10}, sum: {sum}",
            shape=x.shape,
            first_10=x.flatten()[:10],
            sum=jnp.sum(x[:, :5, :]),  # Sum of the first 5 token embeddings
            dtype=x.dtype,
        )

        kv_caches, x = self.model(
            is_prefill,
            kv_caches,
            x,
            attention_metadata,
        )

        hf_config = self.vllm_config.model_config.hf_config
        if hf_config.tie_word_embeddings:
            # self.lm_head.value is (vocab_size, hidden_size)
            logits = jnp.dot(x, self.lm_head.value.T)
        else:
            # self.lm_head.value is (hidden_size, vocab_size)
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

    def load_weights(self, rng_key: jax.Array):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng_key)

        mappings = {
            "model.embed_tokens": ("embed.embedding", ("model", None)),
            "model.layers.*.input_layernorm":
            ("model.layers.*.input_layernorm.scale", (None, )),
            "model.layers.*.mlp.down_proj":
            ("model.layers.*.mlp.down_proj.kernel", ("model", None)),
            "model.layers.*.mlp.gate_proj":
            ("model.layers.*.mlp.gate_proj.kernel", (None, "model")),
            "model.layers.*.mlp.up_proj": ("model.layers.*.mlp.up_proj.kernel",
                                           (None, "model")),
            "model.layers.*.post_attention_layernorm":
            ("model.layers.*.post_attention_layernorm.scale", (None, )),
            "model.layers.*.self_attn.k_proj":
            ("model.layers.*.self_attn.k_proj.kernel", ("model", None, None)),
            "model.layers.*.self_attn.o_proj":
            ("model.layers.*.self_attn.o_proj.kernel", ("model", None, None)),
            "model.layers.*.self_attn.q_proj":
            ("model.layers.*.self_attn.q_proj.kernel", ("model", None, None)),
            "model.layers.*.self_attn.v_proj":
            ("model.layers.*.self_attn.v_proj.kernel", ("model", None, None)),
            "model.layers.*.self_attn.q_proj.bias":
            ("model.layers.*.self_attn.q_proj.bias", ("model", None)),
            "model.layers.*.self_attn.k_proj.bias":
            ("model.layers.*.self_attn.k_proj.bias", ("model", None)),
            "model.layers.*.self_attn.v_proj.bias":
            ("model.layers.*.self_attn.v_proj.bias", ("model", None)),
            "model.norm": ("model.norm.scale", (None, )),
        }

        # Add lm_head mapping only if it's not tied to embeddings
        hf_config = self.vllm_config.model_config.hf_config
        if not hf_config.tie_word_embeddings:
            mappings.update({
                "lm_head": ("lm_head", (None, "model")),
            })

        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        mappings=mappings,
                        mesh=self.mesh)
