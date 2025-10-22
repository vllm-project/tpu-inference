from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Phi3Config, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.attention_interface import attention
from tpu_inference.layers.jax.rope_interface import apply_longrope, apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (MetadataMap,
                                                         load_hf_weights)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class Phi3MLP(nnx.Module):

    def __init__(self, config: Phi3Config, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.gate_up_proj = nnx.Linear(
            hidden_size,
            2 * intermediate_size,
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
        gate_up = self.gate_up_proj(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        fuse = up * self.act_fn(gate)
        result = self.down_proj(fuse)
        return result


class Phi3Attention(nnx.Module):

    def __init__(self, config: Phi3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh, kv_cache_dtype: str):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.max_position_embeddings = config.max_position_embeddings

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        self.qkv_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads + self.num_kv_heads * 2,
             self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
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
        # qkv: (T, N + K * 2, H)
        qkv = self.qkv_proj(x)
        q, k, v = jnp.split(
            qkv, [self.num_heads, self.num_heads + self.num_kv_heads], axis=1)
        if self.rope_scaling:
            q = apply_longrope(q, md.input_positions, self.head_dim_original,
                               self.rope_scaling,
                               self.original_max_position_embeddings,
                               self.max_position_embeddings, self.rope_theta)
            k = apply_longrope(k, md.input_positions, self.head_dim_original,
                               self.rope_scaling,
                               self.original_max_position_embeddings,
                               self.max_position_embeddings, self.rope_theta)
        else:
            q = apply_rope(q, md.input_positions, self.head_dim_original,
                           self.rope_theta, self.rope_scaling)
            k = apply_rope(k, md.input_positions, self.head_dim_original,
                           self.rope_theta, self.rope_scaling)
        # o: (T, N, H)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            # TODO(kyuyeunk/jacobplatin): Enable w8a8 when VREG spill issue is resolved.
            # q_scale = self._q_scale
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = utils.quantize_kv(k, v, self.kv_cache_quantized_dtype,
                                     k_scale, v_scale)
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


class Phi3DecoderLayer(nnx.Module):

    def __init__(self, config: Phi3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh, kv_cache_dtype: str):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.self_attn = Phi3Attention(config=config,
                                       dtype=dtype,
                                       rng=rng,
                                       mesh=mesh,
                                       kv_cache_dtype=kv_cache_dtype)
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.mlp = Phi3MLP(
            config=config,
            dtype=dtype,
            rng=rng,
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


class Phi3Model(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.layers = [
            Phi3DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype)
            for _ in range(hf_config.num_hidden_layers)
        ]
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        if model_config.hf_config.tie_word_embeddings:
            self.lm_head = self.embed.embedding
        else:
            self.lm_head = nnx.Param(
                init_fn(rng.params(), (hidden_size, vocab_size), dtype),
                sharding=(None, "model"),
            )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[List[jax.Array], jax.Array]:
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class Phi3ForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Phi3Model(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
        )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.model.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.model.lm_head.value)
        return logits

    def get_metadata_map(self) -> MetadataMap:
        sharding_size = self.mesh.shape["model"]

        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config

        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        qkv_heads = num_heads + num_kv_heads * 2
        hidden_size = model_config.get_hidden_size()

        # Pad head_dim for kernel performance.
        head_dim_original = model_config.get_head_size()

        # Key: path to a HF layer weight
        # Value: path to a nnx layer weight
        name_map = {
            "model.embed_tokens": "model.embed.embedding",
            "model.layers.*.input_layernorm":
            "model.layers.*.input_layernorm.scale",
            "model.layers.*.mlp.down_proj":
            "model.layers.*.mlp.down_proj.kernel",
            "model.layers.*.mlp.gate_up_proj":
            "model.layers.*.mlp.gate_up_proj.kernel",
            "model.layers.*.post_attention_layernorm":
            "model.layers.*.post_attention_layernorm.scale",
            "model.layers.*.self_attn.qkv_proj":
            "model.layers.*.self_attn.qkv_proj.kernel",
            "model.layers.*.self_attn.o_proj":
            "model.layers.*.self_attn.o_proj.kernel",
            "model.norm": "model.norm.scale",
        }
        if not self.vllm_config.model_config.hf_config.tie_word_embeddings:
            name_map.update({
                "lm_head": "model.lm_head",
            })

        reshape_keys: dict[str, tuple[int, ...]] = {
            "qkv_proj": (qkv_heads, head_dim_original, hidden_size),
            "o_proj": (hidden_size, num_heads, head_dim_original),
        }
        transpose_keys: dict[str, tuple[int, ...]] = {
            "lm_head": (1, 0),
            "gate_up_proj": (1, 0),
            "down_proj": (1, 0),
            "qkv_proj": (2, 0, 1),
            "o_proj": (1, 2, 0),
        }

        # key: (padding_dim, padding_size)
        pad_keys: dict[str, tuple[int, ...]] = {
            "qkv_proj": (1, sharding_size // num_heads),
            "o_proj": (0, sharding_size // num_heads),
        }

        return MetadataMap(name_map=name_map,
                           reshape_map=reshape_keys,
                           bias_reshape_map={},
                           transpose_map=transpose_keys,
                           pad_map=pad_keys,
                           bias_pad_map={})

    def load_weights(self, rng_key: jax.Array):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng_key)

        metadata_map = self.get_metadata_map()
        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        metadata_map=metadata_map,
                        mesh=self.mesh)
