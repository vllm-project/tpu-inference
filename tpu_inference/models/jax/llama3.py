from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (get_default_maps,
                                                         load_hf_weights)

logger = init_logger(__name__)

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
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.MLP_TENSOR, None)),
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
                 mesh: Mesh, kv_cache_dtype: str):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"] * mesh.shape.get("attn_dp", 1)
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.ATTN_HEAD, None, None)),
            rngs=rng,
        )

        self._q_scale = 1.0
        # [total_num_pages, page_size, num_kv_heads, 1]
        # self._k_scale = nnx.Variable(
        #     jnp.zeros((1422, 256, self.num_kv_heads, 1)),
        #     dtype=jnp.float32,
        # )
        # self._v_scale = nnx.Variable(
        #     jnp.zeros((1422, 256, self.num_kv_heads, 1)),
        #     dtype=jnp.float32,
        # )
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
        k_scale_cache: Optional[jax.Array] = None,
        v_scale_cache: Optional[jax.Array] = None,
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
        # if self.kv_cache_quantized_dtype:
        #     # TODO(kyuyeunk/jacobplatin): Enable w8a8 when VREG spill issue is resolved.
        #     # q_scale = self._q_scale
        #     k_scale = 1.0
        #     v_scale = 1.0

        #     k, v = utils.quantize_kv(k, v, self.kv_cache_quantized_dtype,
        #                              k_scale, v_scale)

        new_kv_cache, outputs, updated_k_scale, updated_v_scale = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            q_scale=None,
            k_scale=k_scale_cache,
            v_scale=v_scale_cache,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o, updated_k_scale, updated_v_scale


class LlamaDecoderLayer(nnx.Module):

    def __init__(self, config: LlamaConfig, dtype: jnp.dtype, rng: nnx.Rngs,
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
        self.self_attn = LlamaAttention(config=config,
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
        self.mlp = LlamaMLP(
            config=config,
            dtype=dtype,
            rng=rng,
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
        k_scale_cache: Optional[jax.Array] = None,
        v_scale_cache: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output, k_scale_cache, v_scale_cache = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
            k_scale_cache,
            v_scale_cache,
        )
        attn_output += x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs, k_scale_cache, v_scale_cache


class LlamaModel(nnx.Module):

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
            embedding_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.VOCAB, None)),
            rngs=rng,
        )
        self.layers = [
            LlamaDecoderLayer(
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
                sharding=(None, ShardingAxisName.VOCAB),
            )

        self.aux_hidden_state_layers = []
        if vllm_config.speculative_config and vllm_config.speculative_config.method == "eagle3":
            self.aux_hidden_state_layers = self.get_eagle3_aux_hidden_state_layers(
            )

    def get_eagle3_aux_hidden_state_layers(self):
        num_layers = len(self.layers)
        return (2, num_layers // 2, num_layers - 3)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        x = self.embed(input_ids)
        aux_hidden_states = []
        updated_k_scales = []
        updated_v_scales = []
        for i, layer in enumerate(self.layers):
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(x)
            kv_cache = kv_caches[i]
            k_scales_cache = attention_metadata.k_scale_cache[i]
            v_scales_cache = attention_metadata.v_scale_cache[i]
            kv_cache, x, k_scales_cache, v_scales_cache = layer(
                kv_cache,
                x,
                attention_metadata,
                k_scale_cache=k_scales_cache,
                v_scale_cache=v_scales_cache)
            kv_caches[i] = kv_cache
            updated_k_scales.append(k_scales_cache)
            updated_v_scales.append(v_scales_cache)

        x = self.norm(x)
        return kv_caches, x, aux_hidden_states, updated_k_scales, updated_v_scales


class LlamaForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = LlamaModel(
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
        kv_caches, x, aux_hidden_states, new_k_scales, new_v_scales = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
        )
        return kv_caches, x, aux_hidden_states, new_k_scales, new_v_scales

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.model.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.model.lm_head.value)
        return logits

    def load_weights(self, rng_key: jax.Array):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng_key)

        # Key: path to a HF layer weight
        # Value: path to a nnx layer weight
        mappings = {
            "model.embed_tokens": "model.embed.embedding",
            "model.layers.*.input_layernorm":
            "model.layers.*.input_layernorm.scale",
            "model.layers.*.mlp.down_proj":
            "model.layers.*.mlp.down_proj.kernel",
            "model.layers.*.mlp.gate_proj":
            "model.layers.*.mlp.gate_proj.kernel",
            "model.layers.*.mlp.up_proj": "model.layers.*.mlp.up_proj.kernel",
            "model.layers.*.post_attention_layernorm":
            "model.layers.*.post_attention_layernorm.scale",
            "model.layers.*.self_attn.k_proj":
            "model.layers.*.self_attn.k_proj.kernel",
            "model.layers.*.self_attn.o_proj":
            "model.layers.*.self_attn.o_proj.kernel",
            "model.layers.*.self_attn.q_proj":
            "model.layers.*.self_attn.q_proj.kernel",
            "model.layers.*.self_attn.v_proj":
            "model.layers.*.self_attn.v_proj.kernel",
            "model.norm": "model.norm.scale",
        }
        # Add lm_head mapping only if it's not tied to embeddings
        if not self.vllm_config.model_config.hf_config.tie_word_embeddings:
            mappings.update({
                "lm_head": "model.lm_head",
            })

        metadata_map = get_default_maps(self.vllm_config.model_config,
                                        self.mesh, mappings)
        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        metadata_map=metadata_map,
                        mesh=self.mesh)

        # for layer in self.model.layers:
        #     attn = layer.self_attn
        #     if hasattr(attn._k_scale, 'value') and isinstance(
        #             attn._k_scale.value, jax.ShapeDtypeStruct):
        #         # Using the shape and dtype from the abstract struct to initialize zeros
        #         attn._k_scale.value = jnp.zeros(
        #             attn._k_scale.value.shape, dtype=attn._k_scale.value.dtype)

        #     if hasattr(attn._v_scale, 'value') and isinstance(
        #             attn._v_scale.value, jax.ShapeDtypeStruct):
        #         attn._v_scale.value = jnp.zeros(
        #             attn._v_scale.value.shape, dtype=attn._v_scale.value.dtype)
