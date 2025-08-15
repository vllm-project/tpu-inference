from functools import partial
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_commons import utils
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention import attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.rope import apply_rope
from tpu_commons.models.jax.utils.weight_utils import (
    TpuCommonAnnotation, annotated_module, load_hf_weights,
    load_hf_weights_through_annotation)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

_annotation_map: dict[Any, TpuCommonAnnotation] = {}
_M = partial(annotated_module, _annotation_map)


class LlamaMLP(nnx.Module):

    def __init__(self, config: LlamaConfig, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.gate_proj = _M(nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        ),
                            transpose_param=(1, 0),
                            hf_name="gate_proj")
        self.up_proj = _M(nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        ),
                          transpose_param=(1, 0),
                          hf_name="up_proj")
        self.down_proj = _M(nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        ),
                            transpose_param=(1, 0),
                            hf_name="down_proj")
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

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        self.q_proj = _M(nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        ),
                         reshape_param=(self.num_heads, self.head_dim_original,
                                        self.hidden_size),
                         bias_reshape_param=(self.num_heads,
                                             self.head_dim_original),
                         transpose_param=(2, 0, 1),
                         pad_param=(1, sharding_size // self.num_heads),
                         bias_pad_param=(0, sharding_size // self.num_heads),
                         hf_name="q_proj")
        self.k_proj = _M(
            nnx.Einsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                rngs=rng,
            ),
            reshape_param=(self.num_kv_heads, self.head_dim_original,
                           self.hidden_size),
            bias_reshape_param=(self.num_kv_heads, self.head_dim_original),
            pad_param=(1, sharding_size // self.num_kv_heads),
            bias_pad_param=(0, sharding_size // self.num_kv_heads),
            transpose_param=(2, 0, 1),
            hf_name="k_proj")
        self.v_proj = _M(
            nnx.Einsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                rngs=rng,
            ),
            reshape_param=(self.num_kv_heads, self.head_dim_original,
                           self.hidden_size),
            bias_reshape_param=(self.num_kv_heads, self.head_dim_original),
            pad_param=(1, sharding_size // self.num_kv_heads),
            bias_pad_param=(0, sharding_size // self.num_kv_heads),
            transpose_param=(2, 0, 1),
            hf_name="v_proj")
        self.o_proj = _M(nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
        ),
                         reshape_param=(self.hidden_size, self.num_heads,
                                        self.head_dim_original),
                         transpose_param=(1, 2, 0),
                         pad_param=(0, sharding_size // self.num_heads),
                         hf_name="o_proj")

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
        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class LlamaDecoderLayer(nnx.Module):

    def __init__(self, config: LlamaConfig, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = _M(nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        ),
                                  hf_name="input_layernorm")
        self.self_attn = _M(LlamaAttention(config=config,
                                           dtype=dtype,
                                           rng=rng,
                                           mesh=mesh),
                            hf_name="self_attn")
        self.post_attention_layernorm = _M(nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        ),
                                           hf_name="post_attention_layernorm")
        self.mlp = _M(LlamaMLP(
            config=config,
            dtype=dtype,
            rng=rng,
        ),
                      hf_name="mlp")

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


class LlamaModel(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.layers = _M([
            LlamaDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
            ) for _ in range(hf_config.num_hidden_layers)
        ],
                         hf_name="layers")

        self.norm = _M(nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        ),
                       hf_name="norm")

    def __call__(
        self,
        kv_caches: List[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[List[jax.Array], jax.Array]:
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


class LlamaForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        vocab_size = model_config.get_vocab_size()
        hidden_size = model_config.get_hidden_size()
        dtype = model_config.dtype

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.embed = _M(nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=self.rng,
        ),
                        hf_name="model.embed_tokens")
        self.model = _M(LlamaModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        ),
                        hf_name="model")

        if model_config.hf_config.tie_word_embeddings:
            self.lm_head = self.embed.embedding
        else:
            self.lm_head = _M(nnx.Param(
                init_fn(self.rng.params(), (hidden_size, vocab_size), dtype),
                sharding=(None, "model"),
            ),
                              transpose_param=(1, 0),
                              hf_name="lm_head")

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array]:
        x = self.embed(input_ids)
        kv_caches, x = self.model(
            kv_caches,
            x,
            attention_metadata,
        )
        return kv_caches, x

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.lm_head.value)
        return logits

    def load_weights(self, rng_key: jax.Array):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng_key)

        load_hf_weights_through_annotation(vllm_config=self.vllm_config,
                                           model=self,
                                           mesh=self.mesh,
                                           annotation_map=_annotation_map)
        """
        The name map is {
            'model.embed_tokens': 'embed.embedding',
            'model.layers.*.input_layernorm': 'model.layers.*.input_layernorm.scale',
            'model.layers.*.self_attn.q_proj': 'model.layers.*.self_attn.q_proj.kernel',
            'model.layers.*.self_attn.k_proj': 'model.layers.*.self_attn.k_proj.kernel',
            'model.layers.*.self_attn.v_proj': 'model.layers.*.self_attn.v_proj.kernel',
            'model.layers.*.self_attn.o_proj': 'model.layers.*.self_attn.o_proj.kernel',
            'model.layers.*.post_attention_layernorm': 'model.layers.*.post_attention_layernorm.scale',
            'model.layers.*.mlp.gate_proj': 'model.layers.*.mlp.gate_proj.kernel',
            'model.layers.*.mlp.up_proj': 'model.layers.*.mlp.up_proj.kernel',
            'model.layers.*.mlp.down_proj': 'model.layers.*.mlp.down_proj.kernel',
            'model.norm': 'model.norm.scale',
            'lm_head': 'lm_head'}

        The transpose map is {
            'model.layers.*.self_attn.q_proj': (2, 0, 1),
            'model.layers.*.self_attn.k_proj': (2, 0, 1),
            'model.layers.*.self_attn.v_proj': (2, 0, 1),
            'model.layers.*.self_attn.o_proj': (1, 2, 0),
            'model.layers.*.mlp.gate_proj': (1, 0),
            'model.layers.*.mlp.up_proj': (1, 0),
            'model.layers.*.mlp.down_proj': (1, 0),
            'lm_head': (1, 0)}

        The reshape map is {
            'model.layers.*.self_attn.q_proj': (32, 128, 4096),
            'model.layers.*.self_attn.k_proj': (8, 128, 4096),
            'model.layers.*.self_attn.v_proj': (8, 128, 4096),
            'model.layers.*.self_attn.o_proj': (4096, 32, 128)}

        The bias_reshape map is {
            'model.layers.*.self_attn.q_proj': (32, 128),
            'model.layers.*.self_attn.k_proj': (8, 128),
            'model.layers.*.self_attn.v_proj': (8, 128)}

        The pad map is {
            'model.layers.*.self_attn.q_proj': (1, 0),
            'model.layers.*.self_attn.k_proj': (1, 0),
            'model.layers.*.self_attn.v_proj': (1, 0),
            'model.layers.*.self_attn.o_proj': (0, 0)}

        The bias_pad map is {
            'model.layers.*.self_attn.q_proj': (0, 0),
            'model.layers.*.self_attn.k_proj': (0, 0),
            'model.layers.*.self_attn.v_proj': (0, 0)}
        """

    def _load_weights(self, rng_key: jax.Array):
        # Key: path to a HF layer weight
        # Value: path to a nnx layer weight
        self.rng = nnx.Rngs(rng_key)

        mappings = {
            "model.embed_tokens": "embed.embedding",
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
                "lm_head": "lm_head",
            })
        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        mappings=mappings,
                        mesh=self.mesh)
