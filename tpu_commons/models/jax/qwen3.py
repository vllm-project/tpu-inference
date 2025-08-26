from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Qwen3Config
from vllm.config import VllmConfig

from tpu_commons import utils
from tpu_commons.kernels.ragged_paged_attention.v3.util import (
    align_to, get_dtype_packing)
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention import attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.rope import apply_rope
from tpu_commons.models.jax.qwen2 import Qwen2DecoderLayer
from tpu_commons.models.jax.qwen2 import Qwen2MLP as Qwen3MLP
from tpu_commons.models.jax.qwen2 import Qwen2Model
from tpu_commons.models.jax.utils.weight_utils import (get_default_maps,
                                                       load_hf_weights)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


def _get_reshape_maps(hf_config, dtype):
    # model_config = vllm_config.model_config
    # hf_config = model_config.hf_config

    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    hidden_size = hf_config.hidden_size
    q_packing = get_dtype_packing(dtype)
    actual_num_q_heads_per_kv_head = num_heads // num_kv_heads
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    head_dim_original = getattr(hf_config, "head_dim",
                                hidden_size // num_heads)
    head_dim = utils.get_padded_head_dim(head_dim_original)

    # TODO(cuiq) handle padding instead of not allowing it.
    assert num_heads % num_kv_heads == 0
    assert num_q_heads_per_kv_head % q_packing == 0
    assert head_dim == head_dim_original

    return {
        "K": num_kv_heads,
        "N": num_heads,
        "D": hidden_size,
        "H": head_dim,
        "C": q_packing,
        "R": num_q_heads_per_kv_head // q_packing,
        "num_q_heads_per_kv_head": num_q_heads_per_kv_head,
        "actual_num_q_heads_per_kv_head": actual_num_q_heads_per_kv_head,
    }


class Qwen3Attention(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh
        reshape_map = _get_reshape_maps(config, dtype)
        self.actual_num_q_heads_per_kv_head = reshape_map[
            "actual_num_q_heads_per_kv_head"]

        self.q_proj = nnx.Einsum(
            "TD,KRCHD->KTRCH",
            (
                reshape_map["K"],
                reshape_map["R"],
                reshape_map["C"],
                reshape_map["H"],
                reshape_map["D"],
            ),
            param_dtype=dtype,
            # TODO(cuiq): How partitioning works???
            kernel_init=nnx.with_partitioning(
                init_fn, ("model", None, None, None, None)),
            rngs=rng,
        )
        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
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

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        # q: (K, T, num_q_heads_per_kv_heads_per_packing, packing, H)
        q = self.q_proj(x)

        # TODO(cuiq): does this normal work?
        q = self.q_norm(q)
        q = apply_rope(q, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # k: (T, K, H)
        k = self.k_proj(x)
        k = self.k_norm(k)
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
            self.actual_num_q_heads_per_kv_head,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Qwen3DecoderLayer(Qwen2DecoderLayer):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs,
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
        self.self_attn = Qwen3Attention(config=config,
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
        self.mlp = Qwen3MLP(
            config=config,
            dtype=dtype,
            rng=rng,
        )


class Qwen3Model(Qwen2Model):

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
            Qwen3DecoderLayer(
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


class Qwen3ForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Qwen3Model(
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
    ) -> Tuple[List[jax.Array], jax.Array]:
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
        )
        return kv_caches, x

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
            "model.layers.*.self_attn.k_norm":
            "model.layers.*.self_attn.k_norm.scale",
            "model.layers.*.self_attn.k_proj":
            "model.layers.*.self_attn.k_proj.kernel",
            "model.layers.*.self_attn.o_proj":
            "model.layers.*.self_attn.o_proj.kernel",
            "model.layers.*.self_attn.q_norm":
            "model.layers.*.self_attn.q_norm.scale",
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

        metadata_map = get_default_maps(self.vllm_config, self.mesh, mappings)
        self.update_medata_maps(metadata_map)

        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        metadata_map=metadata_map,
                        mesh=self.mesh)

    def update_medata_maps(self, metadata_map):
        # place update the map by replace the value directly

        # reshape_keys
        reshape_mapping = _get_reshape_maps(
            self.vllm_config.model_config.hf_config,
            self.vllm_config.model_config.dtype)

        # pre reshape
        metadata_map.pre_reshape_map["q_proj"] = (
            reshape_mapping["K"],
            reshape_mapping["actual_num_q_heads_per_kv_head"],
            reshape_mapping["H"], reshape_mapping["D"])
        metadata_map.pre_bias_reshape_map["q_proj.bias"] = (
            reshape_mapping["K"],
            reshape_mapping["actual_num_q_heads_per_kv_head"],
            reshape_mapping["H"],
        )

        # pre pad
        metadata_map.pre_pad_map["q_proj"] = (
            (0, 0),
            (0, reshape_mapping["num_q_heads_per_kv_head"] -
             reshape_mapping["actual_num_q_heads_per_kv_head"]),
            (0, 0),
            (0, 0),
        )
        metadata_map.pre_bias_pad_map["q_proj.bias"] = (
            (0, 0),
            (0, reshape_mapping["num_q_heads_per_kv_head"] -
             reshape_mapping["actual_num_q_heads_per_kv_head"]),
            (0, 0),
        )

        # final reshape
        metadata_map.reshape_map["q_proj"] = (reshape_mapping["K"],
                                              reshape_mapping["R"],
                                              reshape_mapping["C"],
                                              reshape_mapping["H"],
                                              reshape_mapping["D"])

        metadata_map.bias_reshape_map["q_proj.bias"] = (
            reshape_mapping["K"],
            reshape_mapping["R"],
            reshape_mapping["C"],
            reshape_mapping["H"],
        )
        # Don't transpose q_proj
        del metadata_map.transpose_map["q_proj"]
