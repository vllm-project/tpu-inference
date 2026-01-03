# Qwen3 VL MoE - Vision components are identical to Qwen3 VL dense
# Import all vision-related classes and helper functions from qwen3_vl

from dataclasses import InitVar, dataclass

from tpu_inference.models.jax.qwen3_vl import (
    # Constants
    DEFAULT_BLOCK_K_MAJOR,
    # Helper functions
    _infer_pos_embed_grid_hw,
    generate_segment_ids_from_grid_thw,
    pad_segment_ids_for_attention,
    apply_rotary_pos_emb_vision,
    apply_rotary_pos_emb_thd_padded,
    # Types
    SegmentIds,
    Qwen3VLImagePixelInputs,
    Qwen3VLImageEmbeddingInputs,
    Qwen3VLImageInputs,
    # Vision classes
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionMLP,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionTransformer,
    # Text classes
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.typing import Sharding
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata

init_fn = nnx.initializers.uniform()

modeling_flax_utils = FlaxUtils()

@dataclass(kw_only=True)
class Qwen3VLMoeTextExperts(nnx.Module):
    num_experts: int
    intermediate_size: int
    hidden_size: int
    dtype: jnp.dtype = jnp.bfloat16
    rngs: InitVar[nnx.Rngs] = None
    edf_sharding: Sharding = ('expert', None, 'model')
    efd_sharding: Sharding = ('expert', 'model', None)
    random_init: bool = False

    def __post_init__(self, rngs: nnx.Rngs):
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = create_param(
            rngs,
            shape=(self.num_experts, self.hidden_size, 2 * self.expert_dim),
            sharding=self.edf_sharding,
            dtype=self.dtype,
            random_init=self.random_init
        )
        self.down_proj = create_param(
            rngs,
            shape=(self.num_experts, self.expert_dim, self.hidden_size),
            sharding=self.efd_sharding,
            dtype=self.dtype,
            random_init=self.random_init
        )

    def __call__(
            self, hidden_states: jax.Array, routing_weights: jax.Array, router_indices: jax.Array
        ) -> jax.Array:
        """
        Inference-only forward pass for MoE experts.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size) or (batch_size * seq_len, hidden_size)
            routing_weights: (batch_size * seq_len, num_experts) - full routing weights after scatter
            router_indices: (batch_size * seq_len, top_k) - indices of selected experts

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (T, D)

        # Compute gate_up for all experts: (T, D) @ (E, D, 2F) -> (T, E, 2F)
        gate_up = jnp.einsum('TD,EDF -> TEF', hidden_states, self.gate_up_proj.value)

        # Split into gate and up projections
        gate, up = jnp.split(gate_up, 2, axis=-1)  # (T, E, F) each

        # Apply activation and compute expert outputs
        gated_output = up * jax.nn.silu(gate)  # (T, E, F)

        # Down projection: (T, E, F) @ (E, F, D) -> (T, E, D)
        next_states = jnp.einsum('TEF,EFD -> TED', gated_output, self.down_proj.value)

        # Apply routing weights: (T, E, D) * (T, E, 1) -> (T, E, D)
        next_states = next_states * routing_weights[..., None]

        # Sum across experts: (T, E, D) -> (T, D)
        next_states = next_states.sum(axis=1)

        # Reshape back to batch format
        next_states = next_states.reshape(batch_size, -1, self.hidden_size)

        return next_states

@dataclass(kw_only=True)
class Qwen3VLMoeTextSparseMoeBlock(nnx.Module):
    hidden_size: int
    num_experts: int
    top_k: int
    intermediate_size: int
    dtype: jnp.dtype = jnp.bfloat16
    rngs: InitVar[nnx.Rngs] = None
    gate_sharding: Sharding = (None, 'expert')
    edf_sharding: Sharding = ('expert', None, 'model')
    efd_sharding: Sharding = ('expert', 'model', None)
    random_init: bool = False

    def __post_init__(self, rngs: nnx.Rngs):
        self.gate = create_param(
            rngs,
            shape=(self.hidden_size, self.num_experts),
            sharding=self.gate_sharding,
            dtype=self.dtype,
            random_init=self.random_init
        )
        self.experts = Qwen3VLMoeTextExperts(
            num_experts=self.num_experts,
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            rngs=rngs,
            edf_sharding=self.edf_sharding,
            efd_sharding=self.efd_sharding,
            random_init=self.random_init
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        """
        Forward pass for Sparse MoE block.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)  # (T, D)

        # Compute router logits: (T, D) @ (D, E) -> (T, E)
        router_logits = jnp.einsum('TD,DE -> TE', hidden_states_flat, self.gate.value)

        # Softmax to get routing probabilities (in float32 for numerical stability)
        routing_weights = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)

        # Get top-k experts
        top_k_weights, top_k_indices = jax.lax.top_k(routing_weights, self.top_k)

        # Normalize top-k weights (sum to 1)
        top_k_weights = top_k_weights / top_k_weights.sum(axis=-1, keepdims=True)
        top_k_weights = top_k_weights.astype(self.dtype)

        # Scatter normalized weights back to full expert dimension
        # router_weights shape: (T, E)
        num_tokens = hidden_states_flat.shape[0]
        router_weights = jnp.zeros((num_tokens, self.num_experts), dtype=self.dtype)

        # Create indices for scatter
        token_indices = jnp.arange(num_tokens)[:, None]  # (T, 1)
        token_indices = jnp.broadcast_to(token_indices, top_k_indices.shape)  # (T, top_k)

        # Scatter the top_k weights into the full routing weights matrix
        router_weights = router_weights.at[token_indices, top_k_indices].set(top_k_weights)

        # Reshape hidden_states back to batch format for experts
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)

        routed_out = self.experts(hidden_states, router_weights, top_k_indices)

        return routed_out


class Qwen3VLMoeTextAttention(nnx.Module):
    def __init__(
        self,
        config,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str = "auto",
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads, sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads, sharding_size)

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=self.rms_norm_eps, dtype=dtype)

        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=self.rms_norm_eps, dtype=dtype)

        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rngs,
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
        position_embeddings: Tuple[jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        q = self.q_proj(hidden_states)
        q = self.q_norm(q)

        k = self.k_proj(hidden_states)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q = apply_rotary_pos_emb_thd_padded(q, cos, sin, self.head_dim_original)
        k = apply_rotary_pos_emb_thd_padded(k, cos, sin, self.head_dim_original)

        v = self.v_proj(hidden_states)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
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
        o = self.o_proj(outputs)
        return new_kv_cache, o

class Qwen3VLMoeTextMLP(nnx.Module):
    """Dense SwiGLU MLP for non-MoE layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rngs: nnx.Rngs,
        hidden_act: str = "silu",
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.up_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.down_proj = nnx.Linear(
            intermediate_size, hidden_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )

        if hidden_act == "silu":
            self.act_fn = jax.nn.silu
        else:
            raise NotImplementedError(f"Activation function '{hidden_act}' not implemented")

    def __call__(self, x: jax.Array) -> jax.Array:
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_proj = self.down_proj(gate_output * up_output)
        return down_proj


class Qwen3VLMoeTextDecoderLayer(nnx.Module):
    """Decoder layer that conditionally uses MoE or dense MLP based on layer index."""

    def __init__(
        self,
        config,
        layer_idx: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str = "auto",
    ):
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.self_attn = Qwen3VLMoeTextAttention(
            config=config,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
        )

        # Determine if this layer uses MoE or dense MLP
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0)
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)

        use_moe = (
            layer_idx not in mlp_only_layers
            and num_experts > 0
            and (layer_idx + 1) % decoder_sparse_step == 0
        )

        if use_moe:
            moe_intermediate_size = getattr(config, "moe_intermediate_size", config.intermediate_size)
            top_k = getattr(config, "num_experts_per_tok", 2)
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                intermediate_size=moe_intermediate_size,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.mlp = Qwen3VLMoeTextMLP(
                hidden_size=hidden_size,
                intermediate_size=config.intermediate_size,
                rngs=rngs,
                hidden_act=getattr(config, "hidden_act", "silu"),
                dtype=dtype,
            )

        self.input_layernorm = Qwen3VLTextRMSNorm(
            hidden_size, eps=rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            hidden_size, eps=rms_norm_eps, dtype=dtype
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
        position_embeddings: Tuple[jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        kv_cache, hidden_states = self.self_attn(
            kv_cache=kv_cache,
            hidden_states=hidden_states,
            attention_metadata=attention_metadata,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return kv_cache, hidden_states


class Qwen3VLMoeTextModel(nnx.Module):
    """Text model for Qwen3VL MoE with MRoPE and DeepStack support."""

    def __init__(
        self,
        config,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str = "auto",
    ):
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        rms_norm_eps = config.rms_norm_eps
        num_hidden_layers = config.num_hidden_layers

        self.hidden_size = hidden_size
        self.config = config

        self.embed_tokens = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rngs,
        )

        # layer idx govern layer type
        self.layers = [
            Qwen3VLMoeTextDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
                kv_cache_dtype=kv_cache_dtype,
            )
            for layer_idx in range(num_hidden_layers)
        ]

        self.norm = Qwen3VLTextRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            dtype=dtype,
        )

        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        max_position_embeddings = getattr(config, "max_position_embeddings", 128000)
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)

        mrope_section = None
        if rope_scaling is not None:
            mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_type="default",
            mrope_section=mrope_section,
        )

        # LM head
        tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if tie_word_embeddings:
            self.lm_head = self.embed_tokens.embedding
        else:
            self.lm_head = nnx.Param(
                init_fn(rngs.params(), (hidden_size, vocab_size), dtype),
                sharding=(None, "model"),
            )
        self.tie_word_embeddings = tie_word_embeddings

    def _inject_visual_features(
        self,
        hidden_states: jax.Array,
        visual_pos_mask: jax.Array,
        visual_embeds: jax.Array,
    ) -> jax.Array:
        """Add DeepStack visual features at masked positions.

        Args:
            hidden_states: (seq_len, hidden_size) or (batch, seq_len, hidden_size)
            visual_pos_mask: Boolean mask matching hidden_states without the last dim
            visual_embeds: Visual features (num_visual_tokens, hidden_size)

        Returns:
            Updated hidden_states with visual features added
        """
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        mask = jnp.broadcast_to(visual_pos_mask, hidden_states.shape[:-1])
        flat_mask = mask.reshape(-1).astype(jnp.bool_)

        visual_embeds = visual_embeds.astype(flat_hidden.dtype)
        dummy_row = jnp.zeros((1, flat_hidden.shape[-1]), dtype=flat_hidden.dtype)
        padded_embeds = jnp.concatenate([dummy_row, visual_embeds, dummy_row], axis=0)
        gather_indices = jnp.cumsum(flat_mask, dtype=jnp.int32)
        max_index = visual_embeds.shape[0] + 1
        gather_indices = jnp.minimum(gather_indices, max_index)

        updates = padded_embeds[gather_indices] * flat_mask[:, None]
        updated = flat_hidden + updates

        return updated.reshape(hidden_states.shape)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        visual_pos_mask: Optional[jax.Array] = None,
        deepstack_visual_embeds: Optional[List[jax.Array]] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        """Forward pass with KV cache, MRoPE, and DeepStack support."""
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        # Compute position embeddings from attention metadata
        pos = attention_metadata.input_positions
        # Handle (seq_len,), (3, seq_len), or (3, bs, seq_len) inputs
        if pos.ndim == 1:
            # (seq_len,) -> (3, seq_len) for text-only
            pos = jnp.broadcast_to(pos[None, :], (3, pos.shape[0]))
        elif pos.ndim == 2:
            # (3, seq_len) -> (3, 1, seq_len)
            pos = pos[:, None, :]

        cos, sin = self.rotary_emb(pos)
        position_embeddings = (cos, sin)

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache=kv_cache,
                hidden_states=x,
                attention_metadata=attention_metadata,
                position_embeddings=position_embeddings,
            )
            kv_caches[i] = kv_cache

            if (
                deepstack_visual_embeds is not None
                and i < len(deepstack_visual_embeds)
                and visual_pos_mask is not None
            ):
                x = self._inject_visual_features(
                    x, visual_pos_mask, deepstack_visual_embeds[i]
                )

        x = self.norm(x)

        return kv_caches, x

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.lm_head.value)
        return logits