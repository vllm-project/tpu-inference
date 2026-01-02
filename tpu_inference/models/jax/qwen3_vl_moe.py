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
)

from typing import Optional, Tuple

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

        # Call experts
        routed_out = self.experts(hidden_states, router_weights, top_k_indices)

        return routed_out


class Qwen3VLMoeTextAttention(nnx.Module):
    # MHA
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str = "auto",
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.dtype = dtype
        self.mesh = mesh

        self.head_dim_original = head_dim
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim_original ** -0.5

        # Pad heads for sharding
        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads, sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads, sharding_size)

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=rms_norm_eps, dtype=dtype)

        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=rms_norm_eps, dtype=dtype)

        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )

        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rngs,
        )

        # KV cache quantization
        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(kv_cache_dtype)

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: Tuple[jax.Array, jax.Array],
        attention_mask: Optional[jax.Array] = None, # additive mask
    ) -> jax.Array:
        query_states = self.q_proj(hidden_states)  # (T, N, H)
        key_states = self.k_proj(hidden_states)    # (T, K, H)
        value_states = self.v_proj(hidden_states)  # (T, K, H)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb_thd_padded(query_states, cos, sin, self.head_dim_original)
        key_states = apply_rotary_pos_emb_thd_padded(key_states, cos, sin, self.head_dim_original)

        # Repeat KV heads for GQA
        if self.num_key_value_groups > 1:
            key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=1)

        # query_states: (T, N, H), key_states: (T, N, H), value_states: (T, N, H)
        attn_weights = jnp.einsum('TNH,SNH->TNS', query_states, key_states) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(self.dtype)

        attn_output = jnp.einsum('TNS,SNH->TNH', attn_weights, value_states)
        attn_output = self.o_proj(attn_output)

        return attn_output