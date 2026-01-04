# Qwen3 VL MoE - Vision components are identical to Qwen3 VL dense
# Import all vision-related classes and helper functions from qwen3_vl

from dataclasses import InitVar, dataclass

from tpu_inference.models.jax.qwen3_vl import (
    # Constants
    DEFAULT_BLOCK_K_MAJOR,
    # Helper functions
    _infer_pos_embed_grid_hw,
    generate_segment_ids_from_grid_thw,
    get_mrope_input_positions,
    pad_segment_ids_for_attention,
    apply_rotary_pos_emb_vision,
    apply_rotary_pos_emb_thd_padded,
    _ModelConfigAdapter,
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
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.multi_modal_utils import (
    merge_multimodal_embeddings,
)
from tpu_inference.models.jax.utils.weight_utils import (
    get_default_maps,
    load_hf_weights,
)

init_fn = nnx.initializers.uniform()

modeling_flax_utils = FlaxUtils()

logger = init_logger(__name__)

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


class Qwen3VLMoeForConditionalGeneration(nnx.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh,
    ):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        config = vllm_config.model_config.hf_config
        self.config = config
        text_config = getattr(config, "text_config", config)

        self.visual = Qwen3VLVisionTransformer(
            vllm_config=vllm_config,
            rngs=self.rng,
            mesh=mesh,
            norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
        )

        self.language_model = Qwen3VLMoeTextModel(
            config=text_config,
            dtype=vllm_config.model_config.dtype,
            rngs=self.rng,
            mesh=mesh,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
        )

        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

    def get_input_embeddings(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array],
    ) -> jax.Array:
        """Get input embeddings with multimodal content merged.

        Args:
            input_ids: Input token IDs
            multimodal_embeddings: Flattened multimodal embeddings

        Returns:
            Input embeddings with multimodal content merged
        """
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if multimodal_embeddings is not None and multimodal_embeddings.shape[0] != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.image_token_id, self.video_token_id],
            )

        return inputs_embeds

    def embed_input_ids(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute input embeddings and merge multimodal embeddings if present."""
        return self.get_input_embeddings(input_ids, multimodal_embeddings)

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> jax.Array:
        if isinstance(mm_input, list):
            arrays_to_concat = [jnp.asarray(item) for item in mm_input]
            return jnp.concatenate(arrays_to_concat, axis=0)

        if hasattr(mm_input, 'ndim'):
            array_input = jnp.asarray(mm_input)
            if array_input.ndim == 2:
                return array_input
            if array_input.ndim == 3:
                return array_input.reshape(-1, array_input.shape[-1])

        raise ValueError(f"Incorrect type of {name}. "
                         f"Got type: {type(mm_input)}")

    def _normalize_grid_thw(
            self, grid_thw: object) -> Tuple[Tuple[int, int, int], ...]:
        if grid_thw is None:
            return ()
        if isinstance(grid_thw, (list, tuple)):
            if len(grid_thw) == 0:
                return ()
            if len(grid_thw) == 3 and all(
                    isinstance(v, (int, np.integer)) for v in grid_thw):
                return (tuple(int(v) for v in grid_thw), )
            if isinstance(grid_thw[0], (list, tuple)):
                return tuple(tuple(int(v) for v in row) for row in grid_thw)
        if hasattr(grid_thw, 'ndim'):
            array_input = np.asarray(grid_thw)
            if array_input.size == 0:
                return ()
            if array_input.ndim == 1 and array_input.shape[0] == 3:
                return (tuple(int(v) for v in array_input.tolist()), )
            if array_input.ndim == 2 and array_input.shape[1] == 3:
                return tuple(
                    tuple(int(v) for v in row)
                    for row in array_input.tolist())
        raise ValueError("Incorrect type of grid_thw. "
                         f"Got type: {type(grid_thw)}")

    def _parse_and_validate_image_input(
            self, image_grid_thw: Tuple[Tuple[int, int, int], ...],
            **kwargs: object) -> Optional[Qwen3VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            pixel_values = kwargs.pop("pixel_values_videos", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "pixel values")

            if not isinstance(pixel_values, jax.Array):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen3VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw)

        # NOTE: Not supporting image embeddings precomputed. Matches Qwen2.5VL.
        # if image_embeds is not None:
        #     image_embeds = self._validate_and_reshape_mm_tensor(
        #         image_embeds, "image embeds")
        #     if not isinstance(image_embeds, jax.Array):
        #         raise ValueError("Incorrect type of image embeddings. "
        #                          f"Got type: {type(image_embeds)}")
        #     return Qwen3VLImageEmbeddingInputs(
        #         type="image_embeds",
        #         image_embeds=image_embeds,
        #         image_grid_thw=image_grid_thw)

    def _parse_and_validate_multimodal_inputs(self,
                                              image_grid_thw: Tuple[Tuple[int,
                                                                          int,
                                                                          int],
                                                                    ...],
                                              **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "pixel_values_videos",
                             "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(
                        image_grid_thw, **kwargs)
        return mm_input_by_modality

    def _process_image_input(
            self, image_input: Qwen3VLImageInputs
    ) -> tuple[tuple[jax.Array, ...],
               Optional[list[list[jax.Array]]]]:
        grid_thw = image_input["image_grid_thw"]
        if not grid_thw:
            return (), None

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].astype(self.visual.dtype)
            deepstack_embeds = None
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds, deepstack_embeds = self.visual(pixel_values, grid_thw)

        sizes = np.array([
            t * (h // self.spatial_merge_size) * (w // self.spatial_merge_size)
            for t, h, w in grid_thw
        ])

        if sizes.size == 0:
            return (), None
        if sizes.size == 1:
            image_splits = (image_embeds, )
            deepstack_by_item = None
            if deepstack_embeds:
                deepstack_by_item = [
                    [layer_embeds for layer_embeds in deepstack_embeds]
                ]
            return image_splits, deepstack_by_item

        split_indices = np.cumsum(sizes)[:-1]
        image_splits = tuple(jnp.split(image_embeds, split_indices))
        deepstack_by_item = None
        if deepstack_embeds:
            layer_splits = [
                tuple(jnp.split(layer_embeds, split_indices))
                for layer_embeds in deepstack_embeds
            ]
            deepstack_by_item = []
            for item_idx in range(len(image_splits)):
                deepstack_by_item.append(
                    [layer_split[item_idx] for layer_split in layer_splits]
                )
        return image_splits, deepstack_by_item

    def embed_multimodal(
        self,
        image_grid_thw: Tuple[Tuple[int, int, int], ...],
        **kwargs,
    ) -> dict:
        """Get multimodal embeddings from pixel values.

        This method is called by the serving infrastructure (multimodal_manager).
        DeepStack embeddings are returned alongside visual embeddings for caching.

        Args:
            image_grid_thw: Grid dimensions (T, H, W) for each image
            **kwargs: Contains 'pixel_values' for vision encoder

        Returns:
            A dict with:
              - "embeds": Tuple of embeddings, one per image (Qwen 2.5 VL format)
              - "deepstack": Optional list of per-image DeepStack embeddings
        """
        image_grid_thw = self._normalize_grid_thw(image_grid_thw)
        if not image_grid_thw:
            image_grid_thw = self._normalize_grid_thw(
                kwargs.get("video_grid_thw", None))

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            image_grid_thw, **kwargs)
        if not mm_input_by_modality:
            return {}
        if not image_grid_thw:
            return {}

        multimodal_embeddings: tuple[jax.Array, ...] = ()
        deepstack_outputs = None
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_splits, deepstack_by_item = self._process_image_input(
                    multimodal_input)
                multimodal_embeddings += image_splits
                if deepstack_by_item is not None:
                    if deepstack_outputs is None:
                        deepstack_outputs = []
                    deepstack_outputs.extend(deepstack_by_item)

        return {"embeds": multimodal_embeddings, "deepstack": deepstack_outputs}

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        visual_pos_mask = None
        deepstack_embeds = None
        if args:
            candidate = args[-1]
            if isinstance(candidate, (list, tuple)):
                deepstack_embeds = candidate

        if deepstack_embeds and input_ids is not None:
            visual_pos_mask = (input_ids == self.image_token_id) | (
                input_ids == self.video_token_id
            )

        kv_caches, hidden_states = self.language_model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_embeds,
        )

        return kv_caches, hidden_states, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.language_model.compute_logits(hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: List[int],
        hf_config=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths=None,
        use_audio_in_video: bool = False,
    ) -> Tuple[jax.Array, int]:
        """Compute MRoPE 3D position IDs for input sequence.

        This is a wrapper around the module-level get_mrope_input_positions function
        that uses the model's configuration.

        Args:
            input_tokens: List of token IDs for the sequence
            hf_config: Optional HF config (defaults to self.config)
            image_grid_thw: List of (T, H, W) tuples for each image
            video_grid_thw: List of (T, H, W) tuples for each video
            context_len: Context length for slicing positions
            seq_len: Sequence length for slicing positions

        Returns:
            llm_positions: (3, sliced_seq_len) position IDs for [T, H, W]
            mrope_position_delta: Delta for rope calculation
        """
        del second_per_grid_ts, audio_feature_lengths, use_audio_in_video

        if hf_config is None:
            hf_config = self.config

        if video_grid_thw is not None:
            expanded_video = []
            for t, h, w in video_grid_thw:
                t_val = int(t)
                expanded_video.extend([(1, int(h), int(w))] * t_val)
            video_grid_thw = expanded_video

        llm_positions, mrope_position_delta = get_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=getattr(hf_config, "vision_start_token_id",
                                          self.vision_start_token_id),
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )

        llm_positions = llm_positions[:, context_len:seq_len]
        return llm_positions, mrope_position_delta

    def precompile_vision_encoder(
        self,
        run_compilation_fn,
    ) -> None:
        """Precompile vision encoder for warmup.

        Args:
            run_compilation_fn: Function to run compilation with signature
                (name, fn, *args, **kwargs)
        """
        vc = self.config.vision_config
        patch_input_dim = (
            vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        )

        image_shapes = []
        if warmup_config := self.vllm_config.additional_config.get(
            "vision_warmup_config"
        ):
            image_shapes = warmup_config.get("image_shapes", [])

        factor = vc.patch_size * vc.spatial_merge_size
        for input_hw in image_shapes:
            if not isinstance(input_hw, list) or len(input_hw) != 2:
                logger.warning(f"Skipping invalid shape {input_hw}.")
                continue
            h_input, w_input = input_hw
            h_processed = round(h_input / factor) * factor
            w_processed = round(w_input / factor) * factor
            t, h, w = 1, h_processed // vc.patch_size, w_processed // vc.patch_size
            grid_thw = (t, h, w)
            num_patches = t * h * w

            dummy_pixel_values = jnp.ones(
                (num_patches, patch_input_dim),
                self.vllm_config.model_config.dtype,
            )
            dummy_grid_thw = (grid_thw,)

            run_compilation_fn(
                "vision_encoder",
                self.visual.encode_jit,
                dummy_pixel_values,
                dummy_grid_thw,
                image_shape=input_hw,
            )

    def load_weights(self, rng_key: jax.Array) -> None:
        self.rng = nnx.Rngs(rng_key)

        mappings = {
            "model.language_model.embed_tokens": "language_model.embed_tokens.embedding",
            "model.language_model.layers.*.input_layernorm": "language_model.layers.*.input_layernorm.weight",
            "model.language_model.layers.*.mlp.down_proj": "language_model.layers.*.mlp.down_proj.kernel",
            "model.language_model.layers.*.mlp.gate_proj": "language_model.layers.*.mlp.gate_proj.kernel",
            "model.language_model.layers.*.mlp.up_proj": "language_model.layers.*.mlp.up_proj.kernel",
            "model.language_model.layers.*.mlp.gate": "language_model.layers.*.mlp.gate",
            "model.language_model.layers.*.mlp.experts.gate_up_proj": "language_model.layers.*.mlp.experts.gate_up_proj",
            "model.language_model.layers.*.mlp.experts.down_proj": "language_model.layers.*.mlp.experts.down_proj",
            "model.language_model.layers.*.post_attention_layernorm": "language_model.layers.*.post_attention_layernorm.weight",
            "model.language_model.layers.*.self_attn.k_proj": "language_model.layers.*.self_attn.k_proj.kernel",
            "model.language_model.layers.*.self_attn.o_proj": "language_model.layers.*.self_attn.o_proj.kernel",
            "model.language_model.layers.*.self_attn.q_proj": "language_model.layers.*.self_attn.q_proj.kernel",
            "model.language_model.layers.*.self_attn.v_proj": "language_model.layers.*.self_attn.v_proj.kernel",
            "model.language_model.layers.*.self_attn.q_norm": "language_model.layers.*.self_attn.q_norm.weight",
            "model.language_model.layers.*.self_attn.k_norm": "language_model.layers.*.self_attn.k_norm.weight",
            "model.language_model.norm": "language_model.norm.weight",
            "model.visual.patch_embed.proj": "visual.patch_embed.proj.kernel",
            "model.visual.patch_embed.proj.bias": "visual.patch_embed.proj.bias",
            "model.visual.pos_embed": "visual.pos_embed.embedding",
            "model.visual.blocks.*.attn.qkv": "visual.blocks.*.attn.qkv_proj.kernel",
            "model.visual.blocks.*.attn.qkv.bias": "visual.blocks.*.attn.qkv_proj.bias",
            "model.visual.blocks.*.attn.proj": "visual.blocks.*.attn.proj.kernel",
            "model.visual.blocks.*.attn.proj.bias": "visual.blocks.*.attn.proj.bias",
            "model.visual.blocks.*.mlp.linear_fc1": "visual.blocks.*.mlp.fc1.kernel",
            "model.visual.blocks.*.mlp.linear_fc1.bias": "visual.blocks.*.mlp.fc1.bias",
            "model.visual.blocks.*.mlp.linear_fc2": "visual.blocks.*.mlp.fc2.kernel",
            "model.visual.blocks.*.mlp.linear_fc2.bias": "visual.blocks.*.mlp.fc2.bias",
            "model.visual.blocks.*.norm1": "visual.blocks.*.norm1.scale",
            "model.visual.blocks.*.norm1.bias": "visual.blocks.*.norm1.bias",
            "model.visual.blocks.*.norm2": "visual.blocks.*.norm2.scale",
            "model.visual.blocks.*.norm2.bias": "visual.blocks.*.norm2.bias",
            "model.visual.merger.norm": "visual.merger.norm.scale",
            "model.visual.merger.norm.bias": "visual.merger.norm.bias",
            "model.visual.merger.linear_fc1": "visual.merger.linear_fc1.kernel",
            "model.visual.merger.linear_fc1.bias": "visual.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2": "visual.merger.linear_fc2.kernel",
            "model.visual.merger.linear_fc2.bias": "visual.merger.linear_fc2.bias",
        }

        hf_config = self.vllm_config.model_config.hf_config
        if not hf_config.tie_word_embeddings:
            mappings["lm_head"] = "language_model.lm_head"

        vision_config = hf_config.vision_config
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24])
        for i in range(len(deepstack_indexes)):
            mappings[f"model.visual.deepstack_merger_list.{i}.norm"] = f"visual.deepstack_merger_list.{i}.norm.scale"
            mappings[f"model.visual.deepstack_merger_list.{i}.norm.bias"] = f"visual.deepstack_merger_list.{i}.norm.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1"] = f"visual.deepstack_merger_list.{i}.linear_fc1.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1.bias"] = f"visual.deepstack_merger_list.{i}.linear_fc1.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2"] = f"visual.deepstack_merger_list.{i}.linear_fc2.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2.bias"] = f"visual.deepstack_merger_list.{i}.linear_fc2.bias"

        adapted_model_config = _ModelConfigAdapter(self.vllm_config.model_config)
        metadata_map = get_default_maps(
            adapted_model_config, self.mesh, mappings
        )

        transpose_map = {
            "experts.gate_up_proj": (0, 1, 2),
            "experts.down_proj": (0, 1, 2),
        }
        transpose_map.update(metadata_map.transpose_map)
        transpose_map["mlp.gate"] = (1, 0)
        metadata_map.transpose_map = transpose_map

        load_hf_weights(
            vllm_config=self.vllm_config,
            model=self,
            metadata_map=metadata_map,
            mesh=self.mesh,
        )
