import math
from functools import partial
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import (
    attention,
    sharded_flash_attention,
)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.multi_modal_utils import (
    merge_multimodal_embeddings,
)
from tpu_inference.models.jax.utils.weight_utils import (
    get_default_maps,
    load_hf_weights,
)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

DEFAULT_BLOCK_K_MAJOR = 128


class _Qwen3VLConfigAdapter:
    """Adapter that exposes text_config attributes at the top level.

    Qwen3VLConfig has a nested structure where text model attributes
    (hidden_size, num_attention_heads, etc.) are under text_config.
    This adapter makes them accessible directly for compatibility with
    weight loading utilities that expect a flat config structure.
    """

    def __init__(self, config):
        self._config = config
        self._text_config = getattr(config, "text_config", None)

    def __getattr__(self, name):
        # First try the main config
        try:
            return getattr(self._config, name)
        except AttributeError:
            pass
        # Fall back to text_config if available
        if self._text_config is not None:
            try:
                return getattr(self._text_config, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self._config).__name__}' object has no attribute '{name}'"
        )


class _ModelConfigAdapter:
    """Adapter for model_config that wraps hf_config with _Qwen3VLConfigAdapter.

    This allows get_default_maps to access text_config attributes directly
    from hf_config without modifying the weight_utils module.
    """

    def __init__(self, model_config):
        self._model_config = model_config
        self._hf_config_adapter = _Qwen3VLConfigAdapter(model_config.hf_config)

    @property
    def hf_config(self):
        return self._hf_config_adapter

    def __getattr__(self, name):
        return getattr(self._model_config, name)


def _infer_pos_embed_grid_hw(num_position_embeddings: int) -> Tuple[int, int]:
    """Infer a (grid_h, grid_w) pair from a flattened 2D embedding table."""
    if num_position_embeddings <= 0:
        raise ValueError(
            f"num_position_embeddings must be positive, got {num_position_embeddings}"
        )
    root = int(math.sqrt(num_position_embeddings))
    for grid_h in range(root, 0, -1):
        if num_position_embeddings % grid_h == 0:
            return grid_h, num_position_embeddings // grid_h
    return num_position_embeddings, 1


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers). Only tokens with the same id can attend to each other.

    Attributes:
        q: segment ids along the Q sequence.
        kv: segment ids along the KV sequence.
    """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


def generate_segment_ids_from_grid_thw(
    grid_thw: Tuple[Tuple[int, int, int], ...],
) -> jax.Array:
    """Generate segment IDs from grid dimensions for variable-length attention.

    Each frame gets a unique segment ID (starting from 1). For images (t=1),
    this means one segment ID per image. For videos (t>1), each temporal frame
    gets its own segment ID, preventing cross-frame attention. Temporal order
    is preserved through separate tokens that annotate frame positions.

    Args:
        grid_thw: Tuple of (T, H, W) for each image/video

    Returns:
        segment_ids: (total_tokens,) array with segment IDs per frame
    """
    segments = []
    seg_id = 1 # start for 1 to make padding 0

    for (t, h, w) in grid_thw:
        frame_size = h * w
        for _ in range(t):
            segments.append(jnp.full((frame_size,), seg_id, dtype=jnp.int32))
            seg_id += 1

    return jnp.concatenate(segments, axis=0) if segments else jnp.zeros((0,), dtype=jnp.int32)


def pad_segment_ids_for_attention(
    segment_ids: jax.Array,
    padded_seq_len: int,
) -> SegmentIds:
    """Pad segment IDs and format for flash attention.

    Args:
        segment_ids: (seq_len,) segment IDs for valid tokens
        padded_seq_len: Padded sequence length (multiple of block size)

    Returns:
        SegmentIds for flash attention with padding tokens set to 0
    """
    seq_len = segment_ids.shape[0]
    padded_ids = jnp.zeros(padded_seq_len, dtype=jnp.int32)
    padded_ids = padded_ids.at[:seq_len].set(segment_ids)
    padded_ids = padded_ids.reshape(1, -1)
    return SegmentIds(q=padded_ids, kv=padded_ids)


def compute_vision_counts_per_sequence(
    input_ids: jax.Array,
    attention_mask: jax.Array,
    image_token_id: int,
    video_token_id: int,
) -> Tuple[jax.Array, jax.Array]:
    """Compute the number of images and videos per sequence in a batch.

    This is a preprocessing function for MRoPE computation. It does not JIT.

    Args:
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        image_token_id: Token ID for image placeholders
        video_token_id: Token ID for video placeholders

    Returns:
        num_images_per_sequence: (batch_size,)
        num_videos_per_sequence: (batch_size,)
    """
    # Mask invalid positions
    masked_ids = jnp.where(attention_mask == 1, input_ids, -1)

    # Count per sequence
    num_images = jnp.sum(masked_ids == image_token_id, axis=1)
    num_videos = jnp.sum(masked_ids == video_token_id, axis=1)

    return num_images, num_videos


def get_mrope_input_positions(
    input_tokens: List[int],
    image_grid_thw: Optional[List[Tuple[int, int, int]]],
    video_grid_thw: Optional[List[Tuple[int, int, int]]],
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> Tuple[jax.Array, int]:
    """Compute MRoPE 3D position IDs for a single sequence.

    This function computes position IDs for text and vision tokens.
    Text tokens get sequential positions, while vision tokens get
    3D positions based on their temporal, height, and width coordinates.

    Args:
        input_tokens: List of token IDs for the sequence
        image_grid_thw: List of (T, H, W) tuples for each image
        video_grid_thw: List of (T, H, W) tuples for each video
        image_token_id: Token ID for image placeholders
        video_token_id: Token ID for video placeholders
        vision_start_token_id: Token ID marking start of vision tokens
        spatial_merge_size: Spatial merge size from vision config

    Returns:
        llm_positions: (3, seq_len) position IDs for [T, H, W]
        mrope_position_delta: Delta for rope calculation
    """
    input_tokens_np = np.array(input_tokens)
    vision_start_indices = np.where(input_tokens_np == vision_start_token_id)[0]
    # Guard against `vision_start_token_id` appearing at the last position.
    vision_start_indices = vision_start_indices[
        vision_start_indices + 1 < input_tokens_np.size
    ]
    vision_tokens = (
        input_tokens_np[vision_start_indices + 1]
        if vision_start_indices.size
        else np.array([], dtype=input_tokens_np.dtype)
    )
    image_nums = int(np.sum(vision_tokens == image_token_id))
    video_nums = int(np.sum(vision_tokens == video_token_id))

    # If vision placeholder tokens exist, matching grids must be provided.
    if image_nums > 0 and not image_grid_thw:
        raise ValueError("image_grid_thw must be provided when image tokens are present.")
    if video_nums > 0 and not video_grid_thw:
        raise ValueError("video_grid_thw must be provided when video tokens are present.")
    # Validate grid counts match token counts
    if image_grid_thw and len(image_grid_thw) < image_nums:
        raise ValueError(
            f"image_grid_thw requires {image_nums} entries but found {len(image_grid_thw)}"
        )
    if video_grid_thw and len(video_grid_thw) < video_nums:
        raise ValueError(
            f"video_grid_thw requires {video_nums} entries but found {len(video_grid_thw)}"
        )
    llm_pos_ids_list = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    image_index, video_index = 0, 0

    for _ in range(image_nums + video_nums):
        if remain_images > 0:
            try:
                ed_image = input_tokens.index(image_token_id, st)
            except ValueError:
                ed_image = len(input_tokens) + 1
        else:
            ed_image = len(input_tokens) + 1

        if remain_videos > 0:
            try:
                ed_video = input_tokens.index(video_token_id, st)
            except ValueError:
                ed_video = len(input_tokens) + 1
        else:
            ed_video = len(input_tokens) + 1

        if ed_image < ed_video:
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            remain_images -= 1
            ed = ed_image
        else:
            t, h, w = video_grid_thw[video_index]  # t=1
            video_index += 1
            remain_videos -= 1
            ed = ed_video

        # t would always be 1
        llm_grid_t = t
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size
        text_len = ed - st

        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            ) + st_idx
        )

        # t_index alaways zero
        num_vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w

        t_index = jnp.broadcast_to(
            jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1),
            (llm_grid_t, llm_grid_h * llm_grid_w),
        ).flatten()  # llm_grid_t=1 then [0, 0, 0, ...]

        h_index = jnp.broadcast_to(
            jnp.arange(llm_grid_h, dtype=jnp.int32).reshape(1, -1, 1),
            (llm_grid_t, llm_grid_h, llm_grid_w),
        ).flatten()

        w_index = jnp.broadcast_to(
            jnp.arange(llm_grid_w, dtype=jnp.int32).reshape(1, 1, -1),
            (llm_grid_t, llm_grid_h, llm_grid_w),
        ).flatten()

        llm_pos_ids_list.append(
            jnp.stack([t_index, h_index, w_index]) + text_len + st_idx
        )

        st = ed + num_vision_tokens

    # Trailing text
    if st < len(input_tokens):
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            ) + st_idx
        )

    llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = int(llm_positions.max()) + 1 - len(input_tokens)

    return llm_positions, mrope_position_delta


def apply_interleaved_mrope(
    freqs: jax.Array,
    mrope_section: List[int] = [24, 20, 20],
) -> jax.Array:
    """

    :param freqs: MRoPE frequency derived from T, H, W values. (3, bs, seq, 64)
    :param mrope_section: How MRoPE would be placed.
    :return: A final interleaved MRoPE frequency. (bs, seq, 64)
    """
    # Validate mrope_section
    if len(mrope_section) != 3:
        raise ValueError(
            f"mrope_section must have length 3, got {len(mrope_section)}"
        )
    if any(s < 0 for s in mrope_section):
        raise ValueError(
            f"mrope_section values must be non-negative, got {mrope_section}"
        )
    head_dim_half = freqs.shape[-1]
    if sum(mrope_section) != head_dim_half:
        raise ValueError(
            f"mrope_section must sum to {head_dim_half}, got {sum(mrope_section)}"
        )

    t, h, w = mrope_section # 24, 20, 20

    result = freqs[0].copy()

    h_indices = jnp.arange(1, h * 3, 3)
    result = result.at[..., h_indices].set(freqs[1][..., h_indices])

    w_indices = jnp.arange(2, w * 3, 3)  # [2, 5, 8, ..., 59]
    result = result.at[..., w_indices].set(freqs[2][..., w_indices])

    return result


class Qwen3VLTextRMSNorm(nnx.Module):
    """RMSNorm for Qwen3VL text model. Equivalent to T5LayerNorm."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.weight = nnx.Param(jnp.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return self.weight.value * hidden_states.astype(input_dtype)


def rotate_half(x: jax.Array) -> jax.Array:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_thd_padded(
    x: jax.Array,          # (T, H, padded_dim)
    cos: jax.Array,        # (bs, T, head_dim) or (T, head_dim)
    sin: jax.Array,        # (bs, T, head_dim) or (T, head_dim)
    head_dim: int,         # head_dim_original
) -> jax.Array:
    """Match HF/PyTorch RoPE application on Q/K, but with padded head_dim support. The batch size is fake, so packing would be required?"""
    x_dtype = x.dtype
    x_f = x.astype(jnp.float32)

    # Normalize cos/sin shapes to (T, head_dim)
    if cos.ndim == 3:
        # (bs, T, head_dim) -> (T, head_dim) ; your pipeline is effectively bs=1
        cos = cos[0]
        sin = sin[0]

    cos_f = cos.astype(jnp.float32)[:, None, :]  # (T, 1, head_dim)
    sin_f = sin.astype(jnp.float32)[:, None, :]  # (T, 1, head_dim)

    # Rotate only the real (unpadded) head_dim
    x_main = x_f[..., :head_dim]  # (T, H, head_dim)
    x_rot = rotate_half(x_main)   # uses split-half rotate like HF
    out_main = (x_main * cos_f) + (x_rot * sin_f)

    # Reconstruct padded output: padded dims must be zeros (equivalent to "no dims exist" in PyTorch)
    if x.shape[-1] > head_dim:
        out = jnp.zeros_like(x_f)
        out = out.at[..., :head_dim].set(out_main)
    else:
        out = out_main

    return out.astype(x_dtype)


class Qwen3VLTextRotaryEmbedding(nnx.Module):
    """
    Multimodal Rotary Position Embedding (MRoPE) for Qwen3VL text model.
    Supports 3D position encoding with temporal, height, and width dimensions.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 128000,
        rope_theta: float = 5000000.0,
        rope_type: str = "default",
        mrope_section: List[int] = None,
    ):
        """
        Args:
            dim: Head dimension
            max_position_embeddings: Maximum sequence length
            rope_theta: Base frequency for RoPE
            rope_type: Type of RoPE initialization ("default" for now)
            mrope_section: Section sizes for MRoPE interleaving [T_dim, H_dim, W_dim]
        """
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.rope_type = rope_type

        if rope_type != "default":
            raise NotImplementedError(f"RoPE type '{rope_type}' not yet implemented")

        # Compute inverse frequencies: shape (dim // 2,)
        inv_freq = 1.0 / (
            rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )
        self.inv_freq = inv_freq

        # MRoPE section for interleaving [T_dim, H_dim, W_dim]
        # Must sum to dim // 2 = 64
        self.mrope_section = mrope_section if mrope_section is not None else [24, 20, 20]

    def __call__(
        self,
        position_ids: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Compute cos/sin embeddings from 3D position IDs.

        Args:
            position_ids: Position IDs of shape (3, seq_len) or (3, bs, seq_len)
                          where dim 0 is [T, H, W] positions

        Returns:
            cos: Cosine embeddings of shape (bs, seq_len, dim)
            sin: Sine embeddings of shape (bs, seq_len, dim)
        """
        # Handle both (3, seq_len) and (3, bs, seq_len) inputs
        if position_ids.ndim == 2:
            # (3, seq_len) -> (3, 1, seq_len)
            position_ids = position_ids[:, None, :]

        # position_ids: (3, bs, seq_len)
        # inv_freq: (dim // 2,)
        # freqs: (3, bs, seq_len, dim // 2)
        freqs = position_ids[:, :, :, None].astype(jnp.float32) * self.inv_freq[None, None, None, :]

        # Apply MRoPE interleaving: (3, bs, seq_len, dim//2) -> (bs, seq_len, dim//2)
        freqs = apply_interleaved_mrope(freqs, self.mrope_section)
        # `apply_rotary_pos_emb` uses `rotate_half` on the full head dim, so
        # duplicate the half-dim freqs to produce (bs, seq_len, dim).
        freqs = jnp.concatenate([freqs, freqs], axis=-1)

        # Compute cos and sin
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)

        return cos.astype(jnp.bfloat16), sin.astype(jnp.bfloat16)


class Qwen3VLTextAttention(nnx.Module):
    """Multi-headed attention for Qwen3VL text model with KV cache support."""

    def __init__(
        self,
        config,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size //
                                         self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        mrope_section = None
        if self.rope_scaling is not None:
            mrope_section = self.rope_scaling.get("mrope_section", [24, 20, 20]) # should be this always for dense

        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            dim=self.head_dim_original,
            max_position_embeddings=getattr(config, "max_position_embeddings", 128000),
            rope_theta=self.rope_theta,
            rope_type="default",
            mrope_section=mrope_section,
        )

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

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
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        q = self.q_proj(hidden_states)
        q = self.q_norm(q)

        k = self.k_proj(hidden_states)
        k = self.k_norm(k)

        # Ensure positions are treated as Qwen3-VL does: always 3D positions for MRoPE
        pos = md.input_positions
        if pos.ndim == 1:
            # Expand vanilla positions to (3, T) for text-only
            pos = jnp.broadcast_to(pos[None, :], (3, pos.shape[0]))

        cos, sin = self.rotary_emb(pos)  # (bs, T, head_dim_original)

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


class Qwen3VLTextMLP(nnx.Module):
    """SwiGLU MLP for Qwen3VL text model.

    Uses gated activation: down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rngs: nnx.Rngs,
        hidden_act: str = "silu",
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            intermediate_size: Intermediate dimension size
            rngs: Random number generators
            hidden_act: Activation function
            dtype: Data type for parameters
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # All projections have no bias in Qwen3VL
        self.gate_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.up_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.down_proj = nnx.Linear(
            intermediate_size, hidden_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )

        # Activation function (default is SiLU for SwiGLU)
        if hidden_act == "silu":
            self.act_fn = jax.nn.silu
        else:
            raise NotImplementedError(f"Activation function '{hidden_act}' not implemented")

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass using SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_proj = self.down_proj(gate_output * up_output)
        return down_proj


class Qwen3VLTextDecoderLayer(nnx.Module):
    """Transformer decoder layer for Qwen3VL text model with KV cache."""

    def __init__(
        self,
        config,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
    ):
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.self_attn = Qwen3VLTextAttention(
            config=config,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
        )
        self.mlp = Qwen3VLTextMLP(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            rngs=rngs,
            hidden_act=getattr(config, "hidden_act", "silu"),
            dtype=dtype,
        )
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size,
                                                  eps=rms_norm_eps,
                                                  dtype=dtype)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(hidden_size,
                                                           eps=rms_norm_eps,
                                                           dtype=dtype)

    def __call__(
        self,
        kv_cache: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        kv_cache, hidden_states = self.self_attn(
            kv_cache=kv_cache,
            hidden_states=hidden_states,
            attention_metadata=attention_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return kv_cache, hidden_states


def apply_rotary_pos_emb_vision(
    x: jax.Array, rotary_pos_emb: jax.Array
) -> jax.Array:
    """Apply rotary position embedding to vision tensors.

    Args:
        x: Input tensor of shape [B, T, N, H]
        rotary_pos_emb: Rotary embeddings of shape [T, H//2]

    Returns:
        Rotated tensor of shape [B, T, N, H]
    """
    _, _, _, H = x.shape
    half_dim = H // 2

    # [B, T, N, H//2]
    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]

    # [T, H//2]
    cos_emb = jnp.cos(rotary_pos_emb)
    sin_emb = jnp.sin(rotary_pos_emb)

    # [1, T, 1, H//2]
    cos_emb = cos_emb[None, :, None, :]
    sin_emb = sin_emb[None, :, None, :]

    # [B, T, N, H//2]
    x_rotated_real = x_real * cos_emb - x_imag * sin_emb
    x_rotated_imag = x_real * sin_emb + x_imag * cos_emb

    # [B, T, N, H]
    x_rotated = jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)

    return x_rotated


class Qwen3VLVisionRotaryEmbedding(nnx.Module):
    """Rotary position embedding for vision encoder."""

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (
            self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)



class Qwen3VLVisionPatchEmbed(nnx.Module):
    """3D Patch Embedding for video/image input using 3D convolution."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, None, None, None, "model")
            ),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply 3D patch embedding.

        Args:
            x: Input tensor of shape (num_patches, C * T * H * W)

        Returns:
            Embedded patches of shape (num_patches, hidden_size)
        """
        L, dim = x.shape
        C = dim // (
            self.temporal_patch_size * self.patch_size * self.patch_size
        )
        # Reshape to (L, T, H, W, C) for Conv3D with channels_last
        x = x.reshape(
            L, C, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # L, T, H, W, C
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv, shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        return x



class Qwen3VLVisionMLP(nnx.Module):
    """SwiGLU-style MLP for vision encoder."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.fc1 = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        return self.fc2(x)



class Qwen3VLVisionAttention(nnx.Module):
    """Full attention for vision encoder using sharded flash attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads, sharding_size)
        self.head_dim = hidden_size // num_heads  # Original head dim

        self.mesh = mesh

        # QKV projection
        self.qkv_proj = nnx.Linear(
            hidden_size,
            3 * hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )

        # Output projection
        self.proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

        # Qwen3VL's Vision Transformer uses full bidirectional attention.
        self.flash_attention = sharded_flash_attention(
            mesh=mesh,
            causal=False,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            vmem_limit_bytes=128 * 1024 * 1024,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> jax.Array:
        """Apply vision attention with variable-length segment masking.

        Args:
            x: Input tensor of shape (T, B, D)
            rotary_pos_emb: Rotary position embeddings
            segment_ids: Segment IDs (T,) for variable-length attention masking

        Returns:
            Output tensor of shape (T, B, D)
        """
        T, B, D = x.shape
        assert B == 1, "Vision attention currently only supports batch size 1"

        qkv = self.qkv_proj(x)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Head-last reshape
        q = q.reshape(T, B, self.num_heads, self.head_dim)
        k = k.reshape(T, B, self.num_heads, self.head_dim)
        v = v.reshape(T, B, self.num_heads, self.head_dim)

        # Transpose to [B, T, N, H]
        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # Transpose to [B, N, T, H] for flash attention
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Pad sequence length to multiple of block size
        block_k_major = DEFAULT_BLOCK_K_MAJOR
        T_attn = q.shape[2]
        padded_T = (T_attn + block_k_major - 1) // block_k_major * block_k_major
        pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))

        q = jnp.pad(q, pad_width, "constant")
        k = jnp.pad(k, pad_width, "constant")
        v = jnp.pad(v, pad_width, "constant")

        # Pad segment IDs for attention (padding tokens get segment_id=0)
        padded_segment_ids = pad_segment_ids_for_attention(segment_ids, padded_T)

        output = self.flash_attention(q, k, v, padded_segment_ids)

        # Unpad and reshape: [B, N, T, H] -> [T, B, N, H] -> [T, B, D]
        output = output[:, :, :T_attn, :]
        output = jnp.transpose(output, (2, 0, 1, 3))
        output = output.reshape(T, B, D)

        output = self.proj(output)

        return output



class Qwen3VLVisionBlock(nnx.Module):
    """Transformer block for vision encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        norm_eps: float = 1e-6,
    ):
        self.norm1 = nnx.LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )
        self.attn = Qwen3VLVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.norm2 = nnx.LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )
        self.mlp = Qwen3VLVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, segment_ids)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionPatchMerger(nnx.Module):
    """Merge spatial patches and project to language model dimension.

    This module supports both:
    - Final merger: norm before reshape (use_postshuffle_norm=False)
    - DeepStack merger: reshape before norm (use_postshuffle_norm=True)
    """

    def __init__(
        self,
        d_model: int, # in
        context_dim: int, # out
        spatial_merge_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        use_postshuffle_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        # Normalization dimension depends on use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nnx.LayerNorm(
            norm_dim,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )

        self.linear_fc1 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )
        self.linear_fc2 = nnx.Linear(
            self.hidden_size,
            d_model,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply patch merging.

        If use_postshuffle_norm is True, reshaping would happen first.
        """
        if self.use_postshuffle_norm:
            # DeepStack: reshape first, then norm
            x = x.reshape(-1, self.hidden_size)
            x = self.norm(x)
        else:
            # Final merger: norm first, then reshape
            x = self.norm(x)
            x = x.reshape(-1, self.hidden_size)

        x = self.linear_fc1(x)
        x = nnx.gelu(x) # make this configurable?
        x = self.linear_fc2(x)
        return x


class Qwen3VLVisionTransformer(nnx.Module):
    """Vision Transformer for Qwen3VL with DeepStack support."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        rngs: nnx.Rngs,
        mesh: Mesh,
        norm_eps: float = 1e-6,
    ):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vision_config = hf_config.vision_config
        dtype = model_config.dtype

        self.config = vision_config
        self.dtype = dtype

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        # analyze full attn block indexes' purpose by visiting flash attention impl.
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            rngs=rngs,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
            dtype=dtype,
        )

        # Learned PE, 48 x 48 H W
        num_position_embeddings = getattr(
            vision_config, "num_position_embeddings", 2304
        )
        self.pos_embed = nnx.Embed(
            num_embeddings=num_position_embeddings,
            features=self.hidden_size,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rngs,
        )
        # NOTE: Learned PE can be rectangular (H != W); infer a base (H, W) grid.
        image_size = getattr(vision_config, "image_size", None)
        pos_embed_grid_h = pos_embed_grid_w = None
        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            h0, w0 = int(image_size[0]), int(image_size[1])
            if h0 > 0 and w0 > 0 and h0 * w0 == num_position_embeddings:
                pos_embed_grid_h, pos_embed_grid_w = h0, w0
            else:
                h1, w1 = h0 // patch_size, w0 // patch_size
                if h1 > 0 and w1 > 0 and h1 * w1 == num_position_embeddings:
                    pos_embed_grid_h, pos_embed_grid_w = h1, w1
        elif isinstance(image_size, int):
            s0 = int(image_size)
            if s0 > 0 and s0 * s0 == num_position_embeddings:
                pos_embed_grid_h = pos_embed_grid_w = s0
            else:
                s1 = s0 // patch_size
                if s1 > 0 and s1 * s1 == num_position_embeddings:
                    pos_embed_grid_h = pos_embed_grid_w = s1
        if pos_embed_grid_h is None:
            if image_size is not None:
                logger.warning(
                    "Couldn't derive pos_embed grid from vision_config.image_size="
                    f"{image_size!r}; inferring from num_position_embeddings={num_position_embeddings}."
                )
            pos_embed_grid_h, pos_embed_grid_w = _infer_pos_embed_grid_hw(
                num_position_embeddings
            )
        self.pos_embed_grid_h = pos_embed_grid_h
        self.pos_embed_grid_w = pos_embed_grid_w

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        intermediate_size = vision_config.intermediate_size
        self.blocks = [
            Qwen3VLVisionBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=intermediate_size,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
                norm_eps=norm_eps,
            )
            for _ in range(vision_config.depth)
        ]

        # Final merger settings
        # Qwen3VLConfig uses text_config for text model hidden_size
        text_config = getattr(hf_config, "text_config", hf_config)
        out_hidden_size = getattr(vision_config, "out_hidden_size", text_config.hidden_size)

        self.merger = Qwen3VLVisionPatchMerger(
            d_model=out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            use_postshuffle_norm=False,
            norm_eps=norm_eps,
        )

        # DeepStack configuration
        self.deepstack_visual_indexes = getattr(
            vision_config, "deepstack_visual_indexes", [8, 16, 24]
        )
        self.deepstack_merger_list = [
            Qwen3VLVisionPatchMerger(
                d_model=out_hidden_size,
                context_dim=self.hidden_size,
                spatial_merge_size=self.spatial_merge_size,
                dtype=dtype,
                rngs=rngs,
                use_postshuffle_norm=True,  # DeepStack uses postshuffle norm
                norm_eps=norm_eps,
            )
            for _ in range(len(self.deepstack_visual_indexes))
        ]

        # TODO: Setting this to True should make patch module forward to use eager mode.
        # However, transformer blocks(causes most overhead) may be JIT-ed. Padding would be required for efficiency.
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        self.enable_dynamic_image_sizes = additional_config.get(
            "enable_dynamic_image_sizes", False
        )

    def rotary_pos_emb_thw(
        self, t: int, h: int, w: int
    ) -> jax.Array:
        """Compute rotary position embeddings for a grid of patches.

        Args:
            t: Temporal dimension
            h: Height dimension (in patches)
            w: Width dimension (in patches)

        Returns:
            Rotary embeddings of shape (t * merged_h * merged_w, spatial_merge_unit, dim)
        """
        merge_size = self.spatial_merge_size

        hpos_ids, wpos_ids = jnp.indices((h, w))
        hpos_ids = (
            hpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )
        wpos_ids = (
            wpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )
        pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids = jnp.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)

        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )

        return rotary_pos_emb

    def fast_pos_embed_interpolate(
        self, grid_thw: Tuple[Tuple[int, int, int], ...]
    ) -> jax.Array:
        """Bilinear interpolation for learned positional embeddings.

        Args:
            grid_thw: Tuple of (T, H, W) for each image/video

        Returns:
            Position embeddings of shape (total_patches, hidden_size)
        """
        merge_size = self.spatial_merge_size

        idx_list = [jnp.empty((0,), dtype=jnp.int32) for _ in range(4)]
        weight_list = [jnp.empty((0,), dtype=jnp.float32) for _ in range(4)]

        grid_ts = [g[0] for g in grid_thw]
        grid_hs = [g[1] for g in grid_thw]
        grid_ws = [g[2] for g in grid_thw]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            # Create linearly spaced indices for interpolation
            h_idxs = jnp.linspace(0, self.pos_embed_grid_h - 1, h)
            w_idxs = jnp.linspace(0, self.pos_embed_grid_w - 1, w)

            # Floor and ceil indices for bilinear interpolation
            h_idxs_floor = h_idxs.astype(jnp.int32)
            w_idxs_floor = w_idxs.astype(jnp.int32)
            h_idxs_ceil = (h_idxs.astype(jnp.int32) + 1).clip(
                max=self.pos_embed_grid_h - 1
            )
            w_idxs_ceil = (w_idxs.astype(jnp.int32) + 1).clip(
                max=self.pos_embed_grid_w - 1
            )

            # Interpolation weights
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            # Base indices for 2D grid (flattened)
            base_h = h_idxs_floor * self.pos_embed_grid_w
            base_h_ceil = h_idxs_ceil * self.pos_embed_grid_w

            # Compute 4 corner indices for bilinear interpolation
            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).reshape(-1),  # top-left
                (base_h[:, None] + w_idxs_ceil[None, :]).reshape(-1),   # top-right
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).reshape(-1),  # bottom-left
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).reshape(-1),   # bottom-right
            ]

            # Compute weights for bilinear interpolation
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),  # top-left
                ((1 - dh)[:, None] * dw[None, :]).reshape(-1),        # top-right
                (dh[:, None] * (1 - dw)[None, :]).reshape(-1),        # bottom-left
                (dh[:, None] * dw[None, :]).reshape(-1),              # bottom-right
            ]

            for i in range(4):
                idx_list[i] = jnp.concatenate([idx_list[i], indices[i]], axis=0)
                weight_list[i] = jnp.concatenate([weight_list[i], weights[i]], axis=0)

        # Convert to JAX arrays
        idx_tensor = jnp.stack(idx_list, axis=0)  # int32, shape (4, N)
        weight_tensor = jnp.stack(weight_list, axis=0)  # float32, shape (4, N)

        # Lookup embeddings and apply bilinear interpolation
        # TODO: `weight_tensor` is float32 and can promote the embedding dtype.
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Split embeddings for each image/video
        split_sizes = [h * w for h, w in zip(grid_hs, grid_ws)]

        patch_pos_embeds_list = []
        offset = 0
        for size in split_sizes:
            patch_pos_embeds_list.append(patch_pos_embeds[offset: offset + size])
            offset += size

        # Rearrange embeddings to match patch ordering (with merge blocks)
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds_list, grid_ts, grid_hs, grid_ws):
            # Repeat for temporal dimension
            pos_embed = jnp.tile(pos_embed, (t, 1))

            # Rearrange to match merge block ordering
            # TODO: This assumes `h` and `w` are divisible by `spatial_merge_size`.
            h_merged = h // merge_size
            w_merged = w // merge_size
            pos_embed = pos_embed.reshape(
                t, h_merged, merge_size, w_merged, merge_size, -1
            )
            pos_embed = jnp.transpose(pos_embed, (0, 1, 3, 2, 4, 5))
            pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])

            patch_pos_embeds_permute.append(pos_embed)

        patch_pos_embeds = jnp.concatenate(patch_pos_embeds_permute, axis=0)
        return patch_pos_embeds

    def encode(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> Tuple[jax.Array, List[jax.Array]]:
        hidden_states = self.patch_embed(x)

        # Reshape for merge block ordering
        seq_len = x.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )

        # Add batch dimension: (seq, merge_unit, dim) -> (seq * merge_unit, 1, dim)
        hidden_states = hidden_states.reshape(seq_len, 1, -1)

        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                segment_ids=segment_ids,
            )

            # Collect DeepStack features at specified layers
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                # Squeeze batch dim for merger: (seq, 1, dim) -> (seq, dim)
                deepstack_feat = self.deepstack_merger_list[idx](
                    hidden_states.squeeze(1)
                )
                deepstack_features.append(deepstack_feat)

        # Squeeze batch dim and apply final merger
        hidden_states = self.merger(hidden_states.squeeze(1))

        return hidden_states, deepstack_features

    @partial(jax.jit, static_argnames=("grid_thw",))
    def encode_jit(
        self, x: jax.Array, grid_thw: Tuple[Tuple[int, int, int], ...]
    ) -> Tuple[jax.Array, List[jax.Array]]:
        """JIT-compiled encoding with static grid dimensions."""
        # Learned PE interpolate
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        hidden_states = self.patch_embed(x)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb_list = []
        for t, h, w in grid_thw:
            rotary_pos_emb_list.append(self.rotary_pos_emb_thw(t, h, w))
        rotary_pos_emb = jnp.concatenate(rotary_pos_emb_list, axis=0)
        rotary_pos_emb = rotary_pos_emb.reshape(-1, rotary_pos_emb.shape[-1])

        # pre-merge patch segment ids
        segment_ids = generate_segment_ids_from_grid_thw(
            grid_thw
        )
        assert segment_ids.shape[0] == hidden_states.shape[0], (
            "segment_ids must match the patch sequence length. "
            f"Got {segment_ids.shape[0]=} vs {hidden_states.shape[0]=}.")

        # Reshape for transformer
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states.reshape(seq_len, 1, -1)

        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                segment_ids=segment_ids,
            )

            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feat = self.deepstack_merger_list[idx](
                    hidden_states.squeeze(1)
                )
                deepstack_features.append(deepstack_feat)

        hidden_states = self.merger(hidden_states.squeeze(1))

        return hidden_states, deepstack_features

    def __call__(
        self, x: jax.Array, grid_thw: Tuple[Tuple[int, int, int], ...]
    ) -> Tuple[jax.Array, List[jax.Array]]:
        """Forward pass for vision encoder.

        Args:
            x: Pixel values of shape (num_patches, C * T * ps * ps)
            grid_thw: Tuple of (T, H, W) for each image/video

        Returns:
            hidden_states: Final merged features (total_tokens, out_hidden_size)
            deepstack_features: List of intermediate features for DeepStack
        """
        return self.encode_jit(x, grid_thw)



class Qwen3VLModel(nnx.Module):
    """Text model for Qwen3VL with MRoPE and DeepStack support."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        # Qwen3VLConfig uses text_config for text model attributes
        text_config = getattr(hf_config, "text_config", hf_config)
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = text_config.rms_norm_eps
        hidden_size = text_config.hidden_size

        self.hidden_size = hidden_size

        # Embedder
        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )

        # Decoder layers with MRoPE support and KV cache attention
        self.layers = [
            Qwen3VLTextDecoderLayer(
                config=text_config,
                dtype=dtype,
                rngs=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            )
            for _ in range(text_config.num_hidden_layers)
        ]

        self.norm = Qwen3VLTextRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            dtype=dtype,
        )

        # LM head
        if hf_config.tie_word_embeddings:
            self.lm_head = self.embed.embedding
        else:
            self.lm_head = nnx.Param(
                init_fn(rng.params(), (hidden_size, vocab_size), dtype),
                sharding=(None, "model"),
            )

        self.tie_word_embeddings = hf_config.tie_word_embeddings

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
        # Flatten for indexing
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_mask = visual_pos_mask.reshape(-1)

        # Find mask indices and add visual embeddings
        # TODO: This assumes `visual_embeds.shape[0] == visual_pos_mask.sum()`.
        mask_indices = jnp.where(flat_mask)[0]
        if visual_embeds.shape[0] != mask_indices.shape[0]:
            visual_embeds = visual_embeds[:mask_indices.shape[0]]
        updated = flat_hidden.at[mask_indices].add(visual_embeds)

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
            x = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache=kv_cache,
                hidden_states=x,
                attention_metadata=attention_metadata,
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
        """Compute logits from hidden states."""
        if self.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.lm_head.value)
        return logits



class Qwen3VLForConditionalGeneration(nnx.Module):
    """Qwen3VL model for conditional generation."""

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
        # Qwen3VLConfig uses text_config for text model attributes
        text_config = getattr(config, "text_config", config)

        self.visual = Qwen3VLVisionTransformer(
            vllm_config=vllm_config,
            rngs=self.rng,
            mesh=mesh,
            norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
        )

        self.language_model = Qwen3VLModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

        # Special token IDs
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
        inputs_embeds = self.language_model.embed(input_ids)

        if multimodal_embeddings is not None and multimodal_embeddings.shape[0] != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.image_token_id, self.video_token_id],
            )

        return inputs_embeds

    def get_multimodal_embeddings(
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
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return {}
        if not image_grid_thw:
            return {}

        image_embeds, deepstack_embeds = self.visual(pixel_values, image_grid_thw)

        # Split embeddings per image (matching Qwen 2.5 VL return format)
        sizes = np.array([
            t * (h // self.spatial_merge_size) * (w // self.spatial_merge_size)
            for t, h, w in image_grid_thw
        ])

        if sizes.size == 0:
            return {}
        if sizes.size == 1:
            image_splits = (image_embeds,)
            deepstack_by_item = None
            if deepstack_embeds:
                deepstack_by_item = [
                    [layer_embeds for layer_embeds in deepstack_embeds]
                ]
            return {"embeds": image_splits, "deepstack": deepstack_by_item}

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
        return {"embeds": image_splits, "deepstack": deepstack_by_item}

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        """Forward pass for Qwen3VL using KV cache attention."""
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
        # TODO: check if second_per_grid_ts is required for multimodal
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
        del audio_feature_lengths, use_audio_in_video

        if hf_config is None:
            hf_config = self.config

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

        # Get warmup image shapes from config
        image_shapes = []
        # TODO: `additional_config` may be missing depending on integration.
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
        """Load weights from HuggingFace model."""
        # TODO: verify model loading
        self.rng = nnx.Rngs(rng_key)

        mappings = {
            "model.language_model.embed_tokens": "language_model.embed.embedding",
            "model.language_model.layers.*.input_layernorm": "language_model.layers.*.input_layernorm.weight",
            "model.language_model.layers.*.mlp.down_proj": "language_model.layers.*.mlp.down_proj.kernel",
            "model.language_model.layers.*.mlp.gate_proj": "language_model.layers.*.mlp.gate_proj.kernel",
            "model.language_model.layers.*.mlp.up_proj": "language_model.layers.*.mlp.up_proj.kernel",
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

        # Add lm_head mapping if not tied
        hf_config = self.vllm_config.model_config.hf_config
        if not hf_config.tie_word_embeddings:
            mappings["lm_head"] = "language_model.lm_head"

        # Add deepstack_merger_list mappings dynamically based on config
        # weight_utils.py only handles "layers" and "blocks" wildcards,
        # so we need explicit mappings for each deepstack merger index
        vision_config = hf_config.vision_config
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24])
        for i in range(len(deepstack_indexes)):
            mappings[f"model.visual.deepstack_merger_list.{i}.norm"] = f"visual.deepstack_merger_list.{i}.norm.scale"
            mappings[f"model.visual.deepstack_merger_list.{i}.norm.bias"] = f"visual.deepstack_merger_list.{i}.norm.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1"] = f"visual.deepstack_merger_list.{i}.linear_fc1.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1.bias"] = f"visual.deepstack_merger_list.{i}.linear_fc1.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2"] = f"visual.deepstack_merger_list.{i}.linear_fc2.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2.bias"] = f"visual.deepstack_merger_list.{i}.linear_fc2.bias"

        # Use adapter to expose text_config attributes at top level for
        # get_default_maps which expects num_attention_heads, etc. directly
        adapted_model_config = _ModelConfigAdapter(self.vllm_config.model_config)
        metadata_map = get_default_maps(
            adapted_model_config, self.mesh, mappings
        )
        load_hf_weights(
            vllm_config=self.vllm_config,
            model=self,
            metadata_map=metadata_map,
            mesh=self.mesh,
        )
