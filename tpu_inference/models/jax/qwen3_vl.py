# Experimental batch support!!

import math
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import (
    sharded_flash_attention,
)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.qwen3 import Qwen3DecoderLayer
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
    spatial_merge_size: int,
) -> jax.Array:
    """Generate segment IDs from grid dimensions for variable-length attention.

    Each image/video in grid_thw gets a unique segment ID (starting from 1).
    This ensures tokens from different images cannot attend to each other.

    Args:
        grid_thw: Tuple of (T, H, W) for each image/video
        spatial_merge_size: Spatial merge size from vision config

    Returns:
        segment_ids: (total_tokens,) array with segment IDs per token
    """
    segment_ids_list = []
    for idx, (t, h, w) in enumerate(grid_thw):
        num_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
        # Segment IDs start from 1 (0 is reserved for padding)
        segment_ids_list.append(jnp.full(num_tokens, idx + 1, dtype=jnp.int32))

    return jnp.concatenate(segment_ids_list, axis=0)


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
    second_per_grid_ts: Optional[List[float]] = None,
    tokens_per_second: float = 1.0,
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
        second_per_grid_ts: Optional list of temporal spacing for videos
        tokens_per_second: Tokens per second for video temporal positions

    Returns:
        llm_positions: (3, seq_len) position IDs for [T, H, W]
        mrope_position_delta: Delta for rope calculation
    """
    input_tokens_np = np.array(input_tokens)
    vision_start_indices = np.where(input_tokens_np == vision_start_token_id)[0]
    vision_tokens = input_tokens_np[vision_start_indices + 1]
    image_nums = int(np.sum(vision_tokens == image_token_id))
    video_nums = int(np.sum(vision_tokens == video_token_id))

    llm_pos_ids_list = []

    st = 0
    remain_images, remain_videos = image_nums, video_nums
    image_index, video_index = 0, 0

    for _ in range(image_nums + video_nums):
        video_second_per_grid_t = 0.0

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
            t, h, w = video_grid_thw[video_index]
            video_second_per_grid_t = 1.0
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[video_index]
            video_index += 1
            remain_videos -= 1
            ed = ed_video

        llm_grid_t = t
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size
        text_len = ed - st

        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            )
            + st_idx
        )

        t_index = (
            jnp.broadcast_to(
                jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1),
                (llm_grid_t, llm_grid_h * llm_grid_w),
            )
            * video_second_per_grid_t
            * tokens_per_second
        ).astype(jnp.int32).flatten()

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
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            )
            + st_idx
        )

    llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = int(llm_positions.max()) + 1 - len(input_tokens)

    return llm_positions, mrope_position_delta


# ==============================================================================
# Vision Rotary Embedding
# ==============================================================================


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
            use_bias=False,
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


# ==============================================================================
# Vision MLP
# ==============================================================================


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
        self.norm1 = nnx.RMSNorm(
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
        self.norm2 = nnx.RMSNorm(
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
        self.num_grid_per_side = int(num_position_embeddings**0.5)

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
        out_hidden_size = getattr(vision_config, "out_hidden_size", hf_config.hidden_size)

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
            h_idxs = jnp.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = jnp.linspace(0, self.num_grid_per_side - 1, w)

            # Floor and ceil indices for bilinear interpolation
            h_idxs_floor = h_idxs.astype(jnp.int32)
            w_idxs_floor = w_idxs.astype(jnp.int32)
            h_idxs_ceil = (h_idxs.astype(jnp.int32) + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.astype(jnp.int32) + 1).clip(max=self.num_grid_per_side - 1)

            # Interpolation weights
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            # Base indices for 2D grid (flattened)
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

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

    def compute_hidden_states(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> Tuple[jax.Array, List[jax.Array]]:
        """JIT-compatible core computation.

        Args:
            x: Patch embeddings of shape (num_patches, hidden_size)
            rotary_pos_emb: Rotary embeddings
            segment_ids: Segment IDs for variable-length attention

        Returns:
            hidden_states: Final merged features
            deepstack_features: List of intermediate features
        """
        # Patch embedding
        hidden_states = self.patch_embed(x)

        # Reshape for merge block ordering
        seq_len = x.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )

        # Add batch dimension: (seq, merge_unit, dim) -> (seq * merge_unit, 1, dim)
        hidden_states = hidden_states.reshape(seq_len, 1, -1)

        # Process through transformer blocks
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

        # Generate segment IDs for variable-length attention
        segment_ids = generate_segment_ids_from_grid_thw(
            grid_thw, self.spatial_merge_size
        )

        # Reshape for transformer
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states.reshape(seq_len, 1, -1)

        # Process through transformer blocks
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
    """Text model that reuses Qwen3DecoderLayer with DeepStack injection."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        # Embedder
        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )

        # Text part is just Qwen3
        self.layers = [
            Qwen3DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            )
            for _ in range(hf_config.num_hidden_layers)
        ]

        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
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
            hidden_states: (seq_len, hidden_size)
            visual_pos_mask: Boolean mask (seq_len,) indicating visual token positions
            visual_embeds: Visual features (num_visual_tokens, hidden_size)

        Returns:
            Updated hidden_states with visual features added
        """
        # Flatten for indexing
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_mask = visual_pos_mask.reshape(-1)

        # Find mask indices and add visual embeddings
        mask_indices = jnp.where(flat_mask)[0]
        updated = flat_hidden.at[mask_indices].add(visual_embeds)

        return updated.reshape(hidden_states.shape)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        visual_pos_mask: Optional[jax.Array] = None,
        deepstack_visual_embeds: Optional[List[jax.Array]] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        """Forward pass with DeepStack support.

        Args:
            kv_caches: List of KV cache tensors
            input_ids: Input token IDs
            attention_metadata: Attention metadata for paged attention
            inputs_embeds: Optional pre-computed embeddings
            visual_pos_mask: Boolean mask for visual token positions
            deepstack_visual_embeds: List of visual embeddings for DeepStack

        Returns:
            kv_caches: Updated KV caches
            hidden_states: Final hidden states
        """
        # Get embeddings
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            kv_caches[i], x = layer(kv_caches[i], x, attention_metadata)

            # DeepStack injection
            if (
                deepstack_visual_embeds is not None
                and i < len(deepstack_visual_embeds)
                and visual_pos_mask is not None
            ):
                x = self._inject_visual_features(
                    x, visual_pos_mask, deepstack_visual_embeds[i]
                )

        # Final normalization
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

        # Vision encoder
        self.visual = Qwen3VLVisionTransformer(
            vllm_config=vllm_config,
            rngs=self.rng,
            mesh=mesh,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        # Text model with DeepStack
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
        pixel_values: jax.Array,
        image_grid_thw: Tuple[Tuple[int, int, int], ...],
    ) -> Tuple[jax.Array, List[jax.Array]]:
        """Get multimodal embeddings from pixel values.

        Args:
            pixel_values: Pixel values for vision encoder
            image_grid_thw: Grid dimensions (T, H, W) for each image

        Returns:
            image_embeds: Flattened image embeddings
            deepstack_embeds: List of DeepStack embeddings
        """
        image_embeds, deepstack_embeds = self.visual(pixel_values, image_grid_thw)
        return image_embeds, deepstack_embeds

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        pixel_values: Optional[jax.Array] = None,
        image_grid_thw: Optional[Tuple[Tuple[int, int, int], ...]] = None,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        """Forward pass for Qwen3VL.

        Args:
            kv_caches: List of KV cache tensors
            input_ids: Input token IDs
            attention_metadata: Attention metadata
            inputs_embeds: Optional pre-computed embeddings
            pixel_values: Optional pixel values for vision encoder
            image_grid_thw: Grid dimensions for images

        Returns:
            kv_caches: Updated KV caches
            hidden_states: Final hidden states
            aux_outputs: Empty list (for compatibility)
        """
        visual_pos_mask = None
        deepstack_embeds = None

        # Process images if provided
        if pixel_values is not None and image_grid_thw is not None:
            image_embeds, deepstack_embeds = self.get_multimodal_embeddings(
                pixel_values, image_grid_thw
            )
            inputs_embeds = self.get_input_embeddings(input_ids, image_embeds)
            visual_pos_mask = (input_ids == self.image_token_id) | (
                input_ids == self.video_token_id
            )

        # Forward through language model
        kv_caches, x = self.language_model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_embeds,
        )

        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: List[int],
        image_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
        video_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
        second_per_grid_ts: Optional[List[float]] = None,
    ) -> Tuple[jax.Array, int]:
        """Compute MRoPE 3D position IDs for input sequence.

        This is a wrapper around the module-level get_mrope_input_positions function
        that uses the model's configuration.

        Args:
            input_tokens: List of token IDs for the sequence
            image_grid_thw: List of (T, H, W) tuples for each image
            video_grid_thw: List of (T, H, W) tuples for each video
            second_per_grid_ts: Optional list of temporal spacing for videos

        Returns:
            llm_positions: (3, seq_len) position IDs for [T, H, W]
            mrope_position_delta: Delta for rope calculation
        """
        return get_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            spatial_merge_size=self.spatial_merge_size,
            second_per_grid_ts=second_per_grid_ts,
        )

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
        spatial_merge_unit = vc.spatial_merge_size ** 2

        # Get warmup image shapes from config
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
        """Load weights from HuggingFace model."""
        self.rng = nnx.Rngs(rng_key)

        # Weight mappings: HF name -> JAX name
        mappings = {
            # Language model embeddings
            "model.embed_tokens": "language_model.embed.embedding",

            # Language model layers
            "model.layers.*.input_layernorm": "language_model.layers.*.input_layernorm.scale",
            "model.layers.*.mlp.down_proj": "language_model.layers.*.mlp.down_proj.kernel",
            "model.layers.*.mlp.gate_proj": "language_model.layers.*.mlp.gate_proj.kernel",
            "model.layers.*.mlp.up_proj": "language_model.layers.*.mlp.up_proj.kernel",
            "model.layers.*.post_attention_layernorm": "language_model.layers.*.post_attention_layernorm.scale",
            "model.layers.*.self_attn.k_proj": "language_model.layers.*.self_attn.k_proj.kernel",
            "model.layers.*.self_attn.o_proj": "language_model.layers.*.self_attn.o_proj.kernel",
            "model.layers.*.self_attn.q_proj": "language_model.layers.*.self_attn.q_proj.kernel",
            "model.layers.*.self_attn.v_proj": "language_model.layers.*.self_attn.v_proj.kernel",
            "model.layers.*.self_attn.q_norm": "language_model.layers.*.self_attn.q_norm.scale",
            "model.layers.*.self_attn.k_norm": "language_model.layers.*.self_attn.k_norm.scale",
            "model.norm": "language_model.norm.scale",

            # Vision encoder - patch embedding
            "visual.patch_embed.proj": "visual.patch_embed.proj.kernel",

            # Vision encoder - position embedding
            "visual.pos_embed": "visual.pos_embed.embedding",

            # Vision encoder - blocks
            "visual.blocks.*.attn.qkv": "visual.blocks.*.attn.qkv_proj.kernel",
            "visual.blocks.*.attn.qkv.bias": "visual.blocks.*.attn.qkv_proj.bias",
            "visual.blocks.*.attn.proj": "visual.blocks.*.attn.proj.kernel",
            "visual.blocks.*.attn.proj.bias": "visual.blocks.*.attn.proj.bias",
            "visual.blocks.*.mlp.gate_proj": "visual.blocks.*.mlp.gate_proj.kernel",
            "visual.blocks.*.mlp.gate_proj.bias": "visual.blocks.*.mlp.gate_proj.bias",
            "visual.blocks.*.mlp.up_proj": "visual.blocks.*.mlp.up_proj.kernel",
            "visual.blocks.*.mlp.up_proj.bias": "visual.blocks.*.mlp.up_proj.bias",
            "visual.blocks.*.mlp.down_proj": "visual.blocks.*.mlp.down_proj.kernel",
            "visual.blocks.*.mlp.down_proj.bias": "visual.blocks.*.mlp.down_proj.bias",
            "visual.blocks.*.norm1": "visual.blocks.*.norm1.scale",
            "visual.blocks.*.norm2": "visual.blocks.*.norm2.scale",

            # Vision encoder - main merger
            "visual.merger.ln_q": "visual.merger.ln_q.scale",
            "visual.merger.mlp.0": "visual.merger.mlp_fc1.kernel",
            "visual.merger.mlp.0.bias": "visual.merger.mlp_fc1.bias",
            "visual.merger.mlp.2": "visual.merger.mlp_fc2.kernel",
            "visual.merger.mlp.2.bias": "visual.merger.mlp_fc2.bias",

            # Vision encoder - DeepStack mergers
            "visual.deepstack_merger_list.*.ln_q": "visual.deepstack_merger_list.*.ln_q.scale",
            "visual.deepstack_merger_list.*.mlp.0": "visual.deepstack_merger_list.*.mlp_fc1.kernel",
            "visual.deepstack_merger_list.*.mlp.0.bias": "visual.deepstack_merger_list.*.mlp_fc1.bias",
            "visual.deepstack_merger_list.*.mlp.2": "visual.deepstack_merger_list.*.mlp_fc2.kernel",
            "visual.deepstack_merger_list.*.mlp.2.bias": "visual.deepstack_merger_list.*.mlp_fc2.bias",
        }

        # Add lm_head mapping if not tied
        hf_config = self.vllm_config.model_config.hf_config
        if not hf_config.tie_word_embeddings:
            mappings["lm_head"] = "language_model.lm_head"

        metadata_map = get_default_maps(
            self.vllm_config.model_config, self.mesh, mappings
        )
        load_hf_weights(
            vllm_config=self.vllm_config,
            model=self,
            metadata_map=metadata_map,
            mesh=self.mesh,
        )
