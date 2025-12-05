import jax
import jax.numpy as jnp
from flax import nnx

from qwen3_convert.arch.attention_interface import attention_interface # just for splash attention in TPUs


def compute_vision_counts_per_sequence(
    input_ids: jax.Array,
    attention_mask: jax.Array,
    image_token_id: int,
    video_token_id: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Preprocessing function
    Compute the number of images and videos per sequence in a batch.

    Args:
        input_ids: (batch_size, seq_len) - Required
        attention_mask: (batch_size, seq_len) - Required
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


def compute_rope_index(
    input_ids: jax.Array,
    image_grid_thw: jax.Array | None,
    video_grid_thw: jax.Array | None,
    attention_mask: jax.Array,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
    num_images_per_sequence: jax.Array | None = None,
    num_videos_per_sequence: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Preprocessing function
    Compute 3D position IDs for MRoPE. Call BEFORE jitted forward pass.

    This function is NOT JIT-compatible due to Python control flow over
    dynamic input shapes. It should be called eagerly before the model forward pass.

    In Qwen3VL, position IDs are 3D: [temporal, height, width].
    For text tokens: position increases sequentially in all 3 dimensions.
    For image/video tokens: spatial (H, W) positions from grid, temporal from frame index.

    Args:
        input_ids: Token IDs of shape (batch, seq_len) - Required
        image_grid_thw: Image grid dimensions (num_images, 3) [T, H, W], or None.
            H, W must be divisible by spatial_merge_size.
        video_grid_thw: Video grid dimensions (num_videos, 3) [T, H, W], or None.
            H, W must be divisible by spatial_merge_size.
        attention_mask: Attention mask of shape (batch, seq_len) - Required
        image_token_id: Token ID for image placeholders
        video_token_id: Token ID for video placeholders
        vision_start_token_id: Token ID marking start of vision tokens
        spatial_merge_size: Spatial merge size from vision config
        num_images_per_sequence: (batch_size,) - Required when image_grid_thw is provided
        num_videos_per_sequence: (batch_size,) - Required when video_grid_thw is provided

    Returns:
        position_ids: 3D position IDs of shape (3, batch, seq_len)
        mrope_position_deltas: Delta for rope calculation of shape (batch, 1)
    """
    # Handle video frames: split by temporal dimension
    # In Qwen3VL, videos are split into frames with timestamps
    if video_grid_thw is not None:
        # Repeat each video T times (once per frame) and set T=1 for each
        video_grid_thw = jnp.repeat(video_grid_thw, video_grid_thw[:, 0], axis=0)
        video_grid_thw = video_grid_thw.at[:, 0].set(1)

    # If we have multimodal inputs
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        batch_size, seq_len = input_ids.shape

        # Initialize position IDs: (3, batch, seq_len) for [T, H, W]
        position_ids = jnp.ones((3, batch_size, seq_len), dtype=jnp.int32)
        mrope_position_deltas = []

        # Compute cumulative indices for per-sequence image/video access
        cu_image_counts = jnp.pad(jnp.cumsum(num_images_per_sequence), (1, 0), constant_values=0) if image_grid_thw is not None else jnp.zeros(batch_size + 1, dtype=jnp.int32)
        cu_video_counts = jnp.pad(jnp.cumsum(num_videos_per_sequence), (1, 0), constant_values=0) if video_grid_thw is not None else jnp.zeros(batch_size + 1, dtype=jnp.int32)

        # Process each sample in the batch
        for i in range(batch_size):
            # Set per-sequence starting indices using cumulative counts
            image_index = int(cu_image_counts[i])
            video_index = int(cu_video_counts[i])
            # Get valid tokens (not padding)
            sample_input_ids = input_ids[i]
            mask = attention_mask[i] == 1
            sample_input_ids = sample_input_ids[mask]

            # Find vision tokens
            vision_start_indices = jnp.where(sample_input_ids == vision_start_token_id)[0]
            if len(vision_start_indices) > 0:
                vision_tokens = sample_input_ids[vision_start_indices + 1]
                image_nums = jnp.sum(vision_tokens == image_token_id)
                video_nums = jnp.sum(vision_tokens == video_token_id)
            else:
                image_nums = video_nums = 0

            # Convert to Python list for iteration
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = int(image_nums), int(video_nums)

            # Process each image/video
            for _ in range(int(image_nums + video_nums)):
                # Find next image or video token
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # Determine which comes first
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # Compute LLM grid dimensions (after spatial merging)
                llm_grid_t = int(t)
                llm_grid_h = int(h) // spatial_merge_size
                llm_grid_w = int(w) // spatial_merge_size
                text_len = ed - st

                # Add text positions before vision token
                st_idx = int(llm_pos_ids_list[-1].max()) + 1 if len(llm_pos_ids_list) > 0 else 0
                text_pos = jnp.arange(text_len).reshape(1, -1)
                text_pos = jnp.broadcast_to(text_pos, (3, text_len)) + st_idx
                llm_pos_ids_list.append(text_pos)

                # Add vision positions (3D grid)
                t_index = jnp.arange(llm_grid_t)[:, None].repeat(llm_grid_h * llm_grid_w, axis=1).reshape(-1)
                h_index = jnp.arange(llm_grid_h).reshape(1, -1, 1)
                h_index = jnp.broadcast_to(h_index, (llm_grid_t, llm_grid_h, llm_grid_w)).reshape(-1)
                w_index = jnp.arange(llm_grid_w).reshape(1, 1, -1)
                w_index = jnp.broadcast_to(w_index, (llm_grid_t, llm_grid_h, llm_grid_w)).reshape(-1)

                vision_pos = jnp.stack([t_index, h_index, w_index]) + text_len + st_idx
                llm_pos_ids_list.append(vision_pos)

                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # Add remaining text positions
            if st < len(input_tokens):
                st_idx = int(llm_pos_ids_list[-1].max()) + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                text_pos = jnp.arange(text_len).reshape(1, -1)
                text_pos = jnp.broadcast_to(text_pos, (3, text_len)) + st_idx
                llm_pos_ids_list.append(text_pos)

            # Concatenate all positions
            llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1)  # (3, valid_len)

            # Place in full position_ids tensor
            position_ids = position_ids.at[:, i, mask].set(llm_positions)

            # Compute rope delta for this sample
            max_pos = int(llm_positions.max())
            mrope_position_deltas.append(max_pos + 1 - seq_len)

        mrope_position_deltas = jnp.array(mrope_position_deltas)[:, None]
        return position_ids, mrope_position_deltas

    # Text-only case
    else:
        batch_size, seq_len = input_ids.shape

        # Compute positions from attention mask
        position_ids = jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1) - 1
        position_ids = jnp.where(attention_mask == 0, 1, position_ids)
        position_ids = position_ids[None, :, :].repeat(3, axis=0)

        max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True)
        mrope_position_deltas = max_position_ids[0, :, 0:1] + 1 - seq_len

        return position_ids, mrope_position_deltas


class Qwen3VLVisionPatchEmbed(nnx.Module):
    """
    3D Patch Embedding for video/image input
    Uses 3D convolution to convert patches into embeddings
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1024,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        # 3D convolution: (temporal, height, width)
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            padding='VALID',
            use_bias=True,
            param_dtype=dtype,
            rngs=rngs,
        )
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: [batch, channels, temporal, height, width]
        Returns:
            [batch, num_patches, embed_dim]
        """
        # Rearrange for Flax Conv: [batch, temporal, height, width, channels]
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        # Apply convolution
        x = self.proj(x)  # [batch, T', H', W', embed_dim]

        # Flatten spatial dimensions
        batch = x.shape[0]
        x = x.reshape(batch, -1, x.shape[-1])  # [batch, num_patches, embed_dim]

        return x

class Qwen3VLVisionRotaryEmbedding(nnx.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        self.inv_freq = 1.0 / (theta ** jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)

    def __call__(self, seq_len: int) -> jax.Array:
        seq = jnp.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = jnp.outer(seq, self.inv_freq)
        return freqs

class Qwen3VLVisionPatchMerger(nnx.Module):
    def __init__(self,
                 dim: int,
                 out_dim: int,
                 rngs: nnx.Rngs,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm=False,
                 dtype: jnp.dtype = jnp.bfloat16,
                 ):
        self.hidden_size = dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nnx.LayerNorm(self.hidden_size if use_postshuffle_norm else dim, epsilon=1e-6, param_dtype=dtype, rngs=rngs)
        self.linear_fc1 = nnx.Linear(self.hidden_size, self.hidden_size, param_dtype=dtype, rngs=rngs)
        self.linear_fc2 = nnx.Linear(self.hidden_size, out_dim, param_dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """

        Args:
            x: [batch, num_patches, embed_dim]

        Returns:

        """
        # This if statement bothers me. Maybe pass static?
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x).reshape(-1, self.hidden_size)

        x = self.linear_fc1(x)
        x = self.linear_fc2(nnx.gelu(x))
        return x

def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb_vision(
        q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array
):
    """
    Apply rotary position embedding to vision q/k tensors.

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    # Reshape cos/sin from (seq_len, head_dim) to (1, 1, seq_len, head_dim)
    # to broadcast with (batch, num_heads, seq_len, head_dim)
    cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim)
    sin = sin[None, None, :, :]  # (1, 1, seq_len, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    unsqueeze_dim: int = 1
) -> tuple[jax.Array, jax.Array]:
    """
    Apply Rotary Position Embedding to query and key tensors (for text model).

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine component of RoPE, shape (batch, seq_len, head_dim)
        sin: Sine component of RoPE, shape (batch, seq_len, head_dim)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting (default: 1)

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied
    """
    # Expand cos/sin to match q/k shape: (batch, 1, seq_len, head_dim)
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

class Qwen3VLVisionAttention(nnx.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            rngs: nnx.Rngs,
            attn_mode: str = "eager",
            dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_mode = attn_mode

        # Q, K, V projections
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=True, param_dtype=dtype, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, use_bias=True, param_dtype=dtype, rngs=rngs)

    def __call__(
            self,
            x: jax.Array,
            rope_cos: jax.Array,
            rope_sin: jax.Array,
            mesh: jax.sharding.Mesh,
    ):
        """
        Args:
            x: (batch, seq_len, dim) - Required
            rope_cos: (seq_len, head_dim) - Required
            rope_sin: (seq_len, head_dim) - Required
            mesh: JAX device mesh - Required
        """
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_states, k_states = apply_rotary_pos_emb_vision(q, k, rope_cos, rope_sin)

        # Transpose to [batch, seq_len, num_heads, head_dim] for attention_interface
        q_states = jnp.transpose(q_states, (0, 2, 1, 3))
        k_states = jnp.transpose(k_states, (0, 2, 1, 3))
        v_states = jnp.transpose(v, (0, 2, 1, 3))

        # Vision uses full attention
        attention_mask = jnp.ones((batch, 1, seq_len, seq_len), dtype=jnp.bool_)

        # Compute attention using attention_interface
        attn_output = attention_interface(
            q_states, k_states, v_states,
            attention_mask=attention_mask,
            mesh=mesh,
            attn_scaler=self.scale,
            attn_implementation=self.attn_mode,
        )  # Returns [batch, seq_len, num_heads, head_dim]

        # Reshape to [batch, seq_len, dim]
        attn_output = attn_output.reshape(batch, seq_len, dim)

        # Project output
        output = self.proj(attn_output)

        return output

class Qwen3VLVisionMLP(nnx.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            rngs: nnx.Rngs,
            dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.fc1 = nnx.Linear(dim, hidden_dim, use_bias=True, param_dtype=dtype, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, use_bias=True, param_dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.fc2(x)
        return x

class Qwen3VLVisionBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, param_dtype=dtype, rngs=rngs)
        self.attn = Qwen3VLVisionAttention(dim, num_heads, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, param_dtype=dtype, rngs=rngs)
        self.mlp = Qwen3VLVisionMLP(
            dim,
            intermediate_size,
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(
            self,
            x: jax.Array,
            rope_cos: jax.Array,
            rope_sin: jax.Array,
            mesh: jax.sharding.Mesh,
        ) -> jax.Array:
            # Attention with residual
            x = x + self.attn(self.norm1(x), rope_cos, rope_sin, mesh)
            # MLP with residual
            x = x + self.mlp(self.norm2(x))
            return x

class Qwen3VLVisionModel(nnx.Module):
    """
    Vision encoder for Qwen3VL.

    Note: This model is NOT JIT-compiled due to dynamic image dimensions.
    The rot_pos_emb and fast_pos_embed_interpolate methods use Python
    control flow due to images' dynamic shape.
    Moreover, these two functions are not supporting batch dimensions of seqeunces.

    Technically PatchMerger modules can be jitted without further considerations.
    This would be implemented using callbacks
    """

    def __init__(
            self,
            rngs: nnx.Rngs,
            patch_size: int = 16,
            temporal_patch_size: int = 2,
            in_channels: int = 3,
            embed_dim: int = 1152,  # hidden_size in config
            depth: int = 27,
            num_heads: int = 16,
            intermediate_size: int = 4304,
            output_dim: int = 4096,  # out_hidden_size in config
            spatial_merge_size: int = 2,
            num_position_embeddings: int = 2304,
            deepstack_visual_indexes: list[int] = None,
            dtype: jnp.dtype = jnp.bfloat16,
    ):
        # Store config values
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size

        # Patch embedding
        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size, temporal_patch_size, in_channels, embed_dim, rngs=rngs, dtype=dtype
        )

        # Positional embedding (learned spatial grid positions)
        self.pos_embed = nnx.Embed(num_position_embeddings, embed_dim, dtype=dtype, rngs=rngs)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)

        # Rotary positional embedding (keep in float32)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        # Transformer blocks
        self.blocks = [
            Qwen3VLVisionBlock(embed_dim, num_heads, rngs, intermediate_size, dtype=dtype)
            for _ in range(depth)
        ]

        # Main patch merger (reduces spatial dim, projects to language dim)
        self.merger = Qwen3VLVisionPatchMerger(
            embed_dim, output_dim, rngs, spatial_merge_size, use_postshuffle_norm=False, dtype=dtype
        )

        # Deepstack configuration
        self.deepstack_visual_indexes = deepstack_visual_indexes if deepstack_visual_indexes is not None else [8, 16, 24]
        self.deepstack_merger_list = [
            Qwen3VLVisionPatchMerger(
                embed_dim, output_dim, rngs, spatial_merge_size, use_postshuffle_norm=True, dtype=dtype
            )
            for _ in range(len(self.deepstack_visual_indexes))
        ]

    def rot_pos_emb(self, grid_thw: jax.Array) -> jax.Array:
        """
        Compute rotary position embeddings for 2D spatial positions.

        Args:
            grid_thw: Array of shape (num_images_or_videos, 3) where each row is [num_frames, height, width]

        Returns:
            Embeddings of shape (total_tokens, head_dim)
        """
        merge_size = self.spatial_merge_size

        # Get max height/width across all images/videos
        max_hw = int(jnp.max(grid_thw[:, 1:]))
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)

        # Calculate total number of tokens
        total_tokens = int(jnp.sum(jnp.prod(grid_thw, axis=1)))
        pos_ids = jnp.zeros((total_tokens, 2), dtype=jnp.int32)

        offset = 0
        for num_frames, height, width in grid_thw:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size

            # Generate block indices and intra-block offsets
            block_rows = jnp.arange(merged_h)  # block row indices
            block_cols = jnp.arange(merged_w)  # block col indices
            intra_row = jnp.arange(merge_size)  # intra-block row offsets
            intra_col = jnp.arange(merge_size)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            # Expand and reshape to get all positions
            row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
            col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

            coords = jnp.stack((row_idx, col_idx), axis=-1)

            # Repeat for multiple frames if needed
            if num_frames > 1:
                coords = jnp.tile(coords, (num_frames, 1))

            num_tokens = coords.shape[0]
            pos_ids = pos_ids.at[offset : offset + num_tokens].set(coords)
            offset += num_tokens

        # Lookup rotary embeddings and flatten
        embeddings = freq_table[pos_ids]  # (total_tokens, 2, dim // 2)
        embeddings = embeddings.reshape(total_tokens, -1)  # (total_tokens, dim)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: jax.Array) -> jax.Array:
        """
        Interpolate learned positional embeddings for variable resolution inputs using bilinear interpolation.

        Args:
            grid_thw: Array of shape (num_images_or_videos, 3) where each row is [T, H, W]
                     T: temporal patches (number of frames / temporal_patch_size)
                     H: height patches (height / patch_size), must be divisible by spatial_merge_size
                     W: width patches (width / patch_size), must be divisible by spatial_merge_size

        Returns:
            Position embeddings of shape (total_patches, embed_dim)
            Note: This returns embeddings for ALL patches before spatial merging.
            The spatial merge happens in PatchMerger, not here.
        """
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        merge_size = self.spatial_merge_size

        idx_list = [[], [], [], []]  # 4 corners for bilinear interpolation
        weight_list = [[], [], [], []]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            t, h, w = int(t), int(h), int(w)

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
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        # Convert to JAX arrays
        idx_tensor = jnp.array(idx_list, dtype=jnp.int32)
        weight_tensor = jnp.array(weight_list, dtype=jnp.float32)

        # Lookup embeddings and apply bilinear interpolation
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Split embeddings for each image/video
        split_sizes = [int(h * w) for h, w in zip(grid_hs, grid_ws)]
        patch_pos_embeds_list = jnp.split(patch_pos_embeds, jnp.cumsum(jnp.array(split_sizes[:-1])))

        # Rearrange embeddings to match patch ordering (with merge blocks)
        # H, W in grid_thw must be divisible by spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds_list, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t), int(h), int(w)

            # Repeat for temporal dimension
            pos_embed = jnp.tile(pos_embed, (t, 1))

            # Rearrange to match merge block ordering
            # Position embedding must be reordered to match how patches are grouped for merging
            # The merger groups adjacent patches in merge_size x merge_size blocks
            h_merged = h // merge_size
            w_merged = w // merge_size
            pos_embed = pos_embed.reshape(t, h_merged, merge_size, w_merged, merge_size, -1)
            pos_embed = jnp.transpose(pos_embed, (0, 1, 3, 2, 4, 5))
            pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])

            patch_pos_embeds_permute.append(pos_embed)

        patch_pos_embeds = jnp.concatenate(patch_pos_embeds_permute, axis=0)
        return patch_pos_embeds

    def __call__(
            self,
            hidden_states: jax.Array,
            grid_thw: jax.Array,
            mesh: jax.sharding.Mesh,
    ) -> tuple[jax.Array, list[jax.Array]]:
        """
        Forward pass of the vision model.

        Args:
            hidden_states: (total_patches, C, temporal_patch_size, patch_size, patch_size) - Required
            grid_thw: (num_images_or_videos, 3) where each row is [T, H, W] - Required
                     T: temporal patches (1 for images, frames/temporal_patch_size for videos)
                     H: height patches (image_height / patch_size), must be divisible by spatial_merge_size
                     W: width patches (image_width / patch_size), must be divisible by spatial_merge_size
            mesh: JAX device mesh - Required

        Returns:
            Tuple of (hidden_states, deepstack_feature_lists)
            - hidden_states: Final merged features of shape (total_tokens, output_dim)
            - deepstack_feature_lists: List of intermediate features from specified layers
        """
        # Patch embedding: (total_patches, C, tp, ps, ps) -> (total_patches, 1, embed_dim)
        hidden_states = self.patch_embed(hidden_states)

        # Squeeze: (total_patches, 1, embed_dim) -> (total_patches, embed_dim)
        hidden_states = hidden_states.squeeze(1)

        # Compute position embeddings
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        # Add position embeddings
        hidden_states = hidden_states + pos_embeds

        # Add batch dim: (total_patches, embed_dim) -> (1, total_patches, embed_dim)
        hidden_states = hidden_states[None, :, :]

        # Compute rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Prepare position embeddings for attention (cos/sin)
        emb = jnp.concatenate((rotary_pos_emb, rotary_pos_emb), axis=-1)  # (seq_len, head_dim)
        position_embeddings_cos = jnp.cos(emb)
        position_embeddings_sin = jnp.sin(emb)

        # Process through transformer blocks
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                position_embeddings_cos,
                position_embeddings_sin,
                mesh,
            )

            # Collect intermediate features at deepstack indices
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                # Squeeze batch dim for merger: (1, seq, dim) -> (seq, dim)
                deepstack_feature = self.deepstack_merger_list[idx](hidden_states.squeeze(0))
                deepstack_feature_lists.append(deepstack_feature)

        # Squeeze batch dim and apply final merger: (1, seq, dim) -> (seq, dim) -> (merged_tokens, output_dim)
        hidden_states = self.merger(hidden_states.squeeze(0))

        return hidden_states, deepstack_feature_lists

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
        mrope_section: list[int] = None,
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

        # TODO: Add support for other rope types from ROPE_INIT_FUNCTIONS
        if rope_type != "default":
            raise NotImplementedError(f"RoPE type '{rope_type}' not yet implemented in JAX")

        # Compute inverse frequencies
        inv_freq, attention_scaling = self.compute_default_rope_parameters(dim, rope_theta)
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq
        self.attention_scaling = attention_scaling

        # MRoPE section for interleaving 3D position embeddings
        self.mrope_section = mrope_section if mrope_section is not None else [24, 20, 20]

    @staticmethod
    def compute_default_rope_parameters(
        dim: int,
        rope_theta: float = 5000000.0,
    ) -> tuple[jax.Array, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation.

        Args:
            dim: Head dimension
            rope_theta: Base frequency for RoPE

        Returns:
            Tuple of (inv_freq, attention_scaling)
            - inv_freq: Inverse frequencies for RoPE embeddings
            - attention_scaling: Post-processing scaling factor (1.0 for default RoPE)
        """
        attention_factor = 1.0  # Unused in default RoPE type

        # Compute the inverse frequencies
        # inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        inv_freq = 1.0 / (
            rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )

        return inv_freq, attention_factor

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """
        freqs: (3, bs, seq_len, head_dim // 2)
        mrope_section: [T_dim, H_dim, W_dim] = [24, 20, 20]

        interleaving: first min(sections)*3 dimensions follow the pattern [T,H,W,T,H,W,...]
        The remaining dimensions are filled with T values.
        """
        min_section = min(mrope_section)
        interleaved_len = min_section * 3  # 60

        freqs_t = freqs[0]  # Base: temporal frequencies

        # Create interleaved region
        interleaved = jnp.zeros_like(freqs_t[..., :interleaved_len])
        for i in range(min_section):
            interleaved = interleaved.at[..., i*3].set(freqs[0, ..., i])      # T
            interleaved = interleaved.at[..., i*3+1].set(freqs[1, ..., i])    # H
            interleaved = interleaved.at[..., i*3+2].set(freqs[2, ..., i])    # W

        remaining_t = freqs[0, ..., min_section:mrope_section[0]]
        freqs_result = jnp.concatenate([interleaved, remaining_t], axis=-1)

        return freqs_result

    def __call__(
        self,
        x: jax.Array,
        position_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Compute cos/sin rotary position embeddings for 3D position IDs.

        Args:
            x: Dummy input tensor for shape/dtype reference
            position_ids: Position indices of shape (3, batch_size, seq_len) or (batch_size, seq_len)
                         3 dimensions represent [temporal, height, width] positions

        Returns:
            Tuple of (cos, sin) embeddings, each of shape (batch_size, seq_len, head_dim)
        """
        # Ensure position_ids has shape (3, batch_size, seq_len) without conditionals
        # Extract batch_size and seq_len from last two dimensions
        batch_size = position_ids.shape[-2]
        seq_len = position_ids.shape[-1]

        # Reshape to (-1, bs, seq_len) then broadcast to (3, bs, seq_len)
        # This handles both (bs, seq_len) → (1, bs, seq_len) → (3, bs, seq_len)
        # and (3, bs, seq_len) → (3, bs, seq_len) → (3, bs, seq_len) [no-op]
        position_ids = jnp.broadcast_to(
            jnp.reshape(position_ids, (-1, batch_size, seq_len)),
            (3, batch_size, seq_len)
        )

        # Expand inv_freq: (head_dim // 2,) -> (3, batch_size, head_dim // 2, 1)
        inv_freq_expanded = jnp.broadcast_to(
            self.inv_freq[None, None, :, None],
            (3, batch_size, len(self.inv_freq), 1)
        )

        # Expand position_ids: (3, batch_size, seq_len) -> (3, batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].astype(jnp.float32)

        # Compute frequencies: (3, batch_size, head_dim // 2, seq_len)
        # Matrix multiplication: (head_dim // 2, 1) @ (1, seq_len) = (head_dim // 2, seq_len)
        freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded)

        # Transpose to (3, batch_size, seq_len, head_dim // 2)
        freqs = jnp.transpose(freqs, (0, 1, 3, 2))

        # Apply interleaved MRoPE: (3, bs, seq_len, head_dim // 2) -> (bs, seq_len, head_dim // 2)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        # Concatenate to get full head_dim: (bs, seq_len, head_dim)
        emb = jnp.concatenate([freqs, freqs], axis=-1)

        # Compute cos/sin and apply attention scaling
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling

        # Cast to match input dtype
        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)

        return cos, sin

class Qwen3VLTextRMSNorm(nnx.Module):
    def __init__(self, hidden_size, eps: float = 1e-6, dtype: jnp.dtype = jnp.bfloat16):
        """
        Equivalent to T5LayerNorm.
        """
        self.weight = nnx.Param(jnp.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return self.weight.value * hidden_states.astype(input_dtype)


class Qwen3VLTextAttention(nnx.Module):
    """
    Multi-headed attention for Qwen3VL text model.
    Supports Grouped Query Attention (GQA) and Multi-Query Attention (MQA).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rngs: nnx.Rngs,
        head_dim: int = 128,
        attention_bias: bool = False,
        attn_mode: str = "eager",
        rms_norm_eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        """
        Args:
            hidden_size: Hidden dimension size (default: 4096)
            num_attention_heads: Number of query heads (default: 32)
            num_key_value_heads: Number of key/value heads for GQA/MQA (default: 32)
            rngs: Random number generators
            head_dim: Dimension of each attention head (default: 128)
            attention_bias: Whether to use bias in projections (default: False)
            attn_mode: Attention implementation mode: "eager", "sdpa", or "splash"
            rms_norm_eps: Epsilon for RMSNorm (default: 1e-6)
            dtype: Data type for parameters (default: bfloat16)
        """
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_mode = attn_mode

        # Query, Key, Value projections
        self.q_proj = nnx.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=attention_bias,
            param_dtype=dtype,
            rngs=rngs
        )
        self.k_proj = nnx.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            use_bias=attention_bias,
            param_dtype=dtype,
            rngs=rngs
        )
        self.v_proj = nnx.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            use_bias=attention_bias,
            param_dtype=dtype,
            rngs=rngs
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            use_bias=attention_bias,
            param_dtype=dtype,
            rngs=rngs
        )

        # QK normalization (per-head RMSNorm)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=rms_norm_eps, dtype=dtype)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=rms_norm_eps, dtype=dtype)

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
        attention_mask: jax.Array,
        mesh: jax.sharding.Mesh,
    ) -> jax.Array:
        """
        Forward pass for text attention.
        TODO: Cache support probably

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) embeddings from rotary embedding
            attention_mask: Attention mask of shape (batch, 1, seq_len, seq_len)
            mesh: JAX device mesh for sharding

        Returns:
            Attention output of shape (batch, seq_len, hidden_size)

        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to (batch, num_heads, seq_len, head_dim) for RoPE
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Transpose back to (batch, seq_len, num_heads, head_dim) for attention_interface
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))

        # Apply attention (attention_interface handles GQA/MQA internally)
        attn_output = attention_interface(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            mesh=mesh,
            attn_scaler=self.scaling,
            attn_implementation=self.attn_mode,
        )

        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3VLTextMLP(nnx.Module):
    """
    SwiGLU MLP for Qwen3VL text model.
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
            hidden_size: Hidden dimension size (default: 4096)
            intermediate_size: Intermediate dimension size (default: 22016)
            rngs: Random number generators
            hidden_act: Activation function (default: "silu")
            dtype: Data type for parameters (default: bfloat16)
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # All projections have no bias in Qwen3VL
        self.gate_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs)
        self.up_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_size, hidden_size, use_bias=False, param_dtype=dtype, rngs=rngs)

        # Activation function (default is SiLU for SwiGLU)
        if hidden_act == "silu":
            self.act_fn = jax.nn.silu
        else:
            raise NotImplementedError(f"Activation function '{hidden_act}' not implemented")

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass using SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))

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
    """
    Transformer decoder layer for Qwen3VL text model.
    Uses pre-normalization (RMSNorm before attention and MLP).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rngs: nnx.Rngs,
        head_dim: int = 128,
        attention_bias: bool = False,
        attn_mode: str = "eager",
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        """
        Args:
            hidden_size: Hidden dimension size (default: 4096)
            num_attention_heads: Number of query heads (default: 32)
            num_key_value_heads: Number of key/value heads for GQA/MQA (default: 32)
            intermediate_size: MLP intermediate size (default: 22016)
            rngs: Random number generators
            head_dim: Dimension of each attention head (default: 128)
            attention_bias: Whether to use bias in attention projections (default: False)
            attn_mode: Attention implementation mode (default: "eager")
            rms_norm_eps: Epsilon for RMSNorm (default: 1e-6)
            hidden_act: Activation function for MLP (default: "silu")
            dtype: Data type for parameters (default: bfloat16)
        """
        self.hidden_size = hidden_size

        # Self attention
        self.self_attn = Qwen3VLTextAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rngs=rngs,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attn_mode=attn_mode,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
        )

        # MLP
        self.mlp = Qwen3VLTextMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            rngs=rngs,
            hidden_act=hidden_act,
            dtype=dtype,
        )

        # Layer norms
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: tuple[jax.Array, jax.Array],
        attention_mask: jax.Array,
        mesh: jax.sharding.Mesh,
    ) -> jax.Array:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) from rotary embedding
            attention_mask: Attention mask of shape (batch, 1, seq_len, seq_len)
            mesh: JAX device mesh for sharding

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Self Attention with pre-norm and residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            mesh=mesh,
        )
        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3VLTextModel(nnx.Module):
    """
    Text model for Qwen3VL.
    Not a pure text-only model - DeepStack integrates visual features into early hidden states.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rngs: nnx.Rngs,
        head_dim: int = 128,
        max_position_embeddings: int = 128000,
        rope_theta: float = 5000000.0,
        rope_type: str = "default",
        mrope_section: list[int] = None,
        attention_bias: bool = False,
        attn_mode: str = "eager",
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        pad_token_id: int = 151643,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        """
        Args:
            vocab_size: Vocabulary size (default: 151936)
            hidden_size: Hidden dimension (default: 4096)
            num_hidden_layers: Number of decoder layers (default: 32)
            num_attention_heads: Number of query heads (default: 32)
            num_key_value_heads: Number of KV heads for GQA (default: 32)
            intermediate_size: MLP intermediate size (default: 22016)
            rngs: Random number generators
            head_dim: Attention head dimension (default: 128)
            max_position_embeddings: Max sequence length (default: 128000)
            rope_theta: RoPE base frequency (default: 5000000.0)
            rope_type: RoPE type (default: "default")
            mrope_section: MRoPE section sizes (default: [24, 20, 20])
            attention_bias: Use bias in attention (default: False)
            attn_mode: Attention implementation (default: "eager")
            rms_norm_eps: RMSNorm epsilon (default: 1e-6)
            hidden_act: MLP activation (default: "silu")
            pad_token_id: Padding token ID (default: 151643)
            dtype: Data type for parameters (default: bfloat16)
        """
        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Token embeddings
        self.embed_tokens = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        # Decoder layers
        self.layers = [
            Qwen3VLTextDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rngs=rngs,
                head_dim=head_dim,
                attention_bias=attention_bias,
                attn_mode=attn_mode,
                rms_norm_eps=rms_norm_eps,
                hidden_act=hidden_act,
                dtype=dtype,
            )
            for _ in range(num_hidden_layers)
        ]

        # Final layer norm
        self.norm = Qwen3VLTextRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

        # Rotary position embeddings (keep in float32)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_type=rope_type,
            mrope_section=mrope_section,
        )

    def _deepstack_process(
        self,
        hidden_states: jax.Array,
        visual_pos_masks: jax.Array,
        visual_embeds: jax.Array,
    ) -> jax.Array:
        """
        Integrate visual features into text hidden states at positions marked by visual_pos_masks.
        This is the DeepStack mechanism that adds visual information to early decoder layers.

        Args:
            hidden_states: Shape (batch, seq_len, hidden_size)
            visual_pos_masks: Boolean mask of shape (batch, seq_len) indicating visual token positions
            visual_embeds: Visual features of shape (num_visual_tokens, hidden_size)

        Returns:
            Updated hidden_states with visual features added
        """
        # visual_embeds should match the number of True positions in visual_pos_masks
        hidden_states = hidden_states.astype(visual_embeds.dtype)

        # Add visual embeddings only at masked positions
        # Reshape visual_pos_masks to (batch * seq_len,) for indexing
        batch, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_size)
        flat_mask = visual_pos_masks.reshape(-1)

        # Use jnp.where to add visual features at masked positions
        mask_indices = jnp.where(flat_mask)[0]

        # 해당 위치에 visual_embeds 더하기
        updated_flat = flat_hidden.at[mask_indices].add(visual_embeds)

        return updated_flat.reshape(batch, seq_len, hidden_size)


    def __call__(
        self,
        inputs_embeds: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        mesh: jax.sharding.Mesh,
        visual_pos_masks: jax.Array = None,
        deepstack_visual_embeds: list[jax.Array] = None,
    ) -> jax.Array:
        """
        Forward pass for text model.

        Args:
            inputs_embeds: (batch, seq_len, hidden_size) - Required
            attention_mask: (batch, seq_len) - Required, will be converted to 4D causal mask
            position_ids: (3, batch, seq_len) - Required, use compute_rope_index()
            mesh: JAX device mesh - Required
            visual_pos_masks: (batch, seq_len) boolean mask indicating visual token positions
            deepstack_visual_embeds: List of visual embeddings from vision encoder layers

        Returns:
            Final hidden states of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = inputs_embeds.shape

        # Convert 2D mask (batch, seq_len) to 4D causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attention_mask = attention_mask[:, None, None, :] & causal_mask[None, None, :, :]

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across all decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Process through decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                mesh=mesh,
            )

            # Add visual features to hidden states at early layers (DeepStack)
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                if visual_pos_masks is not None:
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states

class Qwen3VLModel(nnx.Module):
    """
    Qwen3VL multimodal model combining vision and language components.
    Integrates visual features into text through DeepStack mechanism.

    JIT Boundaries:
        - Vision encoder (Qwen3VLVisionModel): NOT JITted due to dynamic image dimensions
        - Text model (Qwen3VLTextModel): Can be JITted
        - Position IDs for multimodal inputs: Must be pre-computed using compute_rope_index()
          before the forward pass for JIT compatibility

    For multimodal inference, use:
        position_ids, rope_deltas = compute_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
            model.image_token_id, model.video_token_id,
            model.vision_start_token_id, model.vision_spatial_merge_size,
        )
        outputs = model(input_ids=input_ids, position_ids=position_ids, ...)
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        # Vision config
        vision_patch_size: int = 16,
        vision_temporal_patch_size: int = 2,
        vision_in_channels: int = 3,
        vision_embed_dim: int = 1152,
        vision_depth: int = 27,
        vision_num_heads: int = 16,
        vision_intermediate_size: int = 4304,
        vision_output_dim: int = 4096,
        vision_spatial_merge_size: int = 2,
        vision_num_position_embeddings: int = 2304,
        vision_deepstack_indexes: list[int] = None,
        # Text config
        text_vocab_size: int = 151936,
        text_hidden_size: int = 4096,
        text_num_hidden_layers: int = 36,
        text_num_attention_heads: int = 32,
        text_num_key_value_heads: int = 8,
        text_intermediate_size: int = 12288,
        text_head_dim: int = 128,
        text_max_position_embeddings: int = 262144,
        text_rope_theta: float = 5000000.0,
        text_rope_type: str = "default",
        text_mrope_section: list[int] = None,
        text_attention_bias: bool = False,
        text_attn_mode: str = "eager",
        text_rms_norm_eps: float = 1e-6,
        text_hidden_act: str = "silu",
        text_pad_token_id: int = 151643,
        # Special token IDs
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        # Dtype
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        """
        Args:
            rngs: Random number generators

            Vision config parameters (prefix: vision_):
                - patch_size, temporal_patch_size, in_channels, embed_dim, depth, num_heads
                - intermediate_size, output_dim, spatial_merge_size, num_position_embeddings
                - deepstack_indexes: Layer indices for DeepStack features

            Text config parameters (prefix: text_):
                - vocab_size, hidden_size, num_hidden_layers, num_attention_heads
                - num_key_value_heads, intermediate_size, head_dim
                - max_position_embeddings, rope_theta, rope_type, mrope_section
                - attention_bias, attn_mode, rms_norm_eps, hidden_act, pad_token_id

            Special token IDs:
                - image_token_id: ID for image placeholder tokens
                - video_token_id: ID for video placeholder tokens
                - vision_start_token_id: ID marking start of vision tokens
        """
        # Store config values
        self.vision_spatial_merge_size = vision_spatial_merge_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id

        # Initialize vision model
        self.visual = Qwen3VLVisionModel(
            rngs=rngs,
            patch_size=vision_patch_size,
            temporal_patch_size=vision_temporal_patch_size,
            in_channels=vision_in_channels,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            intermediate_size=vision_intermediate_size,
            output_dim=vision_output_dim,
            spatial_merge_size=vision_spatial_merge_size,
            num_position_embeddings=vision_num_position_embeddings,
            deepstack_visual_indexes=vision_deepstack_indexes,
            dtype=dtype,
        )

        # Initialize text model
        self.language_model = Qwen3VLTextModel(
            vocab_size=text_vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            num_key_value_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            rngs=rngs,
            head_dim=text_head_dim,
            max_position_embeddings=text_max_position_embeddings,
            rope_theta=text_rope_theta,
            rope_type=text_rope_type,
            mrope_section=text_mrope_section,
            attention_bias=text_attention_bias,
            attn_mode=text_attn_mode,
            rms_norm_eps=text_rms_norm_eps,
            hidden_act=text_hidden_act,
            pad_token_id=text_pad_token_id,
            dtype=dtype,
        )

        # Cache for rope_deltas (used during generation)
        self.rope_deltas = None

    def get_input_embeddings(self):
        """Get the token embedding layer from language model."""
        return self.language_model.embed_tokens

    def get_rope_index(
        self,
        input_ids: jax.Array,
        image_grid_thw: jax.Array = None,
        video_grid_thw: jax.Array = None,
        attention_mask: jax.Array = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Deprecated: Use compute_rope_index() standalone function instead.

        This method is kept for backwards compatibility but delegates to the
        standalone function. For JIT compatibility, call compute_rope_index()
        directly before the forward pass.
        """
        import warnings
        warnings.warn(
            "get_rope_index() is deprecated. Use compute_rope_index() standalone function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return compute_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            spatial_merge_size=self.vision_spatial_merge_size,
        )

    def get_image_features(
        self,
        pixel_values: jax.Array,
        image_grid_thw: jax.Array,
        mesh: jax.sharding.Mesh,
    ) -> tuple[tuple[jax.Array, ...], list[jax.Array]]:
        """
        Encode images into continuous embeddings.

        Args:
            pixel_values: Image pixels (total_patches, channels, temp, height, width)
            image_grid_thw: Grid dimensions (num_images, 3) [T, H, W]
            mesh: JAX device mesh

        Returns:
            image_embeds: Tuple of embeddings for each image
            deepstack_image_embeds: List of intermediate features for DeepStack
        """
        # Encode through vision model
        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            mesh=mesh,
        )

        # Split into individual images
        split_sizes = (jnp.prod(image_grid_thw, axis=-1) // (self.vision_spatial_merge_size ** 2)).astype(jnp.int32)
        split_sizes = split_sizes.tolist()

        # Split the concatenated embeddings
        image_embeds_list = jnp.split(image_embeds, jnp.cumsum(jnp.array(split_sizes[:-1])))

        return tuple(image_embeds_list), deepstack_image_embeds

    def get_video_features(
        self,
        pixel_values_videos: jax.Array,
        video_grid_thw: jax.Array,
        mesh: jax.sharding.Mesh,
    ) -> tuple[tuple[jax.Array, ...], list[jax.Array]]:
        """
        Encode videos into continuous embeddings.
        Same implementation as images in Qwen3VL.

        Args:
            pixel_values_videos: Video pixels (same format as images)
            video_grid_thw: Grid dimensions (num_videos, 3) [T, H, W]
            mesh: JAX device mesh

        Returns:
            video_embeds: Tuple of embeddings for each video
            deepstack_video_embeds: List of intermediate features for DeepStack
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, mesh)

    def get_placeholder_mask(
        self,
        input_ids: jax.Array,
        inputs_embeds: jax.Array,
        image_features: jax.Array = None,
        video_features: jax.Array = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Get boolean masks indicating positions of image and video placeholder tokens.

        Args:
            input_ids: Token IDs (batch, seq_len) or None
            inputs_embeds: Embeddings (batch, seq_len, hidden_size)
            image_features: Flattened image features or None
            video_features: Flattened video features or None

        Returns:
            special_image_mask: Boolean mask (batch, seq_len, hidden_size)
            special_video_mask: Boolean mask (batch, seq_len, hidden_size)
        """
        if input_ids is None:
            # Find special tokens by comparing embeddings
            embed_fn = self.get_input_embeddings()

            image_token_embed = embed_fn(jnp.array([self.image_token_id]))[0]
            special_image_mask = jnp.all(inputs_embeds == image_token_embed, axis=-1)

            video_token_embed = embed_fn(jnp.array([self.video_token_id]))[0]
            special_video_mask = jnp.all(inputs_embeds == video_token_embed, axis=-1)
        else:
            # Find special tokens directly from input_ids
            special_image_mask = input_ids == self.image_token_id
            special_video_mask = input_ids == self.video_token_id

        # Count tokens and validate
        n_image_tokens = int(jnp.sum(special_image_mask))
        n_video_tokens = int(jnp.sum(special_video_mask))

        # Expand masks to match embeddings shape
        special_image_mask = special_image_mask[:, :, None].repeat(inputs_embeds.shape[-1], axis=-1)
        special_video_mask = special_video_mask[:, :, None].repeat(inputs_embeds.shape[-1], axis=-1)

        # Validate feature counts
        if image_features is not None:
            if jnp.sum(special_image_mask) != image_features.size:
                raise ValueError(
                    f"Image features and image tokens do not match: "
                    f"tokens: {n_image_tokens}, features: {image_features.shape[0]}"
                )

        if video_features is not None:
            if jnp.sum(special_video_mask) != video_features.size:
                raise ValueError(
                    f"Video features and video tokens do not match: "
                    f"tokens: {n_video_tokens}, features: {video_features.shape[0]}"
                )

        return special_image_mask, special_video_mask

    def __call__(
        self,
        text_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        mesh: jax.sharding.Mesh,
        pixel_values: jax.Array = None,
        pixel_values_videos: jax.Array = None,
        image_grid_thw: jax.Array = None,
        video_grid_thw: jax.Array = None,
        num_images_per_sequence: jax.Array = None,
        num_videos_per_sequence: jax.Array = None,
    ) -> jax.Array:
        """
        Forward pass combining vision and language.

        Args:
            text_ids: (batch, seq_len) - Required
            attention_mask: (batch, seq_len) - Required
            position_ids: (3, batch, seq_len) - Required, use compute_rope_index()
            mesh: JAX device mesh - Required
            pixel_values: Image pixels for vision encoder, or None
            pixel_values_videos: Video pixels for vision encoder, or None
            image_grid_thw: (num_images, 3), H/W divisible by spatial_merge_size - Required when pixel_values provided
            video_grid_thw: (num_videos, 3), H/W divisible by spatial_merge_size - Required when pixel_values_videos provided
            num_images_per_sequence: (batch_size,) - Required when pixel_values provided
            num_videos_per_sequence: (batch_size,) - Required when pixel_values_videos provided

        Returns:
            Final hidden states (batch, seq_len, hidden_size)
        """
        text_embeds = self.get_input_embeddings()(text_ids)
        batch_size = text_embeds.shape[0]

        # Process images if provided
        image_mask = None
        deepstack_image_embeds = None
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw, mesh
            )
            # Concatenate all image embeddings
            image_embeds = jnp.concatenate(image_embeds, axis=0)

            # Compute cumulative token counts for per-sequence scatter
            cu_image_counts = jnp.pad(jnp.cumsum(num_images_per_sequence), (1, 0), constant_values=0)
            merge_factor = self.vision_spatial_merge_size ** 2
            tokens_per_image = jnp.prod(image_grid_thw, axis=-1) // merge_factor
            cu_token_counts = jnp.pad(jnp.cumsum(tokens_per_image), (1, 0), constant_values=0)

            # Scatter image embeddings per sequence
            for b in range(batch_size):
                img_start, img_end = int(cu_image_counts[b]), int(cu_image_counts[b + 1])
                tok_start, tok_end = int(cu_token_counts[img_start]), int(cu_token_counts[img_end])

                seq_image_embeds = image_embeds[tok_start:tok_end]
                seq_mask = (text_ids[b] == self.image_token_id)
                mask_indices = jnp.where(seq_mask)[0]

                text_embeds = text_embeds.at[b, mask_indices].set(seq_image_embeds)

            # Get mask for DeepStack (still need to track image positions)
            image_mask = (text_ids == self.image_token_id)

        # Process videos if provided
        video_mask = None
        deepstack_video_embeds = None
        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw, mesh
            )
            # Concatenate all video embeddings
            video_embeds = jnp.concatenate(video_embeds, axis=0)

            # Compute cumulative token counts for per-sequence scatter
            cu_video_counts = jnp.pad(jnp.cumsum(num_videos_per_sequence), (1, 0), constant_values=0)
            merge_factor = self.vision_spatial_merge_size ** 2
            tokens_per_video = jnp.prod(video_grid_thw, axis=-1) // merge_factor
            cu_token_counts = jnp.pad(jnp.cumsum(tokens_per_video), (1, 0), constant_values=0)

            # Scatter video embeddings per sequence
            for b in range(batch_size):
                vid_start, vid_end = int(cu_video_counts[b]), int(cu_video_counts[b + 1])
                tok_start, tok_end = int(cu_token_counts[vid_start]), int(cu_token_counts[vid_end])

                seq_video_embeds = video_embeds[tok_start:tok_end]
                seq_mask = (text_ids[b] == self.video_token_id)
                mask_indices = jnp.where(seq_mask)[0]

                text_embeds = text_embeds.at[b, mask_indices].set(seq_video_embeds)

            # Get mask for DeepStack (still need to track video positions)
            video_mask = (text_ids == self.video_token_id)

        # Aggregate visual masks and embeddings for DeepStack
        # Note: image_mask and video_mask are now 2D boolean arrays (batch, seq_len)
        visual_pos_masks = None
        deepstack_visual_embeds = None

        if image_mask is not None and video_mask is not None:
            # Both images and videos present
            visual_pos_masks = image_mask | video_mask

            # Combine DeepStack embeddings
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]

            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = jnp.zeros((int(jnp.sum(visual_pos_masks)), img_embed.shape[-1]))
                embed_joint = embed_joint.at[image_mask_joint].set(img_embed)
                embed_joint = embed_joint.at[video_mask_joint].set(vid_embed)
                deepstack_visual_embeds.append(embed_joint)

        elif image_mask is not None:
            # Only images
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds

        elif video_mask is not None:
            # Only videos
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mesh=mesh,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return outputs
