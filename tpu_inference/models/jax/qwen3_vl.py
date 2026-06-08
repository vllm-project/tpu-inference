# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from functools import partial
from itertools import islice
from typing import (Any, Iterable, List, Literal, NamedTuple, Optional, Tuple,
                    TypedDict, Union)

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import (
    attention, sharded_flash_attention)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.conv import JaxConv
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.norm import JaxLayerNorm, JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.qwen2 import Qwen2MLP as Qwen3MLP
from tpu_inference.models.jax.qwen3 import (Qwen3Attention, Qwen3DecoderLayer,
                                            Qwen3Model)
from tpu_inference.models.jax.utils.multi_modal_utils import (
    _merge_multimodal_embeddings, merge_multimodal_embeddings,
    normalize_mm_grid_thw, reshape_mm_tensor, split_mm_embeddings_by_grid)
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator, load_nnx_param_from_reshaped_torch)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

DEFAULT_BLOCK_K_MAJOR = 128
VISION_GRID_SIZE = 48


def _sync_module_param_sharding(module: nnx.Module) -> None:
    for param in jax.tree_util.tree_leaves(nnx.state(module)):
        if isinstance(param, nnx.Param):
            out_sharding = getattr(param, "out_sharding", None)
            if out_sharding is not None and getattr(param, "sharding",
                                                    None) is None:
                param.set_metadata("sharding", out_sharding)


def _safe_convert_torch_to_jax(v: Any) -> Any:
    """Recursively convert PyTorch tensors (including bfloat16) to JAX arrays safely."""
    if isinstance(v, list):
        return [_safe_convert_torch_to_jax(item) for item in v]
    if isinstance(v, torch.Tensor):
        if v.dtype == torch.bfloat16:
            # Cast to float32 first since NumPy does not natively support bfloat16
            return jnp.asarray(v.detach().cpu().to(
                torch.float32).numpy()).astype(jnp.bfloat16)
        return jnp.asarray(v.detach().cpu().numpy())
    return v


class _Qwen3VLConfigAdapter:

    def __init__(self, config):
        self._config = config
        self._text_config = getattr(config, "text_config", None)

    def __getattr__(self, name):
        try:
            return getattr(self._config, name)
        except AttributeError:
            pass
        if self._text_config is not None:
            try:
                return getattr(self._text_config, name)
            except AttributeError:
                pass
        raise AttributeError(f"Attribute '{name}' not found in either"
                             f" '{type(self._config).__name__}' or"
                             f" '{type(self._text_config).__name__}'")


class _ModelConfigAdapter:

    def __init__(self, model_config):
        self._model_config = model_config
        self._hf_config_adapter = _Qwen3VLConfigAdapter(model_config.hf_config)

    @property
    def hf_config(self):
        return self._hf_config_adapter

    @property
    def hf_text_config(self):
        return self._hf_config_adapter

    def __getattr__(self, name):
        return getattr(self._model_config, name)


class _VllmConfigAdapter:

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config
        self.model_config = _ModelConfigAdapter(vllm_config.model_config)
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config

    def __getattr__(self, name):
        return getattr(self._vllm_config, name)


def _infer_pos_embed_grid_hw(num_position_embeddings: int) -> Tuple[int, int]:
    """Infer a (grid_h, grid_w) pair from a flattened 2D embedding table."""
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


class Qwen3VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: jax.Array
    image_grid_thw: Tuple[Tuple[int, int, int], ...]


class Qwen3VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: jax.Array
    image_grid_thw: Tuple[Tuple[int, int, int], ...]


Qwen3VLImageInputs = Union[Qwen3VLImagePixelInputs,
                           Qwen3VLImageEmbeddingInputs]


def generate_segment_ids_from_grid_thw(
    grid_thw: Tuple[Tuple[int, int, int], ...], ) -> jax.Array:
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
    # Start from 1 so that padding positions remain 0, which
    # pad_segment_ids_for_attention uses to mask out padded tokens.
    seg_id = 1

    for (t, h, w) in grid_thw:
        frame_size = h * w
        for _ in range(t):
            segments.append(jnp.full((frame_size, ), seg_id, dtype=jnp.int32))
            seg_id += 1

    return jnp.concatenate(segments, axis=0) if segments else jnp.zeros(
        (0, ), dtype=jnp.int32)


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
    This is not used currently, but would be needed for sequence batches.

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


def build_mrope_input_positions(
    input_tokens: List[int],
    image_grid_thw: Optional[List[Tuple[int, int, int]]],
    video_grid_thw: Optional[List[Tuple[int, int, int]]],
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: Optional[int],
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
        vision_start_token_id: Vision start token ID. Kept for interface
            compatibility with serving wrappers.
        spatial_merge_size: Spatial merge size from vision config

    Returns:
        llm_positions: (3, seq_len) position IDs for [T, H, W]
        mrope_position_delta: Delta for rope calculation
    """
    del vision_start_token_id

    # Use the provided grid_thw lengths as authoritative counts.
    # Note: For Qwen3VL, each temporal frame may have its own vision_start marker while
    # the grid_thw represents the entire video as a single item with t>1.
    image_nums = len(image_grid_thw) if image_grid_thw else 0
    video_nums = len(video_grid_thw) if video_grid_thw else 0
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

        if ed_image == len(input_tokens) + 1 and ed_video == len(
                input_tokens) + 1:
            break

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
            ) + st_idx)

        # t_index always zero
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
            jnp.stack([t_index, h_index, w_index]) + text_len + st_idx)

        st = ed + num_vision_tokens

    # Trailing text
    if st < len(input_tokens):
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            ) + st_idx)

    llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = int(llm_positions.max()) + 1 - len(input_tokens)

    return llm_positions, mrope_position_delta


class Qwen3VLTextAttention(Qwen3Attention):
    """Qwen3 attention with MRoPE (3D position) and interleaved RoPE ordering."""

    def __call__(
        self,
        kv_cache,
        x: jax.Array,
        attention_metadata,
    ):
        md = attention_metadata
        q = self.q_proj(x)
        q = self.q_norm(q)

        k = self.k_proj(x)
        k = self.k_norm(k)

        # Broadcast 1D positions to 3D for MRoPE
        positions = md.input_positions
        if positions.ndim == 1:
            positions = jnp.broadcast_to(positions[None, :],
                                         (3, positions.shape[0]))

        q = apply_rope(q,
                       positions,
                       self.head_dim_original,
                       self.rope_theta,
                       self.rope_scaling,
                       rope_input_ordering="interleaved")
        k = apply_rope(k,
                       positions,
                       self.head_dim_original,
                       self.rope_theta,
                       self.rope_scaling,
                       rope_input_ordering="interleaved")

        v = self.v_proj(x)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)

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


class Qwen3VLTextDecoderLayer(Qwen3DecoderLayer):
    """Decoder layer with MRoPE-aware attention for VL.

    Overrides __init__ to swap Qwen3Attention with Qwen3VLTextAttention.
    Inherits __call__ from Qwen2DecoderLayer unchanged.
    """

    def __init__(self,
                 config,
                 dtype,
                 rng,
                 mesh,
                 kv_cache_dtype,
                 quant_config,
                 prefix=""):
        # Set up all attributes expected by inherited __call__
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Qwen3VLTextAttention(
            config=config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
            quant_config=quant_config,
            prefix=prefix + ".self_attn",
        )
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.mlp = Qwen3MLP(
            config=config,
            dtype=dtype,
            rng=rng,
            quant_config=quant_config,
            prefix=prefix + ".mlp",
        )


def apply_rotary_pos_emb_vision(x: jax.Array,
                                rotary_pos_emb: jax.Array) -> jax.Array:

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


class Qwen3VLVisionRotaryEmbedding(JaxModule):
    """Rotary position embedding for vision encoder."""

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta**(
            jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)


class Qwen3VLVisionPatchEmbed(JaxModule):
    """3D Patch Embedding for video/image input using 3D convolution."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = JaxConv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,  # Unlike 2.5VL, uses bias for the convolution.
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, None, None, None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs,
        )
        self.proj.weight.set_metadata(
            "weight_loader",
            partial(
                load_nnx_param_from_reshaped_torch,
                permute_dims=(2, 3, 4, 1, 0),
                param_name="visual.patch_embed.proj.weight",
            ))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply 3D patch embedding.

        Args:
            x: Input tensor of shape (num_patches, C * T * H * W)

        Returns:
            Embedded patches of shape (num_patches, hidden_size)
        """
        L, dim = x.shape
        patch_volume = self.temporal_patch_size * self.patch_size * self.patch_size
        assert dim % patch_volume == 0, (
            f"Input dim {dim} is not divisible by patch volume {patch_volume}")
        C = dim // patch_volume
        # Reshape to (L, C, T, H, W) then transpose to (L, T, H, W, C)
        # for Conv3D with channels_last layout.
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size,
                      self.patch_size)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv, shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        return x


class Qwen3VLVisionMLP(JaxModule):
    """SwiGLU-style MLP for vision encoder."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.fc1 = JaxLinear(
            hidden_size,
            intermediate_size,
            rngs=rngs,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
        )
        self.fc2 = JaxLinear(
            intermediate_size,
            hidden_size,
            rngs=rngs,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        T, B, D = x.shape

        # Flatten batch dimension to 2D for JaxLinear compatibility: (T, B, D) -> (T * B, D)
        x_2d = x.reshape(T * B, D)
        x_2d = self.fc1(x_2d)
        x_2d = jax.nn.gelu(x_2d, approximate=False)
        x_2d = self.fc2(x_2d)

        # Reshape back to the original (T, B, D) format before returning
        return x_2d.reshape(T, B, D)


class Qwen3VLVisionAttention(JaxModule):
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
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.head_dim = hidden_size // num_heads  # Original head dim

        self.mesh = mesh

        # QKV projection
        self.qkv_projection = JaxLinear(
            hidden_size,
            3 * hidden_size,
            rngs=rngs,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
        )
        # Output projection
        self.proj = JaxLinear(
            hidden_size,
            hidden_size,
            rngs=rngs,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
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

        x_2d = x.reshape(T * B, D)
        qkv = self.qkv_projection(x_2d)

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
        padded_T = (T_attn + block_k_major -
                    1) // block_k_major * block_k_major
        pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))

        q = jnp.pad(q, pad_width, "constant")
        k = jnp.pad(k, pad_width, "constant")
        v = jnp.pad(v, pad_width, "constant")

        # Pad segment IDs for attention (padding tokens get segment_id=0)
        padded_segment_ids = pad_segment_ids_for_attention(
            segment_ids, padded_T)

        output = self.flash_attention(q, k, v, padded_segment_ids)

        # Unpad and reshape: [B, N, T, H] -> [T, B, N, H] -> [T, B, D]
        output = output[:, :, :T_attn, :]
        output = jnp.transpose(output, (2, 0, 1, 3))
        output = output.reshape(T, B, D)

        # Flatten the batch dimension to 2D for JaxLinear compatibility: (T, B, D) -> (T * B, D)
        output_2d = output.reshape(T * B, D)
        output_2d = self.proj(output_2d)

        # Reshape back to the expected (T, B, D) format before returning
        output = output_2d.reshape(T, B, D)

        return output


class Qwen3VLVisionBlock(JaxModule):
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
        self.norm1 = JaxLayerNorm(
            hidden_size,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
        )
        self.attn = Qwen3VLVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.norm2 = JaxLayerNorm(
            hidden_size,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
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


class Qwen3VLVisionPatchMerger(JaxModule):
    """Merge spatial patches and project to language model dimension.

    This module supports both:
    - Final merger: norm before reshape (use_postshuffle_norm=False)
    - DeepStack merger: reshape before norm (use_postshuffle_norm=True)
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        spatial_merge_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        use_postshuffle_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = JaxLayerNorm(
            norm_dim,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
        )

        self.linear_fc1 = JaxLinear(
            self.hidden_size,
            self.hidden_size,
            rngs=rngs,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
        )
        self.linear_fc2 = JaxLinear(
            self.hidden_size,
            d_model,
            rngs=rngs,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
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
        x = nnx.gelu(x)
        x = self.linear_fc2(x)
        return x


def generate_segment_ids_from_grid_thw_np(
    grid_thw: Tuple[Tuple[int, int, int], ...], ) -> np.ndarray:
    """Generate segment IDs from grid dimensions for variable-length attention (NumPy version)."""
    segments = []
    seg_id = 1
    for (t, h, w) in grid_thw:
        frame_size = h * w
        for _ in range(t):
            segments.append(np.full((frame_size, ), seg_id, dtype=np.int32))
            seg_id += 1
    return np.concatenate(segments, axis=0) if segments else np.zeros(
        (0, ), dtype=np.int32)


class Qwen3VLVisionTransformer(JaxModule, LoadableWithIterator):
    """Vision Transformer for Qwen3VL with DeepStack and Static JIT support."""

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

        num_position_embeddings = getattr(vision_config,
                                          "num_position_embeddings",
                                          VISION_GRID_SIZE * VISION_GRID_SIZE)
        self.pos_embed = JaxEmbed(
            num_embeddings=num_position_embeddings,
            features=self.hidden_size,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rngs,
        )
        self.pos_embed.weight.set_metadata(
            "weight_loader",
            partial(
                load_nnx_param_from_reshaped_torch,
                permute_dims=(0, 1),
                param_name="visual.pos_embed.weight",
            ))
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
                num_position_embeddings)
        self.pos_embed_grid_h = pos_embed_grid_h
        self.pos_embed_grid_w = pos_embed_grid_w

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        intermediate_size = vision_config.intermediate_size
        self.blocks = nnx.List([
            Qwen3VLVisionBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=intermediate_size,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
                norm_eps=norm_eps,
            ) for _ in range(vision_config.depth)
        ])

        text_config = getattr(hf_config, "text_config", hf_config)
        out_hidden_size = getattr(vision_config, "out_hidden_size",
                                  text_config.hidden_size)

        self.merger = Qwen3VLVisionPatchMerger(
            d_model=out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            use_postshuffle_norm=False,
            norm_eps=norm_eps,
        )

        self.deepstack_visual_indexes = tuple(
            getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24]))
        self.deepstack_merger_list = nnx.List([
            Qwen3VLVisionPatchMerger(
                d_model=out_hidden_size,
                context_dim=self.hidden_size,
                spatial_merge_size=self.spatial_merge_size,
                dtype=dtype,
                rngs=rngs,
                use_postshuffle_norm=True,
                norm_eps=norm_eps,
            ) for _ in range(len(self.deepstack_visual_indexes))
        ])

        additional_config = getattr(vllm_config, "additional_config",
                                    None) or {}
        self.enable_dynamic_image_sizes = additional_config.get(
            "enable_dynamic_image_sizes", False)
        _sync_module_param_sharding(self)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        def map_name(name: str) -> str:
            if name.startswith("model.visual."):
                name = name.replace("model.visual.", "", 1)
            elif name.startswith("visual."):
                name = name.replace("visual.", "", 1)
            if "blocks." in name:
                if "attn.qkv." in name:
                    name = name.replace("attn.qkv.", "attn.qkv_projection.")
                elif "mlp.linear_fc1." in name:
                    name = name.replace("mlp.linear_fc1.", "mlp.fc1.")
                elif "mlp.linear_fc2." in name:
                    name = name.replace("mlp.linear_fc2.", "mlp.fc2.")
            return name

        def filter_weights(weights_iterator):
            for name, weight in weights_iterator:
                yield map_name(name), weight

        return super().load_weights(filter_weights(weights))

    def compute_aux_arrays(
        self, grid_thw: Tuple[Tuple[int, int, int], ...]
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Computes grid/position auxiliary arrays (RoPE, pos_embeds, segment_ids) on CPU using NumPy."""

        def get_rope_by_thw_np(t: int, h: int, w: int):
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size
            window_index_thw = np.arange(t * llm_h * llm_w)

            hpos_ids, wpos_ids = np.indices((h, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).transpose(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).transpose(0, 2, 1, 3).flatten()
            pos_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids = np.tile(pos_ids, (t, 1))

            max_size = max(h, w)
            inv_freq = 1.0 / (self.rotary_pos_emb.theta**(
                np.arange(0, self.rotary_pos_emb.dim, 2, dtype=np.float32) /
                self.rotary_pos_emb.dim))
            seq = np.arange(max_size, dtype=np.float32)
            rotary_pos_emb_full = np.outer(seq, inv_freq)

            rotary_pos_emb_thw = rotary_pos_emb_full[pos_ids].reshape(
                pos_ids.shape[0], -1)
            rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(
                rotary_pos_emb_thw.shape[0] // self.spatial_merge_unit,
                self.spatial_merge_unit, -1)

            rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
            rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(
                -1, rotary_pos_emb_thw.shape[-1])

            return rotary_pos_emb_thw, window_index_thw

        def pos_embed_interpolate_np(t: int, h: int, w: int):
            embed_weight = np.array(self.pos_embed.weight[...])
            hidden_dim = embed_weight.shape[1]
            m_size = self.spatial_merge_size

            h_idxs = np.linspace(0, self.pos_embed_grid_h - 1, h)
            w_idxs = np.linspace(0, self.pos_embed_grid_w - 1, w)

            h_floor = h_idxs.astype(np.int32)
            w_floor = w_idxs.astype(np.int32)
            h_ceil = np.clip(h_floor + 1, 0, self.pos_embed_grid_h - 1)
            w_ceil = np.clip(w_floor + 1, 0, self.pos_embed_grid_w - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = np.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = np.meshgrid(h_floor,
                                                     w_floor,
                                                     indexing="ij")
            h_ceil_grid, w_ceil_grid = np.meshgrid(h_ceil,
                                                   w_ceil,
                                                   indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1.0 - dh_grid - w01

            h_grid = np.stack(
                [h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = np.stack(
                [w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * self.pos_embed_grid_w

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = np.stack([w00, w01, w10, w11], axis=0).reshape(4, -1, 1)

            embeds = embed_weight[indices]
            embeds *= weights
            combined = embeds.sum(axis=0)

            combined = combined.reshape(h // m_size, m_size, w // m_size,
                                        m_size, hidden_dim)
            combined = np.transpose(combined,
                                    (0, 2, 1, 3, 4)).reshape(-1, hidden_dim)
            repeated = np.tile(combined, (t, 1))

            return repeated

        num_grids = len(grid_thw)

        rotary_pos_emb = []
        pos_embeds = []
        window_index = []

        window_index_id = 0
        for i in range(num_grids):
            t, h, w = grid_thw[i]
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            rotary_pos_emb_thw, window_index_thw = get_rope_by_thw_np(t, h, w)
            repeated = pos_embed_interpolate_np(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            rotary_pos_emb.append(rotary_pos_emb_thw)
            pos_embeds.append(repeated)

        rotary_pos_emb = np.concatenate(rotary_pos_emb, axis=0)
        pos_embeds = np.concatenate(pos_embeds, axis=0)
        window_index = np.concatenate(window_index, axis=0)

        num_patches = rotary_pos_emb.shape[0]
        bucket_num_patches = 1 << (num_patches - 1).bit_length()
        num_tokens = window_index.shape[0]
        bucket_num_tokens = bucket_num_patches // self.spatial_merge_unit

        rotary_pos_emb_padded = np.pad(rotary_pos_emb,
                                       ((0, bucket_num_patches - num_patches),
                                        (0, 0)))
        pos_embeds_padded = np.pad(pos_embeds,
                                   ((0, bucket_num_patches - num_patches),
                                    (0, 0)))
        window_index_padded = np.concatenate([
            window_index,
            np.arange(num_tokens, bucket_num_tokens, dtype=np.int32)
        ])

        segment_ids = generate_segment_ids_from_grid_thw_np(grid_thw)
        segment_ids_padded = np.pad(
            segment_ids, (0, bucket_num_patches - segment_ids.shape[0]))

        return (jnp.array(window_index_padded),
                jnp.array(rotary_pos_emb_padded, dtype=jnp.bfloat16),
                jnp.array(pos_embeds_padded,
                          dtype=self.dtype), jnp.array(segment_ids_padded))

    @jax.jit
    def encode_padded_jit(
        self,
        x_padded: jax.Array,
        window_index: jax.Array,
        rotary_pos_emb: jax.Array,
        pos_embeds: jax.Array,
        segment_ids: jax.Array,
    ) -> Tuple[jax.Array, List[jax.Array]]:
        hidden_states = self.patch_embed(x_padded)
        hidden_states = hidden_states + pos_embeds

        seq_len = x_padded.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        # Add batch dimension: (seq, dim) -> (seq, 1, dim)
        hidden_states = jnp.expand_dims(hidden_states, axis=1)

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
                    hidden_states.squeeze(1))
                deepstack_features.append(deepstack_feat)

        hidden_states = self.merger(hidden_states.squeeze(1))

        return hidden_states, deepstack_features

    def __call__(
        self, x_padded: jax.Array, grid_thw: Tuple[Tuple[int, int, int], ...]
    ) -> Tuple[jax.Array, List[jax.Array]]:
        """Forward pass for vision encoder with static JIT.

        Args:
            x_padded: Padded pixel values of shape (padded_num_patches, C * T * ps * ps)
            grid_thw: Tuple of (T, H, W) for each image/video

        Returns:
            hidden_states: Final merged features (padded_total_tokens, out_hidden_size)
            deepstack_features: List of intermediate features for DeepStack
        """
        window_index, rotary_pos_emb, pos_embeds, segment_ids = self.compute_aux_arrays(
            grid_thw)

        trace_name = (
            f"encode_padded_jit"
            f"-x_padded_{'_'.join(map(str, x_padded.shape))}"
            f"-window_index_{'_'.join(map(str, window_index.shape))}"
            f"-rotary_pos_emb_{'_'.join(map(str, rotary_pos_emb.shape))}"
            f"-pos_embeds_{'_'.join(map(str, pos_embeds.shape))}"
            f"-segment_ids_{'_'.join(map(str, segment_ids.shape))}")

        start_time = time.time()
        jax.debug.print(f"[vision-jit-profile] Entering {trace_name}")
        with jax.profiler.TraceAnnotation(trace_name):
            hidden_states, deepstack_features = self.encode_padded_jit(
                x_padded, window_index, rotary_pos_emb, pos_embeds,
                segment_ids)
            hidden_states.block_until_ready()
        end_time = time.time()
        jax.debug.print(
            f"[vision-jit-profile] Exiting {trace_name}, time spend: {end_time - start_time}"
        )
        return hidden_states, deepstack_features


def _inject_visual_features(
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
    padded_embeds = jnp.concatenate([dummy_row, visual_embeds, dummy_row],
                                    axis=0)
    gather_indices = jnp.cumsum(flat_mask, dtype=jnp.int32)
    max_index = visual_embeds.shape[0] + 1
    gather_indices = jnp.minimum(gather_indices, max_index)

    updates = padded_embeds[gather_indices] * flat_mask[:, None]
    updated = flat_hidden + updates

    return updated.reshape(hidden_states.shape)


class Qwen3VLModel(Qwen3Model):
    """Text model for Qwen3VL with MRoPE and DeepStack support.

    Overrides __init__ to use Qwen3VLTextDecoderLayer (with MRoPE attention).
    Overrides __call__ to add DeepStack visual feature injection.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        adapted = _VllmConfigAdapter(vllm_config)
        model_config = adapted.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size
        prefix = "model"

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=adapted.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Qwen3VLTextDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=adapted.cache_config.cache_dtype,
                quant_config=adapted.quant_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=adapted.quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

        # Store DeepStack layer indices for injection during forward pass.
        vision_config = getattr(vllm_config.model_config.hf_config,
                                "vision_config", None)
        self.deepstack_visual_indexes = tuple(
            getattr(vision_config, "deepstack_visual_indexes",
                    [8, 16, 24])) if vision_config is not None else ()

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

        # Build a mapping from global layer index to deepstack embed index
        ds_index_map = {}
        if deepstack_visual_embeds is not None and visual_pos_mask is not None:
            for ds_idx, layer_idx in enumerate(self.deepstack_visual_indexes):
                if ds_idx < len(deepstack_visual_embeds):
                    ds_index_map[layer_idx] = ds_idx

        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            global_i = self.start_layer + i
            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            kv_caches[i] = kv_cache

            if global_i in ds_index_map:
                x = _inject_visual_features(
                    x, visual_pos_mask,
                    deepstack_visual_embeds[ds_index_map[global_i]])

        if self.is_last_rank:
            x = self.norm(x)

        return kv_caches, x


class Qwen3VLForConditionalGeneration(JaxModule, LoadableWithIterator):

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh,
    ):
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        config = vllm_config.model_config.hf_config
        self.config = config
        text_config = getattr(config, "text_config", config)

        self.visual = Qwen3VLVisionTransformer(
            vllm_config=vllm_config,
            rngs=rng,
            mesh=mesh,
            norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
        )

        self.language_model = Qwen3VLModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
        )
        self.deepstack_visual_indexes = self.language_model.deepstack_visual_indexes
        model_config = vllm_config.model_config
        if not config.tie_word_embeddings:
            vocab_size = model_config.get_vocab_size()
            hidden_size = text_config.hidden_size
            self.lm_head = JaxEinsum(
                einsum_str="TD,DV->TV",
                kernel_shape=(hidden_size, vocab_size),
                dtype=model_config.dtype,
                rngs=rng,
                quant_config=vllm_config.quant_config,
                kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            )

        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = getattr(config, "vision_start_token_id",
                                             151652)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

    def get_input_embeddings(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array],
        *,
        is_multimodal: jax.Array | None = None,
        do_language_embed_multimodal: bool = True,
    ) -> jax.Array:
        """Get input embeddings with multimodal content merged.

        Args:
            input_ids: Input token IDs
            multimodal_embeddings: Flattened multimodal embeddings
            is_multimodal: Optional boolean mask of multimodal token positions.
            do_language_embed_multimodal: Whether to compute language embeddings
                for multimodal placeholder tokens before merging.

        Returns:
            Input embeddings with multimodal content merged
        """
        if do_language_embed_multimodal:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        else:
            text_config = getattr(self.config, "text_config", self.config)
            embed_shape = (*input_ids.shape, text_config.hidden_size)
            inputs_embeds = jnp.zeros(
                embed_shape, dtype=self.vllm_config.model_config.dtype)

        if multimodal_embeddings is not None and multimodal_embeddings.shape[
                0] != 0:
            if is_multimodal is not None:
                inputs_embeds = _merge_multimodal_embeddings(
                    inputs_embeds,
                    is_multimodal,
                    multimodal_embeddings,
                )
            else:
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
        *,
        is_multimodal: jax.Array | None = None,
    ) -> jax.Array:
        """Compute input embeddings and merge multimodal embeddings if present."""
        mm_embeds_actual = multimodal_embeddings
        deepstack_embeds = None

        if multimodal_embeddings is not None:
            text_config = getattr(self.config, "text_config", self.config)
            hidden_size = text_config.hidden_size
            if multimodal_embeddings.shape[-1] > hidden_size:
                mm_embeds_actual = multimodal_embeddings[..., :hidden_size]
                deepstack_embeds = multimodal_embeddings[..., hidden_size:]

        inputs_embeds = self.get_input_embeddings(
            input_ids,
            mm_embeds_actual,
            is_multimodal=is_multimodal,
        )

        if deepstack_embeds is not None:
            inputs_embeds = jnp.concatenate([inputs_embeds, deepstack_embeds],
                                            axis=-1)

        return inputs_embeds

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
            pixel_values = _safe_convert_torch_to_jax(pixel_values)
            pixel_values = reshape_mm_tensor(pixel_values, "pixel values")

            if not isinstance(pixel_values, jax.Array):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen3VLImagePixelInputs(type="pixel_values",
                                           pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw)

        # NOTE: Not supporting image embeddings precomputed. Matches Qwen2.5VL.
        # if image_embeds is not None:
        #     image_embeds = reshape_mm_tensor(image_embeds, "image embeds")
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
            if input_key in (
                    "pixel_values", "pixel_values_videos",
                    "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(
                        image_grid_thw, **kwargs)
        return mm_input_by_modality

    def _process_image_input(
        self, image_input: Qwen3VLImageInputs
    ) -> tuple[tuple[jax.Array, ...], Optional[list[list[jax.Array]]]]:

        if not image_input:
            return (), None
        grid_thw = image_input["image_grid_thw"]
        if not grid_thw:
            return (), None

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].astype(
                self.visual.dtype)
            deepstack_embeds = None
        else:
            pixel_values = image_input["pixel_values"]
            if pixel_values is None:
                return (), None

            # Pad pixel_values to power of 2 eagerly
            num_patches = pixel_values.shape[0]
            bucket_num_patches = 1 << (num_patches - 1).bit_length()
            pixel_values_padded = jnp.pad(
                pixel_values, ((0, bucket_num_patches - num_patches), (0, 0)))

            image_embeds_padded, deepstack_embeds_padded = self.visual(
                pixel_values_padded, grid_thw)

            # Unpad to actual tokens
            actual_num_tokens = sum(t * (h // self.spatial_merge_size) *
                                    (w // self.spatial_merge_size)
                                    for t, h, w in grid_thw)
            image_embeds = image_embeds_padded[:actual_num_tokens, :]
            if deepstack_embeds_padded is not None:
                deepstack_embeds = [
                    x[:actual_num_tokens, :] for x in deepstack_embeds_padded
                ]
            else:
                deepstack_embeds = None

        return split_mm_embeddings_by_grid(image_embeds, grid_thw,
                                           self.spatial_merge_size,
                                           deepstack_embeds)

    def embed_multimodal(
        self,
        image_grid_thw: Tuple[Tuple[int, int, int], ...] = (),
        video_grid_thw: Tuple[Tuple[int, int, int], ...] = (),
        **kwargs,
    ) -> dict:
        """Get multimodal embeddings from pixel values.

        Qwen3-VL feeds image and video frames through the same vision tower
        with identical grid_thw semantics, so we collapse them into one set
        of grids before encoding.
        """
        image_grid_thw = normalize_mm_grid_thw(image_grid_thw)
        if not image_grid_thw:
            image_grid_thw = normalize_mm_grid_thw(video_grid_thw)

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

        if not multimodal_embeddings:
            return {}

        return {
            "embeds": multimodal_embeddings,
            "deepstack": deepstack_outputs
        }

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        _intermediate_tensors=None,
        _is_first_rank: bool = True,
        _is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array],
               Optional[jax.Array]]:

        if inputs_embeds is not None:
            text_config = getattr(self.config, "text_config", self.config)
            num_ds_layers = len(self.deepstack_visual_indexes)
            expected_concat_dim = (1 + num_ds_layers) * text_config.hidden_size
            if inputs_embeds.shape[-1] == expected_concat_dim:
                splits = jnp.split(inputs_embeds, 1 + num_ds_layers, axis=-1)
                inputs_embeds = splits[0]
                deepstack_embeds = list(splits[1:])
            else:
                deepstack_embeds = None
        else:
            deepstack_embeds = None
        visual_pos_mask = None

        if deepstack_embeds is not None and input_ids is not None:
            visual_pos_mask = (input_ids == self.image_token_id) | (
                input_ids == self.video_token_id)

        kv_caches, hidden_states = self.language_model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_embeds,
        )

        if not _is_last_rank:
            hidden_states = JaxIntermediateTensors(
                tensors={"hidden_states": hidden_states})

        return kv_caches, hidden_states, [], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            return self.lm_head(hidden_states)

        return self.language_model.embed_tokens.decode(hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: List[int],
        mm_features: Optional[list] = None,
    ) -> Tuple[jax.Array, int]:
        """Compute MRoPE 3D position IDs for input sequence.

        Args:
            input_tokens: List of token IDs for the sequence
            mm_features: List of MultiModalFeatureSpec from the scheduler

        Returns:
            llm_positions: (3, seq_len) position IDs for [T, H, W]
            mrope_position_delta: Delta for rope calculation
        """
        image_grid_thw = []
        video_grid_thw = []

        if mm_features:
            for mm_feature in mm_features:
                item = mm_feature.data
                if item is None:
                    continue
                mm_input = item.get_data()
                if mm_input.get("image_grid_thw") is not None:
                    image_grid_thw.extend(
                        normalize_mm_grid_thw(mm_input["image_grid_thw"]))
                if mm_input.get("video_grid_thw") is not None:
                    video_grid_thw.extend(
                        normalize_mm_grid_thw(mm_input["video_grid_thw"]))

        hf_config = self.config

        llm_positions, mrope_position_delta = build_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=image_grid_thw or None,
            video_grid_thw=video_grid_thw or None,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=getattr(hf_config, "vision_start_token_id",
                                          self.vision_start_token_id),
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )

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
        patch_input_dim = (vc.in_channels * vc.temporal_patch_size *
                           vc.patch_size * vc.patch_size)

        # We want to precompile the power-of-2 buckets
        bucket_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]

        for B in bucket_sizes:
            # Find H, W such that H * W = B and both are even
            # B is power of 2: B = 2^k
            k = B.bit_length() - 1
            h = 1 << (k // 2)
            w = 1 << (k - k // 2)
            t = 1
            grid_thw = (t, h, w)

            dummy_pixel_values_padded = jnp.ones(
                (B, patch_input_dim),
                dtype=utils.to_jax_dtype(self.vllm_config.model_config.dtype),
            )
            dummy_grid_thw = (grid_thw, )

            # We call self.visual (which calls self.visual.__call__ -> encode_padded_jit)
            run_compilation_fn(
                f"vision_encoder_bucket_{B}",
                self.visual,
                dummy_pixel_values_padded,
                dummy_grid_thw,
                bucket_size=B,
            )

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        if not isinstance(weights, Iterable):
            return super().load_weights(weights)

        def map_name(name: str) -> str:
            # Remap PyTorch vision tower keys
            if name.startswith("model.visual."):
                name = name.replace("model.visual.", "visual.", 1)

                # Attention block remappings
                if "blocks." in name:
                    # QKV projection: name attn.qkv -> attn.qkv_proj
                    if "attn.qkv." in name:
                        name = name.replace("attn.qkv.",
                                            "attn.qkv_projection.")
                    # MLP feedforward layers: PyTorch uses linear_fc1/linear_fc2, JAX uses fc1/fc2
                    elif "mlp.linear_fc1." in name:
                        name = name.replace("mlp.linear_fc1.", "mlp.fc1.")
                    elif "mlp.linear_fc2." in name:
                        name = name.replace("mlp.linear_fc2.", "mlp.fc2.")

            # Remap PyTorch language model keys (replace 'model.language_model.' with 'language_model.')
            elif name.startswith("model.language_model."):
                name = name.replace("model.", "", 1)

            return name

        def filter_weights(weights_iterator):
            for name, weight in weights_iterator:
                yield map_name(name), weight

        return super().load_weights(filter_weights(weights))
