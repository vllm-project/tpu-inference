# Copyright 2025 Google LLC
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
from functools import partial
from typing import (Callable, List, Literal, NamedTuple, Optional, TypedDict,
                    Union)

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from transformers import modeling_flax_utils
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)
from vllm.config import VllmConfig

from tpu_inference import utils as utils
from tpu_inference.layers.common.attention_interface import \
    sharded_flash_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.qwen2 import Qwen2ForCausalLM
# from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from tpu_inference.models.jax.utils.multi_modal_utils import (
    MultiModalEmbeddings, merge_multimodal_embeddings)
from tpu_inference.models.jax.utils.weight_utils import (get_default_maps,
                                                         load_hf_weights)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

DEFAULT_BLOCK_K_MAJOR = 128


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

  SegmentIds are used to generate segment mask, which prevents attention between
  different segments in the input sequence. Each array is a list of ids
  (integers).
  Only the token with the same id can attend to each other.

  Attributes:
    q: segment ids along the Q sequence.
    kv: segment ids along the KV sequence.
  """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


class Qwen2_5_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: jax.Array
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: tuple[tuple[int, int, int], ...]
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


# NOTE: We are not supporting embedding inputs for now
# The code here makes the struture consistent and
# makes iteasier for future implementation
class Qwen2_5_VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: jax.Array
    """Supported types:
    - list[`jax.Array`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `jax.Array`: A tensor holding all images' features (concatenation of
        all images' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: jax.Array
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2_5_VLImageInputs = Union[Qwen2_5_VLImagePixelInputs,
                              Qwen2_5_VLImageEmbeddingInputs]


class Qwen2_5_VisionMLP(nnx.Module):

    def __init__(self, config: Qwen2_5_VLVisionConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        in_features = config.hidden_size
        hidden_features = config.intermediate_size
        act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]
        self.gate_proj = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            hidden_features,
            in_features,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rngs,
        )
        self.act_fn = act_fn

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


def apply_rotary_pos_emb_vision(x: jax.Array,
                                rotary_pos_emb: jax.Array) -> jax.Array:
    # x: [B, T, N, H]
    # rotary_pos_emb: [T, H//2]
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


def generate_window_segment_ids(cu_seqlens: jax.Array, seq_len: int,
                                padded_seq_len: int) -> SegmentIds:
    """Generates segment IDs for windowed attention

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths for each window.
            e.g., [0, len_win0, len_win0+len_win1, ...]

    Returns:
        A SegmentIds object for flash_attention.
    """
    indices = jnp.arange(seq_len, dtype=jnp.int32)
    segment_ids = jnp.searchsorted(cu_seqlens[1:], indices, side='right') + 1
    padding_segment_ids = jnp.zeros(padded_seq_len - seq_len, dtype=jnp.int32)
    segment_ids = jnp.concatenate([segment_ids, padding_segment_ids])
    segment_ids = segment_ids.reshape(1, -1)

    return SegmentIds(q=segment_ids, kv=segment_ids)


class Qwen2_5_VisionAttention(nnx.Module):

    def __init__(self, config: Qwen2_5_VLConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs, mesh: Mesh):
        vision_config = config.vision_config
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_kv_heads = self.num_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.head_dim_original = self.hidden_size // self.num_heads

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        # TODO: Wenlong: Do not consider padding for now
        self.head_dim = self.head_dim_original

        self.mesh = mesh

        self.qkv_proj = nnx.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs,
        )

        self.proj = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rngs,
        )
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
        cu_window_seqlens: Optional[jax.Array] = None,
        use_fullattn: bool = True,
    ) -> jax.Array:
        T, B, D = x.shape
        assert B == 1, "Vision attention currently only supports batch size 1"
        # [T, B, D] -> [T, B, 3 * D]
        qkv = self.qkv_proj(x)

        # Split into Q, K, V.
        # NOTE: simplified from vLLM's split_qkv,
        # may need to revisit for tp>1
        # [T, B, 3 * D] -> 3 *[T, B, D]
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # [T, B, N, H]
        q = q.reshape(T, B, self.num_heads, self.head_dim)
        k = k.reshape(T, B, self.num_heads, self.head_dim)
        v = v.reshape(T, B, self.num_heads, self.head_dim)

        # [T, B, N, H] -> [B, T, N, H]
        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))

        # rotary_pos_emb shape: (T, H)
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # NOTE: an extra transpose because we need to
        # align the correctness with vLLM's design.
        # Might be able to remove one once implemented.
        # [B, T, N, H] -> [B, N, T, H]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Pad the sequence length to be a multiple of 128 for flash_attention
        block_k_major = DEFAULT_BLOCK_K_MAJOR
        T_attn = q.shape[2]
        padded_T = (T_attn + block_k_major -
                    1) // block_k_major * block_k_major
        pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))

        q = jnp.pad(q, pad_width, 'constant')
        k = jnp.pad(k, pad_width, 'constant')
        v = jnp.pad(v, pad_width, 'constant')

        segment_ids = generate_window_segment_ids(cu_window_seqlens, T_attn,
                                                  padded_T)

        # TODO (jacobplatin): add support for quantized KV cache?
        output = self.flash_attention(q, k, v, segment_ids)

        # Unpad the output
        output = output[:, :, :T_attn, :]

        # [B, N, T, H] -> [T, B, N, H]
        output = jnp.transpose(output, (2, 0, 1, 3))

        output = output.reshape(T, B, D)

        output = self.proj(output)

        return output


class Qwen2_5_VisionBlock(nnx.Module):

    def __init__(self, config: Qwen2_5_VLConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs, mesh: Mesh):
        vision_config = config.vision_config
        dim = vision_config.hidden_size
        norm_layer = partial(nnx.RMSNorm,
                             epsilon=config.rms_norm_eps,
                             scale_init=nnx.with_partitioning(
                                 init_fn, (None, )))

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.attn = Qwen2_5_VisionAttention(config=config,
                                            dtype=dtype,
                                            rngs=rngs,
                                            mesh=mesh)
        self.mlp = Qwen2_5_VisionMLP(config=vision_config,
                                     dtype=dtype,
                                     rngs=rngs)

    def __call__(self,
                 x: jax.Array,
                 rotary_pos_emb: jax.Array,
                 cu_window_seqlens: Optional[jax.Array] = None,
                 use_fullattn: bool = True) -> jax.Array:

        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_window_seqlens,
                          use_fullattn)
        x = x + self.mlp(self.norm2(x))

        return x


class Qwen2_5_VisionPatchEmbed(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(in_features=in_channels,
                             out_features=hidden_size,
                             kernel_size=kernel_size,
                             strides=kernel_size,
                             use_bias=False,
                             param_dtype=dtype,
                             kernel_init=nnx.with_partitioning(
                                 init_fn, (None, None, None, None, "model")),
                             rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is (L, C * T * H * W)
        L, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size *
                    self.patch_size)
        # Reshape to (L, T, H, W, C) for Conv3D with channels_last
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size,
                      self.patch_size)
        # L,T,H,W,C
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv, shape is (L, T_out, H_out, W_out, C_out)
        # With stride=kernel_size, T_out=H_out=W_out=1.
        # So shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        return x


class Qwen2_5_VisionPatchMerger(nnx.Module):

    def __init__(self, d_model: int, context_dim: int, norm_layer: Callable,
                 spatial_merge_size: int, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = norm_layer(context_dim,
                               dtype=dtype,
                               rngs=rngs,
                               scale_init=nnx.with_partitioning(
                                   init_fn, (None, )))
        self.mlp_fc1 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs)
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = nnx.Linear(
            self.hidden_size,
            d_model,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.ln_q(x)
        x = x.reshape(-1, self.hidden_size)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        return x


class Qwen2_5_VisionRotaryEmbedding(nnx.Module):

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta**(
            jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)


class Qwen2_5_VisionTransformer(nnx.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 norm_eps: float = 1e-6):
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

        # args for get_window_index_thw
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
            dtype=dtype,
            rngs=rngs)

        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [
            Qwen2_5_VisionBlock(
                config=hf_config,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            ) for _ in range(vision_config.depth)
        ]
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.hidden_size,
            norm_layer=partial(nnx.RMSNorm, epsilon=norm_eps),
            spatial_merge_size=vision_config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs)

        additional_config = getattr(vllm_config, "additional_config",
                                    None) or {}
        self.enable_dynamic_image_sizes = additional_config.get(
            "enable_dynamic_image_sizes", False)

    def rotary_pos_emb_thw(self, t, h, w):
        hpos_ids, wpos_ids = jnp.indices((h, w))
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
        pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids = jnp.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)

        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(
            pos_ids.shape[0], -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit, -1)

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size

        index = jnp.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w)

        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)),
                               constant_values=-100)
        index_padded = index_padded.reshape(grid_t, num_windows_h,
                                            vit_merger_window_size,
                                            num_windows_w,
                                            vit_merger_window_size)
        index_padded = jnp.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
            vit_merger_window_size)
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        # The number of valid indices is static because grid_t, grid_h, grid_w
        # are static.
        num_valid_indices = grid_t * llm_grid_h * llm_grid_w
        valid_indices = jnp.nonzero(index_padded != -100,
                                    size=num_valid_indices)[0]
        index_new = index_padded[valid_indices]
        cu_seqlens_tmp = jnp.cumsum(seqlens) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.astype(jnp.int32)

        # NOTE (wenlong): Pytorch code uses this to reduce replication,
        # but I don't think there is a need here, plus it would cause problem in JIT
        # Please refer here if there is a problem down-stream
        # cu_seqlens_tmp = jnp.unique(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(
            t, h, w)

        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)

        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(
            -1, rotary_pos_emb_thw.shape[-1])
        cu_seqlens_thw = jnp.full(t, h * w, dtype=jnp.int32)

        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw,
                cu_seqlens_thw)

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: jax.Array,
    ) -> tuple[Optional[int], Optional[list[int]]]:
        max_seqlen, seqlens = None
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    def compute_aux_arrays(self, grid_thw: tuple[tuple[int, int, int]]):
        # num of images/videoes
        num_grids = len(grid_thw)

        rotary_pos_emb = []
        window_index: list = []
        cu_window_seqlens: list = [jnp.array([0], dtype=jnp.int32)]
        cu_seqlens: list = []

        window_index_id = 0
        cu_window_seqlens_last = 0
        for i in range(num_grids):
            t, h, w = grid_thw[i]

            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            cu_seqlens_window_thw = (cu_seqlens_window_thw +
                                     cu_window_seqlens_last)
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = jnp.concatenate(rotary_pos_emb, axis=0)
        window_index = jnp.concatenate(window_index, axis=0)
        cu_window_seqlens = jnp.concatenate(cu_window_seqlens, axis=0)

        cu_seqlens = jnp.concatenate(cu_seqlens, axis=0)
        cu_seqlens = jnp.cumsum(cu_seqlens, axis=0, dtype=jnp.int32)
        cu_seqlens = jnp.pad(cu_seqlens, ((1, 0), ),
                             mode='constant',
                             constant_values=0)
        return window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens

    def pad_inputs(self, x, window_index, rotary_pos_emb, cu_seqlens,
                   cu_window_seqlens):
        # padding
        num_patches = int(rotary_pos_emb.shape[0])
        bucket_num_patches = 1 << (num_patches - 1).bit_length()
        num_tokens = window_index.shape[0]
        bucket_num_tokens = bucket_num_patches // self.spatial_merge_unit
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)
        max_windows = (bucket_num_tokens // vit_merger_window_size) + 2

        rotary_pos_emb = jnp.pad(rotary_pos_emb,
                                 ((0, bucket_num_patches - num_patches),
                                  (0, 0)))
        window_index = jnp.concatenate([
            window_index,
            jnp.arange(num_tokens, bucket_num_tokens, dtype=jnp.int32)
        ])
        cu_window_seqlens = jnp.append(cu_window_seqlens, bucket_num_patches)
        pad_w = max(0, max_windows + 1 - cu_window_seqlens.shape[0])
        cu_window_seqlens = jnp.pad(cu_window_seqlens, (0, pad_w), mode='edge')
        cu_seqlens = jnp.append(cu_seqlens, bucket_num_patches)

        x_padded = jnp.pad(x, ((0, bucket_num_patches - x.shape[0]), (0, 0)))

        return x_padded, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens, num_tokens

    def compute_hidden_states(self, x: jax.Array, window_index: jax.Array,
                              rotary_pos_emb: jax.Array, cu_seqlens: jax.Array,
                              cu_window_seqlens: jax.Array) -> jax.Array:
        hidden_states = self.patch_embed(x)

        # num of patches
        seq_len = x.shape[0]

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        hidden_states = jnp.expand_dims(hidden_states, axis=1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                hidden_states = blk(hidden_states,
                                    rotary_pos_emb=rotary_pos_emb,
                                    cu_window_seqlens=cu_seqlens,
                                    use_fullattn=True)
            else:
                hidden_states = blk(hidden_states,
                                    rotary_pos_emb=rotary_pos_emb,
                                    cu_window_seqlens=cu_window_seqlens,
                                    use_fullattn=False)

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = jnp.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    @jax.jit
    def encode_padded_jit(self, x_padded, window_index, rotary_pos_emb,
                          cu_seqlens, cu_window_seqlens):
        return self.compute_hidden_states(x_padded, window_index,
                                          rotary_pos_emb, cu_seqlens,
                                          cu_window_seqlens)

    @partial(
        jax.jit,
        static_argnames=("grid_thw", ),
    )
    def encode_jit(self, x, grid_thw):
        window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens = self.compute_aux_arrays(
            grid_thw)
        return self.compute_hidden_states(x, window_index, rotary_pos_emb,
                                          cu_seqlens, cu_window_seqlens)

    def __call__(self, x: jax.Array, grid_thw: tuple[tuple[int, int,
                                                           int]]) -> jax.Array:
        # x: pixel_values: jax.Array
        # """Shape:
        # `(num_patches, num_channels * patch_size * patch_size)`
        # """

        # grid_thw: image_grid_thw: jax.Array
        # """Shape: `(num_images, 3)`
        # This should be in `(grid_t, grid_h, grid_w)` format.
        # """
        if self.enable_dynamic_image_sizes:
            window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens = self.compute_aux_arrays(
                grid_thw)
            x_padded, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens, num_tokens = self.pad_inputs(
                x, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens)

            hidden_states = self.encode_padded_jit(x_padded, window_index,
                                                   rotary_pos_emb, cu_seqlens,
                                                   cu_window_seqlens)
            return hidden_states[:num_tokens]

        else:
            return self.encode_jit(x, grid_thw)


class Qwen2_5_VLForConditionalGeneration(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen2_5_VisionTransformer(
            vllm_config=vllm_config,
            rngs=self.rng,
            mesh=mesh,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        self.language_model = Qwen2ForCausalLM(vllm_config, rng_key, mesh)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths=None,
        use_audio_in_video: bool = False,
    ) -> tuple[jax.Array, int]:
        """Get mrope input positions and delta value."""

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config,
                                    "tokens_per_second", 1.0)

        input_tokens_tensor = np.array(input_tokens)
        vision_start_indices = np.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = np.sum(vision_tokens == image_token_id)
        video_nums = np.sum(vision_tokens == video_token_id)
        llm_pos_ids_list: list = []

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
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                jnp.broadcast_to(
                    jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                    (3, text_len)) + st_idx)

            t_index = ((jnp.broadcast_to(
                jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1),
                (llm_grid_t, llm_grid_h * llm_grid_w)) *
                        video_second_per_grid_t * tokens_per_second).astype(
                            jnp.int32).flatten())

            h_index = (jnp.broadcast_to(
                jnp.arange(llm_grid_h, dtype=jnp.int32).reshape(1, -1, 1),
                (llm_grid_t, llm_grid_h, llm_grid_w)).flatten())
            w_index = (jnp.broadcast_to(
                jnp.arange(llm_grid_w, dtype=jnp.int32).reshape(1, 1, -1),
                (llm_grid_t, llm_grid_h, llm_grid_w)).flatten())

            llm_pos_ids_list.append(
                jnp.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st

            llm_pos_ids_list.append(
                jnp.broadcast_to(
                    jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                    (3, text_len)) + st_idx)

        llm_positions = jnp.concatenate(llm_pos_ids_list,
                                        axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> jax.Array:
        if isinstance(mm_input, list):
            # Assuming it's a list of arrays (e.g., np.ndarray, torch.Tensor)
            # that can be concatenated.
            arrays_to_concat = [jnp.asarray(item) for item in mm_input]
            return jnp.concatenate(arrays_to_concat, axis=0)

        # Handle single array-like objects (np.ndarray, torch.Tensor, jax.Array)
        if hasattr(mm_input, 'ndim'):
            array_input = jnp.asarray(mm_input)
            if array_input.ndim == 2:
                return array_input
            if array_input.ndim == 3:
                # This reshapes the batched 3D tensor to a 2D tensor.
                return array_input.reshape(-1, array_input.shape[-1])

        raise ValueError(f"Incorrect type of {name}. "
                         f"Got type: {type(mm_input)}")

    def _parse_and_validate_image_input(
            self, image_grid_thw: tuple[tuple[int, int, int], ...],
            **kwargs: object) -> Optional[Qwen2_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        # image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            # image_grid_thw = self._validate_and_reshape_mm_tensor(
            #     image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, jax.Array):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        # Note: comment them out for now and save for future support
        # if image_embeds is not None:
        #     image_embeds = self._validate_and_reshape_mm_tensor(
        #         image_embeds, "image embeds")
        #     image_grid_thw = self._validate_and_reshape_mm_tensor(
        #         image_grid_thw, "image grid_thw")

        #     if not isinstance(image_embeds, jax.Array):
        #         raise ValueError("Incorrect type of image embeddings. "
        #                          f"Got type: {type(image_embeds)}")
        #     return Qwen2_5_VLImageEmbeddingInputs(
        #         type="image_embeds",
        #         image_embeds=image_embeds,
        #         image_grid_thw=image_grid_thw)

    def _parse_and_validate_multimodal_inputs(self,
                                              image_grid_thw: tuple[tuple[int,
                                                                          int,
                                                                          int],
                                                                    ...],
                                              **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(
                        image_grid_thw, **kwargs)
            # if input_key in ("pixel_values_videos", "video_embeds"
            #                  ) and "video" not in mm_input_by_modality:
            #     mm_input_by_modality[
            #         "video"] = self._parse_and_validate_video_input(**kwargs)
        return mm_input_by_modality

    def get_single_image_embedding(self, image_pixel_values, image_grid_thw):
        return self.visual(image_pixel_values, (image_grid_thw, ))

    def _process_image_input(
            self, image_input: Qwen2_5_VLImageInputs) -> tuple[jax.Array, ...]:

        grid_thw = image_input["image_grid_thw"]

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].astype(
                self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = []
            current_idx = 0
            for image_thw in grid_thw:
                t, h, w = image_thw
                image_size = t * h * w
                end_idx = current_idx + image_size
                image_pixel_values = pixel_values[current_idx:end_idx, :]
                image_embeds.append(
                    self.get_single_image_embedding(image_pixel_values,
                                                    image_thw))
                current_idx = end_idx
            image_embeds = jnp.concatenate(image_embeds, axis=0)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.config.spatial_merge_size
        sizes = np.prod(np.array(grid_thw, dtype=np.int64),
                        axis=-1) // merge_size // merge_size

        if sizes.size == 0:
            return ()
        if sizes.size == 1:
            return (image_embeds, )

        split_indices = np.cumsum(sizes)[:-1]
        return tuple(jnp.split(image_embeds, split_indices))

    def embed_multimodal(self, image_grid_thw: tuple[tuple[int, int, int],
                                                     ...],
                         **kwargs: object) -> MultiModalEmbeddings:

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            image_grid_thw, **kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[jax.Array, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            # if modality == "video":
            #     video_embeddings = self._process_video_input(multimodal_input)
            #     multimodal_embeddings += video_embeddings

        return multimodal_embeddings

    def embed_input_ids(
            self, input_ids: jax.Array,
            multimodal_embeddings: Optional[jax.Array]) -> jax.Array:

        inputs_embeds = self.language_model.model.embed(input_ids)


        if multimodal_embeddings is not None \
            and multimodal_embeddings.shape[0] != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])

        return inputs_embeds

    def __call__(
        self,
        kv_caches: list[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        *args,
    ) -> tuple[list[jax.Array], jax.Array, List[jax.Array]]:
        # The logic of choosing between input_ids and inputs_embeds is
        # handled inside self.language_model.__call__
        kv_caches, x, [] = self.language_model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
        )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, rng_key: jax.Array) -> None:
        self.rng = nnx.Rngs(rng_key)
        self.language_model.rng = self.rng

        # Key: path to a HF layer weight
        # Value: a tuple of (path to a nnx layer weight, nnx weight sharding)

        mappings = {
            "model.embed_tokens": "language_model.model.embed.embedding",
            "model.layers.*.input_layernorm":
            "language_model.model.layers.*.input_layernorm.scale",
            "model.layers.*.mlp.down_proj":
            "language_model.model.layers.*.mlp.down_proj.kernel",
            "model.layers.*.mlp.gate_proj":
            "language_model.model.layers.*.mlp.gate_proj.kernel",
            "model.layers.*.mlp.up_proj":
            "language_model.model.layers.*.mlp.up_proj.kernel",
            "model.layers.*.post_attention_layernorm":
            "language_model.model.layers.*.post_attention_layernorm.scale",
            "model.layers.*.self_attn.k_proj":
            "language_model.model.layers.*.self_attn.k_proj.kernel",
            "model.layers.*.self_attn.o_proj":
            "language_model.model.layers.*.self_attn.o_proj.kernel",
            "model.layers.*.self_attn.q_proj":
            "language_model.model.layers.*.self_attn.q_proj.kernel",
            "model.layers.*.self_attn.v_proj":
            "language_model.model.layers.*.self_attn.v_proj.kernel",
            "model.layers.*.self_attn.q_proj.bias":
            "language_model.model.layers.*.self_attn.q_proj.bias",
            "model.layers.*.self_attn.k_proj.bias":
            "language_model.model.layers.*.self_attn.k_proj.bias",
            "model.layers.*.self_attn.v_proj.bias":
            "language_model.model.layers.*.self_attn.v_proj.bias",
            "model.norm": "language_model.model.norm.scale",
            "visual.blocks.*.attn.proj.bias": "visual.blocks.*.attn.proj.bias",
            "visual.blocks.*.attn.proj": "visual.blocks.*.attn.proj.kernel",
            "visual.blocks.*.attn.qkv.bias":
            "visual.blocks.*.attn.qkv_proj.bias",
            "visual.blocks.*.attn.qkv": "visual.blocks.*.attn.qkv_proj.kernel",
            "visual.blocks.*.mlp.down_proj.bias":
            "visual.blocks.*.mlp.down_proj.bias",
            "visual.blocks.*.mlp.down_proj":
            "visual.blocks.*.mlp.down_proj.kernel",
            "visual.blocks.*.mlp.gate_proj.bias":
            "visual.blocks.*.mlp.gate_proj.bias",
            "visual.blocks.*.mlp.gate_proj":
            "visual.blocks.*.mlp.gate_proj.kernel",
            "visual.blocks.*.mlp.up_proj.bias":
            "visual.blocks.*.mlp.up_proj.bias",
            "visual.blocks.*.mlp.up_proj":
            "visual.blocks.*.mlp.up_proj.kernel",
            "visual.blocks.*.norm1": "visual.blocks.*.norm1.scale",
            "visual.blocks.*.norm2": "visual.blocks.*.norm2.scale",
            "visual.merger.ln_q": "visual.merger.ln_q.scale",
            "visual.merger.mlp.0.bias": "visual.merger.mlp_fc1.bias",
            "visual.merger.mlp.0": "visual.merger.mlp_fc1.kernel",
            "visual.merger.mlp.2.bias": "visual.merger.mlp_fc2.bias",
            "visual.merger.mlp.2": "visual.merger.mlp_fc2.kernel",
            "visual.patch_embed.proj": "visual.patch_embed.proj.kernel",
        }

        # Add lm_head mapping only if it's not tied to embeddings
        hf_config = self.vllm_config.model_config.hf_config
        if not hf_config.tie_word_embeddings:
            mappings.update({
                "lm_head": "language_model.model.lm_head",
            })

        metadata_map = get_default_maps(self.vllm_config.model_config,
                                        self.mesh, mappings)
        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        metadata_map=metadata_map,
                        mesh=self.mesh)

    def precompile_vision_encoder(
        self,
        run_compilation_fn: Callable,
    ) -> None:
        vc = self.vllm_config.model_config.hf_config.vision_config
        patch_input_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        if self.visual.enable_dynamic_image_sizes:
            spatial_merge_unit = vc.spatial_merge_size**2
            max_num_batched_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            mm_kwargs = self.vllm_config.model_config.multimodal_config.mm_processor_kwargs or {}
            limit_pixels = float(mm_kwargs.get("max_pixels", float('inf')))

            max_patches = int(
                min(max_num_batched_tokens * spatial_merge_unit,
                    limit_pixels / (vc.patch_size**2)))

            num_patches_paddings = [
                1 << i for i in range(4, (max_patches - 1).bit_length() + 1)
            ]
            rotary_dim = vc.hidden_size // vc.num_heads // 2
            vit_merger_window_size = (vc.window_size //
                                      vc.spatial_merge_size // vc.patch_size)

            for num_patches in num_patches_paddings:
                dummy_x_padded = jnp.ones(
                    (num_patches, patch_input_dim),
                    dtype=self.vllm_config.model_config.dtype)

                num_tokens = num_patches // spatial_merge_unit
                dummy_window_index = jnp.arange(num_tokens, dtype=jnp.int32)

                dummy_rotary_pos_emb = jnp.ones(
                    (num_patches, rotary_dim),
                    dtype=self.vllm_config.model_config.dtype)

                dummy_cu_seqlens = jnp.array([0, num_patches, num_patches],
                                             dtype=jnp.int32)

                max_windows = (num_tokens // vit_merger_window_size) + 2
                patches_per_window = (vit_merger_window_size**
                                      2) * spatial_merge_unit
                dummy_cu_window_seqlens = jnp.arange(
                    max_windows + 1, dtype=jnp.int32) * patches_per_window
                dummy_cu_window_seqlens = jnp.minimum(dummy_cu_window_seqlens,
                                                      num_patches)

                run_compilation_fn("vision_encoder_padded",
                                   self.visual.encode_padded_jit,
                                   dummy_x_padded,
                                   dummy_window_index,
                                   dummy_rotary_pos_emb,
                                   dummy_cu_seqlens,
                                   dummy_cu_window_seqlens,
                                   num_patches=num_patches)
        else:
            image_shapes = []
            if (warmup_config := self.vllm_config.additional_config.get(
                    "vision_warmup_config")):
                image_shapes = warmup_config.get("image_shapes")

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
                dummy_grid_thw = (grid_thw, )

                run_compilation_fn("vision_encoder",
                                   self.visual.encode_jit,
                                   dummy_pixel_values,
                                   dummy_grid_thw,
                                   image_shape=input_hw)
