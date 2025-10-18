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
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig, Qwen2VLVisionConfig)
from vllm.config import VllmConfig

from tpu_inference import utils as utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.attention_interface import \
    sharded_flash_attention
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.qwen2 import Qwen2ForCausalLM
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


class Qwen2VLImagePixelInputs(TypedDict):
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
class Qwen2VLImageEmbeddingInputs(TypedDict):
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


Qwen2VLImageInputs = Union[Qwen2VLImagePixelInputs,
                           Qwen2VLImageEmbeddingInputs]


class Qwen2VisionMLP(nnx.Module):

    def __init__(self, config: Qwen2VLVisionConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        # Use embed_dim for vision MLP
        in_features = config.embed_dim
        # Qwen2VL uses mlp_ratio instead of intermediate_size
        hidden_features = int(in_features * config.mlp_ratio)
        act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]
        # Qwen2VL vision uses a simple 2-layer MLP (fc1 -> activation -> fc2)
        # not a GLU structure
        self.fc1 = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
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
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


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


class Qwen2VisionAttention(nnx.Module):

    def __init__(self, config: Qwen2VLConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs, mesh: Mesh):
        vision_config = config.vision_config
        # Qwen2VL uses embed_dim for the actual hidden size in vision blocks
        self.hidden_size = vision_config.embed_dim  # 1280 instead of 3584
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

        # For Qwen2VL, use simple segment ids without windowing
        segment_ids = SegmentIds(
            q=jnp.ones((1, padded_T), dtype=jnp.int32),
            kv=jnp.ones((1, padded_T), dtype=jnp.int32)
        )

        # TODO (jacobplatin): add support for quantized KV cache?
        output = self.flash_attention(q, k, v, segment_ids)

        # Unpad the output
        output = output[:, :, :T_attn, :]

        # [B, N, T, H] -> [T, B, N, H]
        output = jnp.transpose(output, (2, 0, 1, 3))

        output = output.reshape(T, B, D)

        output = self.proj(output)

        return output


class Qwen2VisionBlock(nnx.Module):

    def __init__(self, config: Qwen2VLConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs, mesh: Mesh):
        vision_config = config.vision_config
        # Use embed_dim for vision blocks
        dim = vision_config.embed_dim
        # Qwen2VL vision uses LayerNorm, not RMSNorm
        norm_layer = partial(nnx.LayerNorm,
                             epsilon=1e-6,
                             scale_init=nnx.with_partitioning(
                                 init_fn, (None, )),
                             bias_init=nnx.with_partitioning(
                                 init_fn, (None, )))

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.attn = Qwen2VisionAttention(config=config,
                                         dtype=dtype,
                                         rngs=rngs,
                                         mesh=mesh)
        self.mlp = Qwen2VisionMLP(config=vision_config,
                                  dtype=dtype,
                                  rngs=rngs)

    def __call__(self,
                 x: jax.Array,
                 rotary_pos_emb: jax.Array) -> jax.Array:

        x = x + self.attn(self.norm1(x), rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))

        return x


class Qwen2VisionPatchEmbed(nnx.Module):

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


class Qwen2VisionPatchMerger(nnx.Module):

    def __init__(self, d_model: int, context_dim: int, norm_layer: Callable,
                 spatial_merge_size: int, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = norm_layer(context_dim,
                               dtype=dtype,
                               rngs=rngs,
                               scale_init=nnx.with_partitioning(
                                   init_fn, (None, )),
                               bias_init=nnx.with_partitioning(
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


class Qwen2VisionRotaryEmbedding(nnx.Module):

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta**(
            jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)


class Qwen2VisionTransformer(nnx.Module):

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
        # Use embed_dim for vision transformer
        self.hidden_size = vision_config.embed_dim
        self.num_heads = vision_config.num_heads

        # args for processing
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen2VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
            dtype=dtype,
            rngs=rngs)

        # Calculate head_dim using embed_dim
        head_dim = vision_config.embed_dim // vision_config.num_heads
        self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [
            Qwen2VisionBlock(
                config=hf_config,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            ) for _ in range(vision_config.depth)
        ]
        # For Qwen2VL, the output dimension should match the language model's hidden size
        self.merger = Qwen2VisionPatchMerger(
            d_model=hf_config.text_config.hidden_size,  # Language model's hidden size
            context_dim=vision_config.embed_dim,  # Use embed_dim for vision context
            norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),
            spatial_merge_size=vision_config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs)

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

        # rotary_pos_emb_full shape: [max_size, head_dim//2]
        # pos_ids shape: [t*h*w, 2]
        # We need to return shape: [t*h*w, head_dim//2] for apply_rotary_pos_emb_vision
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(
            pos_ids.shape[0], -1)
        
        # Don't reshape further - keep it as [seq_len, head_dim//2]
        return rotary_pos_emb

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
        hidden_states = self.patch_embed(x)

        # num of patches
        seq_len = x.shape[0]
        # num of images/videoes
        num_grids = len(grid_thw)

        rotary_pos_emb = []
        cu_seqlens: list = []

        for i in range(num_grids):
            t, h, w = grid_thw[i]

            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)
            cu_seqlens_thw = jnp.full(t, h * w, dtype=jnp.int32)

            rotary_pos_emb.append(rotary_pos_emb_thw)
            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = jnp.concatenate(rotary_pos_emb, axis=0)

        cu_seqlens = jnp.concatenate(cu_seqlens, axis=0)
        cu_seqlens = jnp.cumsum(cu_seqlens, axis=0, dtype=jnp.int32)
        cu_seqlens = jnp.pad(cu_seqlens, ((1, 0), ),
                             mode='constant',
                             constant_values=0)

        # Reshape for spatial merge
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        
        # Reshape back to sequence format
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = jnp.expand_dims(hidden_states, axis=1)

        # Process through transformer blocks without windowing
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, rotary_pos_emb=rotary_pos_emb)

        # adapter
        hidden_states = self.merger(hidden_states)
        
        return hidden_states


class Qwen2VLForConditionalGeneration(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        config: Qwen2VLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen2VisionTransformer(
            vllm_config=vllm_config,
            rngs=self.rng,
            mesh=mesh,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        self.language_model = Qwen2ForCausalLM(vllm_config, rng_key, mesh)

    @classmethod
    def get_mrope_input_positions(
        cls,
        input_tokens: list[int],
        hf_config,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths=None,
        use_audio_in_video: bool = False,
    ):
        # Since vLLM doesn't have qwen2_vl.py yet, we'll use qwen2_5_vl's implementation
        # as they share the same M-ROPE mechanism
        try:
            from vllm.model_executor.models.qwen2_5_vl import \
                Qwen2_5_VLForConditionalGeneration as vllm_model_cls
            return vllm_model_cls.get_mrope_input_positions(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )
        except (ImportError, AttributeError) as e:
            # If we can't import or the method doesn't exist, log and return None
            logger.warning(f"Could not use vLLM's M-ROPE implementation: {e}")
            # Return None to disable M-ROPE support
            return None, None

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
            **kwargs: object) -> Optional[Qwen2VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")

            if not isinstance(pixel_values, jax.Array):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2VLImagePixelInputs(type="pixel_values",
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
        #     return Qwen2VLImageEmbeddingInputs(
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

    @partial(
        jax.jit,
        static_argnames=("image_grid_thw", ),
    )
    def get_single_image_embedding(self, image_pixel_values, image_grid_thw):
        return self.visual(image_pixel_values, (image_grid_thw, ))

    def _process_image_input(
            self, image_input: Qwen2VLImageInputs) -> tuple[jax.Array, ...]:

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

    def get_multimodal_embeddings(self, image_grid_thw: tuple[tuple[int, int,
                                                                    int], ...],
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

    def get_input_embeddings(
            self, input_ids: jax.Array,
            multimodal_embeddings: Optional[MultiModalEmbeddings]
    ) -> jax.Array:

        inputs_embeds = self.language_model.model.embed(input_ids)


        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
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

        # First, let's add some fallback mappings for keys without .weight suffix
        # These will be tried if the exact key match fails
        mappings = {
            # Fallback mappings for norm layers without .weight suffix
            # These map to .scale which is what LayerNorm uses in Flax
            "visual.blocks.*.norm1": "visual.blocks.*.norm1.scale",
            "visual.blocks.*.norm2": "visual.blocks.*.norm2.scale",
            "visual.merger.ln_q": "visual.merger.ln_q.scale",
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
            # Qwen2VL uses fc1/fc2 naming for vision MLP layers
            "visual.blocks.*.mlp.fc1.bias": "visual.blocks.*.mlp.fc1.bias",
            "visual.blocks.*.mlp.fc1.weight": "visual.blocks.*.mlp.fc1.kernel",
            "visual.blocks.*.mlp.fc1": "visual.blocks.*.mlp.fc1.kernel",  # Without .weight suffix
            "visual.blocks.*.mlp.fc2.bias": "visual.blocks.*.mlp.fc2.bias",
            "visual.blocks.*.mlp.fc2.weight": "visual.blocks.*.mlp.fc2.kernel",
            "visual.blocks.*.mlp.fc2": "visual.blocks.*.mlp.fc2.kernel",  # Without .weight suffix
            "visual.blocks.*.norm1.weight": "visual.blocks.*.norm1.scale",
            "visual.blocks.*.norm1.bias": "visual.blocks.*.norm1.bias",
            "visual.blocks.*.norm2.weight": "visual.blocks.*.norm2.scale",
            "visual.blocks.*.norm2.bias": "visual.blocks.*.norm2.bias",
            "visual.merger.ln_q.weight": "visual.merger.ln_q.scale",
            "visual.merger.ln_q.bias": "visual.merger.ln_q.bias",
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

        metadata_map = get_default_maps(self.vllm_config, self.mesh, mappings)
        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        metadata_map=metadata_map,
                        mesh=self.mesh)
        
        # Check if all parameters have been loaded using nnx's built-in functionality
        from jax._src.core import ShapeDtypeStruct
        
        # Use nnx.iter_graph to safely iterate through all parameters
        unloaded_params = []
        for path, node in nnx.iter_graph(self):
            if isinstance(node, nnx.Param) and isinstance(node.value, ShapeDtypeStruct):
                # Convert the path tuple to a string
                path_str = ".".join(str(p) for p in path)
                unloaded_params.append(path_str)
        
        if unloaded_params:
            logger.error(f"Found {len(unloaded_params)} unloaded parameters:")
            
            # Group by prefix to identify patterns
            prefixes = {}
            for path in unloaded_params:
                # Get the first two components of the path
                parts = path.split('.')
                prefix = '.'.join(parts[:2]) if len(parts) > 1 else parts[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
            
            logger.error("Unloaded parameters by prefix:")
            for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1])[:10]:
                logger.error(f"  {prefix}: {count} parameters")
            
            # Show some specific examples
            logger.error("\nExamples of unloaded parameters:")
            for i, path in enumerate(sorted(unloaded_params)[:10]):
                logger.error(f"  {i+1}. {path}")
            
            raise ValueError(f"Not all parameters were loaded. Found {len(unloaded_params)} unloaded parameters.")