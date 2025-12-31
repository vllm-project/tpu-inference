# Qwen3 VL MoE - Vision components are identical to Qwen3 VL dense
# Import all vision-related classes and helper functions from qwen3_vl

from tpu_inference.models.jax.qwen3_vl import (
    # Constants
    DEFAULT_BLOCK_K_MAJOR,
    # Helper functions
    _infer_pos_embed_grid_hw,
    generate_segment_ids_from_grid_thw,
    pad_segment_ids_for_attention,
    apply_rotary_pos_emb_vision,
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
)

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig