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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from jax.sharding import Mesh
from torchax.interop import torch_view
from vllm.v1.attention.backend import AttentionType

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.vllm.backends.flash_attn_mla import (
    PallasMLAttentionBackend, PallasMLAttentionBackendImpl)
from tpu_inference.runner.kv_cache import get_kv_cache_shape_with_mesh

# ---- Test Configuration & Constants ----

# Total number of tokens across all sequences in the batch
TOTAL_TOKENS = 10
# Number of sequences in the batch
NUM_SEQS = 2
# Padded maximum number of sequences
MAX_NUM_SEQS = 4
# Number of attention heads (Query)
NUM_HEADS = 8
# MLA Specific Configurations
Q_LORA_RANK = 64
KV_LORA_RANK = 32
QK_NOPE_HEAD_DIM = 64
QK_ROPE_HEAD_DIM = 32
V_HEAD_DIM = 64

# Total number of blocks in the KV cache
NUM_BLOCKS = 32
# Number of tokens per block
BLOCK_SIZE = 16
# Maximum number of blocks a single sequence can occupy
MAX_BLOCKS_PER_SEQ = 8


def create_mla_inputs(
    mesh: Mesh,
    q_dtype: jnp.dtype = jnp.bfloat16,
    kv_dtype: jnp.dtype = jnp.bfloat16,
    total_tokens: int = TOTAL_TOKENS,
    num_seqs: int = NUM_SEQS,
    max_num_seqs: int = MAX_NUM_SEQS,
    num_heads: int = NUM_HEADS,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    kv_lora_rank: int = KV_LORA_RANK,
    num_blocks: int = NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    max_blocks_per_seq: int = MAX_BLOCKS_PER_SEQ,
):
    key = jax.random.key(0)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    q = jax.random.uniform(key, (total_tokens, num_heads, qk_head_dim),
                           dtype=q_dtype)
    kv_c_normed = jax.random.uniform(key, (total_tokens, 1, kv_lora_rank),
                                     dtype=q_dtype)
    k_pe = jax.random.uniform(key, (total_tokens, 1, qk_rope_head_dim),
                              dtype=q_dtype)

    q = torch_view(q)
    kv_c_normed = torch_view(kv_c_normed)
    k_pe = torch_view(k_pe)

    # For MLA, KV cache relies heavily on specific dimensions.
    # We use a mocked cache mapping using 1 KV Head as parameterized
    head_size = kv_lora_rank + qk_rope_head_dim
    kv_cache_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks, block_size,
                                                  1, head_size, kv_dtype)
    kv_cache = jax.random.normal(key, kv_cache_shape, dtype=kv_dtype)

    positions = jnp.ones((total_tokens, ), dtype=jnp.int32)
    block_tables = jnp.zeros((max_num_seqs * max_blocks_per_seq),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.array([5, 5, 0, 0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )

    return q, kv_c_normed, k_pe, kv_cache, metadata


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    devices = np.array(jax.local_devices())[0:1]
    if not devices.any():
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


class TestPallasMLAttentionBackend:

    def test_get_name(self):
        assert PallasMLAttentionBackend.get_name() == "FLASHMLA"

    def test_get_impl_cls(self):
        assert PallasMLAttentionBackend.get_impl_cls(
        ) == PallasMLAttentionBackendImpl


class TestPallasMLAttentionBackendImpl:

    def test_init_valid_params(self):
        impl = PallasMLAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            scale=0.088,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
        )

        assert impl.num_heads == NUM_HEADS
        assert impl.head_size == KV_LORA_RANK + QK_ROPE_HEAD_DIM
        assert impl.scale == 0.088
        assert impl.num_kv_heads == 1
        assert impl.q_lora_rank == Q_LORA_RANK
        assert impl.kv_lora_rank == KV_LORA_RANK
        assert impl.qk_nope_head_dim == QK_NOPE_HEAD_DIM
        assert impl.qk_rope_head_dim == QK_ROPE_HEAD_DIM
        assert impl.qk_head_dim == QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
        assert impl.v_head_dim == V_HEAD_DIM

    def test_forward_mha_is_pass(self):
        impl = PallasMLAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            scale=0.088,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
        )
        # Verify the interface method correctly falls through without raising unhandled abstractions
        result = impl.forward_mha(None, None, None, None, None, None, None)
        assert result is None

    def test_forward_mqa_is_pass(self):
        impl = PallasMLAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            scale=0.088,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
        )
        # Verify the interface method correctly falls through without raising unhandled abstractions
        result = impl.forward_mqa(None, None, None, None)
        assert result is None

    def test_forward(self, mesh):
        impl = PallasMLAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            scale=0.088,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
        )

        layer = MagicMock()
        layer.W_UK_T = torch.rand((NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK))
        layer.W_UV = torch.rand((NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM))
        layer.kv_cache_quantized_dtype = None

        q, kv_c_normed, k_pe, kv_cache, metadata = create_mla_inputs(mesh)

        with torchax.default_env():
            outputs, new_kv_cache = impl.forward(q, kv_c_normed, k_pe,
                                                 kv_cache, metadata, mesh,
                                                 layer)

        assert outputs is not None
        assert new_kv_cache is not None

    def test_forward_with_fp8_kv_cache(self, mesh):
        impl = PallasMLAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            scale=0.088,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
        )

        layer = MagicMock()
        layer.W_UK_T = torch.rand((NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK))
        layer.W_UV = torch.rand((NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM))
        layer.kv_cache_quantized_dtype = jnp.float8_e4m3fn
        layer._q_scale_float = None
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        q, kv_c_normed, k_pe, kv_cache, metadata = create_mla_inputs(
            mesh, kv_dtype=jnp.float8_e4m3fn)

        with torchax.default_env():
            outputs, new_kv_cache = impl.forward(q, kv_c_normed, k_pe,
                                                 kv_cache, metadata, mesh,
                                                 layer)

        assert outputs is not None
        assert new_kv_cache is not None

    def test_forward_with_w8a8(self, mesh):
        impl = PallasMLAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            scale=0.088,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
        )

        layer = MagicMock()
        layer.W_UK_T = torch.rand((NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK))
        layer.W_UV = torch.rand((NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM))
        layer.kv_cache_quantized_dtype = jnp.float8_e4m3fn
        layer._q_scale_float = 1.0
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        q, kv_c_normed, k_pe, kv_cache, metadata = create_mla_inputs(
            mesh, kv_dtype=jnp.float8_e4m3fn)

        with torchax.default_env():
            outputs, new_kv_cache = impl.forward(q, kv_c_normed, k_pe,
                                                 kv_cache, metadata, mesh,
                                                 layer)

        assert outputs is not None
        assert new_kv_cache is not None
