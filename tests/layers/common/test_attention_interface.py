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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
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
# Number of attention heads (Key/Value) - for Grouped-Query Attention
NUM_KV_HEADS = 4
# Total number of blocks in the KV cache
NUM_BLOCKS = 32
# Number of tokens per block
BLOCK_SIZE = 16
# Maximum number of blocks a single sequence can occupy
MAX_BLOCKS_PER_SEQ = 8


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    # Create a mesh with available devices, useful for running on CPU/GPU/TPU
    # For this test, it will likely be a single CPU device.
    devices = np.array(jax.local_devices()[:1])
    if not devices.any():
        # Add a mock device if no devices are present (e.g., in a CI environment)
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


# ---- Test for `attention` ----


def _test_attention(monkeypatch, mesh, head_dim, use_sinks=False):
    """
    Tests the main `attention` function.

    Verifies that:
    1. It calls the `sharded_ragged_paged_attention` kernel with correct metadata.
    2. The final outputs (kv_cache and attention output) have the correct shapes.
    """
    # 1. Arrange

    # Create input tensors
    q_dtype = jnp.float32
    kv_dtype = jnp.float32
    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, head_dim), dtype=q_dtype)
    k = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, head_dim), dtype=kv_dtype)
    v = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, head_dim), dtype=kv_dtype)
    sinks = jnp.ones((NUM_HEADS, ), dtype=jnp.float32) if use_sinks else None

    kv_cache_shape = get_kv_cache_shape_with_mesh(
        mesh,
        NUM_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        head_dim,
        kv_dtype,
    )
    kv_cache = jnp.zeros(kv_cache_shape, dtype=kv_dtype)

    # Mock ragged_paged_attention to return a tensor of the correct shape
    mock_paged_attn_kernel = MagicMock(return_value=(jnp.ones(
        (TOTAL_TOKENS, NUM_HEADS, head_dim)), kv_cache), )

    if head_dim == 64:
        monkeypatch.setattr(
            "tpu_inference.layers.common.attention_interface.ragged_paged_attention_hd64",
            mock_paged_attn_kernel,
        )
    else:
        monkeypatch.setattr(
            "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
            mock_paged_attn_kernel,
        )

    # Create AttentionMetadata
    attention_metadata = AttentionMetadata(
        input_positions=jnp.arange(TOTAL_TOKENS, dtype=jnp.int32),
        block_tables=jnp.zeros((MAX_NUM_SEQS * MAX_BLOCKS_PER_SEQ, ),
                               dtype=jnp.int32),
        seq_lens=jnp.array([5, 5, 0, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, NUM_SEQS], dtype=jnp.int32),
    )

    # 2. Act
    final_kv_cache, output = attention(
        kv_cache=kv_cache,
        q=q,
        k=k,
        v=v,
        attention_metadata=attention_metadata,
        mesh=mesh,
        head_dim_original=head_dim,
        sinks=sinks,
    )

    # 3. Assert
    # Check that both mocked kernels were called
    mock_paged_attn_kernel.assert_called_once()

    # Check output shapes
    assert final_kv_cache.shape == kv_cache.shape
    assert output.shape == q.shape

    # Check that the output is the one from our mock
    assert jnp.all(output == 1.0)


def test_attention(monkeypatch, mesh):
    _test_attention(monkeypatch, mesh, 128)


def test_attention_hd64(monkeypatch, mesh):
    _test_attention(monkeypatch, mesh, 64)


def test_attention_sink(monkeypatch, mesh):
    _test_attention(monkeypatch, mesh, 64, True)


def test_attention_sink_no_64_raises_error(monkeypatch, mesh):
    with pytest.raises(
            NotImplementedError,
            match="Attention sink support is only available when head_dim==64"
    ):
        _test_attention(monkeypatch, mesh, 128, True)
