from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tpu_commons.models.jax.attention_interface import (attention,
                                                        update_kv_cache)
from tpu_commons.models.jax.attention_metadata import AttentionMetadata

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
# Dimension of each attention head
HEAD_DIM = 64
# Padded head dimension
PADDED_HEAD_DIM = 64
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
    devices = np.array(jax.local_devices())
    if not devices.any():
        # Add a mock device if no devices are present (e.g., in a CI environment)
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1)), ("data", "model"))


# ---- Test for `update_kv_cache` ----


def test_update_kv_cache(monkeypatch, mesh):
    """
    Tests the `update_kv_cache` function.

    Verifies that:
    1. It correctly reshapes and interleaves the k and v tensors.
    2. It calls the underlying `kv_cache_update` kernel with the correct arguments.
    3. It returns a kv_cache with the expected shape.
    """
    # 1. Arrange
    k = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, PADDED_HEAD_DIM),
                 dtype=jnp.float32)
    v = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, PADDED_HEAD_DIM),
                 dtype=jnp.float32)
    kv_cache = jnp.zeros(
        (NUM_BLOCKS, BLOCK_SIZE, 2 * NUM_KV_HEADS, PADDED_HEAD_DIM),
        dtype=jnp.float32)
    slices = jnp.zeros((3, TOTAL_TOKENS), dtype=jnp.int32)
    num_slices = jnp.array([TOTAL_TOKENS], dtype=jnp.int32)

    # Mock the external kernel dependency
    mock_kernel = MagicMock(
        return_value=kv_cache.reshape(-1, 2 * NUM_KV_HEADS, PADDED_HEAD_DIM))
    monkeypatch.setattr(
        "tpu_commons.models.jax.attention_interface.kv_cache_update",
        mock_kernel)

    # 2. Act
    updated_cache = update_kv_cache(k, v, kv_cache, slices, num_slices, mesh)

    # 3. Assert
    # Check that the kernel was called
    mock_kernel.assert_called_once()

    # Check the shape of the output
    assert updated_cache.shape == kv_cache.shape

    # Check the shape and content of the interleaved `kv` tensor passed to the kernel
    call_args, _ = mock_kernel.call_args
    kv_passed_to_kernel = call_args[0]

    # Expected shape is (total_tokens, num_kv_heads * 2, head_dim)
    expected_kv_shape = (TOTAL_TOKENS, 2 * NUM_KV_HEADS, PADDED_HEAD_DIM)
    assert kv_passed_to_kernel.shape == expected_kv_shape

    # Verify the interleaving logic: jnp.concat([k, v], axis=-1).reshape(...)
    # This interleaves k and v vectors for each head index: [k0, v0, k1, v1, ...]
    expected_kv = jnp.concatenate([k, v], axis=-1).reshape(expected_kv_shape)
    assert jnp.array_equal(kv_passed_to_kernel, expected_kv)


# ---- Test for `attention` ----


def test_attention(monkeypatch, mesh):
    """
    Tests the main `attention` function.

    Verifies that:
    1. It correctly calls the `update_kv_cache` logic.
    2. It calls the `sharded_ragged_paged_attention` kernel with correct metadata.
    3. The final outputs (kv_cache and attention output) have the correct shapes.
    """
    # 1. Arrange

    # Mock the underlying custom kernels
    # Mock kv_cache_update to just return the cache, as its logic is tested above
    mock_kv_update_kernel = MagicMock(
        return_value=jnp.zeros((NUM_BLOCKS * BLOCK_SIZE, 2 * NUM_KV_HEADS,
                                PADDED_HEAD_DIM)))
    monkeypatch.setattr(
        "tpu_commons.models.jax.attention_interface.kv_cache_update",
        mock_kv_update_kernel,
    )

    # Mock ragged_paged_attention to return a tensor of the correct shape
    mock_paged_attn_kernel = MagicMock(
        return_value=jnp.ones((TOTAL_TOKENS, NUM_HEADS, PADDED_HEAD_DIM)))
    monkeypatch.setattr(
        "tpu_commons.models.jax.attention_interface.ragged_paged_attention",
        mock_paged_attn_kernel,
    )

    # Create input tensors
    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, PADDED_HEAD_DIM), dtype=jnp.float32)
    k = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, PADDED_HEAD_DIM),
                 dtype=jnp.float32)
    v = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, PADDED_HEAD_DIM),
                 dtype=jnp.float32)
    kv_cache = jnp.zeros(
        (NUM_BLOCKS, BLOCK_SIZE, 2 * NUM_KV_HEADS, PADDED_HEAD_DIM),
        dtype=jnp.float32)

    # Create AttentionMetadata
    attention_metadata = AttentionMetadata(
        input_positions=jnp.arange(TOTAL_TOKENS, dtype=jnp.int32),
        slot_mapping=jnp.zeros((3, TOTAL_TOKENS), dtype=jnp.int32),
        block_tables=jnp.zeros((MAX_NUM_SEQS, MAX_BLOCKS_PER_SEQ),
                               dtype=jnp.int32),
        seq_lens=jnp.array([5, 5, 0, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32),
        num_seqs=jnp.array([NUM_SEQS], dtype=jnp.int32),
        num_slices=jnp.array([TOTAL_TOKENS], dtype=jnp.int32),
    )

    # 2. Act
    final_kv_cache, output = attention(
        kv_cache=kv_cache,
        q=q,
        k=k,
        v=v,
        attention_metadata=attention_metadata,
        mesh=mesh,
        head_dim_original=HEAD_DIM,
    )

    # 3. Assert
    # Check that both mocked kernels were called
    mock_kv_update_kernel.assert_called_once()
    mock_paged_attn_kernel.assert_called_once()

    # Check output shapes
    assert final_kv_cache.shape == kv_cache.shape
    assert output.shape == q.shape

    # Check that the output is the one from our mock
    assert jnp.all(output == 1.0)
