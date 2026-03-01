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

from tpu_inference.layers.common.attention_interface import (
    attention, mla_attention, sharded_ragged_paged_attention)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
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


# ---- Tests for `sharded_ragged_paged_attention` ----


@pytest.fixture
def gqa_mesh():
    """Provides a mock JAX mesh for GQA testing with tensor parallelism."""
    # This mesh has 8 devices for tensor parallelism over heads.
    # We create a 1x8 mesh for ('attn_data', 'attn_head')
    try:
        devices = np.array(jax.local_devices()[:1] * 4)
        if devices.size == 0:
            raise IndexError
    except IndexError:
        # Fails in environments with no devices
        devices = np.array([jax.devices("cpu")[0]] * 4)

    return Mesh(
        devices.reshape((1, 4)),
        (
            ShardingAxisName.ATTN_DATA,
            ShardingAxisName.ATTN_HEAD,
        ),
    )


def test_sharded_ragged_paged_attention_gqa_replication(monkeypatch, gqa_mesh):
    """
    Tests that K and V heads are correctly replicated for GQA in
    `sharded_ragged_paged_attention`.
    """
    # 1. Arrange
    tp_size = gqa_mesh.shape[ShardingAxisName.ATTN_HEAD]
    assert tp_size == 4
    num_kv_heads = 2  # num_kv_heads < tp_size and tp_size % num_kv_heads == 0
    head_dim = 128
    factor = tp_size // num_kv_heads

    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, head_dim))
    # Create K and V with values that can be checked after repeating
    k_content = jnp.arange(TOTAL_TOKENS * num_kv_heads * head_dim).reshape(
        (TOTAL_TOKENS, num_kv_heads, head_dim))
    v_content = -k_content
    k = k_content
    v = v_content

    # The actual shape of kv_cache does not matter as much since we mock the call
    kv_cache = jnp.zeros((num_kv_heads, NUM_BLOCKS, BLOCK_SIZE, head_dim))

    # Other metadata, can be zero/empty for this test's purpose
    kv_lens = jnp.zeros((MAX_NUM_SEQS, ), dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_NUM_SEQS, MAX_BLOCKS_PER_SEQ),
                             dtype=jnp.int32)
    cu_q_lens = jnp.zeros((MAX_NUM_SEQS + 1, ), dtype=jnp.int32)
    distribution = jnp.zeros((3, ), dtype=jnp.int32)
    sm_scale = 1.0

    # Mock jax.shard_map to capture the arguments passed to its mapped function
    mock_shard_map_callable = MagicMock(return_value=(jnp.ones_like(q),
                                                      kv_cache))
    mock_shard_map = MagicMock(return_value=mock_shard_map_callable)
    monkeypatch.setattr("jax.shard_map", mock_shard_map)

    # 2. Act
    sharded_ragged_paged_attention(
        mesh=gqa_mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=sm_scale,
    )

    # 3. Assert
    # Check that shard_map was called
    mock_shard_map.assert_called_once()
    # Check that the function returned by shard_map was called with arguments
    mock_shard_map_callable.assert_called_once()

    # Get the arguments passed to the jitted function inside shard_map
    call_args = mock_shard_map_callable.call_args[0]
    replicated_k = call_args[1]
    replicated_v = call_args[2]

    # Check shapes
    assert replicated_k.shape[1] == tp_size
    assert replicated_v.shape[1] == tp_size
    assert replicated_k.shape[1] == k.shape[1] * factor
    assert replicated_v.shape[1] == v.shape[1] * factor

    # Check content of replicated K
    expected_k = jnp.repeat(k_content, factor, axis=1)
    assert jnp.array_equal(replicated_k, expected_k)

    # Check content of replicated V
    expected_v = jnp.repeat(v_content, factor, axis=1)
    assert jnp.array_equal(replicated_v, expected_v)


def test_sharded_ragged_paged_attention_gqa_incompatible_raises_error(
    gqa_mesh, ):
    """
    Tests that a ValueError is raised for GQA when tp_size is not divisible
    by num_kv_heads.
    """
    # 1. Arrange
    tp_size = gqa_mesh.shape[ShardingAxisName.ATTN_HEAD]
    assert tp_size == 4
    num_kv_heads = 3  # Incompatible with tp_size=4
    head_dim = 128

    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, head_dim))
    k = jnp.ones((TOTAL_TOKENS, num_kv_heads, head_dim))
    v = jnp.ones((TOTAL_TOKENS, num_kv_heads, head_dim))
    kv_cache = jnp.zeros((num_kv_heads, NUM_BLOCKS, BLOCK_SIZE, head_dim))
    # Other metadata
    kv_lens = jnp.zeros((MAX_NUM_SEQS, ), dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_NUM_SEQS, MAX_BLOCKS_PER_SEQ),
                             dtype=jnp.int32)
    cu_q_lens = jnp.zeros((MAX_NUM_SEQS + 1, ), dtype=jnp.int32)
    distribution = jnp.zeros((3, ), dtype=jnp.int32)
    sm_scale = 1.0

    # 2. Act & Assert
    with pytest.raises(
            ValueError,
            match=(f"For GQA/MQA, tp_size {tp_size} must be divisible by "
                   f"num_kv_heads {num_kv_heads}"),
    ):
        sharded_ragged_paged_attention(
            mesh=gqa_mesh,
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sink=None,
            sm_scale=sm_scale,
        )


def test_mla_attention(monkeypatch, mesh):
    """
    Tests the `mla_attention` function.

    Verifies that:
    1. It correctly calculates block sizes using `get_tuned_block_sizes`
    2. It calls `mla_ragged_paged_attention` with the correct arguments
    3. It returns the expected output and updated KV cache
    """
    qk_nope_dim = 32
    qk_rope_dim = 16
    q_lora_rank = 64
    kv_lora_rank = 64

    q_TNA = jnp.ones((TOTAL_TOKENS, NUM_HEADS, q_lora_rank))
    q_rope_TNH = jnp.ones((TOTAL_TOKENS, NUM_HEADS, qk_rope_dim))
    k_SA = jnp.ones((TOTAL_TOKENS, kv_lora_rank))
    k_rope_SH = jnp.ones((TOTAL_TOKENS, qk_rope_dim))

    # Arbitrary cache shape just for testing
    kv_cache_shape = (1, NUM_BLOCKS, BLOCK_SIZE, kv_lora_rank)
    kv_cache = jnp.zeros(kv_cache_shape)

    metadata = AttentionMetadata(
        input_positions=jnp.arange(TOTAL_TOKENS, dtype=jnp.int32),
        block_tables=jnp.zeros((MAX_NUM_SEQS * MAX_BLOCKS_PER_SEQ, ),
                               dtype=jnp.int32),
        seq_lens=jnp.array([5, 5, 0, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, NUM_SEQS], dtype=jnp.int32),
    )

    mock_tuned_block_sizes = MagicMock(return_value=(8, 8))
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.get_tuned_block_sizes",
        mock_tuned_block_sizes)

    expected_output = jnp.full(q_TNA.shape, 0.5)
    expected_new_cache = jnp.full(kv_cache_shape, 0.1)

    mock_mla_kernel = MagicMock(return_value=(expected_output,
                                              expected_new_cache))
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.mla_ragged_paged_attention",
        mock_mla_kernel)

    final_kv_cache, output = mla_attention(
        q_TNA=q_TNA,
        q_rope_TNH=q_rope_TNH,
        k_SA=k_SA,
        k_rope_SH=k_rope_SH,
        kv_cache=kv_cache,
        md=metadata,
        mesh=mesh,
        num_attention_heads=NUM_HEADS,
        qk_nope_head_dim=qk_nope_dim,
        sm_scale=0.1,
    )

    # Verify mocked functions were called
    mock_tuned_block_sizes.assert_called_once()
    mock_mla_kernel.assert_called_once()

    # Verify output correctness
    assert jnp.array_equal(output, expected_output)
    assert jnp.array_equal(final_kv_cache, expected_new_cache)

    _, kernel_kwargs = mock_mla_kernel.call_args
    assert kernel_kwargs["num_kv_pages_per_block"] == 4
    assert kernel_kwargs["num_queries_per_block"] == 4
    assert kernel_kwargs["sm_scale"] == 0.1
