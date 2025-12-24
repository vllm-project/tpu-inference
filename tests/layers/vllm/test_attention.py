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
import torch
import torchax
from jax.sharding import Mesh
from torchax.interop import torch_view
from vllm.attention.backends.abstract import AttentionType

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.vllm.attention import (PallasAttentionBackend,
                                                 PallasAttentionBackendImpl)
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
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
# Dimension of each attention head
HEAD_DIM = 128
# Total number of blocks in the KV cache
NUM_BLOCKS = 32
# Number of tokens per block
BLOCK_SIZE = 16
# Maximum number of blocks a single sequence can occupy
MAX_BLOCKS_PER_SEQ = 8


def create_inputs(
    mesh: Mesh,
    q_dtype: jnp.dtype = jnp.bfloat16,
    kv_dtype: jnp.dtype = jnp.bfloat16,
    total_tokens: int = TOTAL_TOKENS,
    num_seqs: int = NUM_SEQS,
    max_num_seqs: int = MAX_NUM_SEQS,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    num_blocks: int = NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    max_blocks_per_seq: int = MAX_BLOCKS_PER_SEQ,
):
    key = jax.random.key(0)
    q = jax.random.uniform(key, (total_tokens, num_heads * head_dim),
                           dtype=q_dtype)
    k = jax.random.uniform(key, (total_tokens, num_kv_heads * head_dim),
                           dtype=q_dtype)
    v = jax.random.uniform(key, (total_tokens, num_kv_heads * head_dim),
                           dtype=q_dtype)
    q = torch_view(q)
    k = torch_view(k)
    v = torch_view(v)

    kv_cache_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks, block_size,
                                                  num_kv_heads, head_dim,
                                                  kv_dtype)
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

    return q, k, v, kv_cache, metadata


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    # Create a mesh with available devices, useful for running on CPU/GPU/TPU
    # For this test, it will likely be a single CPU device.
    devices = np.array(jax.local_devices())[0:1]
    if not devices.any():
        # Add a mock device if no devices are present (e.g., in a CI environment)
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


class TestPallasAttentionBackend:

    def test_get_name(self):
        assert PallasAttentionBackend.get_name() == "PALLAS"

    def test_get_impl_cls(self):
        assert PallasAttentionBackend.get_impl_cls(
        ) == PallasAttentionBackendImpl


class TestPallasAttentionBackendImpl:

    def test_init_valid_params(self):
        impl = PallasAttentionBackendImpl(
            num_heads=32,
            head_size=128,
            scale=0.088,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        assert impl.num_heads == 32
        assert impl.head_size == 128
        assert impl.scale == 0.088
        assert impl.num_kv_heads == 8
        assert impl.num_queries_per_kv == 4
        assert impl.sliding_window is None

    def test_init_with_alibi_slopes_raises_error(self):
        with pytest.raises(NotImplementedError,
                           match="Alibi slopes is not supported"):
            PallasAttentionBackendImpl(
                num_heads=32,
                head_size=128,
                scale=0.088,
                num_kv_heads=8,
                alibi_slopes=[1.0, 2.0],
                sliding_window=None,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )

    def test_init_with_encoder_attention_raises_error(self):
        with pytest.raises(NotImplementedError,
                           match="Encoder self-attention"):
            PallasAttentionBackendImpl(
                num_heads=32,
                head_size=128,
                scale=0.088,
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
                attn_type=AttentionType.ENCODER,
            )

    def test_forward(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(mesh)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

    def test_forward_with_fp8_kv_cache(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"
        layer._q_scale_float = None
        layer._k_scale_float = 1
        layer._v_scale_float = 1

        query, key, value, kv_cache, metadata = create_inputs(
            mesh, kv_dtype=jnp.float8_e4m3fn)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

    def test_forward_with_w8a8(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"
        layer._q_scale_float = 1
        layer._k_scale_float = 1
        layer._v_scale_float = 1

        query, key, value, kv_cache, metadata = create_inputs(
            mesh, kv_dtype=jnp.float8_e4m3fn)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

    def test_forward_with_vllm_kv_cache_raises_error(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(mesh)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh), pytest.raises(RuntimeError,
                                          match="should be empty but has"):
            impl.forward(layer, query, key, value, torch.tensor([1]), metadata)

    def test_forward_with_output_scale_raises_error(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(mesh)
        output_scale = torch.tensor([1.0])

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh), pytest.raises(NotImplementedError,
                                          match="fused output quantization"):
            impl.forward(layer,
                         query,
                         key,
                         value,
                         torch.tensor([]),
                         metadata,
                         output_scale=output_scale)

    def test_forward_with_attention_sink(self, mesh):
        head_dim = 64
        sinks = torch.rand([NUM_HEADS], dtype=torch.float32)

        impl = PallasAttentionBackendImpl(num_heads=NUM_HEADS,
                                          head_size=head_dim,
                                          scale=0.088,
                                          num_kv_heads=NUM_KV_HEADS,
                                          alibi_slopes=None,
                                          sliding_window=None,
                                          kv_cache_dtype="auto",
                                          attn_type=AttentionType.DECODER,
                                          sinks=sinks)
        impl.process_weights_after_loading(torch.bfloat16)

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(
            mesh, head_dim=head_dim)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            assert impl.sinks is not None
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

    def test_forward_with_attention_sink_head_dim_128_raises_error(self, mesh):
        head_dim = 128
        sinks = torch.rand([NUM_HEADS], dtype=torch.float32)

        impl = PallasAttentionBackendImpl(num_heads=NUM_HEADS,
                                          head_size=head_dim,
                                          scale=0.088,
                                          num_kv_heads=NUM_KV_HEADS,
                                          alibi_slopes=None,
                                          sliding_window=None,
                                          kv_cache_dtype="auto",
                                          attn_type=AttentionType.DECODER,
                                          sinks=sinks)
        impl.process_weights_after_loading(torch.bfloat16)

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(
            mesh, head_dim=head_dim)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}
        ), pytest.raises(
                NotImplementedError,
                match=
                "Attention sink support is only available when head_dim==64"):
            assert impl.sinks is not None
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)
