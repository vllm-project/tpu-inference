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

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4.indexer.streamindex_topk import \
    streamindex_topk
from tpu_inference.kernels.experimental.deepseek_v32.sparse_mla import (
    gather_paged_kv, sparse_mla_decode, sparse_mla_reference)


def test_gather_paged_kv_fragmented_and_padded():
    page_size = 128
    packing = 32
    width = 640
    cache = np.zeros((4, page_size // packing, packing, width), np.float32)
    for page in range(4):
        for offset in range(page_size):
            cache[page, offset // packing, offset % packing,
                  0] = (page * page_size + offset)
    block_tables = jnp.asarray([[2, 0, 3, 1]], dtype=jnp.int32)
    indices = jnp.asarray([[0, 127, 128, 255, 256, 383, -1, -1]],
                          dtype=jnp.int32)
    selected, selected_lens = gather_paged_kv(jnp.asarray(cache), block_tables,
                                              indices)
    np.testing.assert_array_equal(
        np.asarray(selected[0, :, 0]),
        np.asarray([256, 383, 0, 127, 384, 511, 256, 256], np.float32),
    )
    np.testing.assert_array_equal(np.asarray(selected_lens), [6])


def test_sparse_mla_v4_matches_reference():
    rng = np.random.default_rng(11)
    batch = 1
    num_heads = 32
    latent_dim = 512
    rope_dim = 64
    width = 640
    selected_len = 2048
    q_nope = jnp.asarray(
        rng.normal(size=(batch, num_heads, latent_dim)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    q_pe = jnp.asarray(
        rng.normal(size=(batch, num_heads, rope_dim)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    selected = jnp.asarray(
        rng.normal(size=(batch, selected_len, width)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    selected_lens = jnp.asarray([2035], dtype=jnp.int32)
    scale = (latent_dim + rope_dim)**-0.5
    expected = sparse_mla_reference(q_nope,
                                    q_pe,
                                    selected,
                                    selected_lens,
                                    sm_scale=scale)
    actual = sparse_mla_decode(
        q_nope,
        q_pe,
        selected,
        selected_lens,
        sm_scale=scale,
        block_size=512,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        atol=4e-3,
        rtol=4e-3,
    )


def test_sparse_mla_fully_masked_row_is_zero():
    q_nope = jnp.ones((1, 4, 128), dtype=jnp.bfloat16)
    q_pe = jnp.ones((1, 4, 64), dtype=jnp.bfloat16)
    selected = jnp.ones((1, 128, 256), dtype=jnp.bfloat16)
    actual = sparse_mla_decode(
        q_nope,
        q_pe,
        selected,
        jnp.asarray([0], dtype=jnp.int32),
        sm_scale=192**-0.5,
        block_size=128,
    )
    np.testing.assert_array_equal(np.asarray(actual), 0)


@pytest.mark.parametrize("seq_len", [2047, 2048, 2049])
def test_streamindex_to_sparse_mla_boundary(seq_len):
    """Selected positions flow directly into paged sparse MLA at the boundary."""
    rng = np.random.default_rng(23)
    page_size = 128
    packing = 32
    max_seq_len = 2049
    num_pages = (max_seq_len + page_size - 1) // page_size
    topk = 2048
    width = 640

    # A zero scorer makes every valid key tie, so exact semantics select the
    # lowest absolute positions and append -1 only when the sequence is short.
    index_cache = np.zeros((num_pages, page_size // packing, packing, 256),
                           dtype=np.uint8)
    index_cache[..., 128:132] = np.asarray([1.0], np.float32).view(np.uint8)
    block_table = np.roll(np.arange(num_pages, dtype=np.int32), 5)[None, :]
    topk_indices = streamindex_topk(
        q=jnp.zeros((1, 4, 128), dtype=jnp.float8_e4m3fn),
        indexer_weights=jnp.ones((1, 4), dtype=jnp.float32),
        cache_kv=jnp.asarray(index_cache),
        seq_lens=jnp.asarray([seq_len], dtype=jnp.int32),
        page_indices=jnp.asarray(block_table.reshape(-1)),
        cu_q_lens=jnp.asarray([0, 1], dtype=jnp.int32),
        distribution=jnp.asarray([1, 1, 1], dtype=jnp.int32),
        k=topk,
        compression_ratio=1,
        scale_storage="float32",
        exact_topk=True,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        decode_req_batch_size=1,
    )
    valid = min(seq_len, topk)
    expected_indices = np.full((1, topk), -1, dtype=np.int32)
    expected_indices[0, :valid] = np.arange(valid, dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(topk_indices), expected_indices)

    logical_cache = rng.normal(size=(num_pages * page_size,
                                     width)).astype(np.float32)
    physical_cache = np.zeros(
        (num_pages, page_size // packing, packing, width), np.float32)
    for logical_page, physical_page in enumerate(block_table[0]):
        page = logical_cache[logical_page * page_size:(logical_page + 1) *
                             page_size]
        physical_cache[physical_page] = page.reshape(page_size // packing,
                                                     packing, width)
    selected, selected_lens = gather_paged_kv(
        jnp.asarray(physical_cache, dtype=jnp.bfloat16),
        jnp.asarray(block_table),
        topk_indices,
    )
    np.testing.assert_array_equal(np.asarray(selected_lens), [valid])

    q_nope = jnp.asarray(rng.normal(size=(1, 32, 512)).astype(np.float32),
                         dtype=jnp.bfloat16)
    q_pe = jnp.asarray(rng.normal(size=(1, 32, 64)).astype(np.float32),
                       dtype=jnp.bfloat16)
    sm_scale = 576**-0.5
    actual = sparse_mla_decode(
        q_nope,
        q_pe,
        selected,
        selected_lens,
        sm_scale=sm_scale,
        block_size=512,
    )
    expected_selected = np.zeros((1, topk, width), np.float32)
    expected_selected[0, :valid] = logical_cache[:valid]
    expected = sparse_mla_reference(
        q_nope,
        q_pe,
        jnp.asarray(expected_selected, dtype=jnp.bfloat16),
        selected_lens,
        sm_scale=sm_scale,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        atol=4e-3,
        rtol=4e-3,
    )
