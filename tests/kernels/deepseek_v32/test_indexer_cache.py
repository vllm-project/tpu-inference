# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.experimental.deepseek_v4.indexer.streamindex_topk import \
    streamindex_topk
from tpu_inference.kernels.experimental.deepseek_v32.indexer_cache import (
    insert_indexer_k_cache, pack_indexer_k_records, quantize_indexer_k)


def _reference_quantize(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amax = np.maximum(np.max(np.abs(k), axis=-1, keepdims=True), 1e-4)
    scales = np.exp2(np.ceil(np.log2(amax / 448.0))).astype(np.float32)
    quantized = np.asarray(jnp.asarray(k / scales).astype(jnp.float8_e4m3fn))
    return quantized, scales


def test_quantize_indexer_k_matches_v32_contract():
    values = np.linspace(-511.0, 509.0, 3 * 128,
                         dtype=np.float32).reshape(3, 128)
    values[0] = 0
    expected_q, expected_scale = _reference_quantize(values)
    actual_q, actual_scale = quantize_indexer_k(jnp.asarray(values))
    np.testing.assert_array_equal(np.asarray(actual_q), expected_q)
    np.testing.assert_array_equal(np.asarray(actual_scale), expected_scale)


def test_pack_indexer_k_records_has_exact_bytes_and_zero_padding():
    values = np.linspace(-3.0, 5.0, 256, dtype=np.float32).reshape(2, 128)
    expected_q, expected_scale = _reference_quantize(values)
    expected = np.zeros((2, 256), dtype=np.uint8)
    expected[:, :128] = expected_q.view(np.uint8)
    expected[:, 128:132] = expected_scale.view(np.uint8).reshape(2, 4)

    actual = pack_indexer_k_records(jnp.asarray(values), record_width=256)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_insert_indexer_k_cache_fragmented_slots_and_padding():
    page_size = 64
    packing = 32
    cache = jnp.full((4, page_size // packing, packing, 256),
                     0xA5,
                     dtype=jnp.uint8)
    values = np.stack([
        np.full(128, 1.0, np.float32),
        np.full(128, 2.0, np.float32),
        np.full(128, 3.0, np.float32),
        np.full(128, 4.0, np.float32),
    ])
    slots = jnp.asarray([2 * page_size + 7, page_size - 1, -1, 3 * page_size],
                        dtype=jnp.int32)
    actual = insert_indexer_k_cache(cache, jnp.asarray(values), slots)
    actual_flat = np.asarray(actual).reshape(-1, 256)
    expected_records = np.asarray(
        pack_indexer_k_records(jnp.asarray(values), record_width=256))

    np.testing.assert_array_equal(actual_flat[2 * page_size + 7],
                                  expected_records[0])
    np.testing.assert_array_equal(actual_flat[page_size - 1],
                                  expected_records[1])
    np.testing.assert_array_equal(actual_flat[3 * page_size],
                                  expected_records[3])
    assert not np.any(np.all(actual_flat == expected_records[2], axis=-1))
    untouched = np.delete(actual_flat,
                          [2 * page_size + 7, page_size - 1, 3 * page_size],
                          axis=0)
    np.testing.assert_array_equal(untouched, np.full_like(untouched, 0xA5))


def test_insert_indexer_k_cache_supports_unpadded_vllm_layout():
    cache = jnp.zeros((2, 16, 132), dtype=jnp.uint8)
    key = jnp.arange(128, dtype=jnp.float32)[None, :]
    actual = insert_indexer_k_cache(
        cache,
        key,
        jnp.asarray([19], dtype=jnp.int32),
    )
    expected = pack_indexer_k_records(key, record_width=132)
    np.testing.assert_array_equal(
        np.asarray(actual).reshape(-1, 132)[19], np.asarray(expected[0]))


def test_inserted_cache_composes_with_exact_streamindex_topk():
    page_size = 128
    packing = 32
    head_dim = 128
    keys = np.zeros((page_size, head_dim), dtype=np.float32)
    for token in range(page_size):
        keys[token, :token + 1] = 1.0
    cache = insert_indexer_k_cache(
        jnp.zeros((1, page_size // packing, packing, 256), dtype=jnp.uint8),
        jnp.asarray(keys),
        jnp.arange(page_size, dtype=jnp.int32),
    )
    topk = streamindex_topk(
        q=jnp.ones((1, 4, head_dim), dtype=jnp.float8_e4m3fn),
        indexer_weights=jnp.ones((1, 4), dtype=jnp.float32),
        cache_kv=cache,
        seq_lens=jnp.asarray([page_size], dtype=jnp.int32),
        page_indices=jnp.asarray([0], dtype=jnp.int32),
        cu_q_lens=jnp.asarray([0, 1], dtype=jnp.int32),
        distribution=jnp.asarray([1, 1, 1], dtype=jnp.int32),
        k=8,
        compression_ratio=1,
        scale_storage="float32",
        exact_topk=True,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        decode_req_batch_size=1,
    )
    np.testing.assert_array_equal(np.asarray(topk),
                                  np.arange(127, 119, -1)[None, :])
