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
"""Numerical verification of the GLM-5.2 indexer top-k Pallas kernel
(`tpu_inference/kernels/experimental/glm5/indexer/topk.py`, wrapping
DeepSeek-V4's `streamindex_topk` with `compression_ratio=1`) against the
Phase 1 pure-JAX reference.

`streamindex_topk` selects via `jax.lax.approx_max_k(recall_target=1.0)`,
not exact `jax.lax.top_k` -- per the Phase 2 task spec, verification here is
recall-based (fraction of the reference's top-k indices also selected by the
kernel), not exact index-set equality, since near-ties can legitimately
select different (equally-scored) indices.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.glm5.indexer.kv_writer import \
    index_qk_rope_quant
from tpu_inference.kernels.experimental.glm5.indexer.reference import (
    INDEX_HEAD_DIM, INDEX_N_HEADS, lightning_indexer_scores,
    select_topk_indices)
from tpu_inference.kernels.experimental.glm5.indexer.topk import (
    build_single_sequence_batch_metadata, glm5_indexer_topk)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _per_row_recall(got, ref):
    """Mean, over rows with >=1 valid reference index, of
    |got_row ∩ ref_row| / |ref_row|."""
    recalls = []
    for t in range(ref.shape[0]):
        ref_set = {int(x) for x in ref[t] if x != -1}
        if not ref_set:
            continue
        got_set = {int(x) for x in got[t] if x != -1}
        recalls.append(len(ref_set & got_set) / len(ref_set))
    assert recalls, "no rows with valid reference indices"
    return float(np.mean(recalls))


def _make_case(num_tokens, num_heads, num_kv, *, seed=0):
    rng = np.random.default_rng(seed)
    head_dim = INDEX_HEAD_DIM
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_kv, head_dim)).astype(np.float32))
    raw_weights = jnp.asarray(
        rng.uniform(0.1, 1.0, size=(num_tokens, num_heads)).astype(np.float32))
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)
    return q, k, raw_weights, q_positions, k_positions


def _run_kernel_and_reference(q,
                              k,
                              raw_weights,
                              q_positions,
                              k_positions,
                              *,
                              topk,
                              page_size,
                              num_kv_pages_per_block,
                              num_queries_per_block,
                              seq_len=None):
    num_tokens, num_heads, head_dim = q.shape
    num_kv = k.shape[0]

    q_fp8, q_scale, _, _ = index_qk_rope_quant(q,
                                               jnp.zeros(
                                                   (num_tokens, head_dim),
                                                   jnp.float32),
                                               q_positions,
                                               rope_dim=64,
                                               do_k_norm=False)
    _, _, k_fp8, k_scale = index_qk_rope_quant(jnp.zeros((num_kv, 1, head_dim),
                                                         jnp.float32),
                                               k,
                                               k_positions,
                                               rope_dim=64,
                                               do_k_norm=True)

    # `streamindex_topk` derives each query's absolute causal position from
    # seq_lens/cu_q_lens alone (queries == the last q_len positions of a
    # seq_len-long sequence) -- see build_single_sequence_batch_metadata's
    # docstring. Callers whose `q_positions` don't follow that convention
    # must pass an explicit `seq_len`.
    seq_lens, page_indices, cu_q_lens, distribution = (
        build_single_sequence_batch_metadata(num_tokens,
                                             num_kv,
                                             page_size=page_size,
                                             seq_len=seq_len))

    got = glm5_indexer_topk(q_fp8,
                            q_scale,
                            raw_weights,
                            k_fp8,
                            k_scale,
                            seq_lens,
                            page_indices,
                            cu_q_lens,
                            distribution,
                            topk=topk,
                            page_size=page_size,
                            num_kv_pages_per_block=num_kv_pages_per_block,
                            num_queries_per_block=num_queries_per_block)

    k_normed = k * jax.lax.rsqrt(jnp.mean(k**2, axis=-1, keepdims=True) + 1e-6)
    ref_scores = lightning_indexer_scores(q,
                                          k_normed,
                                          raw_weights,
                                          q_positions,
                                          k_positions,
                                          use_fp8_quant=True)
    ref = select_topk_indices(ref_scores, topk)
    return np.asarray(got), np.asarray(ref)


@pytest.mark.parametrize(
    "num_tokens,num_heads,num_kv,topk,page_size,bkv_p,bq_sz",
    [
        (6, 4, 20, 8, 128, 1, 8),
        (1, INDEX_N_HEADS, 40, 16, 128, 1, 1),  # decode, real index_n_heads.
        (10, 8, 100, 32, 128, 2, 16),
        (8, INDEX_N_HEADS, 300, 64, 128, 4, 8),
    ],
)
def test_glm5_indexer_topk_high_recall_vs_reference(num_tokens, num_heads,
                                                    num_kv, topk, page_size,
                                                    bkv_p, bq_sz):
    q, k, raw_weights, q_positions, k_positions = _make_case(
        num_tokens, num_heads, num_kv)
    got, ref = _run_kernel_and_reference(q,
                                         k,
                                         raw_weights,
                                         q_positions,
                                         k_positions,
                                         topk=topk,
                                         page_size=page_size,
                                         num_kv_pages_per_block=bkv_p,
                                         num_queries_per_block=bq_sz)

    assert got.shape == (num_tokens, topk)
    recall = _per_row_recall(got, ref)
    # approx_max_k(recall_target=1.0) should be at/near-exact for these
    # (non-adversarially-tied) random-float test sizes.
    assert recall >= 0.99, f"recall {recall} too low"


def test_glm5_indexer_topk_causal_masking():
    """A decode query token at position 3 (physical kv cache holds 8 raw
    rows, but only the first 4 -- positions 0..3 -- are causally "live") must
    never select a KV index > 3."""
    num_tokens, num_heads, num_kv, topk = 1, 4, 8, 8
    rng = np.random.default_rng(3)
    head_dim = INDEX_HEAD_DIM
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_kv, head_dim)).astype(np.float32))
    raw_weights = jnp.asarray(
        rng.uniform(0.1, 1.0, size=(num_tokens, num_heads)).astype(np.float32))
    q_positions = jnp.array([3], dtype=jnp.int32)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    got, ref = _run_kernel_and_reference(q,
                                         k,
                                         raw_weights,
                                         q_positions,
                                         k_positions,
                                         topk=topk,
                                         page_size=128,
                                         num_kv_pages_per_block=1,
                                         num_queries_per_block=1,
                                         seq_len=4)
    valid_got = [i for i in got[0] if i != -1]
    assert all(0 <= i <= 3 for i in valid_got)
    valid_ref = [i for i in ref[0] if i != -1]
    assert sorted(valid_got) == sorted(valid_ref)


def test_pack_indexer_kv_cache_shape():
    from tpu_inference.kernels.experimental.glm5.indexer.topk import (
        indexer_kv_cache_width, pack_indexer_kv_cache)

    num_tokens, page_size = 130, 128
    k_fp8 = jnp.zeros((num_tokens, INDEX_HEAD_DIM), dtype=jnp.float8_e4m3fn)
    k_scale = jnp.ones((num_tokens, ), dtype=jnp.float32)
    cache = pack_indexer_kv_cache(k_fp8, k_scale, page_size=page_size)
    width = indexer_kv_cache_width(INDEX_HEAD_DIM)
    assert width == 256
    assert cache.shape == (2, page_size // 4, 4, width)
    assert cache.dtype == jnp.uint8
