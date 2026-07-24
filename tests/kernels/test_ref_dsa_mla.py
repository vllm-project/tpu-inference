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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from tpu_inference.kernels.mla.v1.kernel import ref_mla_ragged_paged_attention
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)


def generate_dsa_mla_inputs(
    seq_lens,  # List[(q_len, kv_len)]
    num_heads,
    lkv_dim,
    r_dim,
    page_size,
    q_dtype,
    kv_dtype,
    num_pages,
    topk,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(1234)

    def gen_random(shape, dtype):
        return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)

    padded_r_dim = align_to(r_dim, 128)
    padded_lkv_dim = align_to(lkv_dim, 128)
    padded_kv_dim = padded_lkv_dim + padded_r_dim
    packing = get_dtype_packing(kv_dtype)
    q_lens = [s[0] for s in seq_lens]
    kv_lens_list = [s[1] for s in seq_lens]
    total_q_len = sum(q_lens)
    cu_q_lens_list = [0]
    for q_len in q_lens:
        cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

    max_kv_len = max(kv_lens_list) if kv_lens_list else 0
    pages_per_seq = cdiv(max_kv_len, page_size)

    page_indices_list = []
    page_count = 0
    for kv_len in kv_lens_list:
        num_seq_pages = cdiv(kv_len, page_size)
        indices = list(range(page_count, page_count + num_seq_pages))
        page_indices_list.extend(indices + [-1] *
                                 (pages_per_seq - num_seq_pages))
        page_count += num_seq_pages

    total_num_pages = max(num_pages, page_count)

    ql_nope = gen_random((total_q_len, num_heads, lkv_dim), q_dtype)
    q_pe = gen_random((total_q_len, num_heads, r_dim), q_dtype)
    new_kv_c = gen_random((total_q_len, lkv_dim), kv_dtype)
    new_k_pe = gen_random((total_q_len, r_dim), kv_dtype)

    cache_kv = gen_random(
        (total_num_pages, page_size // packing, packing, padded_kv_dim),
        kv_dtype,
    )
    kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
    page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
    cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

    num_decode_seqs = 0
    for s in seq_lens:
        if s[0] == 1:
            num_decode_seqs += 1
        else:
            break
    distribution = jnp.array(
        [num_decode_seqs, num_decode_seqs, len(seq_lens)], dtype=jnp.int32)

    topk_indices_list = []
    for seq_idx, (q_len, kv_len) in enumerate(seq_lens):
        for _ in range(q_len):
            if topk >= kv_len:
                idx = list(range(kv_len)) + [-1] * (topk - kv_len)
            else:
                idx = list(rng.choice(kv_len, size=topk, replace=False))
            topk_indices_list.append(idx)
    topk_indices = jnp.array(topk_indices_list, dtype=jnp.int32)

    return (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        topk_indices,
    )


class TestRefDsaMla(parameterized.TestCase):

    @parameterized.parameters(
        {"topk": 8, "page_size": 16},
        {"topk": 16, "page_size": 32},
        {"topk": 32, "page_size": 16},
    )
    def test_ref_dsa_mla_all_keys(self, topk, page_size):
        """Test that topk_indices containing all keys produces same output as topk_indices=None."""
        seq_lens = [(1, 16), (4, 16), (2, 24)]
        num_heads = 4
        lkv_dim = 128
        r_dim = 64
        q_dtype = jnp.float32
        kv_dtype = jnp.float32
        num_pages = 20

        rng = np.random.default_rng(42)
        (
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            _,
        ) = generate_dsa_mla_inputs(
            seq_lens, num_heads, lkv_dim, r_dim, page_size, q_dtype, kv_dtype,
            num_pages, topk, rng
        )

        max_kv_len = max(s[1] for s in seq_lens)
        full_topk = max_kv_len
        topk_indices_list = []
        for q_len, kv_len in seq_lens:
            for _ in range(q_len):
                idx = list(range(kv_len)) + [-1] * (full_topk - kv_len)
                topk_indices_list.append(idx)
        full_topk_indices = jnp.array(topk_indices_list, dtype=jnp.int32)

        out_dsa, _ = ref_mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv.copy(),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            topk_indices=full_topk_indices,
        )

        out_dense, _ = ref_mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv.copy(),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            topk_indices=None,
        )

        np.testing.assert_allclose(out_dsa, out_dense, rtol=1e-5, atol=1e-5)

    @parameterized.parameters(
        {"topk": 4},
        {"topk": 8},
    )
    def test_ref_dsa_mla_sparse(self, topk):
        """Test sparse attention with topk_indices smaller than kv_len."""
        seq_lens = [(1, 32), (8, 32)]
        num_heads = 4
        lkv_dim = 128
        r_dim = 64
        page_size = 16
        q_dtype = jnp.float32
        kv_dtype = jnp.float32
        num_pages = 20
        rng = np.random.default_rng(123)

        (
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            topk_indices,
        ) = generate_dsa_mla_inputs(
            seq_lens, num_heads, lkv_dim, r_dim, page_size, q_dtype, kv_dtype,
            num_pages, topk, rng
        )

        out_dsa, updated_cache_kv = ref_mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            topk_indices=topk_indices,
        )

        self.assertEqual(out_dsa.shape, ql_nope.shape)

        q_start, q_end = cu_q_lens[0], cu_q_lens[1]
        topk_0 = topk_indices[q_start:q_end]

        p0_idx = page_indices[0]
        p1_idx = page_indices[1]
        page0 = updated_cache_kv[p0_idx, ..., :lkv_dim].reshape(-1, lkv_dim)
        page1 = updated_cache_kv[p1_idx, ..., :lkv_dim].reshape(-1, lkv_dim)
        flat_kv_c_0 = jnp.concatenate([page0, page1], axis=0)[:32]

        padded_r_dim = align_to(r_dim, 128)
        p0_pe = updated_cache_kv[p0_idx, ..., lkv_dim:].reshape(-1, padded_r_dim)
        p1_pe = updated_cache_kv[p1_idx, ..., lkv_dim:].reshape(-1, padded_r_dim)
        flat_k_pe_0 = jnp.concatenate([p0_pe, p1_pe], axis=0)[:32]

        k_0 = jnp.concatenate([flat_kv_c_0, flat_k_pe_0], axis=-1)
        v_0 = flat_kv_c_0
        q_pe_padded = jnp.pad(q_pe[0:1], ((0, 0), (0, 0), (0, padded_r_dim - r_dim)))
        q_0 = jnp.concatenate([ql_nope[0:1], q_pe_padded], axis=-1)

        attn_0 = jnp.einsum("qnh,kh->nqk", q_0, k_0)
        valid_keys = set(np.array(topk_0[0]))
        mask_0 = np.array([k_idx not in valid_keys for k_idx in range(32)])
        default_mask_val = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
        attn_0_masked = np.where(mask_0[None, None, :], default_mask_val, np.array(attn_0))
        attn_weights = jax.nn.softmax(jnp.array(attn_0_masked), axis=-1)
        expected_out_0 = jnp.einsum("nqk,kl->qnl", attn_weights, v_0)

        np.testing.assert_allclose(out_dsa[0:1], expected_out_0, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    absltest.main()
