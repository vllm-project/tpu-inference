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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

import tpu_inference.kernels.mla.v1.kernel as mla
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MlaRaggedPagedAttentionKernelTest(jtu.JaxTestCase):

    def _test_mla_ragged_paged_attention(
        self,
        seq_lens,  # List[(q_len, kv_len)]
        num_heads,
        lkv_dim,
        r_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        *,
        num_kv_pages_per_block=8,
        num_queries_per_block=8,
        vmem_limit_bytes=100 * 1024 * 1024,
        sm_scale=1.0,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
    ):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        rng = np.random.default_rng(1234)

        def gen_random(shape, dtype):
            return jnp.array(rng.random(size=shape,
                                        dtype=np.float32)).astype(dtype)

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
        distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)

        ql_nope_for_kernel = ql_nope.copy()
        q_pe_for_kernel = q_pe.copy()

        expected_out, expected_updated_kv = (
            mla.ref_mla_ragged_paged_attention(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                cache_kv.copy(),
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
            ))

        kernel_out, kernel_updated_kv = (mla.mla_ragged_paged_attention(
            ql_nope_for_kernel,
            q_pe_for_kernel,
            new_kv_c,
            new_k_pe,
            cache_kv.copy(),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
        ))

        self.assertEqual(expected_out.shape,
                         (total_q_len, num_heads, padded_lkv_dim))
        self.assertEqual(
            expected_updated_kv.shape,
            (total_num_pages, page_size // packing, packing, padded_kv_dim),
        )
        self.assertEqual(expected_out.dtype, kv_dtype)
        self.assertEqual(expected_updated_kv.dtype, kv_dtype)

        self.assertAllClose(expected_out, kernel_out, atol=0.2, rtol=0.2)
        self.assertAllClose(expected_updated_kv,
                            kernel_updated_kv,
                            atol=0.2,
                            rtol=0.2)

    def test_update_kv_cache(self):
        lkv_dim = 4
        r_dim = 4
        padded_lkv_dim = align_to(lkv_dim, 128)
        padded_r_dim = align_to(r_dim, 128)
        kv_dtype = jnp.bfloat16
        new_kv_c = jnp.arange(16, dtype=kv_dtype).reshape((4, lkv_dim))
        new_k_pe = (jnp.arange(16, dtype=kv_dtype).reshape((4, r_dim)) + 100)
        total_num_pages = 2
        page_size = 4
        cache_kv_shape = mla.get_kv_cache_shape(
            total_num_pages,
            page_size,
            padded_lkv_dim + padded_r_dim,
            kv_dtype,
        )
        cache_kv = jnp.zeros(cache_kv_shape, dtype=kv_dtype)

        # two sequences, first with 3 tokens, second with 1 token
        kv_lens = jnp.array([3, 1], dtype=jnp.int32)
        # first seq uses page 0, second uses page 1
        page_indices = jnp.array([0, -1, 1, -1], dtype=jnp.int32)
        # three tokens for first seq, one for second
        cu_q_lens = jnp.array([0, 3, 4], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

        # manually compute the expected cache
        padded_new_kv_c = jnp.pad(new_kv_c,
                                  ((0, 0), (0, padded_lkv_dim - lkv_dim)),
                                  constant_values=0)
        padded_new_k_pe = jnp.pad(new_k_pe,
                                  ((0, 0), (0, padded_r_dim - r_dim)),
                                  constant_values=0)

        expected_cache = cache_kv
        # First sequence
        # token 0
        page_idx, row, col = 0, 0, 0
        expected_cache = expected_cache.at[page_idx, row,
                                           col, :padded_lkv_dim].set(
                                               padded_new_kv_c[0])
        expected_cache = expected_cache.at[page_idx, row, col,
                                           padded_lkv_dim:padded_lkv_dim +
                                           padded_r_dim].set(
                                               padded_new_k_pe[0])
        # token 1
        page_idx, row, col = 0, 0, 1
        expected_cache = expected_cache.at[page_idx, row,
                                           col, :padded_lkv_dim].set(
                                               padded_new_kv_c[1])
        expected_cache = expected_cache.at[page_idx, row, col,
                                           padded_lkv_dim:padded_lkv_dim +
                                           padded_r_dim].set(
                                               padded_new_k_pe[1])
        # token 2
        page_idx, row, col = 0, 1, 0
        expected_cache = expected_cache.at[page_idx, row,
                                           col, :padded_lkv_dim].set(
                                               padded_new_kv_c[2])
        expected_cache = expected_cache.at[page_idx, row, col,
                                           padded_lkv_dim:padded_lkv_dim +
                                           padded_r_dim].set(
                                               padded_new_k_pe[2])

        # Second sequence
        # token 0
        page_idx, row, col = 1, 0, 0
        expected_cache = expected_cache.at[page_idx, row,
                                           col, :padded_lkv_dim].set(
                                               padded_new_kv_c[3])
        expected_cache = expected_cache.at[page_idx, row, col,
                                           padded_lkv_dim:padded_lkv_dim +
                                           padded_r_dim].set(
                                               padded_new_k_pe[3])

        updated_cache = mla.update_kv_cache(
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

        self.assertAllClose(updated_cache, expected_cache)

    def test_get_kv_cache_shape(self):
        total_num_pages = 10
        page_size = 16
        lkv_dim = 128
        kv_dtype = jnp.bfloat16
        # The calculation for the expected shape is as follows:
        # kv_packing is determined by the dtype, which is 2 for bfloat16.
        # The second dimension is page_size / kv_packing = 16 / 2 = 8
        # The third dimension is kv_packing = 2
        # The fourth dimension is lkv_dim aligned to 128, which is 128
        expected_shape = (10, 8, 2, 128)
        self.assertEqual(
            mla.get_kv_cache_shape(total_num_pages, page_size, lkv_dim,
                                   kv_dtype), expected_shape)

    def test_ragged_paged_attention_basic(self):
        dtype = jnp.bfloat16
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        page_size = 16
        num_pages = 1000

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(dtype=[jnp.bfloat16], )
    def test_ragged_paged_attention_decode_only(self, dtype):
        seq_lens = [
            (1, 18),
            (1, 129),
            (1, 597),
            (1, 122),
            (1, 64),
            (1, 322),
            (1, 463),
            (1, 181),
            (1, 1107),
            (1, 123),
            (1, 31),
            (1, 18),
            (1, 1229),
            (1, 229),
            (1, 87),
            (1, 1328),
        ]
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        page_size = 16
        num_pages = 1000

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(dtype=[jnp.bfloat16], )
    def test_ragged_paged_attention_prefill_only(self, dtype):
        seq_lens = [
            (5, 18),
            (15, 129),
            (120, 597),
            (100, 122),
            (21, 64),
            (32, 322),
            (251, 463),
            (40, 181),
            (64, 1107),
            (99, 123),
            (10, 31),
            (5, 18),
            (3, 1229),
            (120, 229),
            (9, 87),
            (2, 1328),
        ]
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        page_size = 16
        num_pages = 1000

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(dtype=[jnp.bfloat16], )
    def test_ragged_paged_attention_mixed(self, dtype):
        seq_lens = [
            (5, 18),
            (1, 129),
            (120, 597),
            (1, 122),
            (1, 64),
            (32, 322),
            (251, 463),
            (1, 181),
            (1, 1107),
            (99, 123),
            (1, 31),
            (5, 18),
            (3, 1229),
            (117, 229),
            (1, 87),
            (1, 1328),
        ]
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        page_size = 16
        num_pages = 1000

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(sliding_window=[None, 5, 128], )
    def test_ragged_paged_attention_sliding_window(
        self,
        sliding_window: int | None,
    ):
        num_seqs = 5
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        dtype = jnp.float32
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        page_size = 16
        num_pages = 1000

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            sliding_window=sliding_window,
        )

    @parameterized.product(soft_cap=[None, 50.0], )
    def test_ragged_paged_attention_logit_soft_capping(
        self,
        soft_cap: float | None,
    ):
        num_heads = 128
        num_seqs = 2
        dtype = jnp.float32
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        lkv_dim = 512
        r_dim = 64
        page_size = 16
        num_pages = 1000

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            soft_cap=soft_cap,
        )

    def test_ragged_paged_attention_sliding_window_should_be_positive(self):
        dtype = jnp.float32
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        page_size = 16
        num_pages = 1000

        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._test_mla_ragged_paged_attention(
                seq_lens,
                num_heads,
                lkv_dim,
                r_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                sliding_window=0,
            )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._test_mla_ragged_paged_attention(
                seq_lens,
                num_heads,
                lkv_dim,
                r_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                sliding_window=-1,
            )

    def test_ragged_paged_attention_soft_cap_cannot_be_zero(self):
        dtype = jnp.float32
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        page_size = 16
        num_pages = 1000

        with self.assertRaisesRegex(ValueError, "must not be 0.0"):
            self._test_mla_ragged_paged_attention(
                seq_lens,
                num_heads,
                lkv_dim,
                r_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                soft_cap=0.0,
            )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
