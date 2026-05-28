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
from jax._src import dtypes
from jax._src import test_util as jtu

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention, ref_ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedPagedAttentionKernelTest(jtu.JaxTestCase):

    def _test_ragged_paged_attention(
        self,
        seq_lens,  # List[(q_len, kv_len)]
        num_heads,  # [num_q_heads, num_kv_heads]
        head_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        *,
        bq_sz=64,
        bkv_sz=256,
        bq_csz=32,
        bkv_csz=128,
        vmem_limit_bytes=100 * 1024 * 1024,
        max_num_batched_tokens=512,
        max_num_seq=8,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        use_causal_mask: bool = True,
    ):
        rng = np.random.default_rng(1234)

        def gen_random(shape, dtype):
            return jnp.array(rng.random(size=shape,
                                        dtype=np.float32)).astype(dtype)

        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        cu_q_lens = [0]
        kv_lens = []
        for q_len, kv_len in seq_lens:
            assert q_len <= kv_len
            cu_q_lens.append(cu_q_lens[-1] + q_len)
            kv_lens.append(kv_len)

        max_num_batched_tokens = max(align_to(cu_q_lens[-1], 128),
                                     max_num_batched_tokens)
        max_num_seq = max(align_to(len(seq_lens), 8), max_num_seq)
        max_kv_len = max(kv_lens)
        pages_per_seq = cdiv(max_kv_len, page_size)
        num_q_heads, num_kv_heads = num_heads

        q = gen_random((max_num_batched_tokens, num_q_heads, head_dim),
                       q_dtype)
        k = gen_random((max_num_batched_tokens, num_kv_heads, head_dim),
                       kv_dtype)
        v = gen_random((max_num_batched_tokens, num_kv_heads, head_dim),
                       kv_dtype)
        page_cnt = 0
        page_indices_list = []
        kv_pages_list = []
        kv_packing = get_dtype_packing(kv_dtype)
        padded_head_dim = align_to(head_dim, 128)
        num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)
        for kv_len in kv_lens:
            kv = gen_random(
                (
                    kv_len,
                    num_kv_heads_x2 // kv_packing,
                    kv_packing,
                    padded_head_dim,
                ),
                kv_dtype,
            )
            kv = jnp.pad(
                kv,
                (
                    (
                        0,
                        cdiv(kv_len, page_size) * page_size - kv_len,
                    ),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=jnp.nan,
            ).reshape(
                -1,
                page_size,
                num_kv_heads_x2 // kv_packing,
                kv_packing,
                padded_head_dim,
            )
            indices = page_cnt + jnp.arange(kv.shape[0], dtype=jnp.int32)
            indices = jnp.pad(
                indices,
                ((0, pages_per_seq - indices.shape[0]), ),
                constant_values=jnp.nan,
            )
            page_indices_list.append(indices)
            page_cnt += kv.shape[0]
            kv_pages_list.append(kv)

        kv_cache = jnp.concatenate(kv_pages_list, axis=0)
        kv_cache = jnp.pad(
            kv_cache,
            ((0, num_pages - kv_cache.shape[0]), (0, 0), (0, 0), (0, 0),
             (0, 0)),
            constant_values=jnp.nan,
        )
        page_indices = jnp.stack(page_indices_list, axis=0)
        page_indices = jnp.pad(
            page_indices,
            ((0, max_num_seq - page_indices.shape[0]), (0, 0)),
            constant_values=jnp.nan,
        )
        page_indices = page_indices.reshape(-1)

        cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, max_num_seq + 1 - cu_q_lens.shape[0]))
        kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
        kv_lens = jnp.pad(kv_lens, (0, max_num_seq - kv_lens.shape[0]))
        distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)

        args = (
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

        kwargs = {
            "use_causal_mask": use_causal_mask,
            "sliding_window": sliding_window,
            "soft_cap": soft_cap,
            "q_scale": q_scale,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }

        expected, expected_kv_cache = ref_ragged_paged_attention(
            *args,
            **kwargs,
        )

        output, updated_kv_cache = ragged_paged_attention(
            *args,
            **kwargs,
            m_block_sizes=(bq_sz, bkv_sz, bq_csz, bkv_csz),
            vmem_limit_bytes=vmem_limit_bytes,
        )
        output = output[:cu_q_lens[distribution[-1]]]

        dtype_bits = dtypes.itemsize_bits(jnp.dtype(kv_dtype))
        tols = {
            32: 0.15,
            16: 0.2,
            8: 0.2,
            4: 0.2,
        }
        tol = tols[dtype_bits]
        self.assertAllClose(output, expected, atol=tol, rtol=tol)
        mask = ~jnp.isnan(expected_kv_cache)
        self.assertArraysEqual(updated_kv_cache[mask], expected_kv_cache[mask])
        self.assertEqual(output.shape[-1], head_dim)

    @parameterized.product(
        dtype=[jnp.float32, jnp.bfloat16],
        block_sizes=[
            # (bq_sz, bkv_sz, bq_csz, bkv_csz)
            (64, 256, 32, 128),
            (60, 48, 30, 48),
        ],
        use_causal_mask=[True, False],
    )
    def test_ragged_paged_attention_basic(self, dtype, block_sizes,
                                          use_causal_mask):
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            bq_csz=bq_csz,
            bkv_csz=bkv_csz,
            use_causal_mask=use_causal_mask,
        )

    # TODO: support integer (int8, int4) and fp4 kv cache
    @parameterized.product(
        q_dtype=[jnp.bfloat16],
        kv_dtype=[jnp.float8_e5m2, jnp.float8_e4m3fn],
        kv_scales=[(0.5, 0.5), (None, None)],
    )
    def test_ragged_paged_attention_quantized_kv_cache(self, q_dtype, kv_dtype,
                                                       kv_scales):
        if not jtu.is_device_tpu_at_least(version=5):
            self.skipTest("Expect TPUv5+")
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000
        k_scale, v_scale = kv_scales

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            q_dtype,
            kv_dtype,
            num_pages,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    @parameterized.product(
        q_dtype=[jnp.bfloat16],
        kv_dtype=[jnp.float8_e5m2, jnp.float8_e4m3fn],
        q_scale=[0.5],
        kv_scales=[(0.5, 0.5), (None, None)],
    )
    def test_ragged_paged_attention_quantized_attention(
            self, q_dtype, kv_dtype, q_scale, kv_scales):
        if not jtu.is_device_tpu_at_least(version=5):
            self.skipTest("Expect TPUv5+")
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000
        k_scale, v_scale = kv_scales

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            q_dtype,
            kv_dtype,
            num_pages,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], )
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
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], )
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
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], )
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
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
        )

    @parameterized.product(
        num_seqs=[1, 17],
        num_heads=[(32, 8), (12, 2), (5, 1), (3, 3)],
        head_dim=[80, 240],
        dtype=[jnp.float32, jnp.bfloat16],
    )
    def test_ragged_paged_attention_complex(
        self,
        num_seqs,
        num_heads,
        head_dim,
        dtype,
    ):
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
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
        num_heads = (4, 4)
        dtype = jnp.float32
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
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
        num_heads = (16, 2)
        num_seqs = 2
        dtype = jnp.float32
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        head_dim = 128
        page_size = 16
        num_pages = 1000

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            soft_cap=soft_cap,
        )

    def test_ragged_paged_attention_sliding_window_should_be_positive(self):
        dtype = jnp.float32
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._test_ragged_paged_attention(
                seq_lens,
                num_heads,
                head_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                sliding_window=0,
            )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            self._test_ragged_paged_attention(
                seq_lens,
                num_heads,
                head_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                sliding_window=-1,
            )

    def test_ragged_paged_attention_soft_cap_cannot_be_zero(self):
        dtype = jnp.float32
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        with self.assertRaisesRegex(ValueError, "must not be 0.0"):
            self._test_ragged_paged_attention(
                seq_lens,
                num_heads,
                head_dim,
                page_size,
                dtype,
                dtype,
                num_pages,
                soft_cap=0.0,
            )

    # ------------------------------------------------------------------
    # KV-share path (`update_kv_cache=False`) regression tests.
    #
    # Used by gemma-4 KV-shared layers: the cache slot is redirected to
    # a source layer that has already written its normed/roped K,V, and
    # the shared layer must read attention K,V *only* from the cache.
    # The kernel must (1) compute attention using cache K,V (2) ignore
    # the input `keys` / `values` arrays entirely (3) leave the cache
    # unchanged. The pre-fix kernel split each block into
    # `(past from cache, current from input k,v)`, producing a corrupt
    # mix of source-normed-roped-K with shared-raw-K. The fix is in
    # kernel.py `_fetch_bkv`: when `update_kv_cache=False`, force all of
    # `kv_left` to come from the cache.
    #
    # Note on path coverage: the non-shared (`update_kv_cache=True`)
    # path's `_fetch_bkv` expression is unchanged from before the fix,
    # so the existing prefill / decode / mixed tests above continue to
    # cover it.
    # ------------------------------------------------------------------

    def _build_kv_share_inputs(
        self,
        *,
        q_len: int,
        kv_len: int,
        kv_input_seed: int,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 128,
        page_size: int = 16,
        num_pages: int = 8,
        max_num_seqs: int = 8,
        cache_seed: int = 42,
        q_seed: int = 123,
        dtype=jnp.bfloat16,
    ):
        """Build a single-seq kernel input tuple with a pre-populated cache.

        Cache contents are determined by `cache_seed`; q is determined by
        `q_seed`. Input k,v are determined by `kv_input_seed` — varying
        this between calls lets us check the output is invariant to input
        k,v when `update_kv_cache=False`.
        """
        rng_q = np.random.default_rng(q_seed)
        rng_cache = np.random.default_rng(cache_seed)
        rng_input = np.random.default_rng(kv_input_seed)

        pages_per_seq = cdiv(kv_len, page_size)
        max_num_batched_tokens = max(align_to(q_len, 128), 128)
        kv_packing = get_dtype_packing(dtype)
        num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)
        padded_hd = align_to(head_dim, 128)

        q = jnp.array(
            rng_q.random((max_num_batched_tokens, num_q_heads, head_dim),
                         dtype=np.float32)).astype(dtype)
        k = jnp.array(
            rng_input.random((max_num_batched_tokens, num_kv_heads, head_dim),
                             dtype=np.float32)).astype(dtype)
        v = jnp.array(
            rng_input.random((max_num_batched_tokens, num_kv_heads, head_dim),
                             dtype=np.float32)).astype(dtype)

        cache_data = jnp.array(
            rng_cache.random(
                (pages_per_seq * page_size, num_kv_heads_x2 // kv_packing,
                 kv_packing, padded_hd),
                dtype=np.float32)).astype(dtype)
        cache_pages = cache_data.reshape(pages_per_seq, page_size,
                                         num_kv_heads_x2 // kv_packing,
                                         kv_packing, padded_hd)
        # Padding pages stay nan to surface any out-of-bounds reads.
        kv_cache = jnp.pad(
            cache_pages,
            ((0, num_pages - pages_per_seq), (0, 0), (0, 0), (0, 0), (0, 0)),
            constant_values=jnp.nan,
        )

        page_indices = jnp.zeros((max_num_seqs * pages_per_seq, ),
                                 dtype=jnp.int32)
        page_indices = page_indices.at[:pages_per_seq].set(
            jnp.arange(pages_per_seq, dtype=jnp.int32))

        kv_lens_arr = jnp.zeros((max_num_seqs, ),
                                dtype=jnp.int32).at[0].set(kv_len)
        cu_q_lens_arr = jnp.zeros((max_num_seqs + 1, ),
                                  dtype=jnp.int32).at[1].set(q_len)
        # distribution[3] = (decode_end, prefill_end, mixed_end). One seq:
        # decode if q_len==1 else prefill.
        if q_len == 1:
            distribution = jnp.array([1, 1, 1], dtype=jnp.int32)
        else:
            distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

        return (q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_lens_arr,
                distribution)

    def _kv_share_kwargs(self, head_dim: int = 128):
        return dict(
            sm_scale=1.0 / float(head_dim)**0.5,
            update_kv_cache=False,
            m_block_sizes=(64, 256, 32, 128),
        )

    def test_kv_share_prefill_input_kv_is_ignored(self):
        """q_len == kv_len. Two calls with different input k,v but the same
        pre-populated cache and same q produce bit-identical outputs."""
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        args1 = self._build_kv_share_inputs(q_len=16,
                                            kv_len=16,
                                            kv_input_seed=11)
        args2 = self._build_kv_share_inputs(q_len=16,
                                            kv_len=16,
                                            kv_input_seed=99)
        # Sanity (must happen BEFORE the kernel call — kernel donates
        # queries/keys/values/kv_cache). Skip the kv_cache equality check:
        # _build_kv_share_inputs zero-pads unused trailing pages with NaN,
        # and assert_array_equal treats NaN!=NaN. Cache is identical by
        # construction (same cache_seed).
        np.testing.assert_array_equal(args1[0], args2[0])
        self.assertFalse(np.array_equal(args1[1], args2[1]))
        self.assertFalse(np.array_equal(args1[2], args2[2]))
        cache_before = np.asarray(args1[3])

        out1, cache_after_1 = ragged_paged_attention(*args1,
                                                     **self._kv_share_kwargs())
        out2, cache_after_2 = ragged_paged_attention(*args2,
                                                     **self._kv_share_kwargs())

        # Output invariant to input k,v.
        self.assertArraysEqual(out1, out2)
        # Sanity: outputs are real attention values, not all-zero / NaN
        # (regression catch for a kernel that silently fails closed).
        out1_np = np.asarray(out1[:16]).astype(np.float32)
        assert np.all(np.isfinite(out1_np)), "outputs contain non-finite"
        assert float(np.abs(out1_np).max()) > 0.0, (
            "outputs are all zero — kernel likely failed closed")
        # Cache unchanged in both runs (use the pre-donation snapshot).
        mask = ~np.isnan(cache_before)
        np.testing.assert_array_equal(
            np.asarray(cache_after_1)[mask], cache_before[mask])
        np.testing.assert_array_equal(
            np.asarray(cache_after_2)[mask], cache_before[mask])

    def test_kv_share_chunked_prefill_input_kv_is_ignored(self):
        """q_len < kv_len (chunked / continued prefill). This is the regime
        the pre-fix kernel got wrong: cache holds source's normed/roped
        K,V for the past portion, and the kernel must NOT mix in the
        layer's own raw input k,v for the 'current step' portion."""
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        args1 = self._build_kv_share_inputs(q_len=8,
                                            kv_len=24,
                                            kv_input_seed=11)
        args2 = self._build_kv_share_inputs(q_len=8,
                                            kv_len=24,
                                            kv_input_seed=99)
        cache_before = np.asarray(args1[3])

        out1, cache_after_1 = ragged_paged_attention(*args1,
                                                     **self._kv_share_kwargs())
        out2, cache_after_2 = ragged_paged_attention(*args2,
                                                     **self._kv_share_kwargs())

        # Output invariant to input k,v. The pre-fix kernel would mix
        # source K,V (past 16 positions from cache) with shared raw K,V
        # (current 8 positions from input k,v), so different input k,v
        # would give different outputs.
        self.assertArraysEqual(out1[:8], out2[:8])
        # Sanity: outputs are real (not all-zero / NaN).
        out1_np = np.asarray(out1[:8]).astype(np.float32)
        assert np.all(np.isfinite(out1_np))
        assert float(np.abs(out1_np).max()) > 0.0
        # Cache unchanged.
        mask = ~np.isnan(cache_before)
        np.testing.assert_array_equal(
            np.asarray(cache_after_1)[mask], cache_before[mask])

    def test_kv_share_decode_input_kv_is_ignored(self):
        """q_len == 1, kv_len > 1 (decode step). Same invariance."""
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        args1 = self._build_kv_share_inputs(q_len=1,
                                            kv_len=33,
                                            kv_input_seed=11)
        args2 = self._build_kv_share_inputs(q_len=1,
                                            kv_len=33,
                                            kv_input_seed=99)
        cache_before = np.asarray(args1[3])

        out1, cache_after_1 = ragged_paged_attention(*args1,
                                                     **self._kv_share_kwargs())
        out2, cache_after_2 = ragged_paged_attention(*args2,
                                                     **self._kv_share_kwargs())

        # Decode emits q_len = 1 token. Compare just that token (the rest of
        # the max_num_batched_tokens buffer is junk padding).
        self.assertArraysEqual(out1[:1], out2[:1])
        # Sanity: output is real.
        out1_np = np.asarray(out1[:1]).astype(np.float32)
        assert np.all(np.isfinite(out1_np))
        assert float(np.abs(out1_np).max()) > 0.0
        mask = ~np.isnan(cache_before)
        np.testing.assert_array_equal(
            np.asarray(cache_after_1)[mask], cache_before[mask])


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
