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

import os

os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "") +
                                  " --xla_tpu_scoped_vmem_limit_kib=65536")

import jax
import jax.numpy as jnp
import numpy as np
from absl import flags, logging
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.mla.v1 import kernel as kernel_v1
from tpu_inference.kernels.mla.v2 import kernel as kernel_v2


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def generate_mla_inputs(
    seq_lens,  # List[(q_len, kv_len)]
    num_heads,
    lkv_dim,
    r_dim,
    page_size,
    q_dtype,
    kv_dtype,
    num_pages,
    rng=None,
):
    """Generates inputs for the MLA kernel.

  Args:
    seq_lens: List of (q_len, kv_len) for each sequence.
    num_heads: Number of attention heads.
    lkv_dim: Dimension of the linear KV part.
    r_dim: Dimension of the rotary embedding part.
    page_size: Size of each page in the KV cache.
    q_dtype: Data type for queries.
    kv_dtype: Data type for keys and values.
    num_pages: Total number of pages in the cache.
    rng: Optional numpy random number generator.

  Returns:
    A tuple containing:
      - ql_nope: Query linear part without positional encoding.
      - q_pe: Query positional encoding part.
      - new_kv_c: New KV cache data.
      - new_k_pe: New Key positional encoding.
      - cache_kv: The existing KV cache.
      - kv_lens: Array of KV lengths for each sequence.
      - page_indices: Indices mapping sequence pages to cache pages.
      - cu_q_lens: Cumulative query lengths.
      - distribution: Mode distribution (e.g., prefill, decode, mixed).
  """
    if rng is None:
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

    # Find the number of decode sequences at the beginning of the batch.
    num_decode_seqs = 0
    for s in seq_lens:
        if s[0] == 1:
            num_decode_seqs += 1
        else:
            break
    distribution = jnp.array([num_decode_seqs, num_decode_seqs,
                              len(seq_lens)],
                             dtype=jnp.int32)

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
    )


FLAGS = flags.FLAGS
flags.DEFINE_bool("debug_mode", False, "Run in debug mode.")

jax.config.parse_flags_with_absl()

# Sq          as Q sequence length
# Skv         as KV sequence length
# C           Context length, `Skv - Sq`
# H           hidden size
# N           number of attention heads
# Lq          latent dimension for Q              1536 in DSV3
# Lkv         latent dimension for K/V            512 in DSV3
# P           nope dimension, no rope.            128 in DSV3
# R           rope dimension, goes through rope.  64 in DSV3
# V           V head dim.                         128 in DSV3

# h_t         hidden states (input to attention)  shape [Sq, H]
# q_c         latent/compressed Q                 shape [Sq, Lq]
# q_nope      uncompressed Q (no-rope)            shape [Sq, N, P]
# q_pe        uncompressed Q (rope)               shape [Sq, N, R]
# kv_c        latent/compressed KV                shape [Skv, Lkv]
# k_pe        decoupled k position embeddings     shape [Skv, R]
# new_kv_c    new kv_c from current iter          shape [Sq, Lkv]
# new_k_pe    new k_pe from current iter          shape [Sq, R]
# cache_kv_c  cached k_c from previous iters      shape [C, Lkv]
# cache_k_pe  cached k_pe from previous iters     shape [C, R]
# W_DQ        project h_t to q_c                  shape [H, Lq]
# W_UQ        project q_c to q_nope               shape [Lq, N * P]
# W_QR        project q_c to q_pe                 shape [Lq, N * R]
# W_DKV       project h_t to kv_c                 shape [H, Lkv]
# W_UK        project kv_c to k_nope              shape [Lkv, N, P]
# W_KR        project h_t to k_pe                 shape [H, R]
# W_UV        project kv_c to v                   shape [Lkv, N, V]
# W_O         project v to h_t                    shape [N * V, H]

# Runtime

# q_c      = h_t @ W_DQ   # [Sq, Lq]
# q_nope   = (q_c @ W_UQ).view(-1, N, P)  # [Sq, N, P]
# ql_nope  = einsum("snh,lnh->snl", q_nope, W_UK)  # [Sq, N, P] @ [Lkv, N, P] -> [Sq, N, Lkv]
# q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)  # [Sq, N, R]
# new_kv_c = h_t @ W_DKV  # [Sq, H] @ [H, Lkv] -> [Sq, Lkv]
# new_k_pe = RoPE(h_t @ W_KR)  # [Sq, H] @ [H, R] -> [Sq, R]
# kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)  # [Skv, Lkv]
# k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)  # [Skv, R]

# // MQA with QK headdim = Lkv + R
# //           V headdim = Lkv
# //      spda_o shape [Sq, N, Lkv]
# // NOTE: this is less compute-friendly since Lkv > P
# //       but is more data-movement friendly since its MQA vs MHA
# spda_o = scaled_dot_product_attention(
#     torch.cat([ql_nope, q_pe], dim=-1),
#     torch.cat([kv_c, k_pe], dim=-1),
#     kv_c
# )  # [Sq, N, Lkv]

# o = einsum("snl,lnv->snv", spda_o.reshape(-1, N, Lkv), W_UV)  # [Sq, N, Lkv] @ [Lkv, N, V] -> [Sq, N, V]
# return o.view(-1, N * V) @ self.num_heads @ W_O  # [Sq, N * V] @ [N * V, H] -> [Sq, H]


class MlaRaggedPagedAttentionTestBase(jtu.JaxTestCase):
    mla_module = kernel_v2
    # Note: Currently we only test FP8 KV cache. Because V2 is now only support
    # FP8 KV cache for KV cache update logic.
    kv_dtype = jnp.float8_e4m3fn

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
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        rng = np.random.default_rng(1234)

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
        ) = generate_mla_inputs(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            q_dtype,
            kv_dtype,
            num_pages,
            rng=rng,
        )

        padded_r_dim = align_to(r_dim, 128)
        padded_lkv_dim = align_to(lkv_dim, 128)
        padded_kv_dim = padded_lkv_dim + padded_r_dim
        packing = get_dtype_packing(kv_dtype)
        total_q_len = sum(s[0] for s in seq_lens)
        kv_lens_list = [s[1] for s in seq_lens]
        max_kv_len = max(kv_lens_list) if kv_lens_list else 0
        total_num_pages = max(
            num_pages, sum(cdiv(kv_len, page_size) for kv_len in kv_lens_list))

        ql_nope_for_kernel = ql_nope.copy()
        q_pe_for_kernel = q_pe.copy()

        expected_out, expected_updated_kv = (
            kernel_v1.ref_mla_ragged_paged_attention(
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
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            ))

        logging.vlog(1, "DEBUG ---------------------------------")
        logging.vlog(1, "ql_nope_for_kernel.shape: %s",
                     ql_nope_for_kernel.shape)
        logging.vlog(1, "q_pe_for_kernel.shape: %s", q_pe_for_kernel.shape)
        logging.vlog(1, "new_kv_c.shape: %s", new_kv_c.shape)
        logging.vlog(1, "new_k_pe.shape: %s", new_k_pe.shape)
        logging.vlog(1, "cache_kv.shape: %s", cache_kv.shape)
        logging.vlog(1, "kv_lens.shape: %s", kv_lens.shape)
        logging.vlog(1, "kv_lens: %s", kv_lens)
        logging.vlog(1, "page_indices.shape: %s", page_indices.shape)
        logging.vlog(1, "page_indices: %s", page_indices)
        logging.vlog(1, "cu_q_lens.shape: %s", cu_q_lens.shape)
        logging.vlog(1, "cu_q_lens: %s", cu_q_lens)
        logging.vlog(1, "distribution.shape: %s", distribution.shape)
        logging.vlog(1, "distribution: %s", distribution)
        logging.vlog(1, "sm_scale: %s", sm_scale)
        logging.vlog(1, "sliding_window: %s", sliding_window)
        logging.vlog(1, "soft_cap: %s", soft_cap)
        logging.vlog(1, "q_scale: %s", q_scale)
        logging.vlog(1, "k_scale: %s", k_scale)
        logging.vlog(1, "v_scale: %s", v_scale)
        logging.vlog(1, "num_kv_pages_per_block: %s", num_kv_pages_per_block)
        logging.vlog(1, "num_queries_per_block: %s", num_queries_per_block)
        logging.vlog(1, "vmem_limit_bytes: %s", vmem_limit_bytes)
        kernel_out, kernel_updated_kv = kernel_v2.mla_ragged_paged_attention(
            jnp.transpose(ql_nope_for_kernel, (1, 0, 2)),
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
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            s_dtype=jnp.float32,
            decode_batch_size=4,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            debug_mode=FLAGS.debug_mode,
        )
        kernel_out = jnp.transpose(kernel_out, (1, 0, 2))
        with np.printoptions(threshold=np.inf):
            logging.vlog(2, "new_kv_c: %s", new_kv_c)
            logging.vlog(2, "new_k_pe: %s", new_k_pe)
            logging.vlog(2, "expected_updated_kv.shape: %s",
                         expected_updated_kv.shape)
            logging.vlog(2, "expected_updated_kv[..., 0]: %s",
                         expected_updated_kv[..., 0])
            logging.vlog(2, "kernel_updated_kv.shape: %s",
                         kernel_updated_kv.shape)
            logging.vlog(2, "kernel_updated_kv[..., 0]: %s",
                         kernel_updated_kv[..., 0])
        print("DEBUG ---------------------------------")

        self.assertEqual(expected_out.shape,
                         (total_q_len, num_heads, padded_lkv_dim))
        self.assertEqual(
            expected_updated_kv.shape,
            (total_num_pages, page_size // packing, packing, padded_kv_dim),
        )
        self.assertEqual(expected_out.dtype, q_dtype)
        self.assertEqual(expected_updated_kv.dtype, kv_dtype)

        # TODO(miliu): Need a more sophiscated checker here to only check valid
        # entries.
        mask = np.zeros_like(expected_updated_kv, dtype=np.bool_)
        pages_per_seq = cdiv(max_kv_len, page_size)
        for i, kv_len in enumerate(kv_lens_list):
            start_page_idx_in_pages_list = i * pages_per_seq
            num_pages_for_seq = cdiv(kv_len, page_size)
            for j in range(num_pages_for_seq):
                page_idx = page_indices[start_page_idx_in_pages_list + j]
                if page_idx == -1:
                    logging.warning(
                        "Sequence %d page %d has invalid page index -1.", i, j)
                    continue

                is_last_page = j == num_pages_for_seq - 1
                tokens_on_this_page = (kv_len %
                                       page_size if is_last_page and kv_len %
                                       page_size != 0 else page_size)

                for token_idx_in_page in range(tokens_on_this_page):
                    row = token_idx_in_page // packing
                    col = token_idx_in_page % packing
                    mask[page_idx, row, col, :] = True
        true_count = np.sum(mask)
        logging.info("Number of True values in KV checker mask: %d",
                     true_count)
        self.assertEqual(true_count, sum(kv_lens_list) * padded_kv_dim)
        np.testing.assert_allclose(
            np.array(expected_updated_kv) * mask,
            np.array(kernel_updated_kv) * mask,
            rtol=0,
            atol=0,
            err_msg="Updated KV cache mismatch",
        )

        # TODO(miliu): Lower tolerance here or find a better metric for comparison.
        self.assertAllClose(expected_out, kernel_out, atol=0.1, rtol=0.2)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MlaRaggedPagedAttentionKernelV2Test(MlaRaggedPagedAttentionTestBase):

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
            self.mla_module.get_kv_cache_shape(total_num_pages, page_size,
                                               lkv_dim, kv_dtype),
            expected_shape,
        )

    def test_ragged_paged_attention_basic(self):
        dtype = jnp.bfloat16
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        page_size = 128
        num_pages = 1024

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
            num_pages,
        )

    def test_ragged_paged_attention_basic_with_small_page_size(self):
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
            self.kv_dtype,
            num_pages,
        )

    def test_ragged_paged_attention_decode_only(self, dtype=jnp.bfloat16):
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
        page_size = 128
        num_pages = 1024

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
            num_pages,
        )

    def test_ragged_paged_attention_prefill_only(self, dtype=jnp.bfloat16):
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
        page_size = 128
        num_pages = 1024

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
            num_pages,
        )

    def test_ragged_paged_attention_mixed(self, dtype=jnp.bfloat16):
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
        page_size = 128
        num_pages = 1024

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
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
        page_size = 128
        num_pages = 10240

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
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
        page_size = 128
        num_pages = 10240

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
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
                self.kv_dtype,
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
                self.kv_dtype,
                num_pages,
                sliding_window=-1,
            )

    def test_ragged_paged_attention_with_scales(self):
        num_heads = 128
        num_seqs = 2
        dtype = jnp.float32
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 100, num_seqs)
        kv_lens = q_lens + rng.integers(0, 50, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        lkv_dim = 512
        r_dim = 64
        page_size = 128
        num_pages = 10240

        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
            num_pages,
            q_scale=0.5,
            k_scale=0.5,
            v_scale=0.7,
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
                self.kv_dtype,
                num_pages,
                soft_cap=0.0,
            )

    @parameterized.named_parameters(
        dict(testcase_name="default"),
        dict(testcase_name="batch_size_8", batch_size=8),
        dict(testcase_name="batch_size_16", batch_size=16),
        dict(testcase_name="batch_size_32", batch_size=32),
        dict(testcase_name="kv_len_random", kv_len_range=(0, 5120 + 1)),
        dict(
            testcase_name="seq_len_random",
            kv_len_range=(4096, 4096 + 1),
            seq_len_range=(1, 1024 + 1),
        ),
        dict(
            testcase_name="len_random",
            kv_len_range=(0, 4096 + 1),
            seq_len_range=(1, 1024 + 1),
        ),
        dict(testcase_name="page_size_16", page_size=16),
        dict(testcase_name="page_size_256", page_size=256),
        dict(testcase_name="num_pages_10240", num_pages=10240),
        dict(testcase_name="num_kv_pages_per_block_1",
             num_kv_pages_per_block=1),
        dict(testcase_name="num_kv_pages_per_block_2",
             num_kv_pages_per_block=2),
        dict(testcase_name="num_kv_pages_per_block_4",
             num_kv_pages_per_block=4),
        dict(testcase_name="num_kv_pages_per_block_8",
             num_kv_pages_per_block=8),
        dict(testcase_name="num_queries_per_block_1", num_queries_per_block=1),
        dict(testcase_name="num_queries_per_block_2", num_queries_per_block=2),
        dict(testcase_name="num_queries_per_block_4", num_queries_per_block=4),
        dict(testcase_name="num_queries_per_block_8", num_queries_per_block=8),
        dict(
            testcase_name="decode_bs4_kv10_ps16_kvb16_qb1",
            seq_len_range=(1, 2),
            batch_size=4,
            kv_len_range=(10, 11),
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=1,
        ),
        dict(
            testcase_name="decode_bs4_kv128_ps16_kvb16_qb1",
            seq_len_range=(1, 2),
            batch_size=4,
            kv_len_range=(128, 129),
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=1,
        ),
        dict(
            testcase_name="decode_bs1_kv128_ps16_kvb16_qb1",
            seq_len_range=(1, 2),
            batch_size=1,
            kv_len_range=(128, 129),
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=1,
        ),
        dict(
            testcase_name="decode_bs8_kv123_ps16_kvb16_qb1",
            seq_len_range=(1, 2),
            batch_size=8,
            kv_len_range=(123, 124),
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=1,
        ),
        dict(
            testcase_name="decode_bs2_kv128_ps16_kvb16_qb16",
            batch_size=2,
            seq_len_range=(1, 2),
            kv_len_range=(128, 129),
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="decode_bs5_kv124_ps16_kvb16_qb16",
            batch_size=5,
            seq_len_range=(1, 2),
            kv_len_range=(124, 125),
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="decode_bs13_kv124_ps16_kvb16_qb16",
            batch_size=13,
            seq_len_range=(1, 2),
            kv_len_range=(124, 125),
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="decode_bs16_kv124_ps16_kvb16_qb16",
            batch_size=16,
            seq_len_range=(1, 2),
            kv_len_range=(124, 125),
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="decode_bs1_kv5_ps4_kvb1_qb1",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(5, 6),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        ),
        dict(
            testcase_name="decode_bs1_kv5_ps16_kvb8_qb1",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(5, 6),
            page_size=16,
            num_kv_pages_per_block=8,
            num_queries_per_block=1,
        ),
        # Corner cases for small page size (page_size=4, block capacity=4)
        dict(
            testcase_name="decode_bs1_kv3_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(3, 4),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv4_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(4, 5),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv5_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(5, 6),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv7_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(7, 8),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len2_kv7_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(2, 3),
            kv_len_range=(7, 8),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len3_kv7_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(3, 4),
            kv_len_range=(7, 8),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len4_kv7_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(4, 5),
            kv_len_range=(7, 8),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len5_kv7_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(7, 8),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv8_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(8, 9),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv9_ps4_kvb1_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(9, 10),
            page_size=4,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        ),
        # Corner cases around page boundaries (page_size=16)
        dict(
            testcase_name="decode_bs1_kv15_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(15, 16),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv17_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(17, 18),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Corner cases around KV block boundaries
        # (num_kv_pages_per_block=2 -> block capacity=32)
        dict(
            testcase_name="decode_bs1_kv31_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(31, 32),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv32_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(32, 33),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv33_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(33, 34),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Corner cases for larger sequences spanning multiple blocks
        dict(
            testcase_name="decode_bs1_kv63_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(63, 64),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv64_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(64, 65),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs1_kv65_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(1, 2),
            kv_len_range=(65, 66),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Corner cases around query block boundaries (num_queries_per_block=4)
        dict(
            testcase_name="prefill_bs1_seq_len3_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(3, 4),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len4_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(4, 5),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len5_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len7_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(7, 8),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len8_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(8, 9),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len9_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(9, 10),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Corner cases around query block boundaries (num_queries_per_block=16)
        dict(
            testcase_name="prefill_bs1_seq_len15_kv32_ps16_kvb2_qb16",
            batch_size=1,
            seq_len_range=(15, 16),
            kv_len_range=(32, 33),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len16_kv32_ps16_kvb2_qb16",
            batch_size=1,
            seq_len_range=(16, 17),
            kv_len_range=(32, 33),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len17_kv32_ps16_kvb2_qb16",
            batch_size=1,
            seq_len_range=(17, 18),
            kv_len_range=(32, 33),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len31_kv64_ps16_kvb2_qb16",
            batch_size=1,
            seq_len_range=(31, 32),
            kv_len_range=(64, 65),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len32_kv64_ps16_kvb2_qb16",
            batch_size=1,
            seq_len_range=(32, 33),
            kv_len_range=(64, 65),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="prefill_bs1_seq_len33_kv64_ps16_kvb2_qb16",
            batch_size=1,
            seq_len_range=(33, 34),
            kv_len_range=(64, 65),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=16,
        ),
        dict(
            testcase_name="decode_bs3_kv16_ps16_kvb2_qb4",
            batch_size=3,
            seq_len_range=(1, 2),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs4_kv16_ps16_kvb2_qb4",
            batch_size=4,
            seq_len_range=(1, 2),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs5_kv16_ps16_kvb2_qb4",
            batch_size=5,
            seq_len_range=(1, 2),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs4_mixed_kv_lens_ps16_1",
            batch_size=4,
            seq_len_range=(1, 2),
            kv_len_range=[(10, 11), (128, 129), (5, 6), (33, 34)],
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=1,
        ),
        dict(
            testcase_name="decode_bs4_mixed_kv_lens_ps16_2",
            batch_size=4,
            seq_len_range=(1, 2),
            kv_len_range=[(1, 2), (16, 17), (32, 33), (65, 66)],
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs4_mixed_kv_lens_ps16_3",
            batch_size=4,
            seq_len_range=(1, 2),
            kv_len_range=[(7, 8), (15, 16), (17, 18), (63, 64)],
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="decode_bs8_mixed_kv_lens",
            batch_size=8,
            seq_len_range=(1, 2),
            kv_len_range=[(2, 3), (12, 13), (32, 33), (64, 65), (10, 11),
                          (128, 129), (5, 6), (33, 34)],
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Mixed cases around page boundaries (page_size=16)
        dict(
            testcase_name="mixed_bs1_seq_len5_kv15_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(15, 16),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len5_kv17_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(17, 18),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Mixed cases around KV block boundaries (num_kv_pages_per_block=2 -> block capacity=32)
        dict(
            testcase_name="mixed_bs1_seq_len5_kv31_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(31, 32),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len5_kv32_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(32, 33),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len5_kv33_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(33, 34),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Mixed cases for larger sequences spanning multiple blocks
        dict(
            testcase_name="mixed_bs1_seq_len5_kv63_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(63, 64),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len5_kv64_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(64, 65),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len5_kv65_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(65, 66),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Mixed cases around query block boundaries (num_queries_per_block=4)
        dict(
            testcase_name="mixed_bs1_seq_len3_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(3, 4),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len4_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(4, 5),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len5_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(5, 6),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len7_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(7, 8),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len8_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(8, 9),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs1_seq_len9_kv16_ps16_kvb2_qb4",
            batch_size=1,
            seq_len_range=(9, 10),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs3_seq_len5_kv16_ps16_kvb2_qb4",
            batch_size=3,
            seq_len_range=(5, 6),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs4_seq_len5_kv16_ps16_kvb2_qb4",
            batch_size=4,
            seq_len_range=(5, 6),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs5_seq_len5_kv16_ps16_kvb2_qb4",
            batch_size=5,
            seq_len_range=(5, 6),
            kv_len_range=(16, 17),
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Mixed cases with varying kv lengths
        dict(
            testcase_name="mixed_bs4_mixed_kv_lens_ps16_1",
            batch_size=4,
            seq_len_range=(5, 6),
            kv_len_range=[(10, 11), (128, 129), (5, 6), (33, 34)],
            page_size=16,
            num_kv_pages_per_block=16,
            num_queries_per_block=1,
        ),
        dict(
            testcase_name="mixed_bs4_mixed_kv_lens_ps16_2",
            batch_size=4,
            seq_len_range=(5, 6),
            kv_len_range=[(1, 2), (16, 17), (32, 33), (65, 66)],
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs4_mixed_kv_lens_ps16_3",
            batch_size=4,
            seq_len_range=(5, 6),
            kv_len_range=[(7, 8), (15, 16), (17, 18), (63, 64)],
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_bs8_mixed_kv_lens",
            batch_size=8,
            seq_len_range=(5, 6),
            kv_len_range=[(2, 3), (12, 13), (32, 33), (64, 65), (10, 11),
                          (128, 129), (5, 6), (33, 34)],
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        # Mixed cases with varying seq lengths (decode and prefill combinations)
        dict(
            testcase_name="mixed_batch_bs4_ps16_kvb2_qb4",
            batch_size=4,
            seq_len_range=[(1, 2), (5, 6), (1, 2), (5, 6)],
            kv_len_range=[(15, 16), (31, 32), (32, 33), (65, 66)],
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
        dict(
            testcase_name="mixed_batch_bs8_ps16_kvb2_qb4",
            batch_size=8,
            seq_len_range=[(1, 2), (1, 2), (4, 5), (5, 6), (1, 2), (8, 9),
                           (1, 2), (2, 3)],
            kv_len_range=[(15, 16), (17, 18), (31, 32), (33, 34), (63, 64),
                          (65, 66), (10, 11), (5, 6)],
            page_size=16,
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        ),
    )
    def test_ragged_paged_attention_deepseekv3(
        self,
        batch_size=4,
        seq_len_range=(1, 1 + 1),  # default by decode only
        kv_len_range=(5120, 5120 + 1),
        page_size=128,
        num_pages=1024,
        num_kv_pages_per_block=8,
        num_queries_per_block=16,
    ):
        rng = np.random.default_rng(1234)
        if isinstance(seq_len_range, list):
            assert len(seq_len_range) == batch_size
            q_lens = np.array(
                [rng.integers(low, high) for low, high in seq_len_range])
        else:
            q_lens = rng.integers(seq_len_range[0], seq_len_range[1],
                                  batch_size)

        if isinstance(kv_len_range, list):
            assert len(kv_len_range) == batch_size
            kv_lens = q_lens + np.array(
                [rng.integers(low, high) for low, high in kv_len_range])
        else:
            kv_lens = q_lens + rng.integers(kv_len_range[0], kv_len_range[1],
                                            batch_size)
        logging.info("Generated kv_lens: %s", kv_lens)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        dtype = jnp.bfloat16
        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
            num_pages,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
        )

    def test_ragged_paged_attention_deepseekv3_batch_decode(
        self,
        batch_size=128,
        seq_len_range=(1, 1 + 1),  # default by decode only
        kv_len_range=(9216, 9216 + 1),
        page_size=1024,
        num_pages=128,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
    ):
        rng = np.random.default_rng(1234)
        if isinstance(seq_len_range, list):
            assert len(seq_len_range) == batch_size
            q_lens = np.array(
                [rng.integers(low, high) for low, high in seq_len_range])
        else:
            q_lens = rng.integers(seq_len_range[0], seq_len_range[1],
                                  batch_size)

        if isinstance(kv_len_range, list):
            assert len(kv_len_range) == batch_size
            kv_lens = q_lens + np.array(
                [rng.integers(low, high) for low, high in kv_len_range])
        else:
            kv_lens = q_lens + rng.integers(kv_len_range[0], kv_len_range[1],
                                            batch_size)
        logging.info("Generated kv_lens: %s", kv_lens)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        num_heads = 128
        lkv_dim = 512
        r_dim = 64
        dtype = jnp.bfloat16
        self._test_mla_ragged_paged_attention(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            dtype,
            self.kv_dtype,
            num_pages,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
