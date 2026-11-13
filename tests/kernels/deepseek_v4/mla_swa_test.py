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
"""Correctness test for MLA kernels."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

# Import the kernels
from tpu_inference.kernels.experimental.deepseek_v4.core_attention import \
    mla_swa


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = jax.dtypes.itemsize_bits(dtype)
    return 32 // bits


@jax.jit(donate_argnames="cache_kv")
def update_kv_cache(
        new_kv: jax.Array,  # [num_tokens, actual_lkv_dim]
        cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        distribution: jax.Array,  # i32[3]
) -> tuple[jax.Array, jax.Array]:
    """Update KV cache with new tokens."""
    actual_lkv_dim = new_kv.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if actual_lkv_dim != lkv_dim:
        new_kv = jnp.pad(new_kv, ((0, 0), (0, lkv_dim - actual_lkv_dim)),
                         constant_values=0)
    kv_dim = lkv_dim
    _, page_size_per_kv_packing, kv_packing, cache_kv_dim = cache_kv.shape
    assert kv_dim == cache_kv_dim
    page_size = page_size_per_kv_packing * kv_packing

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs

    def seq_loop_body(i, cache_kv):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]

        def token_loop_body(j, cache_kv_):
            token_idx_in_seq = kv_len - q_len + j
            page_num_in_seq = token_idx_in_seq // page_size
            page_indices_start = i * pages_per_seq
            page_idx = page_indices[page_indices_start + page_num_in_seq]
            row = (token_idx_in_seq % page_size) // kv_packing
            col = (token_idx_in_seq % page_size) % kv_packing

            cache_kv_ = cache_kv_.at[page_idx, row, col,
                                     ..., :lkv_dim].set(new_kv[q_start + j])
            return cache_kv_

        return jax.lax.fori_loop(0, q_len, token_loop_body, cache_kv)

    cache_kv = jax.lax.fori_loop(0, distribution[-1], seq_loop_body, cache_kv)

    return cache_kv


def reformat_swc_cache(swc_cache):
    # Raw bf16 layout: uint8 [total_num_pages, slots, kv_packing, lkv_dim], where
    # each token owns `slots_per_token` consecutive slots. Reinterpret the bytes
    # as bf16
    total_num_pages, slots, *_ = swc_cache.shape
    slots_per_token = 2
    tokens = slots // slots_per_token
    row = swc_cache.reshape(total_num_pages, tokens, -1)
    nb = row.shape[-1] // 256  # number of 128-lane (lo, hi) block pairs
    row = row.reshape(total_num_pages, tokens, nb, 2, 128)
    swc_cache_lo = row[..., 0, :]
    swc_cache_hi = row[..., 1, :]
    swc_cache = (swc_cache_hi.astype(jnp.uint16) << 8) | swc_cache_lo.astype(
        jnp.uint16)
    swc_cache = swc_cache.reshape(total_num_pages, tokens, nb * 128)
    return jax.lax.bitcast_convert_type(swc_cache, jnp.bfloat16)


def ref_implementation(
    q: jax.Array,  # [num_tokens, actual_num_q_heads, actual_lkv_dim]
    new_kv: jax.Array,  # [num_tokens, actual_lkv_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    *,
    sliding_window: int,
    sm_scale: float = 1.0,
    mask_value: float | None = DEFAULT_MASK_VALUE,
):

    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    updated_cache_kv = update_kv_cache(
        new_kv,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )
    # Pad q and q_pe to make the last dimension 128-byte aligned.
    actual_lkv_dim = q.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if lkv_dim != actual_lkv_dim:
        q = jnp.pad(
            q,
            ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
            constant_values=0,
        )

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
        updated_cache_kv.shape)
    page_size = page_size_per_kv_packing * kv_packing
    assert lkv_dim == q.shape[-1]

    kv_c_cache = updated_cache_kv[..., :lkv_dim]

    # Quantize and dequantize kv_c_cache to simulate the loss of quantization
    kv_c_cache = kv_c_cache.reshape(total_num_pages, page_size, lkv_dim)

    outputs = []
    ls = []
    ms = []

    for i in range(distribution[-1]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]

        q_i = q[q_start:q_end]  # [q_len, actual_num_q_heads, lkv_dim+r_dim]

        indices_start = i * pages_per_seq
        num_pages_i = cdiv(kv_len, page_size)
        indices_end = indices_start + num_pages_i
        indices = page_indices[indices_start:indices_end]

        # Gather paged kv_c and k_pe
        gathered_kv_c = kv_c_cache[
            indices]  # [num_pages_i, page_size, lkv_dim]

        # Flatten pages to sequence
        flat_kv_c = gathered_kv_c.reshape(
            -1, lkv_dim)  # [num_pages_i * page_size, lkv_dim]

        # Prepare k and v for attention
        k_i = flat_kv_c[:kv_len]  # [kv_len, lkv_dim]
        v_i = flat_kv_c[:kv_len]  # [kv_len, lkv_dim]

        # MQA attention:
        # q:[q_len, actual_num_q_heads, lkv_dim+r_dim]
        # k:[kv_len, lkv_dim+r_dim]
        # v:[kv_len, lkv_dim]
        # attn: [actual_num_q_heads, q_len, kv_len]
        attn = jnp.einsum("qnh,kh->nqk",
                          q_i,
                          k_i,
                          preferred_element_type=jnp.float32)
        attn *= sm_scale

        # Causal mask
        q_span = kv_len - q_len + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        attn = jnp.where(mask, mask_value, attn)
        m = jnp.max(attn, axis=-1, keepdims=True)
        l_sum = jnp.sum(jnp.exp(attn - m), axis=-1, keepdims=True)
        l_sinks = jnp.exp(attention_sinks[..., None, None] - m)
        l_final = l_sum + l_sinks
        attn = jnp.exp(attn - m) / l_final

        # out_i: [q_len, actual_num_q_heads, lkv_dim]
        out_i = jnp.einsum("nqk,kl->qnl", attn, v_i).astype(q_i.dtype)
        outputs.append(out_i)
        ls.append(jnp.transpose(l_sum[..., 0]))
        ms.append(jnp.transpose(m[..., 0]))

    return (
        jnp.concatenate(outputs, axis=0),
        updated_cache_kv,
        jnp.concatenate(ls, axis=0),
        jnp.concatenate(ms, axis=0),
    )


class CorrectnessTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(1234)
        self.rng_key = jax.random.PRNGKey(1234)

        self.kv_dtype = jnp.bfloat16
        self.q_dtype = jnp.bfloat16
        self.kv_packing = get_dtype_packing(self.kv_dtype)

        # Configuration (Smaller for correctness test)
        self.batch_size = 12
        self.num_heads = 128
        self.head_dim = 512
        self.lkv_dim = 640
        self.sliding_window = 16
        self.page_size = 16
        self.attention_sinks = jnp.array(
            self.rng.random(size=(self.num_heads, ), dtype=np.float32))

        self.ref_pages_per_seq = cdiv(self.sliding_window * 10, self.page_size)
        self.ref_page_indices = jnp.arange(self.batch_size *
                                           self.ref_pages_per_seq,
                                           dtype=jnp.int32)
        head_dim = self.head_dim
        self.ref_cache = jnp.zeros(
            (
                self.batch_size * self.ref_pages_per_seq,
                self.page_size // self.kv_packing,
                self.kv_packing,
                head_dim,
            ),
            dtype=self.kv_dtype,
        )

        # DSv4's SWA cache overlay onto the CSA's compressed kv cache.
        # shape: [total_num_pages, page_size_csa_compressed_cache, 4, 128]
        # Physical page holds a few extra tokens beyond the logical page_size.
        self.sw_physical_page_size = self.page_size + 4
        self.swc_cache = jnp.zeros(
            (
                self.batch_size * self.ref_pages_per_seq,
                self.sw_physical_page_size * 2,  # each token takes 2 slots
                4,
                128,
            ),
            dtype=jnp.uint8,
        )
        self.swc_page_indices = self.ref_page_indices
        self.kv_lens = jnp.zeros((self.batch_size, ), dtype=jnp.int32)

    def compare_cache(self, kv_lens):
        print("Comparing output cache...")
        # Construct valid token mask for cache comparison
        batch_size = self.batch_size
        pages_per_seq = self.ref_pages_per_seq
        page_size = self.page_size
        kv_packing = self.kv_packing

        # Create a grid of token indices for each position in the cache
        pages = np.arange(pages_per_seq)[:, None, None]
        rows = np.arange(page_size // kv_packing)[None, :, None]
        cols = np.arange(kv_packing)[None, None, :]
        token_indices = pages * page_size + rows * kv_packing + cols

        # kv_lens shape is (batch_size,). Reshape for broadcasting
        kv_lens_np = np.array(kv_lens)[:, None, None, None]
        valid_mask = token_indices[None, ...] < kv_lens_np
        valid_mask = valid_mask.reshape(batch_size * pages_per_seq,
                                        page_size // kv_packing, kv_packing, 1)

        ref_cache_masked = np.where(valid_mask, self.ref_cache, 0)
        swa_cache = reformat_swc_cache(self.swc_cache[:, :page_size * 2, :, :])
        swa_cache = swa_cache.reshape(batch_size * pages_per_seq,
                                      page_size // kv_packing, kv_packing, 512)
        swa_cache_masked = np.where(valid_mask, swa_cache, 0)

        diff_cache = np.abs(ref_cache_masked - swa_cache_masked)
        print(f"Max Diff Cache: {np.max(diff_cache)}")
        np.testing.assert_allclose(ref_cache_masked,
                                   swa_cache_masked,
                                   rtol=0.1,
                                   atol=0.1)

    def run_and_compare_outputs(self, q, new_kv, kv_lens, cu_q_lens,
                                distribution):
        total_tokens = q.shape[0]
        out_base, self.ref_cache, l_base, m_base = ref_implementation(
            q,
            new_kv,
            self.ref_cache,
            kv_lens,
            self.ref_page_indices,
            cu_q_lens,
            distribution,
            self.attention_sinks,
            sm_scale=1.0,
            sliding_window=self.sliding_window,
        )

        out, self.swc_cache, l_sum, m = (
            mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                self.swc_cache,
                kv_lens,
                self.swc_page_indices,
                cu_q_lens,
                distribution,
                self.attention_sinks,
                sm_scale=1.0,
                sliding_window=self.sliding_window,
                num_queries_per_block=8,
                num_kv_pages_per_block=2,
                q_compute_block_size=2,
                logical_page_size=self.page_size,
            ))

        # Compare output
        print("Comparing output attention...")
        out_base.block_until_ready()
        out.block_until_ready()
        diff_out = np.abs(out_base - out)
        print(f"Max Diff Out: {np.max(diff_out)}")
        print(f"kv_lens: {kv_lens}")
        print(f"cu_q_lens: {cu_q_lens}")
        np.testing.assert_allclose(out_base, out, rtol=0.1, atol=0.1)

        l_sum.block_until_ready()
        m.block_until_ready()
        l_base.block_until_ready()
        m_base.block_until_ready()
        assert l_sum.shape == (total_tokens, self.num_heads)
        assert m.shape == (total_tokens, self.num_heads)
        np.testing.assert_allclose(l_base, l_sum, rtol=0.1, atol=0.1)
        np.testing.assert_allclose(m_base, m, rtol=0.1, atol=0.1)

        # Cache comparison
        self.compare_cache(kv_lens)

    def gen_random(self, shape, dtype):
        return jnp.array(self.rng.random(size=shape,
                                         dtype=np.float32)).astype(dtype)

    def gen_random_int(self, shape, low, high):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return jax.random.randint(subkey,
                                  shape=shape,
                                  minval=low,
                                  maxval=high,
                                  dtype=jnp.int32)

    def test_correctness_rng(self):
        print(f"JAX Backend: {jax.default_backend()}")

        # First step, contains variable length prefill
        new_kv_lens = self.gen_random_int((self.batch_size, ),
                                          self.sliding_window // 2,
                                          self.sliding_window * 2)
        cu_q_lens = jnp.concatenate(
            [jnp.array([0]),
             jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)])
        self.kv_lens += new_kv_lens
        total_tokens = jnp.sum(new_kv_lens)
        q = self.gen_random((total_tokens, self.num_heads, self.head_dim),
                            self.q_dtype)
        new_kv = self.gen_random((total_tokens, self.head_dim), self.kv_dtype)
        distribution = jnp.array([0, 0, self.batch_size], dtype=jnp.int32)

        self.run_and_compare_outputs(q, new_kv, self.kv_lens, cu_q_lens,
                                     distribution)

        # Second step, contains half decode and half prefill
        num_decode_seqs = self.batch_size // 2
        new_kv_lens = self.gen_random_int(
            (self.batch_size - num_decode_seqs, ),
            self.sliding_window // 2,
            self.sliding_window * 2,
        )
        new_kv_lens = jnp.concatenate([
            jnp.ones((num_decode_seqs, ), dtype=jnp.int32),
            new_kv_lens,
        ])
        cu_q_lens = jnp.concatenate(
            [jnp.array([0]),
             jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)])
        self.kv_lens += new_kv_lens
        total_tokens = jnp.sum(new_kv_lens)
        q = self.gen_random((total_tokens, self.num_heads, self.head_dim),
                            self.q_dtype)
        new_kv = self.gen_random((total_tokens, self.head_dim), self.kv_dtype)
        distribution = jnp.array(
            [num_decode_seqs, num_decode_seqs, self.batch_size],
            dtype=jnp.int32)

        self.run_and_compare_outputs(q, new_kv, self.kv_lens, cu_q_lens,
                                     distribution)

        # Third step, contains full decode
        new_kv_lens = jnp.ones((self.batch_size, ), dtype=jnp.int32)
        cu_q_lens = jnp.concatenate(
            [jnp.array([0]),
             jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)])
        self.kv_lens += new_kv_lens
        total_tokens = jnp.sum(new_kv_lens)
        q = self.gen_random((total_tokens, self.num_heads, self.head_dim),
                            self.q_dtype)
        new_kv = self.gen_random((total_tokens, self.head_dim), self.kv_dtype)
        distribution = jnp.array(
            [self.batch_size, self.batch_size, self.batch_size],
            dtype=jnp.int32)
        self.run_and_compare_outputs(q, new_kv, self.kv_lens, cu_q_lens,
                                     distribution)


if __name__ == "__main__":
    absltest.main()
