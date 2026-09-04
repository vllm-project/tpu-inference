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
"""FlashAttention-2 style ragged paged attention, plus its equivalence test.

This single file holds both the implementation and the test that checks it
against `ref_ragged_paged_attention` from
`tpu_inference.kernels.ragged_paged_attention.v2.kernel`.

`ref_ragged_paged_attention_fa2` is a drop-in replacement for that reference
which never materializes the full `[num_q_heads, q_len, kv_len]` attention
matrix. Instead it follows the FlashAttention-2 schedule:

  * the outer loop walks blocks of queries (rows of the attention matrix),
  * the inner loop walks blocks of KV (columns), keeping a running max `m`,
    a running (unnormalized) softmax denominator `l` and an unnormalized
    output accumulator `acc` per query row,
  * each inner step rescales `acc`/`l` by `exp(m_prev - m_new)` only (FA2
    keeps the accumulator unnormalized and divides by `l` exactly once, at
    the very end, unlike FA1 which renormalizes every step),
  * KV blocks that are fully masked out by causality / the sliding window
    are skipped entirely.

Run with `pytest` or directly: `python <this file>`.
"""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()

# -----------------------------------------------------------------------------
# Copied verbatim from the v2 kernel module (`DEFAULT_MASK_VALUE`,
# `static_validate_inputs`, `ref_ragged_paged_attention`) so this file stands
# alone. Keep in sync with kernel.py.
# -----------------------------------------------------------------------------

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def static_validate_inputs(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.
    Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    # These inputs are optional. If not specified, we will not validate them.
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    _, num_q_heads, head_dim = q.shape
    _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
    assert num_combined_kv_heads % 2 == 0
    assert isinstance(k_scale, float) or k_scale is None
    assert isinstance(v_scale, float) or v_scale is None
    num_kv_heads = num_combined_kv_heads // 2
    max_num_seqs, pages_per_seq = page_indices.shape
    if num_seqs.shape != (1, ):
        raise ValueError(f"{num_seqs.shape=} must be (1,)")
    if head_dim_k != head_dim:
        raise ValueError(
            f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}."
        )
    if kv_lens.shape != (max_num_seqs, ):
        raise ValueError(
            f"Expected {kv_lens.shape=} to be ({max_num_seqs},) where"
            " `max_num_seqs` is `page_indices.shape[0]`.")
    if cu_q_lens.shape != (max_num_seqs + 1, ):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)  where"
            " `max_num_seqs` is `page_indices.shape[0]`.")
    if (kv_lens.dtype != jnp.int32 or page_indices.dtype != jnp.int32
            or cu_q_lens.dtype != jnp.int32):
        raise ValueError(
            "The dtype of `kv_lens`, `page_indices`, and `cu_q_lens` must be"
            f" int32. Got {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}.")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_q_heads=} must be divisible by {num_kv_heads=}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if (num_kv_pages_per_block is not None
            and not 0 < num_kv_pages_per_block <= pages_per_seq):
        raise ValueError(
            f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}]."
        )
    if num_queries_per_block is not None and num_queries_per_block <= 0:
        raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")
    del sm_scale  # No constraints on sm_scale.
    del mask_value  # No consstraints on mask_value.


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.
    Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    static_validate_inputs(
        queries,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    _, _, num_combined_kv_heads, head_dim = kv_pages.shape
    assert num_combined_kv_heads % 2 == 0
    num_kv_heads = num_combined_kv_heads // 2
    num_q_heads = queries.shape[1]
    assert num_q_heads % num_kv_heads == 0
    num_query_per_kv = num_q_heads // num_kv_heads
    outputs = []
    for i in range(num_seqs[0]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]
        q = queries[q_start:q_end]
        k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads,
                                                  head_dim)[:kv_len]
        v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads,
                                                  head_dim)[:kv_len]
        if k_scale is not None:
            k = k.astype(jnp.float32) * k_scale
            k = k.astype(q.dtype)
        if v_scale is not None:
            v = v.astype(jnp.float32) * v_scale
            v = v.astype(q.dtype)
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)
        attn = jnp.einsum("qhd,khd->hqk",
                          q,
                          k,
                          preferred_element_type=jnp.float32)
        attn *= sm_scale
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)

# -----------------------------------------------------------------------------
# FlashAttention-2 style implementation.
# -----------------------------------------------------------------------------

def _flash_attention_v2_one_seq(
    q,  # [q_len, num_q_heads, head_dim]
    k,  # [kv_len, num_kv_heads, head_dim]
    v,  # [kv_len, num_kv_heads, head_dim]
    *,
    sm_scale,
    sliding_window,
    soft_cap,
    mask_value,
    q_block_size,
    kv_block_size,
):
    """Runs the FA2 loop for a single sequence and returns [q_len, num_q_heads, head_dim]."""
    q_len, num_q_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape
    num_query_per_kv = num_q_heads // num_kv_heads

    # Group the query heads that share a KV head so that the KV block is read
    # once per group instead of once per query head (i.e. no `jnp.repeat` of
    # K/V): [q_len, num_kv_heads, num_query_per_kv, head_dim].
    q = q.reshape(q_len, num_kv_heads, num_query_per_kv, head_dim)

    # Row `i` of this sequence attends to KV positions `[0, kv_offset + i]`.
    kv_offset = kv_len - q_len

    out_blocks = []
    for q_start in range(0, q_len, q_block_size):
        q_end = min(q_start + q_block_size, q_len)
        bq = q_end - q_start
        # [bq, num_kv_heads, num_query_per_kv, head_dim]
        q_blk = q[q_start:q_end]
        q_span = kv_offset + q_start + jax.lax.broadcasted_iota(
            jnp.int32, (bq, 1, 1, 1), 0)

        stats_shape = (bq, num_kv_heads, num_query_per_kv)
        m_i = jnp.full(stats_shape, -jnp.inf, dtype=jnp.float32)
        l_i = jnp.zeros(stats_shape, dtype=jnp.float32)
        acc = jnp.zeros(stats_shape + (head_dim, ), dtype=jnp.float32)

        # Causality bounds the KV columns this query block can see; with a
        # sliding window it is bounded from below as well. Blocks outside
        # that range contribute nothing, so skip them.
        kv_upper = kv_offset + q_end  # exclusive
        kv_lower = 0  # inclusive
        if sliding_window is not None:
            kv_lower = max(0, kv_offset + q_start - sliding_window + 1)

        for kv_start in range(
            (kv_lower // kv_block_size) * kv_block_size,
                kv_upper,
                kv_block_size,
        ):
            kv_end = min(kv_start + kv_block_size, kv_len)
            k_blk = k[kv_start:kv_end]  # [bk, num_kv_heads, head_dim]
            v_blk = v[kv_start:kv_end]

            # [bq, num_kv_heads, num_query_per_kv, bk]
            s = jnp.einsum("qhgd,khd->qhgk",
                           q_blk,
                           k_blk,
                           preferred_element_type=jnp.float32)
            s *= sm_scale
            if soft_cap is not None:
                s = soft_cap * jnp.tanh(s / soft_cap)

            kv_span = kv_start + jax.lax.broadcasted_iota(
                jnp.int32, (1, 1, 1, kv_end - kv_start), 3)
            mask = q_span < kv_span
            if sliding_window is not None:
                mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
            # NOTE: `mask_value` is only accepted for API compatibility.
            # Masked logits must not participate in the running max, so they
            # are set to -inf (which the plain reference's large negative
            # `mask_value` is numerically equivalent to after the softmax).
            s = jnp.where(mask, -jnp.inf, s)

            # ---- FA2 online softmax update ----
            m_new = jnp.maximum(m_i, jnp.max(s, axis=-1)) # [bq, num_kv_heads, num_query_per_kv]
            # A fully masked block leaves `m_new == -inf`; in that case `l_i`
            # and `acc` are still zero so any finite stand-in works, and it
            # keeps `s - m_new` from producing NaNs.
            m_new_safe = jnp.where(jnp.isneginf(m_new), 0.0, m_new)
            alpha = jnp.exp(m_i - m_new_safe)  # rescaling of the past [bq, num_kv_heads, num_query_per_kv]
            p = jnp.exp(s - m_new_safe[..., None]) # [bq, num_kv_heads, num_query_per_kv, kv_end - kv_start]

            l_i = alpha * l_i + jnp.sum(p, axis=-1) # [bq, num_kv_heads, num_query_per_kv]
            # Mirror the plain reference, which casts the probabilities to the
            # value dtype before the second matmul.
            acc = alpha[..., None] * acc + jnp.einsum(
                "qhgk,khd->qhgd",
                p.astype(v.dtype),
                v_blk,
                preferred_element_type=jnp.float32,
            )
            m_i = m_new

        # FA2 divides by the softmax denominator exactly once, here.
        out = acc / jnp.where(l_i == 0.0, 1.0, l_i)[..., None]
        # Undo the GQA grouping: [bq, h, g, d] -> [bq, num_q_heads, head_dim].
        out_blocks.append(out.reshape(bq, num_q_heads, head_dim))

    return jnp.concatenate(out_blocks, axis=0)


def ref_ragged_paged_attention_fa2(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.
    Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    q_block_size: int = 128,
    kv_block_size: int = 256,
):
    """FlashAttention-2 style ragged paged attention reference.

    Same semantics and signature as `ref_ragged_paged_attention`, plus the two
    FA2 tiling knobs `q_block_size` / `kv_block_size` (they only change the
    blocking schedule, not the result).
    """
    static_validate_inputs(
        queries,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError(
            f"{q_block_size=} and {kv_block_size=} must be positive.")
    _, _, num_combined_kv_heads, head_dim = kv_pages.shape
    assert num_combined_kv_heads % 2 == 0
    num_kv_heads = num_combined_kv_heads // 2
    num_q_heads = queries.shape[1]
    assert num_q_heads % num_kv_heads == 0

    outputs = []
    for i in range(num_seqs[0]):
        q_start = int(cu_q_lens[i])
        q_end = int(cu_q_lens[i + 1])
        kv_len = int(kv_lens[i])
        indices = page_indices[i]
        q = queries[q_start:q_end]
        k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads,
                                                  head_dim)[:kv_len]
        v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads,
                                                  head_dim)[:kv_len]
        if k_scale is not None:
            k = (k.astype(jnp.float32) * k_scale).astype(q.dtype)
        if v_scale is not None:
            v = (v.astype(jnp.float32) * v_scale).astype(q.dtype)
        out = _flash_attention_v2_one_seq(
            q,
            k,
            v,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
        )
        outputs.append(out.astype(queries.dtype))

    return jnp.concatenate(outputs, axis=0)


# -----------------------------------------------------------------------------
# Test: the FA2 schedule must reproduce the plain reference.
# -----------------------------------------------------------------------------


def ceil_div(x, a):
    assert a != 0
    return (x + a - 1) // a


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedPagedAttentionFa2RefTest(jtu.JaxTestCase):

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16])
    def test_fa2_ref_matches_ref_ragged_paged_attention(self, dtype):
        # A mixed prefill + decode batch, the same shape family the v2 kernel
        # test uses.
        seq_lens = [(192, 328), (1, 129), (120, 597), (1, 1328), (64, 255)]
        num_q_heads, num_kv_heads = 32, 8
        head_dim = 128
        page_size = 16
        num_pages = 1000
        max_num_seqs = 8
        sm_scale = head_dim**-0.5

        cu_q_lens = [0]
        kv_lens = []
        for q_len, kv_len in seq_lens:
            assert q_len <= kv_len
            cu_q_lens.append(cu_q_lens[-1] + q_len)
            kv_lens.append(kv_len)
        max_num_batched_tokens = max(cu_q_lens[-1], 512)
        pages_per_seq = ceil_div(max(kv_lens), page_size)

        cu_q_lens = jnp.pad(jnp.array(cu_q_lens, dtype=jnp.int32),
                            (0, max_num_seqs + 1 - len(cu_q_lens)))
        kv_lens = jnp.pad(jnp.array(kv_lens, dtype=jnp.int32),
                          (0, max_num_seqs - len(kv_lens)))
        num_seqs = jnp.array([len(seq_lens)], dtype=jnp.int32)

        k0, k1, k2 = jax.random.split(jax.random.key(1234), 3)
        q = jax.random.normal(k0,
                              (max_num_batched_tokens, num_q_heads, head_dim),
                              dtype=dtype)
        kv_pages = jax.random.normal(
            k1, (num_pages, page_size, num_kv_heads * 2, head_dim),
            dtype=dtype)
        page_indices = jax.random.randint(k2, (max_num_seqs, pages_per_seq),
                                          0,
                                          num_pages,
                                          dtype=jnp.int32)

        kwargs = dict(
            sm_scale=sm_scale,
            sliding_window=None,
            soft_cap=None,
        )
        # The two implementations contract over the KV axis in a different
        # order, so TPU's default (reduced precision) f32 matmul alone accounts
        # for ~2e-3 of drift. Pin the matmul precision so the comparison
        # measures the algorithm rather than the MXU passes.
        with jax.default_matmul_precision("highest"):
            expected = ref_ragged_paged_attention(q,
                                                  kv_pages,
                                                  kv_lens,
                                                  page_indices,
                                                  cu_q_lens,
                                                  num_seqs=num_seqs,
                                                  **kwargs)
            actual = ref_ragged_paged_attention_fa2(q,
                                                    kv_pages,
                                                    kv_lens,
                                                    page_indices,
                                                    cu_q_lens,
                                                    num_seqs=num_seqs,
                                                    q_block_size=64,
                                                    kv_block_size=128,
                                                    **kwargs)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        tol = 1e-5 if dtype == jnp.float32 else 2e-2
        self.assertAllClose(actual, expected, atol=tol, rtol=tol)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
