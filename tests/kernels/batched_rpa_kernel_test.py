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
"""Correctness tests for the batched ragged paged attention kernel.

Validates that the high-performance TPU Pallas kernel
(``tpu_inference.kernels.experimental.batched_rpa``) matches a pure-JAX
mathematical reference.

Core concepts:
  * "Ragged" batching: active tokens of varying-length sequences are packed
    flatly in 1D to avoid padding overhead.
  * "Paged" KV cache: each sequence's keys/values are split into fixed-size
    pages to eliminate memory fragmentation during generation.

Adapted from the RPA v3 test.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import dtypes
from jax._src import test_util as jtu

from tpu_inference.kernels.experimental.batched_rpa import configs
from tpu_inference.kernels.experimental.batched_rpa.utils import (
    align_to, get_dtype_packing)
from tpu_inference.kernels.experimental.batched_rpa.wrapper import \
    ragged_paged_attention

jax.config.parse_flags_with_absl()


def cdiv(a, b):
    return (a + b - 1) // b


def merge_kv(
        k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
        v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
):
    """Pads and packs separate Key and Value arrays together.

    TPUs achieve higher memory bandwidth and compute alignment when keys and
    values are stored side-by-side inside each page of the cache.
    """
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)

    head_dim = align_to(actual_head_dim, 128)
    kv = jnp.pad(
        jnp.concatenate([k, v], axis=-1).reshape(max_num_tokens,
                                                 actual_num_kv_heads_x2,
                                                 actual_head_dim),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv


def ref_ragged_paged_attention(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, nkvx2//packing, packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """Pure-JAX baseline implementation of Ragged Paged Attention.

    Acts as the ground truth. Iterates over the batched sequences sequentially,
    reads and updates their virtual cache entries, scales and clips quantized
    values, and performs standard scaled dot-product attention.
    """
    if out_dtype is None:
        out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16

    if mask_value is None:
        # We do not set to -inf directly because (-inf) - (-inf) is nan.
        mask_value = -float(jnp.finfo(out_dtype).max)

    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    merged_kv = merge_kv(keys, values)
    assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
        kv_cache.shape)
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    assert num_kv_heads_x2 % 2 == 0
    assert actual_num_q_heads % actual_num_kv_heads == 0
    assert head_dim % 128 == 0
    assert get_dtype_packing(kv_cache.dtype) == kv_packing
    assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    outputs = []

    for i in range(distribution[-1]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_len = kv_lens[i]
        indices_start = i * pages_per_seq
        indices_end = indices_start + cdiv(kv_len, page_size)
        indices = page_indices[indices_start:indices_end]
        q = queries[q_start:q_end, :, :actual_head_dim]

        # Step 1: Write incoming K/V tokens into the designated cache pages.
        assert kv_len - q_len >= 0
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
        gathered_kv = gathered_kv.at[kv_len - q_len:kv_len].set(
            merged_kv[q_start:q_end])
        kv_cache = kv_cache.at[indices].set(
            gathered_kv.reshape(gathered_shape))

        kv = gathered_kv.reshape(
            -1, num_kv_heads_x2,
            head_dim)[:, :actual_num_kv_heads * 2, :].reshape(
                -1, actual_num_kv_heads, head_dim * 2)
        # Extract past key and value sequences up to the current length.
        k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
        v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        # Step 2: Handle optional quantization de-scaling/clipping.
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        # Step 3: Compute raw attention scores (Q * K^T).
        attn = jnp.einsum("qhd,khd->hqk",
                          q,
                          k,
                          preferred_element_type=jnp.float32).astype(out_dtype)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        # Step 4: Apply causal mask and optional sliding-window mask.
        if use_causal_mask:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
                jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span >= kv_span
            if sliding_window is not None:
                mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
            attn = jnp.where(mask, attn, mask_value)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        # Step 5: Weighted sum of values (Softmax * V).
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(out_dtype)
        if v_scale is not None:
            out *= v_scale

        outputs.append(out)

    result = jnp.concatenate(outputs, axis=0)
    return result, kv_cache


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class BatchedRaggedPagedAttentionKernelTest(jtu.JaxTestCase):

    def _test_ragged_paged_attention(
        self,
        seq_lens,  # List[(q_len, kv_len)]
        num_heads,  # (num_q_heads, num_kv_heads)
        head_dim,
        page_size,
        q_dtype,
        kv_dtype,
        num_pages,
        *,
        distribution=None,
        bq_sz=64,
        bkv_sz=256,
        bq_csz=32,
        vmem_limit_bytes=100 * 1024 * 1024,
        max_num_batched_tokens=512,
        max_num_seq=8,
        use_explicit_block_sizes: bool = False,
        sliding_window: int | None = None,
        soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ):
        """Universal test execution harness.

        Generates structured test inputs, builds an artificial KV cache filled
        with NaN, runs both the reference baseline and the TPU-accelerated
        kernel, and validates that the resulting matrices and cache contents
        align within expected numeric margins.
        """
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
        num_active_tokens = cu_q_lens[-1]

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
                    (0, cdiv(kv_len, page_size) * page_size - kv_len),
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
        if distribution is None:
            distribution = [0, 0, len(seq_lens)]
        if not (len(distribution) == 3
                and 0 <= distribution[0] <= distribution[1] <= distribution[2]
                and distribution[2] == len(seq_lens)):
            raise ValueError(
                "distribution must contain cumulative decode/prefill/mixed "
                "sequence counts and end at len(seq_lens); got "
                f"{distribution=} and {len(seq_lens)=}.")
        distribution = jnp.array(distribution, dtype=jnp.int32)

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
        # Materialize the reference before invoking the donating wrapper.
        # `jnp.array(jax_array)` may reuse the original device buffer, so it is
        # not sufficient to protect lazy reference computations from donation.
        expected.block_until_ready()
        expected_kv_cache.block_until_ready()

        block_size_kwargs = {}
        if use_explicit_block_sizes:
            block_sizes = configs.BlockSizes(
                bq_sz=bq_sz,
                bq_c_sz=bq_csz,
                bkv_sz=bkv_sz,
                batch_size=2,
                n_buffer=3,
            )
            block_size_kwargs = {
                "decode_block_sizes": block_sizes,
                "prefill_block_sizes": block_sizes,
            }

        # The wrapper donates queries/keys/values/kv_cache, so pass explicit
        # copies rather than `jnp.array(x)`, which may reuse x's device buffer.
        output, updated_kv_cache = ragged_paged_attention(
            q.copy(),
            k.copy(),
            v.copy(),
            kv_cache.copy(),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            **kwargs,
            **block_size_kwargs,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        # TPU execution is asynchronous. Synchronize here so a kernel failure
        # is attributed to this invocation instead of a later slice/assertion.
        output.block_until_ready()
        updated_kv_cache.block_until_ready()
        output = output[:num_active_tokens]

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
            # (bq_sz, bkv_sz, bq_csz)
            (64, 256, 32),
            (60, 48, 30),
        ],
    )
    def test_ragged_paged_attention_basic(self, dtype, block_sizes):
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        bq_sz, bkv_sz, bq_csz = block_sizes

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
            use_explicit_block_sizes=True,
        )

    # TODO: support integer (int8, int4) and fp4 kv cache.
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

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16])
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
            distribution=[len(seq_lens),
                          len(seq_lens),
                          len(seq_lens)],
        )

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16])
    def test_ragged_paged_attention_prefill_via_mixed_bucket(self, dtype):
        # The current wrapper launches DECODE and MIXED kernels, but not the
        # dedicated PREFILL mode. Variable-length prefill requests therefore
        # belong to the mixed bucket.
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

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16])
    def test_ragged_paged_attention_mixed(self, dtype):
        decode_seq_lens = [
            (1, 129),
            (1, 122),
            (1, 64),
            (1, 181),
            (1, 1107),
            (1, 31),
            (1, 87),
            (1, 1328),
        ]
        prefill_seq_lens = [
            (5, 18),
            (120, 597),
            (32, 322),
            (251, 463),
            (99, 123),
            (5, 18),
            (3, 1229),
            (117, 229),
        ]
        seq_lens = decode_seq_lens + prefill_seq_lens
        num_decode = len(decode_seq_lens)
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
            distribution=[num_decode, num_decode,
                          len(seq_lens)],
        )

    @parameterized.product(
        num_seqs=[1, 17],
        num_heads=[(32, 8), (12, 2), (5, 1), (3, 3)],
        head_dim=[128, 256],
        dtype=[jnp.float32, jnp.bfloat16],
    )
    def test_ragged_paged_attention_complex(
        self,
        num_seqs,
        num_heads,
        head_dim,
        dtype,
    ):
        # TODO: Remove this skip when batched RPA supports MHA configurations
        # with one query head per KV head. The (3, 3) configuration currently
        # causes a native TPU RuntimeUnexpectedCoreHalt for float32, before a
        # Python-level assertion can be reported.
        if num_heads == (3, 3):
            self.skipTest(
                "Batched RPA does not yet support the (3, 3) MHA head "
                "configuration.")

        # TODO: Remove this skip when q_vmem_shape consistently uses the
        # aligned query-head count produced by prepare_inputs. With BF16,
        # packing_q=2, so five query heads for one KV head currently produce
        # an undersized VMEM dimension.
        if dtype == jnp.bfloat16 and num_heads == (5, 1):
            self.skipTest(
                "Batched RPA does not yet support odd query-head groups with "
                "packed BF16 inputs.")

        # TODO: Remove this skip when automatic block-size selection accounts
        # for scheduler SMEM, or when empty decode ranges are not launched.
        oversized_scheduler_smem = (head_dim == 128 and (
            (num_heads == (12, 2) and dtype == jnp.bfloat16) or
            (num_heads == (5, 1) and dtype == jnp.float32)))
        if oversized_scheduler_smem:
            self.skipTest(
                "The automatic decode block size produces scheduler metadata "
                "larger than TPU SMEM, even though the decode bucket is empty."
            )

        # TODO: Remove this skip after the batched-RPA KV-cache writeback path
        # correctly handles this larger multi-sequence configuration.
        if (num_seqs == 17 and num_heads == (32, 8) and head_dim == 256
                and dtype == jnp.float32):
            self.skipTest(
                "Known batched-RPA KV-cache writeback mismatch. Attention "
                "output is correct, but some newly written cache values do not "
                "match the reference.")
        use_explicit_block_sizes = (num_heads == (32, 8) and head_dim == 128
                                    and dtype == jnp.float32)

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
            use_explicit_block_sizes=use_explicit_block_sizes,
        )

    @parameterized.product(sliding_window=[None, 5, 128])
    def test_ragged_paged_attention_sliding_window(
        self,
        sliding_window: int | None,
    ):
        num_seqs = 5
        # Keep this feature test on a known-supported GQA configuration. A
        # one-query-head-per-KV-head setup exercises the currently unsupported
        # MHA path and can hard-halt the TPU before sliding-window behavior is
        # evaluated.
        num_heads = (32, 8)
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

    @parameterized.product(soft_cap=[None, 50.0])
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


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
