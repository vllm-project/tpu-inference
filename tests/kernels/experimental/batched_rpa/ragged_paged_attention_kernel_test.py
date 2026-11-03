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
"""Correctness tests for batched ragged paged attention."""

# This test is adapted from the RPA v3 kernel test.

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import dtypes
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import (configs, utils,
                                                            wrapper)

jax.config.parse_flags_with_absl()


@jax.jit
def _assert_cache_equal(actual: jax.Array, expected: jax.Array) -> jax.Array:
    return jnp.all(jnp.logical_or(jnp.isnan(expected), actual == expected))


def cdiv(a, b):
    return (a + b - 1) // b


def merge_kv(
        k: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
        v: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = utils.get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = utils.align_to(actual_num_kv_heads_x2, kv_packing)

    head_dim = utils.align_to(actual_head_dim, 128)
    kv = jnp.pad(
        jnp.concat([k, v],
                   axis=-1).reshape(max_num_tokens, actual_num_kv_heads_x2,
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
    kv_cache: jax.Array,
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing,
    #  head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sink: jax.Array | None = None,  # f32[actual_num_q_heads]
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
    assert utils.get_dtype_packing(kv_cache.dtype) == kv_packing
    assert num_kv_heads_x2 == utils.align_to(actual_num_kv_heads * 2,
                                             kv_packing)
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

        # Update the kv cache.
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
        k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
        v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

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

        if use_causal_mask:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
                jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span >= kv_span
            if sliding_window is not None:
                mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
            attn = jnp.where(mask, attn, mask_value)

        if attention_sink is not None:
            sink = attention_sink.astype(attn.dtype).reshape(-1, 1, 1)
            sink = jnp.broadcast_to(sink, (attn.shape[0], attn.shape[1], 1))
            attn = jnp.concat([sink, attn], axis=2)
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
            attn = attn[..., 1:]
        else:
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(out_dtype)
        if v_scale is not None:
            out *= v_scale

        outputs.append(out)

    result = jnp.concatenate(outputs, axis=0)
    return result, kv_cache


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
            vmem_limit_bytes=64 * 1024 * 1024,
            max_num_batched_tokens=512,
            max_num_seq=8,
            sliding_window: int | None = None,
            soft_cap: float | None = None,
            q_scale: float | None = None,
            k_scale: float | None = None,
            v_scale: float | None = None,
            kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
            decode_query_size: int = 1,
            distribution: jax.Array | None = None,
            sm_scale: float = 1.0,
            nan_kv_seq_idx: int | None = None,
            attention_sink: jax.Array | None = None,  # f32[num_q_heads]
    ):
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            # Tokens live on the lane dim, so page_size must be a multiple of 128.
            page_size = utils.align_to(page_size, 128)
        bkv_sz = utils.align_to(bkv_sz, page_size)

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

        max_num_batched_tokens = max(utils.align_to(cu_q_lens[-1], 128),
                                     max_num_batched_tokens)
        max_num_seq = max(utils.align_to(len(seq_lens), 8), max_num_seq)
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
        seq_page_ranges = []
        kv_packing = utils.get_dtype_packing(kv_dtype)
        padded_head_dim = utils.align_to(head_dim, 128)
        num_kv_heads_x2 = utils.align_to(num_kv_heads * 2, kv_packing)
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
            seq_page_ranges.append((page_cnt, page_cnt + kv.shape[0]))
            page_cnt += kv.shape[0]
            kv_pages_list.append(kv)

        kv_cache = jnp.concatenate(kv_pages_list, axis=0)
        kv_cache = jnp.pad(
            kv_cache,
            ((0, num_pages - kv_cache.shape[0]), (0, 0), (0, 0), (0, 0),
             (0, 0)),
            constant_values=jnp.nan,
        )
        nan_rows = None
        if nan_kv_seq_idx is not None:
            # Fill this sequence's cached KV with NaN, so it attends to NaN.
            first, last = seq_page_ranges[nan_kv_seq_idx]
            kv_cache = kv_cache.at[first:last].set(jnp.nan)
            nan_rows = slice(cu_q_lens[nan_kv_seq_idx],
                             cu_q_lens[nan_kv_seq_idx + 1])
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
            attention_sink,
        )

        kwargs = {
            "sm_scale": sm_scale,
            "sliding_window": sliding_window,
            "soft_cap": soft_cap,
            "q_scale": q_scale,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }

        expected, expected_kv_cache = ref_ragged_paged_attention(
            *jax.tree.map(lambda x: x.copy(), args),
            **kwargs,
        )

        rpa_kv_cache = kv_cache
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        aligned_head_dim_rpa = utils.align_to(
            head_dim,
            num_sublanes * kv_packing
            if kv_layout == configs.KVLayout.SEQ_ALONG_LANE else 128,
        )

        if aligned_head_dim_rpa != 128:
            rpa_kv_cache = rpa_kv_cache[..., :aligned_head_dim_rpa]

        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            # kv_cache shape: [num_pages, page_size, num_kv_heads_x2 // kv_packing,
            #                  kv_packing, aligned_head_dim_rpa]
            rpa_kv_cache = rpa_kv_cache.reshape(
                num_pages,
                page_size,
                num_kv_heads_x2,
                aligned_head_dim_rpa // kv_packing,
                kv_packing,
            ).transpose(0, 2, 3, 4, 1)

        rpa_args = (
            q,
            k,
            v,
            rpa_kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sink,
        )

        rpa_bkv_sz = bkv_sz
        rpa_batch_size = 2
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            rpa_batch_size = 1

        has_decode = int(distribution[0]) > 0
        decode_block_sizes = configs.BlockSizes(
            bq_sz=decode_query_size if has_decode else bq_sz,
            bq_c_sz=decode_query_size if has_decode else bq_csz,
            bkv_sz=rpa_bkv_sz,
            batch_size=rpa_batch_size,
        )
        prefill_block_sizes = configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=bq_csz,
            bkv_sz=rpa_bkv_sz,
            batch_size=rpa_batch_size,
        )

        output, updated_kv_cache = wrapper.ragged_paged_attention(
            *jax.tree.map(lambda x: x.copy(), rpa_args),
            **kwargs,
            decode_block_sizes=decode_block_sizes,
            prefill_block_sizes=prefill_block_sizes,
            vmem_limit_bytes=vmem_limit_bytes,
            kv_layout=kv_layout,
            decode_query_size=decode_query_size,
        )
        output = output[:cu_q_lens[distribution[-1]]]

        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            # updated_kv_cache shape: [num_pages, num_kv_heads_x2,
            #                          aligned_head_dim_rpa // kv_packing, kv_packing,
            #                          page_size]
            updated_kv_cache = updated_kv_cache.transpose(
                0, 4, 1, 2, 3).reshape(
                    num_pages,
                    page_size,
                    num_kv_heads_x2 // kv_packing,
                    kv_packing,
                    aligned_head_dim_rpa,
                )

        if aligned_head_dim_rpa != 128:
            expected_kv_cache = expected_kv_cache[..., :aligned_head_dim_rpa]

        dtype_bits = dtypes.itemsize_bits(jnp.dtype(kv_dtype))
        tols = {
            32: 0.15,
            16: 0.2,
            8: 0.2,
            4: 0.2,
        }
        tol = tols[dtype_bits]
        if nan_rows is not None:
            # Every other sequence must stay finite and match the reference.
            keep = np.ones(output.shape[0], dtype=bool)
            keep[nan_rows] = False
            output, expected = output[keep], expected[keep]
            self.assertTrue(jnp.isfinite(output).all().item())
        self.assertAllClose(output, expected, atol=tol, rtol=tol)
        self.assertTrue(
            _assert_cache_equal(updated_kv_cache, expected_kv_cache).item(),
            "updated_kv_cache does not match expected_kv_cache",
        )
        self.assertEqual(output.shape[-1], head_dim)

    @parameterized.product(
        dtype=[jnp.float32, jnp.bfloat16],
        block_sizes=[
            # (bq_sz, bkv_sz, bq_csz)
            (64, 256, 32),
            (60, 48, 30),
        ],
        kv_layout=[
            configs.KVLayout.HEAD_ALONG_SUBLANE,
            configs.KVLayout.SEQ_ALONG_LANE,
        ],
        page_size=[16, 256, 512],
        # 1.0 (e.g. Gemma4) folds the kernel's scale multiply away; 128**-0.5 is
        # head_dim**-0.5 (e.g. Qwen3).
        sm_scale=[1.0, 128**-0.5],
    )
    def test_ragged_paged_attention_basic(self, dtype, block_sizes, kv_layout,
                                          page_size, sm_scale):
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        num_heads = (32, 8)
        head_dim = 128
        num_pages = 100

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
            kv_layout=kv_layout,
            sm_scale=sm_scale,
        )

    @parameterized.product(
        page_size=[128, 256, 512],
        bq_sz=[1, 4, 64],
    )
    def test_ragged_paged_attention_seq_along_lane_page_size(
            self, page_size, bq_sz):
        # page_size > 128 only matters for SEQ_ALONG_LANE, where tokens are packed
        # along the lane dim. bq_sz in (1, 4) in DECODE mode takes the decode stitch
        # path; bq_sz=64 in MIXED mode takes the prefill roll path.
        is_decode = bq_sz <= 4
        q_len = bq_sz if is_decode else 5
        seq_lens = [(q_len, n) for n in (18, 129, 597, 1107, 31, 1328)]
        n_seqs = len(seq_lens)
        dist = (jnp.array([n_seqs, n_seqs, n_seqs], dtype=jnp.int32)
                if is_decode else jnp.array([0, 0, n_seqs], dtype=jnp.int32))
        self._test_ragged_paged_attention(
            seq_lens,
            (32, 8),
            128,
            page_size,
            jnp.bfloat16,
            jnp.bfloat16,
            1000,
            bq_sz=bq_sz,
            bq_csz=bq_sz,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
            decode_query_size=bq_sz if is_decode else 1,
            distribution=dist,
        )

    @parameterized.product(kv_layout=[
        configs.KVLayout.HEAD_ALONG_SUBLANE,
        configs.KVLayout.SEQ_ALONG_LANE,
    ], )
    def test_ragged_paged_attention_large_schedule(self, kv_layout):
        # The 8 GiB KV cache is held several times over (input, reference output,
        # kernel input and output), which exceeds the HBM of chips before v7.
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        seq_lens = [(1, 8191)] * 1024
        num_heads = (16, 1)
        head_dim = 256
        page_size = 256
        q_dtype = jnp.bfloat16.dtype
        kv_dtype = jnp.bfloat16.dtype
        num_pages = cdiv(8192, page_size) * len(seq_lens) + 32
        bq_sz = 1
        bq_csz = 1
        bkv_sz = 1792

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            q_dtype,
            kv_dtype,
            num_pages,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            bq_csz=bq_csz,
            kv_layout=kv_layout,
        )

    # TODO: b/000000000 - Support integer (int8, int4) and fp4 kv cache
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
    def test_ragged_paged_attention_spec_decode(self, dtype):
        # Multi-token decode queries (e.g. speculative decoding verification)
        # where K > 1 new tokens are added to sequences at various existing lengths
        # including page-boundary crossings (page_size = 16).
        seq_lens = [
            (4, 18),
            (4, 129),
            (4, 597),
            (4, 122),
            (4, 64),
            (4, 322),
            (4, 463),
            (4, 181),
            (4, 1107),
            (4, 123),
            (4, 31),
            (4, 18),
            (4, 1229),
            (4, 229),
            (4, 87),
            (4, 1328),
        ]
        num_heads = (32, 8)
        head_dim = 128
        page_size = 16
        num_pages = 1000

        distribution = jnp.array(
            [len(seq_lens), len(seq_lens),
             len(seq_lens)], dtype=jnp.int32)

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            distribution=distribution,
            decode_query_size=4,
        )

    @parameterized.product(
        dtype=[jnp.float32, jnp.bfloat16],
        kv_layout=[
            configs.KVLayout.HEAD_ALONG_SUBLANE,
            configs.KVLayout.SEQ_ALONG_LANE,
        ],
        q_len=[1, 4],
    )
    def test_ragged_paged_attention_nan_kv_stays_in_its_sequence(
            self, dtype, kv_layout, q_len):
        # The kernel carries its running max, sum and output from one sequence
        # to the next. NaN in one sequence's cached KV (e.g. written mid-batch by a
        # corrupted MTP draft, not produced by the kernel itself) must not leak
        # into the sequences after it. Those later sequences fill whole KV blocks:
        # a partly filled block can also read NaN from stale V rows that an earlier
        # sequence left in VMEM, which is a separate issue.
        seq_lens = [(q_len, kv_len)
                    for kv_len in (129, 597, 256, 512, 768, 256, 1024, 512)]
        self._test_ragged_paged_attention(
            seq_lens,
            (32, 8),
            128,
            16,
            dtype,
            dtype,
            1000,
            bq_sz=q_len,
            bq_csz=q_len,
            kv_layout=kv_layout,
            distribution=jnp.array([len(seq_lens)] * 3, dtype=jnp.int32),
            decode_query_size=q_len,
            nan_kv_seq_idx=1,
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

    @parameterized.product(
        kv_layout=[
            configs.KVLayout.HEAD_ALONG_SUBLANE,
            configs.KVLayout.SEQ_ALONG_LANE,
        ],
        dtype=[jnp.float32, jnp.bfloat16],
        head_dim=[64, 128],
        num_heads=[(6, 2), (4, 4), (5, 1)],
        sliding_window=[None, 128],
    )
    def test_ragged_paged_attention_attention_sink(
        self,
        kv_layout: configs.KVLayout,
        dtype: jnp.dtype,
        head_dim: int,
        num_heads: tuple[int, int],
        sliding_window: int | None,
    ):
        num_seqs = 5
        rng = np.random.default_rng(1234)
        q_lens = rng.integers(1, 300, num_seqs)
        kv_lens = q_lens + rng.integers(0, 300, num_seqs)
        seq_lens = list(zip(q_lens.tolist(), kv_lens.tolist()))
        page_size = 128
        num_pages = 1000
        attention_sink = jnp.asarray(
            head_dim / 4 +
            (rng.random(num_heads[0], dtype=np.float32) - 0.5) * 16,
            dtype=jnp.float32,
        )

        self._test_ragged_paged_attention(
            seq_lens,
            num_heads,
            head_dim,
            page_size,
            dtype,
            dtype,
            num_pages,
            sliding_window=sliding_window,
            kv_layout=kv_layout,
            attention_sink=attention_sink,
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


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
