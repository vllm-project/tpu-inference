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
"""Numerical verification of the GLM-5.2 sparse (gather + online-softmax
attend) Pallas TPU kernel
(`tpu_inference/kernels/experimental/glm5/core_attention/sparse_attention.py`)
against the Phase 1 pure-JAX reference
(`tpu_inference/kernels/experimental/glm5/core_attention/reference.py`).
"""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tpu_inference.kernels.experimental.glm5.core_attention.reference import (
    dense_causal_attention_reference, sparse_causal_attention)
from tpu_inference.kernels.experimental.glm5.core_attention.sparse_attention import (
    flatten_paged_kv_cache, resolve_topk_to_physical_slots,
    sparse_causal_attention_kernel)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_FP8_E4M3_MIN = float(jnp.finfo(jnp.float8_e4m3fn).min)
_FP8_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)


def _make_topk_indices(rng, num_tokens, num_kv, topk, q_positions):
    """Random causal top-k indices, `-1`-padded, mirroring the style used in
    tests/kernels/glm5/sparse_attention_reference_test.py."""
    topk_indices = np.full((num_tokens, topk), -1, dtype=np.int32)
    for t in range(num_tokens):
        max_valid = int(np.asarray(q_positions)[t]) + 1
        num_valid = min(topk, max_valid)
        topk_indices[t, :num_valid] = rng.choice(max_valid,
                                                 size=num_valid,
                                                 replace=False)
    return topk_indices


@pytest.mark.parametrize(
    "num_tokens,num_heads,qk_dim,v_dim,num_kv,topk,chunk_n",
    [
        (4, 2, 32, 16, 24, 8, 4),
        (1, 4, 128, 64, 40, 16, 8),  # single decode token, MLA-ish dims.
        (6, 8, 64, 32, 100, 32, 8),
        (3, 32, 128, 128, 50, 40, 8),  # GLM-5.2's real index_n_heads.
    ],
)
def test_sparse_causal_attention_kernel_matches_reference(
        num_tokens, num_heads, qk_dim, v_dim, num_kv, topk, chunk_n):
    rng = np.random.default_rng(0)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, qk_dim)).astype(np.float32))
    cache_kv = jnp.asarray(
        rng.normal(size=(num_kv, qk_dim)).astype(np.float32))
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    topk_indices = jnp.asarray(
        _make_topk_indices(rng, num_tokens, num_kv, topk, q_positions))

    out = sparse_causal_attention_kernel(q,
                                         cache_kv,
                                         topk_indices,
                                         v_head_dim=v_dim,
                                         topk=topk,
                                         chunk_n=chunk_n)
    ref = sparse_causal_attention(q, cache_kv, cache_kv[:, :v_dim],
                                  topk_indices, q_positions, k_positions)

    assert out.shape == (num_tokens, num_heads, v_dim)
    np.testing.assert_allclose(np.asarray(out),
                               np.asarray(ref),
                               atol=1e-4,
                               rtol=1e-4)


def test_sparse_causal_attention_kernel_bf16_within_production_tolerance():
    """bf16 end-to-end tolerance target from
    tests/v1/attention/test_sparse_mla_backends.py (rtol=0.01, atol=0.01),
    same target used by the Phase 1 JAX reference's own bf16 test.

    `cache_kv` is now a *real* bf16 array (not an upcast-to-float32
    workaround): the kernel's per-row DMA gather uses a leading-axis VMEM/HBM
    layout (see `sparse_causal_attention_kernel`'s docstring "Quantized (FP8)
    cache support" section) that works for any floating dtype, including
    sub-32-bit ones -- this was previously blocked by a Mosaic VMEM tiling
    constraint, fixed as part of adding FP8 cache support below."""
    num_tokens, num_heads, qk_dim, v_dim, num_kv, topk, chunk_n = (8, 4, 64,
                                                                   64, 40, 16,
                                                                   8)
    rng = np.random.default_rng(1)
    q_f32 = rng.normal(size=(num_tokens, num_heads, qk_dim)).astype(np.float32)
    cache_f32 = rng.normal(size=(num_kv, qk_dim)).astype(np.float32)
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)

    topk_indices = jnp.asarray(
        _make_topk_indices(rng, num_tokens, num_kv, topk, q_positions))

    q_bf16 = jnp.asarray(q_f32).astype(jnp.bfloat16)
    cache_bf16 = jnp.asarray(cache_f32).astype(jnp.bfloat16)

    bf16_out = sparse_causal_attention_kernel(q_bf16,
                                              cache_bf16,
                                              topk_indices,
                                              v_head_dim=v_dim,
                                              topk=topk,
                                              chunk_n=chunk_n)
    fp32_out = sparse_causal_attention_kernel(jnp.asarray(q_f32),
                                              jnp.asarray(cache_f32),
                                              topk_indices,
                                              v_head_dim=v_dim,
                                              topk=topk,
                                              chunk_n=chunk_n)

    np.testing.assert_allclose(np.asarray(bf16_out, dtype=np.float32),
                               np.asarray(fp32_out, dtype=np.float32),
                               rtol=0.01,
                               atol=0.01)


def _static_per_tensor_quantize_fp8(x_f32, scale):
    """Same formula as
    `tpu_inference/layers/common/quantization/__init__.py::static_per_tensor_quantize_tensor`
    (``q = clip(x / scale, dtype_min, dtype_max).astype(float8_e4m3fn)``) --
    the quantization scheme GLM-5.2's *main* attention cache actually uses
    (confirmed via `layers/vllm/backends/flash_attn_mla.py`), a single
    scalar scale per tensor, not DSv4's/the indexer's per-token `ue8m0`
    block scale. Plain numpy (via `ml_dtypes`), not `jnp.clip`/`.astype`,
    since this helper builds test fixtures, not kernel code."""
    return np.clip(x_f32 / scale, _FP8_E4M3_MIN,
                   _FP8_E4M3_MAX).astype(ml_dtypes.float8_e4m3fn)


@pytest.mark.parametrize(
    "num_tokens,num_heads,qk_dim,v_dim,num_kv,topk,chunk_n",
    [
        # NOTE: `topk`/`num_kv` here are deliberately not tiny (unlike this
        # file's other, exact-precision test cases): a static per-tensor FP8
        # scale calibrated off the whole cache's max magnitude quantizes
        # coarsely, and with only a handful of KV positions contributing to
        # the softmax-weighted sum, per-element rounding noise doesn't
        # average out and can occasionally exceed even this fp8-appropriate
        # tolerance for 1-2 output elements (verified empirically across
        # random seeds) -- topk>=64 with num_kv proportionally larger is
        # robust across seeds and still far below GLM-5.2's real topk=2048
        # (i.e. this remains a conservative, not favorable, test).
        (8, 4, 64, 64, 120, 64, 8),
        (1, 32, 128, 128, 128, 64, 8),  # decode, GLM-5.2's real index_n_heads.
        (5, 8, 64, 32, 150, 64, 8),
    ],
)
def test_sparse_causal_attention_kernel_fp8_within_tolerance(
        num_tokens, num_heads, qk_dim, v_dim, num_kv, topk, chunk_n):
    """FP8 main-attention-cache tolerance target from
    tests/v1/attention/test_sparse_mla_backends.py (rtol=0.065, atol=0.05),
    same target used by the Phase 1 JAX reference's own FP8 test -- this is
    the numerical path that unblocks `nvidia/GLM-5.2-NVFP4`'s auto-selected
    FP8 KV cache (see this module's "Quantized (FP8) cache support"
    docstring section and PallasMLASparseAttentionBackendImpl.forward's
    former NotImplementedError branch)."""
    rng = np.random.default_rng(4)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, qk_dim)).astype(np.float32))
    cache_f32 = rng.normal(size=(num_kv, qk_dim)).astype(np.float32)
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    topk_indices = jnp.asarray(
        _make_topk_indices(rng, num_tokens, num_kv, topk, q_positions))

    # A "calibrated" static per-tensor scale: map the cache's dynamic range
    # onto float8_e4m3fn's representable range, same spirit as a real
    # kv-cache-dtype=fp8 calibration pass.
    k_scale = float(np.abs(cache_f32).max()) / _FP8_E4M3_MAX
    v_scale = k_scale  # same physical latent serves as both K and V here.

    cache_fp8 = jnp.asarray(_static_per_tensor_quantize_fp8(
        cache_f32, k_scale))

    fp8_out = sparse_causal_attention_kernel(q,
                                             cache_fp8,
                                             topk_indices,
                                             v_head_dim=v_dim,
                                             topk=topk,
                                             chunk_n=chunk_n,
                                             k_scale=k_scale,
                                             v_scale=v_scale)
    fp32_ref = sparse_causal_attention(q, jnp.asarray(cache_f32),
                                       jnp.asarray(cache_f32[:, :v_dim]),
                                       topk_indices, q_positions, k_positions)

    assert fp8_out.dtype == q.dtype
    np.testing.assert_allclose(np.asarray(fp8_out, dtype=np.float32),
                               np.asarray(fp32_ref, dtype=np.float32),
                               rtol=0.065,
                               atol=0.05)


def test_sparse_causal_attention_kernel_fp8_scale_actually_applied():
    """A regression guard for the dequant step itself: an incorrect (e.g.
    forgotten) scale application should produce a visibly different result,
    not silently coincide with the correctly-scaled one."""
    num_tokens, num_heads, qk_dim, v_dim, num_kv, topk, chunk_n = (4, 2, 32,
                                                                   32, 24, 8,
                                                                   4)
    rng = np.random.default_rng(5)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, qk_dim)).astype(np.float32))
    # Deliberately large-magnitude values so a scale of 1.0 (i.e. "forgot to
    # dequantize") would clip almost everything to fp8_max and be obviously
    # wrong relative to the correctly-scaled output.
    cache_f32 = (rng.normal(size=(num_kv, qk_dim)) * 50.0).astype(np.float32)
    q_positions = jnp.arange(num_tokens,
                             dtype=jnp.int32) + (num_kv - num_tokens)

    topk_indices = jnp.asarray(
        _make_topk_indices(rng, num_tokens, num_kv, topk, q_positions))

    real_scale = float(np.abs(cache_f32).max()) / _FP8_E4M3_MAX
    cache_fp8 = jnp.asarray(
        _static_per_tensor_quantize_fp8(cache_f32, real_scale))

    correctly_scaled = sparse_causal_attention_kernel(q,
                                                      cache_fp8,
                                                      topk_indices,
                                                      v_head_dim=v_dim,
                                                      topk=topk,
                                                      chunk_n=chunk_n,
                                                      k_scale=real_scale,
                                                      v_scale=real_scale)
    wrongly_scaled = sparse_causal_attention_kernel(q,
                                                    cache_fp8,
                                                    topk_indices,
                                                    v_head_dim=v_dim,
                                                    topk=topk,
                                                    chunk_n=chunk_n,
                                                    k_scale=1.0,
                                                    v_scale=1.0)

    assert not np.allclose(np.asarray(correctly_scaled),
                           np.asarray(wrongly_scaled),
                           rtol=0.1,
                           atol=0.1)


def test_sparse_causal_attention_kernel_reduces_to_dense_when_topk_covers_all(
):
    num_tokens, num_heads, qk_dim, v_dim, num_kv, chunk_n = 5, 4, 32, 32, 8, 8
    topk = num_kv
    rng = np.random.default_rng(2)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, qk_dim)).astype(np.float32))
    cache_kv = jnp.asarray(
        rng.normal(size=(num_kv, qk_dim)).astype(np.float32))
    q_positions = jnp.arange(num_tokens, dtype=jnp.int32)
    k_positions = jnp.arange(num_kv, dtype=jnp.int32)

    # Every query token selects every KV position (causal masking must still
    # exclude the future ones via -1... but since num_tokens < num_kv here,
    # emulate full coverage per token by masking future positions with -1).
    idx = np.broadcast_to(np.arange(num_kv, dtype=np.int32),
                          (num_tokens, num_kv)).copy()
    for t in range(num_tokens):
        idx[t, idx[t] > t] = -1
    topk_indices = jnp.asarray(idx)

    kernel_out = sparse_causal_attention_kernel(q,
                                                cache_kv,
                                                topk_indices,
                                                v_head_dim=v_dim,
                                                topk=topk,
                                                chunk_n=chunk_n)
    dense_out = dense_causal_attention_reference(q, cache_kv,
                                                 cache_kv[:, :v_dim],
                                                 q_positions, k_positions)

    np.testing.assert_allclose(np.asarray(kernel_out),
                               np.asarray(dense_out),
                               atol=1e-4,
                               rtol=1e-4)


def test_sparse_causal_attention_kernel_all_padding_row_is_finite():
    num_heads, qk_dim, v_dim, num_kv, topk, chunk_n = 2, 16, 16, 4, 8, 4
    q = jnp.ones((1, num_heads, qk_dim), dtype=jnp.float32)
    cache_kv = jnp.ones((num_kv, qk_dim), dtype=jnp.float32)
    topk_indices = jnp.full((1, topk), -1, dtype=jnp.int32)

    out = sparse_causal_attention_kernel(q,
                                         cache_kv,
                                         topk_indices,
                                         v_head_dim=v_dim,
                                         topk=topk,
                                         chunk_n=chunk_n)
    assert np.all(np.isfinite(np.asarray(out)))
    np.testing.assert_array_equal(np.asarray(out), np.zeros_like(out))


def test_flatten_paged_kv_cache_matches_manual_reshape():
    num_pages, page_size_per_kv_packing, kv_packing, head_dim = 3, 4, 2, 16
    rng = np.random.default_rng(3)
    cache = jnp.asarray(
        rng.normal(size=(num_pages, page_size_per_kv_packing, kv_packing,
                         head_dim)).astype(np.float32))
    flat = flatten_paged_kv_cache(cache)
    expected = np.asarray(cache).reshape(
        num_pages * page_size_per_kv_packing * kv_packing, head_dim)
    np.testing.assert_array_equal(np.asarray(flat), expected)
    # Idempotent on an already-flat cache.
    np.testing.assert_array_equal(np.asarray(flatten_paged_kv_cache(flat)),
                                  np.asarray(flat))


def test_resolve_topk_to_physical_slots_matches_manual_formula():
    block_size = 4
    block_table = jnp.array([[2, 0, 5], [1, 3, -1]], dtype=jnp.int32)
    req_ids = jnp.array([0, 0, 1], dtype=jnp.int32)
    topk_indices = jnp.array([
        [0, 3, 4, -1],
        [7, -1, 1, 9],
        [0, 5, -1, 4],
    ],
                             dtype=jnp.int32)

    got = resolve_topk_to_physical_slots(topk_indices,
                                         block_table,
                                         req_ids,
                                         block_size=block_size)
    got_np = np.asarray(got)

    block_table_np = np.asarray(block_table)
    req_ids_np = np.asarray(req_ids)
    topk_np = np.asarray(topk_indices)
    expected = np.full_like(topk_np, -1)
    for t in range(topk_np.shape[0]):
        for i in range(topk_np.shape[1]):
            idx = topk_np[t, i]
            if idx < 0:
                continue
            block = block_table_np[req_ids_np[t], idx // block_size]
            expected[t, i] = block * block_size + idx % block_size

    np.testing.assert_array_equal(got_np, expected)
