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
"""Numerical verification of the GLM-5.2 indexer Q/K writer Pallas kernel
(`tpu_inference/kernels/experimental/glm5/indexer/kv_writer.py`) against the
Phase 1 pure-JAX reference
(`tpu_inference/kernels/experimental/glm5/indexer/reference.py`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.glm5.indexer.kv_writer import \
    index_qk_rope_quant
from tpu_inference.kernels.experimental.glm5.indexer.reference import (
    apply_indexer_rope, quantize_fp8_ue8m0)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _reference_qk_rope_quant(q, k, positions, *, rope_dim, do_k_norm,
                             rmsnorm_eps):
    """Same pipeline as the Pallas kernel, computed with the Phase 1 JAX
    reference's RoPE/quant primitives (precision=HIGHEST already baked into
    `apply_indexer_rope`/`quantize_fp8_ue8m0`'s callers is not needed here --
    these are elementwise ops, not matmuls, so TPU's default matmul-precision
    downcast does not apply)."""
    if do_k_norm:
        k = k.astype(jnp.float32) * jax.lax.rsqrt(
            jnp.mean(k.astype(jnp.float32)**2, axis=-1, keepdims=True) +
            rmsnorm_eps)
    q_roped = apply_indexer_rope(q, positions, rope_dim=rope_dim)
    k_roped = apply_indexer_rope(k, positions, rope_dim=rope_dim)
    q_fp8, q_scale = quantize_fp8_ue8m0(q_roped, 128)
    k_fp8, k_scale = quantize_fp8_ue8m0(k_roped, 128)
    return q_fp8, jnp.squeeze(q_scale, -1), k_fp8, jnp.squeeze(k_scale, -1)


@pytest.mark.parametrize(
    "num_tokens,num_heads",
    [
        (8, 32),  # GLM-5.2's real index_n_heads.
        (1, 1),
        (16, 4),
        (4, 8),
    ])
def test_index_qk_rope_quant_matches_reference(num_tokens, num_heads):
    head_dim = 128
    rope_dim = 64
    rng = np.random.default_rng(0)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_tokens, head_dim)).astype(np.float32))
    positions = jnp.arange(num_tokens, dtype=jnp.int32) + 5

    q_fp8, q_scale, k_fp8, k_scale = index_qk_rope_quant(q,
                                                         k,
                                                         positions,
                                                         rope_dim=rope_dim,
                                                         do_k_norm=True)

    ref_q_fp8, ref_q_scale, ref_k_fp8, ref_k_scale = _reference_qk_rope_quant(
        q, k, positions, rope_dim=rope_dim, do_k_norm=True, rmsnorm_eps=1e-6)

    np.testing.assert_array_equal(np.asarray(q_scale), np.asarray(ref_q_scale))
    np.testing.assert_array_equal(np.asarray(k_scale), np.asarray(ref_k_scale))

    q_dequant = q_fp8.astype(jnp.float32) * q_scale[..., None]
    ref_q_dequant = ref_q_fp8.astype(jnp.float32) * ref_q_scale[..., None]
    k_dequant = k_fp8.astype(jnp.float32) * k_scale[..., None]
    ref_k_dequant = ref_k_fp8.astype(jnp.float32) * ref_k_scale[..., None]

    np.testing.assert_allclose(np.asarray(q_dequant),
                               np.asarray(ref_q_dequant),
                               atol=0,
                               rtol=0)
    np.testing.assert_allclose(np.asarray(k_dequant),
                               np.asarray(ref_k_dequant),
                               atol=0,
                               rtol=0)


def test_index_qk_rope_quant_no_k_norm():
    num_tokens, num_heads, head_dim, rope_dim = 4, 2, 128, 64
    rng = np.random.default_rng(1)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_tokens, head_dim)).astype(np.float32))
    positions = jnp.arange(num_tokens, dtype=jnp.int32)

    _, _, k_fp8, k_scale = index_qk_rope_quant(q,
                                               k,
                                               positions,
                                               rope_dim=rope_dim,
                                               do_k_norm=False)
    _, _, ref_k_fp8, ref_k_scale = _reference_qk_rope_quant(q,
                                                            k,
                                                            positions,
                                                            rope_dim=rope_dim,
                                                            do_k_norm=False,
                                                            rmsnorm_eps=1e-6)

    np.testing.assert_array_equal(np.asarray(k_scale), np.asarray(ref_k_scale))
    k_dequant = k_fp8.astype(jnp.float32) * k_scale[..., None]
    ref_k_dequant = ref_k_fp8.astype(jnp.float32) * ref_k_scale[..., None]
    np.testing.assert_allclose(np.asarray(k_dequant),
                               np.asarray(ref_k_dequant),
                               atol=0,
                               rtol=0)


def test_index_qk_rope_quant_nope_tail_unrotated():
    """The trailing (head_dim - rope_dim) NoPE channels should be identical
    (up to RMSNorm + quantization) to the un-roped input -- RoPE must only
    touch the leading `rope_dim` channels."""
    num_tokens, num_heads, head_dim, rope_dim = 4, 2, 128, 64
    rng = np.random.default_rng(2)
    q = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(num_tokens, head_dim)).astype(np.float32))
    positions = jnp.arange(num_tokens, dtype=jnp.int32) + 100

    q_fp8, q_scale, _, _ = index_qk_rope_quant(q,
                                               k,
                                               positions,
                                               rope_dim=rope_dim,
                                               do_k_norm=True)
    q_dequant_tail = (q_fp8.astype(jnp.float32) *
                      q_scale[..., None])[..., rope_dim:]
    # Un-rotated reference tail, quantized with the *same* per-row scale the
    # kernel derived from the full (roped) row (block_size == head_dim).
    q_tail_scaled = (q.astype(jnp.float32) / q_scale[..., None])[...,
                                                                 rope_dim:]
    # Re-quantize/dequantize the tail through the fp8 grid the same way the
    # kernel would, to compare apples-to-apples (rounding, not exactness).
    q_tail_fp8 = q_tail_scaled.astype(jnp.float8_e4m3fn)
    q_tail_dequant_expected = q_tail_fp8.astype(jnp.float32) * q_scale[...,
                                                                       None]

    np.testing.assert_array_equal(np.asarray(q_dequant_tail),
                                  np.asarray(q_tail_dequant_expected))
