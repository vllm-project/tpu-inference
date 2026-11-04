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
"""Numerical tests for kernel-side symmetric (encoder) ALiBi.

Compares `encoder_only_flash_attention(..., alibi_slopes)` against a dense
jnp reference (scaled logits + symmetric ALiBi bias + segment masking) on
both kernel paths (multi-step kv loop and single-step). Requires TPU.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from tpu_inference.kernels.flash_attention.kernel import (
    BlockSizes, encoder_only_flash_attention)

jax.config.parse_flags_with_absl()


def _alibi_slopes(n_heads: int) -> list[float]:
    """Standard ALiBi slope schedule (matches JinaBert reference)."""

    def pow2_slopes(n):
        start = 2**(-(2**-(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return pow2_slopes(n_heads)
    closest = 2**math.floor(math.log2(n_heads))
    return (pow2_slopes(closest) +
            _alibi_slopes(2 * closest)[0::2][:n_heads - closest])


def _reference(q, k, v, seq_lens, slopes, sm_scale):
    """Dense softmax(sm_scale * qk^T - slope * |i - j| + segment mask) @ v."""
    total_len = q.shape[0]
    seg = np.full((total_len, ), len(seq_lens), dtype=np.int32)
    offset = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        seg[offset:offset + seq_len] = seq_idx
        offset += seq_len
    seg = jnp.asarray(seg)
    # precision=HIGHEST: the kernel accumulates in float32; the default MXU
    # precision (bf16 multiplies) would add ~1e-3 error to the reference.
    logits = jnp.einsum(
        "qhd,khd->hqk", q, k, precision=jax.lax.Precision.HIGHEST).astype(
            jnp.float32) * sm_scale
    distance = jnp.abs(
        jnp.arange(total_len)[:, None] - jnp.arange(total_len)[None, :])
    logits = logits - slopes[:, None, None] * distance[None].astype(
        jnp.float32)
    mask = (seg[:, None] == seg[None, :])[None]
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("hqk,khd->qhd",
                      probs,
                      v.astype(jnp.float32),
                      precision=jax.lax.Precision.HIGHEST)


def _assert_close(out: np.ndarray, ref: np.ndarray) -> None:
    """Tolerances sized for TPU MXU precision.

    The kernel's f32 dots use default MXU precision (bf16 multiply passes),
    while the reference runs at Precision.HIGHEST — that alone yields ~1e-3
    to ~1e-2 elementwise noise after softmax (the no-alibi path shows the
    same). The tile-indexing logic is verified exactly (1e-6) in Pallas
    interpret mode. A real bias/indexing bug shifts logits by O(slope *
    block) and craters cosine similarity, so the cosine gate below is the
    functional check; the elementwise bound just caps outliers.
    """
    a, b = out.ravel(), ref.ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.9999, f"cosine similarity too low: {cos}"
    assert float(np.mean(np.abs(out - ref))) < 2e-3
    np.testing.assert_allclose(out, ref, atol=2e-2, rtol=2e-2)


class EncoderAlibiKernelTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ("multi_step_ragged", 512, 8, 64, (200, 180, 100), 128, 128),
        ("single_step_full", 512, 8, 64, (512, ), 128, 512),
        ("multi_step_six_heads", 384, 6, 64, (131, 97), 128, 128),
        ("long_context", 8192, 8, 64, (5000, 3000), 512, 512),
    )
    def test_alibi_matches_dense_reference(self, total_len, num_heads,
                                           head_dim, seq_lens, block_q,
                                           block_k):
        rng = np.random.default_rng(0)
        q = jnp.asarray(rng.standard_normal((total_len, num_heads, head_dim)),
                        dtype=jnp.float32)
        k = jnp.asarray(rng.standard_normal((total_len, num_heads, head_dim)),
                        dtype=jnp.float32)
        v = jnp.asarray(rng.standard_normal((total_len, num_heads, head_dim)),
                        dtype=jnp.float32)
        slopes = jnp.asarray(_alibi_slopes(num_heads), dtype=jnp.float32)
        sm_scale = head_dim**-0.5
        block_sizes = BlockSizes(block_q=block_q,
                                 block_k_major=block_k,
                                 block_k=block_k,
                                 block_b=1)

        out = encoder_only_flash_attention(q,
                                           k,
                                           v,
                                           jnp.asarray(seq_lens,
                                                       dtype=jnp.int32),
                                           slopes,
                                           causal=False,
                                           sm_scale=sm_scale,
                                           block_sizes=block_sizes)
        ref = _reference(q, k, v, seq_lens, slopes, sm_scale)

        num_real = sum(seq_lens)  # padding rows carry garbage by design
        _assert_close(np.asarray(out[:num_real]), np.asarray(ref[:num_real]))

    def test_no_alibi_unchanged(self):
        """slopes=None must reproduce plain segment-masked attention."""
        rng = np.random.default_rng(1)
        q = jnp.asarray(rng.standard_normal((256, 4, 64)), dtype=jnp.float32)
        k = jnp.asarray(rng.standard_normal((256, 4, 64)), dtype=jnp.float32)
        v = jnp.asarray(rng.standard_normal((256, 4, 64)), dtype=jnp.float32)
        block_sizes = BlockSizes(block_q=128,
                                 block_k_major=128,
                                 block_k=128,
                                 block_b=1)
        out = encoder_only_flash_attention(q,
                                           k,
                                           v,
                                           jnp.asarray([256], dtype=jnp.int32),
                                           None,
                                           causal=False,
                                           sm_scale=64**-0.5,
                                           block_sizes=block_sizes)
        ref = _reference(q, k, v, (256, ), jnp.zeros((4, )), 64**-0.5)
        _assert_close(np.asarray(out), np.asarray(ref))


if __name__ == "__main__":
    absltest.main()
