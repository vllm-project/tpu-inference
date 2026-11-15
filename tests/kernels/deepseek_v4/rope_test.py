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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from tpu_inference.kernels.experimental.deepseek_v4.rope import (qnorm_rope,
                                                                 rope,
                                                                 rope_quant)

# DeepSeek-V4-Flash: head_dim 512, qk_rope_head_dim 64, 64 attention heads.
HEAD_DIM = 512
ROTARY_DIM = 64
NUM_HEADS = 64


def rope_ref_impl(x, positions, cos_sin_cache, inverse=False):
    """Reference implementation of ``DeepseekV4ScalingRotaryEmbedding.forward_native``.

  GPT-J interleaved rotation over the *trailing* ``rotary_dim`` channels.
  """
    rotary_dim = cos_sin_cache.shape[1]
    orig_dtype = x.dtype
    xf = jnp.asarray(x).astype(jnp.float32)

    cos, sin = jnp.split(jnp.asarray(cos_sin_cache)[positions].astype(
        jnp.float32),
                         2,
                         axis=-1)
    cos = jnp.repeat(cos, 2, axis=-1)  # repeat_interleave
    sin = jnp.repeat(sin, 2, axis=-1)
    if inverse:
        sin = -sin
    if xf.ndim == 3:  # [num_tokens, num_heads, head_dim]: broadcast over heads
        cos = cos[:, None, :]
        sin = sin[:, None, :]

    x_pass, x_rot = xf[..., :-rotary_dim], xf[..., -rotary_dim:]
    # rotate_gptj: [-x1, x0, -x3, x2, ...]
    rotated = jnp.stack([-x_rot[..., 1::2], x_rot[..., 0::2]],
                        axis=-1).reshape(x_rot.shape)
    out_rot = x_rot * cos + rotated * sin
    return jnp.concatenate([x_pass, out_rot], axis=-1).astype(orig_dtype)


def quant_ref_impl(y, quant_dtype):
    """Reference implementation of ``quantization.quantize_tensor(..., axis=-1)``."""
    dtype_max = float(jnp.finfo(quant_dtype).max)
    yf = jnp.asarray(y).astype(jnp.float32)
    abs_max = jnp.max(jnp.abs(yf), axis=-1, keepdims=True)
    scale = abs_max / dtype_max
    # quantize_tensor leaves an all-zero row at NaN; the kernel returns zeros.
    scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
    return (yf * scale_inv).astype(quant_dtype), jnp.squeeze(scale, -1)


def qnorm_rope_ref_impl(x, positions, cos_sin_cache, eps, inverse=False):
    """Reference implementation of ``DeepseekV4Attention.qnorm_rope``.

  Per-head RMSNorm (no weight) over the whole ``head_dim``, in float32, then
  the same RoPE ``rope_ref_impl`` applies.
  """
    xf = jnp.asarray(x).astype(jnp.float32)
    rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return rope_ref_impl(xf * rms, positions, cos_sin_cache,
                         inverse=inverse).astype(x.dtype)


def assert_bits_equal(actual, expected):
    """Byte-for-byte equality, for dtypes NumPy will not compare directly."""
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint8),
        np.asarray(expected).view(np.uint8))


def make_cos_sin_cache(max_position, rotary_dim, seed=0):
    """A cos/sin cache with the layout DeepSeek-V4 builds: ``[cos | sin]``."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.0, 2.0 * np.pi,
                        (max_position, rotary_dim // 2)).astype(np.float32)
    return np.concatenate([np.cos(freqs), np.sin(freqs)],
                          axis=-1).astype(np.float32)


class RopeTest(parameterized.TestCase):

    def _run(
        self,
        shape,
        num_positions,
        inverse,
        dtype,
        seed=0,
        max_position=4096,
    ):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=shape).astype(dtype)
        positions = rng.integers(0, max_position,
                                 size=(num_positions, )).astype(np.int32)
        cos_sin_cache = make_cos_sin_cache(max_position, ROTARY_DIM, seed)

        expected = rope_ref_impl(x, positions, cos_sin_cache, inverse=inverse)
        actual = rope(
            jnp.asarray(x),
            jnp.asarray(positions),
            jnp.asarray(cos_sin_cache),
            inverse=inverse,
        )
        return np.asarray(actual), expected

    @parameterized.product(
        num_tokens=(8, 64, 256),
        inverse=(False, True),
    )
    def test_matches_reference_multi_head(self, num_tokens, inverse):
        """``qnorm_rope`` / ``_o_proj`` shape: [tokens, heads, head_dim]."""
        actual, expected = self._run((num_tokens, NUM_HEADS, HEAD_DIM),
                                     num_tokens, inverse, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    @parameterized.product(
        num_tokens=(8, 128),
        inverse=(False, True),
    )
    def test_matches_reference_single_head(self, num_tokens, inverse):
        """``kv_rope`` shape: [tokens, head_dim]."""
        actual, expected = self._run((num_tokens, HEAD_DIM), num_tokens,
                                     inverse, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_nope_channels_untouched(self):
        """Everything before the trailing ``rotary_dim`` must pass through."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=(64, 8, HEAD_DIM)).astype(np.float32)
        positions = rng.integers(0, 4096, size=(64, )).astype(np.int32)
        cos_sin_cache = make_cos_sin_cache(4096, ROTARY_DIM)

        actual = np.asarray(
            rope(jnp.asarray(x), jnp.asarray(positions),
                 jnp.asarray(cos_sin_cache)))
        np.testing.assert_array_equal(actual[..., :-ROTARY_DIM],
                                      x[..., :-ROTARY_DIM])


class RopeQuantTest(parameterized.TestCase):
    """RoPE fused with the indexer's per-row fp8 quantization."""

    @parameterized.product(
        num_tokens=(8, 64, 256),
        inverse=(False, True),
    )
    def test_matches_reference(self, num_tokens, inverse):
        """Indexer query shape: [tokens, index_n_heads, index_head_dim]."""
        indexer_head_dim = 128
        indexer_num_heads = 64
        shape = (num_tokens, indexer_num_heads, indexer_head_dim)
        seed = 0
        rng = np.random.default_rng(seed)
        x = rng.normal(size=shape).astype(jnp.bfloat16)
        positions = rng.integers(0, 4096, size=(shape[0], )).astype(np.int32)
        cos_sin_cache = make_cos_sin_cache(4096, ROTARY_DIM, seed)

        # The kernel quantizes the float32 rotation directly -- it never rounds to
        # ``x.dtype`` in between -- so neither does the reference.
        q_expected, scales_expected = quant_ref_impl(
            rope_ref_impl(x.astype(np.float32),
                          positions,
                          cos_sin_cache,
                          inverse=inverse),
            jnp.float8_e4m3fn,
        )
        q, scales = rope_quant(
            jnp.asarray(x),
            jnp.asarray(positions),
            jnp.asarray(cos_sin_cache),
            inverse=inverse,
            quant_dtype=jnp.float8_e4m3fn,
        )
        np.testing.assert_allclose(np.asarray(scales),
                                   np.asarray(scales_expected),
                                   rtol=1e-6,
                                   atol=1e-6)
        assert_bits_equal(q, q_expected)


class QNormRopeTest(parameterized.TestCase):
    """RoPE fused with the per-head RMSNorm that precedes it on ``q``."""

    @parameterized.product(num_tokens=(8, 64, 256), )
    def test_matches_reference_multi_head(self, num_tokens):
        """``qnorm_rope`` shape: [tokens, heads, head_dim]."""
        num_heads = 128
        head_dim = 512
        seed = 0
        shape = (num_tokens, num_heads, head_dim)
        max_position = 4096
        dtype = jnp.bfloat16
        rng = np.random.default_rng(seed)
        x = rng.normal(size=shape).astype(np.float32)
        positions = rng.integers(0, max_position,
                                 size=(shape[0], )).astype(np.int32)
        cos_sin_cache = make_cos_sin_cache(max_position, ROTARY_DIM, seed)
        eps = 1e-6
        x = jnp.asarray(x, dtype=dtype)
        positions = jnp.asarray(positions)
        cos_sin_cache = jnp.asarray(cos_sin_cache)

        expected = qnorm_rope_ref_impl(x,
                                       positions,
                                       cos_sin_cache,
                                       eps,
                                       inverse=False)
        actual = qnorm_rope(
            jnp.asarray(x),
            jnp.asarray(positions),
            jnp.asarray(cos_sin_cache),
            eps=eps,
            inverse=False,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "highest")
    absltest.main()
