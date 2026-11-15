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
import torch
import vllm.model_executor.kernels.mhc as mhc_kernels
from absl.testing import absltest, parameterized

from tpu_inference.kernels.experimental.deepseek_v4.mhc import utils

HC_MULT = 4
# Values from the official DeepSeek-V4 config.json: rms_norm_eps=1e-6,
# hc_eps=1e-6 (feeds BOTH pre_eps and sinkhorn_eps), hc_sinkhorn_iters=20;
# hc_post_alpha=2.0 is hardcoded in the model (deepseek_v4.py).
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0

# f32 outputs: absorb summation-order differences of the (D=hc_mult*H)-long
# f32 accumulations between torch and XLA.
F32_RTOL, F32_ATOL = 1e-4, 1e-5
# bf16 outputs: ~1-2 ulp at magnitude O(1).
BF16_RTOL, BF16_ATOL = 2e-2, 2e-2
# Fused-path f32 gate outputs: the seam kernel's recombine and the
# reference's einsum can round near-boundary stream values to different
# adjacent bf16 values before the mix GEMM, so gates carry ~1 bf16-ulp of
# extra input noise (worst at small H, where each element weighs more).
FUSED_F32_RTOL, FUSED_F32_ATOL = 1e-3, 1e-5
# At the tiny test H=256 each flipped element weighs 28x more than at the
# real H=7168, and the flip pattern is backend-dependent (measured up to
# 2.4e-3 on the TPU backend); the small-shape fused tests use this
# looser bound, the real-shape tests keep FUSED_F32_RTOL.
FUSED_SMALL_H_RTOL = 5e-3


def _ref_mhc_pre_mixes(x2d: jax.Array,
                       fn: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = x2d.astype(jnp.float32)
    mixes = jax.lax.dot_general(
        x,
        fn,
        dimension_numbers=(((1, ), (1, )), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
    )
    sqrsum = jnp.sum(x * x, axis=-1, keepdims=True)
    return mixes, sqrsum


def _ref_mhc_pre_collapse(
    pre_mix: jax.Array,
    x2d: jax.Array,
    hc_mult: int,
    hidden_size: int,
) -> jax.Array:
    out = pre_mix[:, 0:1] * x2d[:, :hidden_size].astype(jnp.float32)
    for i in range(1, hc_mult):
        out = out + (pre_mix[:, i:i + 1] *
                     x2d[:, i * hidden_size:
                         (i + 1) * hidden_size].astype(jnp.float32))
    return out.astype(jnp.bfloat16)


def _ref_mhc_pre(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert residual.dtype == jnp.bfloat16
    assert fn.dtype == jnp.float32

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    x2d = residual.reshape(-1, hc_mult * hidden_size)

    mixes, sqrsum = _ref_mhc_pre_mixes(x2d, fn)
    pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
        mixes, sqrsum, hc_mult, hidden_size, hc_scale, hc_base, rms_eps,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
    layer_input = _ref_mhc_pre_collapse(pre_mix, x2d, hc_mult, hidden_size)

    return (post_mix.reshape(*outer_shape, hc_mult, 1),
            comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
            layer_input.reshape(*outer_shape, hidden_size))


def _ref_mhc_post(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    *,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:
    mixed_residual = jnp.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.astype(jnp.float32),
        residual.astype(jnp.float32),
        precision=precision,
    )
    post_term = (post_layer_mix.astype(jnp.float32) *
                 x[..., None, :].astype(jnp.float32))
    return (mixed_residual + post_term).astype(residual.dtype)


def _ref_mhc_fused_post_pre(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    residual_cur = _ref_mhc_post(x,
                                 residual,
                                 post_layer_mix,
                                 comb_res_mix,
                                 precision=jax.lax.Precision.HIGHEST)
    post_mix_cur, comb_mix_cur, layer_input_cur = _ref_mhc_pre(
        residual_cur, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _to_np(x) -> np.ndarray:
    """jax array (any dtype incl. bf16) -> float32 numpy."""
    return np.asarray(jnp.asarray(x).astype(jnp.float32))


def _make_pre_inputs(rng, num_tokens, hidden_size):
    hc_mult3 = 2 * HC_MULT + HC_MULT * HC_MULT
    residual = rng.standard_normal((num_tokens, HC_MULT, hidden_size),
                                   dtype=np.float32)
    fn = (rng.standard_normal(
        (hc_mult3, HC_MULT * hidden_size), dtype=np.float32) * 0.02)
    hc_scale = (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32)
    hc_base = (0.5 * rng.standard_normal(hc_mult3)).astype(np.float32)
    return residual, fn, hc_scale, hc_base


class MhcPreParityTest(parameterized.TestCase):

    @parameterized.product(
        num_tokens=[1, 8, 17, 128],
        hidden_size=[256, 7168],
        sinkhorn_repeat=[1, 2, 20],
    )
    def test_matches_torch_reference(self, num_tokens, hidden_size,
                                     sinkhorn_repeat):

        rng = np.random.default_rng(0)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(
            rng, num_tokens, hidden_size)

        want_post, want_comb, want_layer = mhc_kernels.mhc_pre_torch(
            torch.from_numpy(residual).to(torch.bfloat16),
            torch.from_numpy(fn),
            torch.from_numpy(hc_scale),
            torch.from_numpy(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            sinkhorn_repeat,
        )

        got_post, got_comb, got_layer = _ref_mhc_pre(
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            sinkhorn_repeat,
        )

        self.assertEqual(got_post.shape, (num_tokens, HC_MULT, 1))
        self.assertEqual(got_comb.shape, (num_tokens, HC_MULT, HC_MULT))
        self.assertEqual(got_layer.shape, (num_tokens, hidden_size))
        self.assertEqual(got_layer.dtype, jnp.bfloat16)

        np.testing.assert_allclose(_to_np(got_post),
                                   want_post.float().numpy(),
                                   rtol=F32_RTOL,
                                   atol=F32_ATOL)
        np.testing.assert_allclose(_to_np(got_comb),
                                   want_comb.float().numpy(),
                                   rtol=F32_RTOL,
                                   atol=F32_ATOL)
        np.testing.assert_allclose(_to_np(got_layer),
                                   want_layer.float().numpy(),
                                   rtol=BF16_RTOL,
                                   atol=BF16_ATOL)

    def test_padded_zero_rows_are_finite(self):
        """vLLM pads token buckets with all-zero rows; they must not NaN."""

        rng = np.random.default_rng(1)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(rng, 8, 256)
        residual[4:] = 0.0  # padded tail

        got_post, got_comb, got_layer = _ref_mhc_pre(
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(fn),
            jnp.asarray(hc_scale),
            jnp.asarray(hc_base),
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            2,
        )
        for out in (got_post, got_comb, got_layer):
            self.assertTrue(bool(jnp.isfinite(out.astype(jnp.float32)).all()))


class MhcPostParityTest(parameterized.TestCase):

    @parameterized.product(num_tokens=[1, 17, 128], hidden_size=[256, 7168])
    def test_matches_torch_reference(self, num_tokens, hidden_size):

        rng = np.random.default_rng(2)
        x = rng.standard_normal((num_tokens, hidden_size), dtype=np.float32)
        residual = rng.standard_normal((num_tokens, HC_MULT, hidden_size),
                                       dtype=np.float32)
        # Mimic mhc_pre outputs: positive doubly-normalized-ish comb mix and
        # sigmoid-scaled post mix, both f32.
        post_mix = (2.0 / (1.0 + np.exp(-rng.standard_normal(
            (num_tokens, HC_MULT, 1))))).astype(np.float32)
        comb_raw = np.abs(rng.standard_normal(
            (num_tokens, HC_MULT, HC_MULT))).astype(np.float32) + 0.01
        comb_mix = comb_raw / comb_raw.sum(axis=-1, keepdims=True)

        want = mhc_kernels.mhc_post_torch(
            torch.from_numpy(x).to(torch.bfloat16),
            torch.from_numpy(residual).to(torch.bfloat16),
            torch.from_numpy(post_mix),
            torch.from_numpy(comb_mix),
        )

        got = _ref_mhc_post(
            jnp.asarray(x).astype(jnp.bfloat16),
            jnp.asarray(residual).astype(jnp.bfloat16),
            jnp.asarray(post_mix),
            jnp.asarray(comb_mix),
        )

        self.assertEqual(got.shape, (num_tokens, HC_MULT, hidden_size))
        self.assertEqual(got.dtype, jnp.bfloat16)
        np.testing.assert_allclose(_to_np(got),
                                   want.float().numpy(),
                                   rtol=BF16_RTOL,
                                   atol=BF16_ATOL)


def _make_post_inputs(rng, num_tokens, hidden_size):
    x = rng.standard_normal((num_tokens, hidden_size), dtype=np.float32)
    residual = rng.standard_normal((num_tokens, HC_MULT, hidden_size),
                                   dtype=np.float32)
    post_mix = (
        2.0 /
        (1.0 + np.exp(-rng.standard_normal((num_tokens, HC_MULT, 1))))).astype(
            np.float32)
    comb_raw = np.abs(rng.standard_normal(
        (num_tokens, HC_MULT, HC_MULT))).astype(np.float32) + 0.01
    comb_mix = comb_raw / comb_raw.sum(axis=-1, keepdims=True)
    return (jnp.asarray(x).astype(jnp.bfloat16),
            jnp.asarray(residual).astype(jnp.bfloat16), jnp.asarray(post_mix),
            jnp.asarray(comb_mix))


class MhcPostPallasSmallShapeTest(parameterized.TestCase):
    """Post kernel vs the jnp oracle at small shapes and ragged token
    counts (block indexing and padding); needs TPU hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(num_tokens=[1, 17, 64, 200], hidden_size=[256])
    def test_matches_reference(self, num_tokens, hidden_size):
        from tpu_inference.kernels.experimental.deepseek_v4.mhc import \
            post_kernel

        rng = np.random.default_rng(7)
        args = _make_post_inputs(rng, num_tokens, hidden_size)

        want = _ref_mhc_post(*args)
        got = post_kernel.mhc_post(*args, token_block_size=64)

        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, jnp.bfloat16)
        np.testing.assert_allclose(_to_np(got),
                                   _to_np(want),
                                   rtol=BF16_RTOL,
                                   atol=BF16_ATOL)


class MhcPostPallasTpuTest(parameterized.TestCase):
    """Compiled (Mosaic) post kernel vs the jnp reference. Needs TPU."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(
        num_tokens=[16, 17, 128, 2048],
        token_block_size=[32, 64],
    )
    def test_matches_reference(self, num_tokens, token_block_size):
        from tpu_inference.kernels.experimental.deepseek_v4.mhc import \
            post_kernel

        rng = np.random.default_rng(8)
        args = _make_post_inputs(rng, num_tokens, 7168)

        want = _ref_mhc_post(*args)
        got = post_kernel.mhc_post(*args, token_block_size=token_block_size)

        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(_to_np(got),
                                   _to_np(want),
                                   rtol=BF16_RTOL,
                                   atol=BF16_ATOL)


def _make_fused_inputs(rng, num_tokens, hidden_size):
    """x/residual/gates (post-style) plus fn/scale/base (pre-style)."""
    x, residual, post_mix, comb_mix = _make_post_inputs(
        rng, num_tokens, hidden_size)
    _, fn, hc_scale, hc_base = _make_pre_inputs(rng, num_tokens, hidden_size)
    return (x, residual, post_mix, comb_mix, jnp.asarray(fn),
            jnp.asarray(hc_scale), jnp.asarray(hc_base))


def _fused_args(rng, num_tokens, hidden_size, sinkhorn_repeat=20):
    x, res, plm, crm, fn, sc, hb = _make_fused_inputs(rng, num_tokens,
                                                      hidden_size)
    return (x, res, plm, crm, fn, sc, hb, RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE, sinkhorn_repeat)


class MhcFusedPostPrePallasSmallShapeTest(parameterized.TestCase):
    """Fused seam kernel vs the sequential oracle at small shapes and
    ragged token counts; needs TPU hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(num_tokens=[1, 17, 64, 200], hidden_size=[256])
    def test_matches_sequential_reference(self, num_tokens, hidden_size):
        from tpu_inference.kernels.experimental.deepseek_v4.mhc import \
            fused_post_pre_kernel

        args = _fused_args(np.random.default_rng(9), num_tokens, hidden_size)
        fused_fn = fused_post_pre_kernel.mhc_fused_post_pre

        want_res, want_post, want_comb, want_layer = (_ref_mhc_fused_post_pre(
            *args))
        got_res, got_post, got_comb, got_layer = (fused_fn(
            *args, token_block_size=64))

        self.assertEqual(got_res.shape, want_res.shape)
        self.assertEqual(got_res.dtype, jnp.bfloat16)
        self.assertEqual(got_layer.dtype, jnp.bfloat16)
        np.testing.assert_allclose(_to_np(got_res),
                                   _to_np(want_res),
                                   rtol=BF16_RTOL,
                                   atol=BF16_ATOL)
        np.testing.assert_allclose(_to_np(got_post),
                                   _to_np(want_post),
                                   rtol=FUSED_SMALL_H_RTOL,
                                   atol=FUSED_F32_ATOL)
        np.testing.assert_allclose(_to_np(got_comb),
                                   _to_np(want_comb),
                                   rtol=FUSED_SMALL_H_RTOL,
                                   atol=FUSED_F32_ATOL)
        np.testing.assert_allclose(_to_np(got_layer),
                                   _to_np(want_layer),
                                   rtol=BF16_RTOL,
                                   atol=BF16_ATOL)


class MhcFusedPostPrePallasTpuTest(parameterized.TestCase):
    """Compiled (Mosaic) fused seam kernel vs the sequential reference.
    Needs TPU hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(
        num_tokens=[16, 17, 128, 2048],
        token_block_size=[32, 64],
    )
    def test_matches_sequential_reference(self, num_tokens, token_block_size):
        from tpu_inference.kernels.experimental.deepseek_v4.mhc import \
            fused_post_pre_kernel

        args = _fused_args(np.random.default_rng(10), num_tokens, 7168)
        fused_fn = fused_post_pre_kernel.mhc_fused_post_pre

        want = _ref_mhc_fused_post_pre(*args)
        got = fused_fn(*args, token_block_size=token_block_size)

        for g, w, tol in zip(got, want, ("bf16", "f32", "f32", "bf16")):
            rtol, atol = ((BF16_RTOL, BF16_ATOL) if tol == "bf16" else
                          (FUSED_F32_RTOL, FUSED_F32_ATOL))
            np.testing.assert_allclose(_to_np(g),
                                       _to_np(w),
                                       rtol=rtol,
                                       atol=atol)


class MhcPreMixesPallasSmallShapeTest(parameterized.TestCase):
    """Mix-GEMM kernel vs the jnp oracle at small shapes and ragged
    token counts, validating block indexing and padding; needs TPU
    hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(num_tokens=[1, 17, 64, 200], hidden_size=[256])
    def test_matches_reference(self, num_tokens, hidden_size):
        from tpu_inference.kernels.experimental.deepseek_v4.mhc import \
            pre_kernel

        rng = np.random.default_rng(4)
        residual, fn, _, _ = _make_pre_inputs(rng, num_tokens, hidden_size)
        x2d = (jnp.asarray(residual).astype(jnp.bfloat16).reshape(
            num_tokens, HC_MULT * hidden_size))
        fn_j = jnp.asarray(fn)

        want_mixes, want_sqrsum = _ref_mhc_pre_mixes(x2d, fn_j)
        got_mixes, got_sqrsum = pre_kernel.mhc_pre_mixes(
            x2d,
            fn_j,
            token_block_size=64,
        )

        self.assertEqual(got_mixes.shape, want_mixes.shape)
        self.assertEqual(got_sqrsum.shape, want_sqrsum.shape)
        np.testing.assert_allclose(_to_np(got_mixes),
                                   _to_np(want_mixes),
                                   rtol=F32_RTOL,
                                   atol=F32_ATOL)
        np.testing.assert_allclose(_to_np(got_sqrsum),
                                   _to_np(want_sqrsum),
                                   rtol=F32_RTOL,
                                   atol=F32_ATOL)


class MhcPrePallasTpuTest(parameterized.TestCase):
    """Compiled (Mosaic) kernel vs the jnp reference. Needs TPU hardware."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "tpu":
            self.skipTest("Pallas TPU kernel needs TPU hardware")

    @parameterized.product(
        num_tokens=[16, 17, 128, 2048],
        token_block_size=[64, 128],
    )
    def test_mixes_matches_reference(self, num_tokens, token_block_size):
        from tpu_inference.kernels.experimental.deepseek_v4.mhc import \
            pre_kernel

        hidden_size = 7168  # DeepSeek-V4 shape; D = 28672
        rng = np.random.default_rng(5)
        residual, fn, _, _ = _make_pre_inputs(rng, num_tokens, hidden_size)
        x2d = (jnp.asarray(residual).astype(jnp.bfloat16).reshape(
            num_tokens, HC_MULT * hidden_size))
        fn_j = jnp.asarray(fn)

        want_mixes, want_sqrsum = _ref_mhc_pre_mixes(x2d, fn_j)
        got_mixes, got_sqrsum = pre_kernel.mhc_pre_mixes(
            x2d, fn_j, token_block_size=token_block_size)

        np.testing.assert_allclose(_to_np(got_mixes),
                                   _to_np(want_mixes),
                                   rtol=F32_RTOL,
                                   atol=F32_ATOL)
        np.testing.assert_allclose(_to_np(got_sqrsum),
                                   _to_np(want_sqrsum),
                                   rtol=F32_RTOL,
                                   atol=F32_ATOL)

    @parameterized.product(num_tokens=[16, 128], sinkhorn_repeat=[1, 20])
    def test_full_op_matches_reference(self, num_tokens, sinkhorn_repeat):
        from tpu_inference.kernels.experimental.deepseek_v4.mhc import \
            pre_kernel

        hidden_size = 7168
        rng = np.random.default_rng(6)
        residual, fn, hc_scale, hc_base = _make_pre_inputs(
            rng, num_tokens, hidden_size)
        args = (jnp.asarray(residual).astype(jnp.bfloat16), jnp.asarray(fn),
                jnp.asarray(hc_scale), jnp.asarray(hc_base), RMS_EPS,
                HC_PRE_EPS, HC_SINKHORN_EPS, HC_POST_MULT_VALUE,
                sinkhorn_repeat)

        want_post, want_comb, want_layer = _ref_mhc_pre(*args)
        got_post, got_comb, got_layer = pre_kernel.mhc_pre(*args)

        self.assertEqual(got_layer.dtype, jnp.bfloat16)
        np.testing.assert_allclose(_to_np(got_post),
                                   _to_np(want_post),
                                   rtol=F32_RTOL,
                                   atol=F32_ATOL)
        np.testing.assert_allclose(_to_np(got_comb),
                                   _to_np(want_comb),
                                   rtol=F32_RTOL,
                                   atol=F32_ATOL)
        np.testing.assert_allclose(_to_np(got_layer),
                                   _to_np(want_layer),
                                   rtol=BF16_RTOL,
                                   atol=BF16_ATOL)


if __name__ == "__main__":
    absltest.main()
