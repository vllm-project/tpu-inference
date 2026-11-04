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

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from flax import nnx
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.core import dot_general, qarray
from qwix._src.providers import ptq

from tpu_inference.kernels.fused_rmsnorm_fp8 import kernel as fused_rmsnorm_fp8


def _has_tpu() -> bool:
    return bool(jax.devices()) and all(d.platform == "tpu"
                                       for d in jax.devices())


class FusedRmsnormFp8KernelTest(parameterized.TestCase):

    @parameterized.named_parameters(
        {
            "testcase_name": "m_1024_pref_512",
            "m": 1024,
            "pref": 512,
            "expected": 512
        },
        {
            "testcase_name": "m_512_pref_512",
            "m": 512,
            "pref": 512,
            "expected": 512
        },
        {
            "testcase_name": "m_384_pref_512",
            "m": 384,
            "pref": 512,
            "expected": 128
        },
        {
            "testcase_name": "m_64_pref_256",
            "m": 64,
            "pref": 256,
            "expected": 64
        },
        {
            "testcase_name": "m_8_pref_512",
            "m": 8,
            "pref": 512,
            "expected": 8
        },
        {
            "testcase_name": "m_4_too_small",
            "m": 4,
            "pref": 512,
            "expected": None
        },
        {
            "testcase_name": "m_100_unaligned",
            "m": 100,
            "pref": 512,
            "expected": None
        },
    )
    def test_select_block_m(self, m, pref, expected):
        self.assertEqual(fused_rmsnorm_fp8.select_block_m(m, pref), expected)

    @parameterized.named_parameters(
        {
            "testcase_name": "1d_per_token",
            "shape": (128, ),
            "block_k": None
        },
        {
            "testcase_name": "2d_per_token",
            "shape": (16, 128),
            "block_k": None
        },
        {
            "testcase_name": "3d_per_token",
            "shape": (2, 8, 256),
            "block_k": None
        },
        {
            "testcase_name": "2d_subchannel",
            "shape": (16, 256),
            "block_k": 128
        },
    )
    def test_xla_shapes_and_qarray_contract(self, shape, block_k):
        k = shape[-1]
        x = jax.random.normal(jax.random.key(0), shape, dtype=jnp.bfloat16)
        scale = jax.random.uniform(jax.random.key(1), (k, ),
                                   dtype=jnp.bfloat16,
                                   minval=0.5,
                                   maxval=1.5)
        q_out = fused_rmsnorm_fp8.fused_rmsnorm_fp8(x,
                                                    scale,
                                                    block_k=block_k,
                                                    implementation="xla")
        self.assertIsInstance(q_out, qarray.QArray)
        self.assertEqual(q_out.qvalue.dtype, jnp.float8_e4m3fn)
        self.assertEqual(q_out.scale.dtype, jnp.bfloat16)
        self.assertEqual(q_out.qvalue.shape, shape)
        expected_num_scales = 1 if block_k is None else k // block_k
        self.assertEqual(q_out.scale.shape, (*shape[:-1], expected_num_scales))

    def test_xla_residual_and_return_residual(self):
        x = jax.random.normal(jax.random.key(0), (16, 128), dtype=jnp.bfloat16)
        res = jax.random.normal(jax.random.key(1), (16, 128),
                                dtype=jnp.bfloat16)
        scale = jnp.ones((128, ), dtype=jnp.bfloat16)

        q_out, updated_res = fused_rmsnorm_fp8.fused_rmsnorm_fp8(
            x, scale, residual=res, return_residual=True, implementation="xla")
        self.assertIsInstance(q_out, qarray.QArray)
        np.testing.assert_array_equal(
            np.asarray(updated_res), np.asarray(
                (x + res).astype(jnp.bfloat16)))

        q_sum = fused_rmsnorm_fp8.fused_rmsnorm_fp8(
            (x + res).astype(jnp.bfloat16), scale, implementation="xla")
        np.testing.assert_array_equal(
            np.asarray(q_out.qvalue).view(np.uint8),
            np.asarray(q_sum.qvalue).view(np.uint8),
        )
        np.testing.assert_array_equal(np.asarray(q_out.scale),
                                      np.asarray(q_sum.scale))

    def test_ptq_provider_consumes_prequantized_qarray(self):
        layer = nnx.Linear(128,
                           64,
                           use_bias=False,
                           param_dtype=jnp.bfloat16,
                           rngs=nnx.Rngs(0))
        q_rules = [
            qconfig.QuantizationRule(
                module_path=".*",
                weight_qtype=jnp.float8_e4m3fn,
                act_qtype=jnp.float8_e4m3fn,
            ),
        ]
        x = jax.random.normal(jax.random.key(0), (16, 128), dtype=jnp.bfloat16)
        scale = jax.random.uniform(jax.random.key(1), (128, ),
                                   dtype=jnp.bfloat16,
                                   minval=0.5,
                                   maxval=1.5)
        ptq_layer = qwix_model.quantize_model(layer, ptq.PtqProvider(q_rules),
                                              x)

        q_act = fused_rmsnorm_fp8.fused_rmsnorm_fp8(x,
                                                    scale,
                                                    eps=1e-6,
                                                    implementation="xla")
        expected = dot_general.dot_general(q_act, ptq_layer.kernel.array,
                                           (((1, ), (0, )), ((), ())))
        with self.enter_context(
                mock.patch.object(
                    qarray,
                    "quantize",
                    wraps=qarray.quantize,
                )) as mock_quantize:
            actual = ptq_layer(q_act)
            mock_quantize.assert_not_called()

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_validation_errors(self):
        x = jnp.ones((16, 128), dtype=jnp.bfloat16)
        scale = jnp.ones((128, ), dtype=jnp.bfloat16)
        with self.assertRaisesRegex(ValueError,
                                    "return_residual=True requires"):
            fused_rmsnorm_fp8.fused_rmsnorm_fp8(x, scale, return_residual=True)
        with self.assertRaisesRegex(ValueError, "scale shape"):
            fused_rmsnorm_fp8.fused_rmsnorm_fp8(
                x, jnp.ones((64, ), dtype=jnp.bfloat16))
        with self.assertRaisesRegex(ValueError, "residual shape"):
            fused_rmsnorm_fp8.fused_rmsnorm_fp8(x,
                                                scale,
                                                residual=jnp.ones(
                                                    (8, 128),
                                                    dtype=jnp.bfloat16))
        bad_kwargs = {"implementation": "invalid"}
        with self.assertRaisesRegex(ValueError, "implementation must be"):
            fused_rmsnorm_fp8.fused_rmsnorm_fp8(x, scale, **bad_kwargs)
        with self.assertRaisesRegex(ValueError, "not divisible by any valid"):
            fused_rmsnorm_fp8.fused_rmsnorm_fp8(
                jnp.ones((4, 128), dtype=jnp.bfloat16),
                scale,
                implementation="pallas",
            )

    @parameterized.named_parameters(
        {
            "testcase_name": "qwen_1k_per_token",
            "m": 1024,
            "k": 5120,
            "block_m": 256,
            "block_k": None,
            "has_residual": False,
            "return_residual": False,
        },
        {
            "testcase_name": "qwen_1k_residual_add",
            "m": 1024,
            "k": 5120,
            "block_m": 256,
            "block_k": None,
            "has_residual": True,
            "return_residual": True,
        },
        {
            "testcase_name": "qwen_4k_bm512_per_token",
            "m": 4096,
            "k": 5120,
            "block_m": 512,
            "block_k": None,
            "has_residual": False,
            "return_residual": False,
        },
        {
            "testcase_name": "qwen_4k_bm512_residual",
            "m": 4096,
            "k": 5120,
            "block_m": 512,
            "block_k": None,
            "has_residual": True,
            "return_residual": True,
        },
        {
            "testcase_name": "qwen_1k_subchannel_bk128",
            "m": 1024,
            "k": 5120,
            "block_m": 256,
            "block_k": 128,
            "has_residual": False,
            "return_residual": False,
        },
        {
            "testcase_name": "qwen_1k_subchannel_bk512",
            "m": 1024,
            "k": 5120,
            "block_m": 256,
            "block_k": 512,
            "has_residual": True,
            "return_residual": True,
        },
    )
    def test_pallas_matches_reference_on_tpu(self, m, k, block_m, block_k,
                                             has_residual, return_residual):
        if not _has_tpu():
            self.skipTest("Pallas TPU kernel test requires TPU backend.")

        k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
        x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16) * 2.5
        scale = jax.random.uniform(k2, (k, ),
                                   dtype=jnp.bfloat16,
                                   minval=0.5,
                                   maxval=1.5)
        residual = (jax.random.normal(k3, (m, k), dtype=jnp.bfloat16) *
                    1.5 if has_residual else None)

        ref_outs = jax.jit(lambda x_in, s_in, r_in: fused_rmsnorm_fp8.
                           fused_rmsnorm_fp8_reference(
                               x_in,
                               s_in,
                               r_in,
                               block_k=block_k,
                               return_residual=return_residual,
                           ))(x, scale, residual)

        pallas_outs = jax.jit(lambda x_in, s_in, r_in: fused_rmsnorm_fp8.
                              fused_rmsnorm_fp8_pallas(
                                  x_in,
                                  s_in,
                                  r_in,
                                  block_m=block_m,
                                  block_k=block_k,
                                  return_residual=return_residual,
                              ))(x, scale, residual)

        ref_q_bytes = np.asarray(ref_outs[0]).view(np.uint8)
        pal_q_bytes = np.asarray(pallas_outs[0]).view(np.uint8)
        exact_match_ratio = float(np.mean(ref_q_bytes == pal_q_bytes))
        self.assertGreaterEqual(exact_match_ratio, 0.999)

        max_scale_diff = float(
            np.max(
                np.abs(
                    np.asarray(ref_outs[1], dtype=np.float32) -
                    np.asarray(pallas_outs[1], dtype=np.float32))))
        self.assertLessEqual(max_scale_diff, 1e-4)

        if return_residual:
            np.testing.assert_array_equal(np.asarray(ref_outs[2]),
                                          np.asarray(pallas_outs[2]))


if __name__ == "__main__":
    absltest.main()
