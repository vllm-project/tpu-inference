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

import dataclasses
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

_MIB = 1024 * 1024
_FP8_RULE = qconfig.QuantizationRule(
    module_path=".*",
    weight_qtype=jnp.float8_e4m3fn,
    act_qtype=jnp.float8_e4m3fn,
)


def _has_tpu() -> bool:
    return bool(jax.devices()) and all(d.platform == "tpu"
                                       for d in jax.devices())


def _random_inputs(m, k, has_residual=False):
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16) * 2.5
    scale = jax.random.uniform(k2, (k, ),
                               dtype=jnp.bfloat16,
                               minval=0.5,
                               maxval=1.5)
    residual = (jax.random.normal(k3, (m, k), dtype=jnp.bfloat16) *
                1.5 if has_residual else None)
    return x, scale, residual


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
            "testcase_name": "m_32_smallest_tile",
            "m": 32,
            "pref": 512,
            "expected": 32
        },
        {
            "testcase_name": "m_16_single_full_block",
            "m": 16,
            "pref": 512,
            "expected": 16
        },
        {
            "testcase_name": "m_8_single_full_block",
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
            "testcase_name": "m_12_unaligned",
            "m": 12,
            "pref": 512,
            "expected": None
        },
        {
            "testcase_name": "m_48_not_fp8_tile_aligned",
            "m": 48,
            "pref": 512,
            "expected": None
        },
        {
            "testcase_name": "m_100_unaligned",
            "m": 100,
            "pref": 512,
            "expected": None
        },
        {
            "testcase_name": "m_4096_vmem_cap_300",
            "m": 4096,
            "pref": 512,
            "max_block_m": 300,
            "expected": 256
        },
        {
            "testcase_name": "m_4096_vmem_cap_below_smallest_tile",
            "m": 4096,
            "pref": 512,
            "max_block_m": 16,
            "expected": None
        },
        {
            "testcase_name": "m_16_vmem_cap_below_single_block",
            "m": 16,
            "pref": 512,
            "max_block_m": 8,
            "expected": None
        },
    )
    def test_select_block_m(self, m, pref, expected, max_block_m=None):
        self.assertEqual(
            fused_rmsnorm_fp8.select_block_m(m, pref, max_block_m=max_block_m),
            expected)

    @parameterized.named_parameters(
        {
            "testcase_name": "64mib_vmem",
            "vmem_capacity_bytes": 64 * _MIB,
            "has_residual": False,
            "expected": 512
        },
        {
            "testcase_name": "64mib_vmem_residual",
            "vmem_capacity_bytes": 64 * _MIB,
            "has_residual": True,
            "expected": 256
        },
        {
            "testcase_name": "128mib_vmem_residual",
            "vmem_capacity_bytes": 128 * _MIB,
            "has_residual": True,
            "expected": 512
        },
    )
    def test_block_m_respects_vmem_budget(self, vmem_capacity_bytes,
                                          has_residual, expected):
        max_block_m = fused_rmsnorm_fp8.max_block_m_for_vmem(
            5120,
            x_itemsize=2,
            residual_itemsize=2 if has_residual else None,
            return_residual=has_residual,
            vmem_capacity_bytes=vmem_capacity_bytes,
        )
        self.assertEqual(
            fused_rmsnorm_fp8.select_block_m(4096,
                                             512,
                                             max_block_m=max_block_m),
            expected)

    def test_vmem_budget_defaults_to_tpu_vmem_capacity(self):
        tpu_info = mock.Mock(vmem_capacity_bytes=64 * _MIB)
        with mock.patch.object(fused_rmsnorm_fp8.pltpu,
                               "get_tpu_info",
                               return_value=tpu_info):
            self.assertEqual(
                fused_rmsnorm_fp8.max_block_m_for_vmem(5120, x_itemsize=2),
                fused_rmsnorm_fp8.max_block_m_for_vmem(5120,
                                                       x_itemsize=2,
                                                       vmem_capacity_bytes=64 *
                                                       _MIB))

    @parameterized.named_parameters(
        {
            "testcase_name": "fp8_dynamic_per_token",
            "rule_kwargs": {},
            "expected": True
        },
        {
            "testcase_name": "fp8_qtype_strings",
            "rule_kwargs": {
                "weight_qtype": "float8_e4m3fn",
                "act_qtype": "float8_e4m3fn"
            },
            "expected": True
        },
        {
            "testcase_name": "weight_only",
            "rule_kwargs": {
                "act_qtype": None
            },
            "expected": False
        },
        {
            "testcase_name": "activation_only",
            "rule_kwargs": {
                "weight_qtype": None
            },
            "expected": False
        },
        {
            "testcase_name": "int8_activations",
            "rule_kwargs": {
                "weight_qtype": jnp.int8,
                "act_qtype": jnp.int8
            },
            "expected": False
        },
        {
            "testcase_name": "static_scale",
            "rule_kwargs": {
                "act_static_scale": True
            },
            "expected": False
        },
        {
            "testcase_name": "subchannel_tiling",
            "rule_kwargs": {
                "tile_size": 128
            },
            "expected": False
        },
        {
            "testcase_name": "clipped_absmax",
            "rule_kwargs": {
                "act_calibration_method": "absmax,0.8"
            },
            "expected": False
        },
        {
            "testcase_name": "minmax",
            "rule_kwargs": {
                "act_calibration_method": "minmax"
            },
            "expected": False
        },
    )
    def test_is_fusable_act_quant_rule(self, rule_kwargs, expected):
        rule = dataclasses.replace(_FP8_RULE, **rule_kwargs)
        self.assertEqual(fused_rmsnorm_fp8.is_fusable_act_quant_rule(rule),
                         expected)
        self.assertFalse(fused_rmsnorm_fp8.is_fusable_act_quant_rule(None))

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

    @parameterized.named_parameters(
        {
            "testcase_name": "per_token",
            "block_k": None
        },
        {
            "testcase_name": "subchannel_128",
            "block_k": 128
        },
    )
    def test_bf16_scale_rounding_matches_stored_scale(self, block_k):
        x = jax.random.normal(jax.random.key(7), (32, 256), dtype=jnp.bfloat16)
        scale = jax.random.uniform(jax.random.key(8), (256, ),
                                   dtype=jnp.bfloat16,
                                   minval=0.5,
                                   maxval=2.0)
        q_out = fused_rmsnorm_fp8.fused_rmsnorm_fp8(x,
                                                    scale,
                                                    eps=1e-6,
                                                    block_k=block_k,
                                                    implementation="xla")
        x_f32 = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = (x_f32 * jax.lax.rsqrt(var + 1e-6)) * scale.astype(
            jnp.float32)
        if block_k is None:
            expected_q = jnp.clip(x_norm / q_out.scale.astype(jnp.float32),
                                  -448.0, 448.0).astype(jnp.float8_e4m3fn)
        else:
            x_blocked = jnp.reshape(x_norm, (32, 256 // block_k, block_k))
            s_blocked = q_out.scale[:, :, None].astype(jnp.float32)
            expected_q = jnp.reshape(
                jnp.clip(x_blocked / s_blocked, -448.0, 448.0),
                (32, 256)).astype(jnp.float8_e4m3fn)
        np.testing.assert_array_equal(
            np.asarray(q_out.qvalue).view(np.uint8),
            np.asarray(expected_q).view(np.uint8),
        )

    def test_ptq_provider_consumes_prequantized_qarray(self):
        layer = nnx.Linear(128,
                           64,
                           use_bias=False,
                           param_dtype=jnp.bfloat16,
                           rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(0), (16, 128), dtype=jnp.bfloat16)
        scale = jax.random.uniform(jax.random.key(1), (128, ),
                                   dtype=jnp.bfloat16,
                                   minval=0.5,
                                   maxval=1.5)
        ptq_layer = qwix_model.quantize_model(layer,
                                              ptq.PtqProvider([_FP8_RULE]), x)

        q_act = fused_rmsnorm_fp8.fused_rmsnorm_fp8(x,
                                                    scale,
                                                    eps=1e-6,
                                                    implementation="xla")
        expected = dot_general.dot_general(q_act, ptq_layer.kernel.array,
                                           (((1, ), (0, )), ((), ())))
        with mock.patch.object(qarray, "quantize",
                               wraps=qarray.quantize) as mock_quantize:
            actual = ptq_layer(q_act)
            mock_quantize.assert_not_called()

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_ptq_provider_requantizes_mismatched_qarray(self):
        """A consumer quantizing activations differently gets requantized input."""
        layer = nnx.Linear(256,
                           64,
                           use_bias=False,
                           param_dtype=jnp.bfloat16,
                           rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(0), (16, 256), dtype=jnp.bfloat16)
        scale = jnp.ones((256, ), dtype=jnp.bfloat16)
        subchannel_rule = dataclasses.replace(_FP8_RULE, tile_size=128)
        ptq_layer = qwix_model.quantize_model(
            layer, ptq.PtqProvider([subchannel_rule]), x)

        q_act = fused_rmsnorm_fp8.fused_rmsnorm_fp8(x,
                                                    scale,
                                                    implementation="xla")
        expected = ptq_layer(qarray.dequantize(q_act))
        with mock.patch.object(qarray, "quantize",
                               wraps=qarray.quantize) as mock_quantize:
            actual = ptq_layer(q_act)
            mock_quantize.assert_called()

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_promote_dtype_hook_only_changes_qarray_operands(self):
        fused_rmsnorm_fp8.enable_qwix_qarray_passthrough()
        provider = ptq.PtqProvider([_FP8_RULE])
        original_promote_dtype = type(provider).promote_dtype.__wrapped__

        f32 = jnp.ones((2, 4), dtype=jnp.float32)
        bf16 = jnp.ones((4, 3), dtype=jnp.bfloat16)
        np_f32 = np.ones((4, ), dtype=np.float32)
        # Without QArray operands, qwix's own promotion runs unchanged.
        for operands, kwargs in (
            ((f32, bf16), {}),
            ((f32, bf16), {
                "dtype": jnp.bfloat16
            }),
            ((np_f32, bf16), {}),
            ((bf16, 2.0, None), {}),
        ):
            expected = original_promote_dtype(provider, operands, **kwargs)
            actual = provider.promote_dtype(operands, **kwargs)
            self.assertLen(actual, len(expected))
            for got, want in zip(actual, expected):
                self.assertIs(type(got), type(want))
                if isinstance(want, jax.Array):
                    self.assertEqual(got.dtype, want.dtype)
                else:
                    self.assertIs(got, want)

        # QArray operands pass through untouched; the rest are still promoted.
        q_act = fused_rmsnorm_fp8.fused_rmsnorm_fp8(jnp.ones(
            (2, 128), dtype=jnp.bfloat16),
                                                    jnp.ones(
                                                        (128, ),
                                                        dtype=jnp.bfloat16),
                                                    implementation="xla")
        bias = jnp.ones((3, ), dtype=jnp.float32)
        promoted = provider.promote_dtype((q_act, None, bias),
                                          dtype=jnp.bfloat16)
        self.assertIs(promoted[0], q_act)
        self.assertIsNone(promoted[1])
        self.assertEqual(promoted[2].dtype, jnp.bfloat16)
        promoted = provider.promote_dtype(q_act, None)
        self.assertIs(promoted[0], q_act)
        self.assertIsNone(promoted[1])

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
        with self.assertRaisesRegex(ValueError, "must be a multiple of 32"):
            fused_rmsnorm_fp8.fused_rmsnorm_fp8_pallas(jnp.ones(
                (64, 128), dtype=jnp.bfloat16),
                                                       scale,
                                                       block_m=16)

    @parameterized.named_parameters(
        {
            "testcase_name": "decode_m8_single_block",
            "m": 8,
            "k": 5120,
            "block_m": 8,
            "block_k": None,
            "has_residual": False,
            "return_residual": False,
        },
        {
            "testcase_name": "decode_m16_single_block",
            "m": 16,
            "k": 5120,
            "block_m": 16,
            "block_k": None,
            "has_residual": False,
            "return_residual": False,
        },
        {
            "testcase_name": "decode_m32_residual",
            "m": 32,
            "k": 5120,
            "block_m": 32,
            "block_k": None,
            "has_residual": True,
            "return_residual": True,
        },
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

        x, scale, residual = _random_inputs(m, k, has_residual)
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

    @parameterized.named_parameters(
        {
            "testcase_name": "decode_m16",
            "m": 16,
            "has_residual": False
        },
        {
            "testcase_name": "prefill_m4096",
            "m": 4096,
            "has_residual": False
        },
        {
            "testcase_name": "prefill_m4096_residual",
            "m": 4096,
            "has_residual": True
        },
    )
    def test_auto_block_selection_matches_reference_on_tpu(
            self, m, has_residual):
        if not _has_tpu():
            self.skipTest("Pallas TPU kernel test requires TPU backend.")

        x, scale, residual = _random_inputs(m, 5120, has_residual)
        with mock.patch.object(fused_rmsnorm_fp8,
                               "fused_rmsnorm_fp8_pallas",
                               wraps=fused_rmsnorm_fp8.fused_rmsnorm_fp8_pallas
                               ) as mock_pallas:
            q_out = jax.jit(
                lambda x_in, s_in, r_in: fused_rmsnorm_fp8.fused_rmsnorm_fp8(
                    x_in, s_in, r_in, return_residual=has_residual))(x, scale,
                                                                     residual)
            mock_pallas.assert_called_once()
        if has_residual:
            q_out = q_out[0]
        ref_q, ref_scale = fused_rmsnorm_fp8.fused_rmsnorm_fp8_reference(
            x, scale, residual)[:2]
        exact_match_ratio = float(
            np.mean(
                np.asarray(ref_q).view(np.uint8) == np.asarray(
                    q_out.qvalue).view(np.uint8)))
        self.assertGreaterEqual(exact_match_ratio, 0.999)
        np.testing.assert_allclose(np.asarray(q_out.scale, dtype=np.float32),
                                   np.asarray(ref_scale, dtype=np.float32),
                                   atol=1e-4)


if __name__ == "__main__":
    absltest.main()
