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

import functools

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.layers.common.quantization import (
    dequantize_tensor, dequantize_tensor_from_mxfp4_packed, quantize_kv,
    quantize_tensor, quantize_tensor_to_mxfp4_packed)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class QuantizationTest(jtu.JaxTestCase):

    @parameterized.product(axis=[-1, 0, (0, 1)])
    def test_mxfp4_quantization(self, axis):
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("mxfp4 is only supported in TPUv7+")

        key = jax.random.key(0)

        shape = (128, 128, 128)
        original = jax.random.normal(key, shape, jnp.bfloat16)

        tensor_q, scale = quantize_tensor_to_mxfp4_packed(original, axis)
        dequantized = dequantize_tensor_from_mxfp4_packed(
            tensor_q, scale, axis)

        self.assertAllClose(dequantized, original, rtol=0.5, atol=0.5)

    @parameterized.product(dtype=[jnp.float8_e4m3fn, jnp.int8],
                           axis=[None, -1, 1, (0, 1)])
    def test_quantization(self, dtype, axis):
        key = jax.random.key(0)

        shape = (128, 128, 128)
        original = jax.random.normal(key, shape, jnp.bfloat16)

        tensor_q, scale = quantize_tensor(dtype, original, axis)
        dequantized = dequantize_tensor(tensor_q, scale, axis)

        self.assertAllClose(dequantized, original, rtol=0.1, atol=0.1)

    @parameterized.product(dtype=[jnp.float8_e4m3fn, jnp.int8],
                           axis=[-1, 1],
                           block_size=[32, 64])
    def test_block_quantization(self, dtype, axis, block_size):
        key = jax.random.key(0)

        shape = (128, 128, 128)
        original = jax.random.normal(key, shape, jnp.bfloat16)

        tensor_q, scale = quantize_tensor(dtype, original, axis, block_size)
        dequantized = dequantize_tensor(tensor_q, scale, axis)

        self.assertAllClose(dequantized, original, rtol=0.1, atol=0.1)

    @parameterized.product(dtype=[jnp.float8_e4m3fn, jnp.int8],
                           axis=[(0, 1), (-1, 0)],
                           block_size=[32, (64, 32)])
    def test_multi_block_quantization(self, dtype, axis, block_size):
        key = jax.random.key(0)

        shape = (128, 128, 128)
        original = jax.random.normal(key, shape, jnp.bfloat16)

        tensor_q, scale = quantize_tensor(dtype, original, axis, block_size)
        dequantized = dequantize_tensor(tensor_q, scale, axis)

        self.assertAllClose(dequantized, original, rtol=0.1, atol=0.1)

    def test_unaligned_block_quantization_raises_error(self):
        key = jax.random.key(0)

        shape = (128, 128)
        tensor = jax.random.normal(key, shape, jnp.bfloat16)
        block_size = 100
        axis = 0

        self.assertRaises(
            ValueError,
            functools.partial(quantize_tensor, jnp.int8, tensor, axis,
                              block_size))

    def test_block_quantization_padding(self):
        key = jax.random.key(0)

        shape = (128, 128)

        original = jax.random.normal(key, shape, jnp.bfloat16)
        block_size = 100
        axis = 0

        tensor_q, scale = quantize_tensor(jnp.int8, original, axis, block_size,
                                          True)

        dequantized = dequantize_tensor(tensor_q, scale, axis)

        padded_size = ((shape[axis] + block_size) // block_size) * block_size
        self.assertEqual(tensor_q.shape[axis], padded_size)
        self.assertTrue((tensor_q[shape[0]:] == 0).all())
        self.assertAllClose(dequantized[:shape[0]],
                            original,
                            rtol=0.1,
                            atol=0.1)

    @parameterized.product(kv_quant_dtype=[jnp.float8_e4m3fn, jnp.int8])
    def test_quantize_kv(self, kv_quant_dtype):
        """Tests the quantize_kv function with float8_e4m3fn dtype."""
        key = jax.random.key(0)

        shape = (128, 128)
        k_original = jax.random.normal(key, shape, jnp.bfloat16)
        v_original = jax.random.normal(key, shape, jnp.bfloat16)
        k_scale = 0.1
        v_scale = 0.2

        k_quantized, v_quantized = quantize_kv(
            kv_quant_dtype,
            k_original,
            v_original,
            k_scale,
            v_scale,
        )

        k_dequantized = k_quantized.astype(jnp.bfloat16) * k_scale
        v_dequantized = v_quantized.astype(jnp.bfloat16) * v_scale

        self.assertAllClose(k_dequantized, k_original, rtol=0.2, atol=0.2)
        self.assertAllClose(v_dequantized, v_original, rtol=0.2, atol=0.2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
