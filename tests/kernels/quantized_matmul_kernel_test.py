# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.quantized_matmul import (kernel, tuned_block_sizes,
                                                    util)

xla_quantized_matmul = kernel.xla_quantized_matmul
quantized_matmul_kernel = kernel.quantized_matmul_kernel
quantize_tensor = util.quantize_tensor
get_tuned_block_sizes = tuned_block_sizes.get_tuned_block_sizes

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class QuantizedMatmulKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(6):
            self.skipTest("Expect TPUv6+")

    def _test_quantized_matmul(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quantize_activation: bool,
        tuned_value=None,
        atol=0.5,
        rtol=0.5,
    ):

        prng_key = jax.random.key(1234)
        k0, k1 = jax.random.split(prng_key, 2)
        x = jax.random.uniform(k0, (bs, n_input_features),
                               dtype=dtype,
                               minval=0,
                               maxval=1)
        w = jax.random.uniform(
            k1,
            (n_output_features, n_input_features),
            dtype=dtype,
            minval=-1,
            maxval=1,
        )
        w_q, w_scale = quantize_tensor(w, q_dtype)
        w_scale = jnp.squeeze(w_scale)
        assert w_scale.shape == (n_output_features, )

        x_q_dtype = w_q.dtype if quantize_activation else dtype
        output = quantized_matmul_kernel(
            x,
            w_q,
            w_scale,
            x_q_dtype=x_q_dtype,
            tuned_value=tuned_value,
        )
        expected = xla_quantized_matmul(
            x, w_q, w_scale, quantize_activation=quantize_activation)

        self.assertAllClose(output,
                            expected,
                            rtol=rtol,
                            atol=atol,
                            check_dtypes=True)

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        q_dtype=[jnp.int8, jnp.float8_e4m3fn],
        bs=[128, 256, 512],
        n_input_features=[128, 256, 512],
        n_output_features=[128, 256, 512],
        quantize_activation=[True],
    )
    def test_quantized_matmul_various_input_shapes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quantize_activation: bool,
    ):
        self._test_quantized_matmul(
            dtype,
            q_dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            tuned_value=None,
        )

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        q_dtype=[jnp.int8, jnp.float8_e4m3fn],
        bs=[64, 192],
        n_input_features=[64, 192],
        n_output_features=[64, 192],
        quantize_activation=[True],
    )
    def test_quantized_matmul_unaligned_input_shapes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quantize_activation: bool,
    ):
        self._test_quantized_matmul(
            dtype,
            q_dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            tuned_value=None,
        )

    @parameterized.parameters(
        (jnp.bfloat16, jnp.int8, 128, 1280, 8192, True),
        (jnp.bfloat16, jnp.int8, 128, 28672, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 4096, 14336, True),
        (jnp.bfloat16, jnp.int8, 128, 4096, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 6144, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 7168, 8192, True),
        (jnp.bfloat16, jnp.int8, 128, 8192, 1024, True),
        (jnp.bfloat16, jnp.int8, 128, 8192, 3584, True),
    )
    def test_quantized_matmul_use_tuned_block_sizes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quantize_activation: bool,
    ):
        self._test_quantized_matmul(
            dtype,
            q_dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            tuned_value=None,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
